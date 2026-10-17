// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Helpers shared by the PLE memory server, its benchmark and the PLE row
// reader: byte sizes, address resolution, the shared-secret token and the
// UCX transport policy.

#pragma once

#include <netdb.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ple_mem {

// Parses "123", "1.5G", "64MiB", "4k" (binary multiples) into bytes; rejects
// negative, non-finite and >= 2^64 values.
inline bool parse_size(const char* s, uint64_t* out) {
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(s, &end);
  if (errno != 0 || end == s || !std::isfinite(v) || v < 0) return false;
  double mult = 1;
  switch (*end) {
    case 'k':
    case 'K':
      mult = double(1ull << 10);
      break;
    case 'm':
    case 'M':
      mult = double(1ull << 20);
      break;
    case 'g':
    case 'G':
      mult = double(1ull << 30);
      break;
    case 't':
    case 'T':
      mult = double(1ull << 40);
      break;
    default:
      break;
  }
  if (mult != 1) ++end;
  if (*end == 'i' || *end == 'I') ++end;
  if (*end == 'b' || *end == 'B') ++end;
  if (*end != '\0') return false;
  const double bytes = v * mult;
  if (!(bytes < 18446744073709551616.0)) return false;
  *out = static_cast<uint64_t>(bytes);
  return true;
}

inline std::string human_bytes(uint64_t bytes) {
  static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(bytes);
  int u = 0;
  while (v >= 1024 && u < 4) {
    v /= 1024;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), u ? "%.2f %s" : "%.0f %s", v, units[u]);
  return buf;
}

// Resolves "host:port" or "[ipv6]:port" into every address getaddrinfo
// returns, in order. Returns false with `err` set.
inline bool resolve_addresses(const std::string& s, bool passive,
                              std::vector<sockaddr_storage>* out,
                              std::vector<socklen_t>* lens, std::string* err) {
  std::string host, port;
  if (!s.empty() && s[0] == '[') {
    const size_t close = s.find(']');
    if (close == std::string::npos || close + 1 >= s.size() ||
        s[close + 1] != ':') {
      *err = "invalid address " + s;
      return false;
    }
    host = s.substr(1, close - 1);
    port = s.substr(close + 2);
  } else {
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos) {
      *err = "address " + s + " has no port";
      return false;
    }
    host = s.substr(0, colon);
    port = s.substr(colon + 1);
  }
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_NUMERICSERV | (passive ? AI_PASSIVE : 0);
  addrinfo* res = nullptr;
  const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if (rc != 0 || res == nullptr) {
    *err = "cannot resolve " + s + ": " + gai_strerror(rc);
    return false;
  }
  for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
    if (a->ai_addrlen > sizeof(sockaddr_storage)) continue;
    sockaddr_storage ss{};
    std::memcpy(&ss, a->ai_addr, a->ai_addrlen);
    out->push_back(ss);
    lens->push_back(a->ai_addrlen);
  }
  freeaddrinfo(res);
  if (out->empty()) *err = "no usable address for " + s;
  return !out->empty();
}

// Reads a shared-secret token: the file content without surrounding
// whitespace, 1 to kMaxTokenBytes bytes.
inline bool read_token_file(const std::string& path, size_t max_bytes,
                            std::string* token, std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *err = "cannot read token file " + path + ": " + std::strerror(errno);
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string t = ss.str();
  const char* ws = " \t\r\n";
  const size_t b = t.find_first_not_of(ws);
  const size_t e = t.find_last_not_of(ws);
  t = b == std::string::npos ? "" : t.substr(b, e - b + 1);
  if (t.empty() || t.size() > max_bytes) {
    *err = "token file " + path + " must hold 1 to " +
           std::to_string(max_bytes) + " bytes";
    return false;
  }
  *token = std::move(t);
  return true;
}

inline bool constant_time_equal(const std::string& a, const char* b,
                                size_t b_len) {
  unsigned char diff = a.size() == b_len ? 0 : 1;
  const size_t n = a.size() < b_len ? a.size() : b_len;
  for (size_t i = 0; i < n; ++i)
    diff |= static_cast<unsigned char>(a[i] ^ b[i]);
  return diff == 0;
}

// Reads a decimal byte count such as a cgroup memory file; false for "max".
inline bool read_u64_file(const char* path, uint64_t* out) {
  std::ifstream f(path);
  std::string s;
  if (!(f >> s)) return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

// Bytes this process's cgroup (v2, else v1) memory limit still allows, or
// UINT64_MAX without a limit. /proc/meminfo ignores container limits.
inline uint64_t cgroup_memory_headroom() {
  static const char* const kFiles[][2] = {
      {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"},
      {"/sys/fs/cgroup/memory/memory.limit_in_bytes",
       "/sys/fs/cgroup/memory/memory.usage_in_bytes"},
  };
  for (const auto& files : kFiles) {
    uint64_t limit = 0, usage = 0;
    if (!read_u64_file(files[0], &limit)) continue;
    if (!read_u64_file(files[1], &usage)) usage = 0;
    return limit > usage ? limit - usage : 0;
  }
  return UINT64_MAX;
}

// MemAvailable from /proc/meminfo in bytes, capped by the cgroup memory
// limit's headroom; 0 if unknown.
inline uint64_t mem_available() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  uint64_t kib = 0;
  std::string unit;
  while (f >> key >> kib >> unit)
    if (key == "MemAvailable:")
      return std::min(kib * 1024, cgroup_memory_headroom());
  return 0;
}

// The effective value of `name` (e.g. "TLS") in a UCP configuration.
inline std::string ucp_config_value(const ucp_config_t* config,
                                    const char* name) {
  char* buf = nullptr;
  size_t len = 0;
  FILE* f = open_memstream(&buf, &len);
  if (f == nullptr) return "";
  ucp_config_print(config, f, nullptr, UCS_CONFIG_PRINT_CONFIG);
  std::fclose(f);
  std::string value;
  std::istringstream lines(std::string(buf, len));
  std::free(buf);
  const std::string key = std::string("UCX_") + name + "=";
  for (std::string line; std::getline(lines, line);) {
    if (line.compare(0, key.size(), key) == 0) {
      value = line.substr(key.size());
      break;
    }
  }
  return value;
}

// UCX's tcp transport writes incoming PUT requests to the raw address the
// peer sends (uct_tcp_ep_handle_put_req), so a process whose context opens it
// can be written by any peer that reaches it. Removes tcp from the transport
// list of `config` (honoring the user's UCX_TLS) unless `allow_tcp`.
inline bool exclude_tcp_transport(ucp_config_t* config, bool allow_tcp,
                                  std::string* err) {
  if (allow_tcp) return true;
  std::string tls = ucp_config_value(config, "TLS");
  std::vector<std::string> items;
  bool negate = !tls.empty() && tls[0] == '^';
  std::stringstream ss(negate ? tls.substr(1) : tls);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) items.push_back(item);
  auto base = [](const std::string& tl) { return tl.substr(0, tl.find(':')); };
  bool all = items.empty();
  for (const auto& it : items) all |= !negate && it == "all";
  std::string value;
  if (all) {
    value = "^tcp";
  } else if (negate) {
    value = tls;
    bool listed = false;
    for (const auto& it : items) listed |= base(it) == "tcp";
    if (!listed) value += ",tcp";
  } else {
    for (const auto& it : items)
      if (base(it) != "tcp") value += (value.empty() ? "" : ",") + it;
    if (value.empty()) {
      *err = "UCX_TLS=" + tls +
             " selects only the tcp transport, which is refused: UCX's tcp "
             "transport lets any peer that reaches this process write its "
             "memory. Use RDMA transports, or explicitly allow tcp on a "
             "trusted network (--allow-tcp / allow_insecure_transport)";
      return false;
    }
  }
  const ucs_status_t st = ucp_config_modify(config, "TLS", value.c_str());
  if (st != UCS_OK) {
    *err = "cannot set UCX_TLS=" + value + ": " + ucs_status_string(st);
    return false;
  }
  return true;
}

// Sets `*tcp` when a transport lane of `ep` is UCX's tcp transport (the
// connection-manager lane has an empty device name and is not one).
inline ucs_status_t ep_uses_tcp(ucp_ep_h ep, bool* tcp) {
  ucp_transport_entry_t entries[16];
  ucp_ep_attr_t attr{};
  attr.field_mask = UCP_EP_ATTR_FIELD_TRANSPORTS;
  attr.transports.entries = entries;
  attr.transports.num_entries = 16;
  attr.transports.entry_size = sizeof(ucp_transport_entry_t);
  const ucs_status_t st = ucp_ep_query(ep, &attr);
  if (st != UCS_OK) return st;
  *tcp = false;
  for (unsigned i = 0; i < attr.transports.num_entries; ++i) {
    const ucp_transport_entry_t& e = entries[i];
    if (e.transport_name != nullptr && e.device_name != nullptr &&
        e.device_name[0] != '\0' && std::strcmp(e.transport_name, "tcp") == 0)
      *tcp = true;
  }
  return UCS_OK;
}

}  // namespace ple_mem
