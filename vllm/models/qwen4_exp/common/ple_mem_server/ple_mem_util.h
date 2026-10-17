// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Small helpers shared by the PLE memory server, its benchmark and the PLE
// row reader: byte-size parsing / formatting and address resolution.

#pragma once

#include <netdb.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ple_mem {

// Parses "123", "1.5G", "64MiB", "4k" (binary multiples) into bytes.
inline bool parse_size(const char* s, uint64_t* out) {
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(s, &end);
  if (errno != 0 || end == s || v < 0) return false;
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
  *out = static_cast<uint64_t>(v * mult);
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

// Resolves "host:port" or "[ipv6]:port". Returns false with `err` set.
inline bool resolve_address(const std::string& s, sockaddr_storage* out,
                            socklen_t* len, std::string* err) {
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
  addrinfo* res = nullptr;
  const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if (rc != 0 || res == nullptr) {
    *err = "cannot resolve " + s + ": " + gai_strerror(rc);
    return false;
  }
  std::memcpy(out, res->ai_addr, res->ai_addrlen);
  *len = res->ai_addrlen;
  freeaddrinfo(res);
  return true;
}

}  // namespace ple_mem
