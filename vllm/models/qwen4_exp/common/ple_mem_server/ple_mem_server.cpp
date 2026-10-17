// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PLE memory server: lends registered RAM to PLE row readers over UCX.
//
// Clients connect with a sockaddr endpoint, borrow regions with ALLOC_REQ,
// fill and read them with one-sided RMA, and release them with FREE_REQ or by
// disconnecting. The server knows nothing about models and needs only UCX;
// build it with the Makefile next to this file (see README.md).

#include <ucp/api/ucp.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "ple_mem_protocol.h"
#include "ple_mem_util.h"

namespace {

volatile sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

void log_msg(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "ple_mem_server: ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

std::string sockaddr_str(const sockaddr_storage& ss) {
  char host[INET6_ADDRSTRLEN] = "?";
  uint16_t port = 0;
  if (ss.ss_family == AF_INET) {
    const auto* a = reinterpret_cast<const sockaddr_in*>(&ss);
    inet_ntop(AF_INET, &a->sin_addr, host, sizeof(host));
    port = ntohs(a->sin_port);
    return std::string(host) + ":" + std::to_string(port);
  }
  if (ss.ss_family == AF_INET6) {
    const auto* a = reinterpret_cast<const sockaddr_in6*>(&ss);
    inet_ntop(AF_INET6, &a->sin6_addr, host, sizeof(host));
    port = ntohs(a->sin6_port);
    return "[" + std::string(host) + "]:" + std::to_string(port);
  }
  return "?";
}

struct Options {
  std::string host = "0.0.0.0";
  std::string port = "18515";
  uint64_t max_memory = 0;
  bool busy_poll = false;
  bool hugepages = false;
  bool verbose = false;
};

void usage(const char* prog) {
  std::fprintf(
      stderr,
      "usage: %s [--host ADDR] [--port PORT] [--max-memory BYTES]\n"
      "          [--busy-poll] [--hugepages] [--verbose]\n"
      "  --host        listen address (default 0.0.0.0)\n"
      "  --port        listen port (default 18515, 0 picks a free port)\n"
      "  --max-memory  cap on lent memory, K/M/G/T suffixes (0 = no limit)\n"
      "  --busy-poll   spin instead of sleeping on UCX wakeup events\n"
      "  --hugepages   madvise(MADV_HUGEPAGE) on lent regions\n"
      "  --verbose     log every request\n",
      prog);
}

bool parse_args(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char** v) {
      if (i + 1 >= argc) return false;
      *v = argv[++i];
      return true;
    };
    const char* v = nullptr;
    if (a == "--host" && value(&v)) {
      o->host = v;
    } else if (a == "--port" && value(&v)) {
      o->port = v;
    } else if (a == "--max-memory" && value(&v)) {
      if (!ple_mem::parse_size(v, &o->max_memory)) {
        log_msg("invalid --max-memory %s", v);
        return false;
      }
    } else if (a == "--busy-poll") {
      o->busy_poll = true;
    } else if (a == "--hugepages") {
      o->hugepages = true;
    } else if (a == "--verbose") {
      o->verbose = true;
    } else {
      if (a != "-h" && a != "--help") log_msg("unknown argument %s", a.c_str());
      return false;
    }
  }
  return true;
}

struct Region {
  uint64_t id;
  uint64_t conn;
  void* addr;
  size_t bytes;
  size_t mapped;
  ucp_mem_h memh;
};

struct Conn {
  uint64_t id;
  std::string peer;
  uint64_t regions = 0;
  uint64_t bytes = 0;
};

struct Event {
  enum Kind { kConnect, kMessage, kEpError } kind;
  ucp_conn_request_h conn_request = nullptr;
  ucp_ep_h ep = nullptr;
  ucs_status_t status = UCS_OK;
  std::vector<char> payload;
};

struct PendingSend {
  void* request;
  std::vector<char> payload;
};

struct Closing {
  void* request;
  uint64_t conn;
};

class Server {
 public:
  explicit Server(const Options& o) : opt_(o) {}

  int run() {
    if (!init()) {
      shutdown();
      return 1;
    }
    loop();
    shutdown();
    return 0;
  }

 private:
  bool init() {
    ucp_config_t* config = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) return fail("ucp_config_read", st);
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_AM;
    if (!opt_.busy_poll) params.features |= UCP_FEATURE_WAKEUP;
    st = ucp_init(&params, config, &ctx_);
    ucp_config_release(config);
    if (st != UCS_OK) return fail("ucp_init", st);

    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(ctx_, &wp, &worker_);
    if (st != UCS_OK) return fail("ucp_worker_create", st);
    if (!opt_.busy_poll) {
      st = ucp_worker_get_efd(worker_, &efd_);
      if (st != UCS_OK) return fail("ucp_worker_get_efd", st);
    }

    ucp_am_handler_param_t am{};
    am.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am.id = ple_mem::kAmRequestId;
    am.flags = UCP_AM_FLAG_WHOLE_MSG;
    am.cb = &Server::am_cb;
    am.arg = this;
    st = ucp_worker_set_am_recv_handler(worker_, &am);
    if (st != UCS_OK) return fail("ucp_worker_set_am_recv_handler", st);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo* res = nullptr;
    const char* host = opt_.host.empty() ? nullptr : opt_.host.c_str();
    const int rc = getaddrinfo(host, opt_.port.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
      log_msg("cannot resolve %s:%s: %s", opt_.host.c_str(), opt_.port.c_str(),
              gai_strerror(rc));
      return false;
    }
    ucp_listener_params_t lp{};
    lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                    UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lp.sockaddr.addr = res->ai_addr;
    lp.sockaddr.addrlen = res->ai_addrlen;
    lp.conn_handler.cb = &Server::conn_cb;
    lp.conn_handler.arg = this;
    st = ucp_listener_create(worker_, &lp, &listener_);
    freeaddrinfo(res);
    if (st != UCS_OK) return fail("ucp_listener_create", st);

    ucp_listener_attr_t la{};
    la.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    st = ucp_listener_query(listener_, &la);
    if (st != UCS_OK) return fail("ucp_listener_query", st);
    log_msg("listening on %s (max memory %s, %s)",
            sockaddr_str(la.sockaddr).c_str(),
            opt_.max_memory ? ple_mem::human_bytes(opt_.max_memory).c_str()
                            : "unlimited",
            opt_.busy_poll ? "busy poll" : "event wait");
    return true;
  }

  bool fail(const char* what, ucs_status_t st) {
    log_msg("%s failed: %s", what, ucs_status_string(st));
    return false;
  }

  void loop() {
    while (!g_stop) {
      while (ucp_worker_progress(worker_) != 0) {
      }
      const bool did_work = !events_.empty();
      handle_events();
      reap_requests();
      if (did_work || opt_.busy_poll) continue;
      const ucs_status_t st = ucp_worker_arm(worker_);
      if (st == UCS_ERR_BUSY) continue;
      if (st != UCS_OK) {
        log_msg("ucp_worker_arm failed: %s", ucs_status_string(st));
        break;
      }
      pollfd pfd{efd_, POLLIN, 0};
      const bool waiting = !sends_.empty() || !closing_.empty();
      poll(&pfd, 1, waiting ? 1 : 100);
    }
  }

  static void conn_cb(ucp_conn_request_h req, void* arg) {
    Event e;
    e.kind = Event::kConnect;
    e.conn_request = req;
    static_cast<Server*>(arg)->events_.push_back(std::move(e));
  }

  static void ep_err_cb(void* arg, ucp_ep_h ep, ucs_status_t status) {
    Event e;
    e.kind = Event::kEpError;
    e.ep = ep;
    e.status = status;
    static_cast<Server*>(arg)->events_.push_back(std::move(e));
  }

  static ucs_status_t am_cb(void* arg, const void*, size_t, void* data,
                            size_t length, const ucp_am_recv_param_t* param) {
    auto* self = static_cast<Server*>(arg);
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      log_msg("dropping a rendezvous request (%zu bytes)", length);
      return UCS_OK;
    }
    Event e;
    e.kind = Event::kMessage;
    if (param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP)
      e.ep = param->reply_ep;
    const char* p = static_cast<const char*>(data);
    e.payload.assign(p, p + length);
    self->events_.push_back(std::move(e));
    return UCS_OK;
  }

  void handle_events() {
    while (!events_.empty()) {
      Event e = std::move(events_.front());
      events_.pop_front();
      switch (e.kind) {
        case Event::kConnect:
          accept(e.conn_request);
          break;
        case Event::kEpError:
          drop_conn(e.ep, e.status);
          break;
        case Event::kMessage:
          handle_message(e.ep, e.payload);
          break;
      }
    }
  }

  void accept(ucp_conn_request_h req) {
    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    std::string peer = "?";
    if (ucp_conn_request_query(req, &attr) == UCS_OK)
      peer = sockaddr_str(attr.client_address);
    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep.conn_request = req;
    ep.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep.err_handler.cb = &Server::ep_err_cb;
    ep.err_handler.arg = this;
    ucp_ep_h h = nullptr;
    const ucs_status_t st = ucp_ep_create(worker_, &ep, &h);
    if (st != UCS_OK) {
      log_msg("cannot accept %s: %s", peer.c_str(), ucs_status_string(st));
      return;
    }
    Conn c;
    c.id = next_conn_++;
    c.peer = peer;
    log_msg("client %" PRIu64 " connected from %s", c.id, peer.c_str());
    conns_[h] = std::move(c);
  }

  void drop_conn(ucp_ep_h ep, ucs_status_t status) {
    auto it = conns_.find(ep);
    if (it == conns_.end()) return;
    const Conn c = std::move(it->second);
    conns_.erase(it);
    log_msg("client %" PRIu64 " (%s) disconnected (%s); releasing %" PRIu64
            " regions, %s",
            c.id, c.peer.c_str(), ucs_status_string(status), c.regions,
            ple_mem::human_bytes(c.bytes).c_str());
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_EP_CLOSE_FLAG_FORCE;
    void* req = ucp_ep_close_nbx(ep, &p);
    if (UCS_PTR_IS_PTR(req)) {
      closing_.push_back(Closing{req, c.id});
    } else {
      release_conn_regions(c.id);
    }
  }

  void release_conn_regions(uint64_t conn) {
    for (auto it = regions_.begin(); it != regions_.end();) {
      if (it->second.conn == conn) {
        free_region(it->second);
        it = regions_.erase(it);
      } else {
        ++it;
      }
    }
    log_totals();
  }

  void handle_message(ucp_ep_h ep, const std::vector<char>& m) {
    const uint16_t type = ple_mem::message_type(m.data(), m.size());
    auto it = ep == nullptr ? conns_.end() : conns_.find(ep);
    if (it == conns_.end()) {
      log_msg("dropping a message from an unknown endpoint");
      return;
    }
    Conn& c = it->second;
    if (type == ple_mem::kAllocReq && m.size() >= sizeof(ple_mem::AllocReq)) {
      ple_mem::AllocReq req;
      std::memcpy(&req, m.data(), sizeof(req));
      alloc(ep, c, req);
    } else if (type == ple_mem::kFreeReq &&
               m.size() >= sizeof(ple_mem::FreeReq)) {
      ple_mem::FreeReq req;
      std::memcpy(&req, m.data(), sizeof(req));
      auto r = regions_.find(req.region_id);
      if (r == regions_.end() || r->second.conn != c.id) {
        log_msg("client %" PRIu64 " freed unknown region %" PRIu64, c.id,
                req.region_id);
        return;
      }
      c.regions -= 1;
      c.bytes -= r->second.bytes;
      if (opt_.verbose)
        log_msg("client %" PRIu64 " freed region %" PRIu64 " (%zu bytes)", c.id,
                req.region_id, r->second.bytes);
      free_region(r->second);
      regions_.erase(r);
    } else {
      log_msg("client %" PRIu64 " sent an invalid message (%zu bytes)", c.id,
              m.size());
    }
  }

  void alloc(ucp_ep_h ep, Conn& c, const ple_mem::AllocReq& req) {
    ple_mem::AllocRep rep{};
    rep.hdr = ple_mem::make_header(ple_mem::kAllocRep);
    rep.cookie = req.cookie;
    void* rkey = nullptr;
    size_t rkey_len = 0;
    std::string err = try_alloc(req.bytes, c.id, &rep, &rkey, &rkey_len);
    if (err.empty()) {
      c.regions += 1;
      c.bytes += req.bytes;
      log_msg("client %" PRIu64 " borrowed region %" PRIu64 " (%" PRIu64
              " bytes); client total %s",
              c.id, rep.region_id, req.bytes,
              ple_mem::human_bytes(c.bytes).c_str());
    } else {
      rep.status = -1;
      std::snprintf(rep.error, sizeof(rep.error), "%s", err.c_str());
      log_msg("client %" PRIu64 " allocation of %" PRIu64 " bytes failed: %s",
              c.id, req.bytes, err.c_str());
    }
    rep.rkey_len = static_cast<uint32_t>(rkey_len);
    std::vector<char> payload(sizeof(rep) + rkey_len);
    std::memcpy(payload.data(), &rep, sizeof(rep));
    if (rkey_len) std::memcpy(payload.data() + sizeof(rep), rkey, rkey_len);
    if (rkey != nullptr) ucp_rkey_buffer_release(rkey);
    send(ep, std::move(payload));
  }

  std::string try_alloc(uint64_t bytes, uint64_t conn, ple_mem::AllocRep* rep,
                        void** rkey, size_t* rkey_len) {
    if (bytes == 0) return "zero-byte region";
    if (opt_.max_memory != 0 && lent_ + bytes > opt_.max_memory)
      return "exceeds --max-memory (" + ple_mem::human_bytes(lent_) +
             " lent of " + ple_mem::human_bytes(opt_.max_memory) + ")";
    const size_t page = opt_.hugepages ? (size_t{2} << 20) : 4096;
    const size_t mapped = (bytes + page - 1) / page * page;
    void* addr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (addr == MAP_FAILED)
      return std::string("mmap failed: ") + std::strerror(errno);
    if (opt_.hugepages) madvise(addr, mapped, MADV_HUGEPAGE);
    std::string err = prefault(addr, mapped);
    if (!err.empty()) {
      munmap(addr, mapped);
      return err;
    }
    ucp_mem_map_params_t mp{};
    mp.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = addr;
    mp.length = mapped;
    ucp_mem_h memh = nullptr;
    ucs_status_t st = ucp_mem_map(ctx_, &mp, &memh);
    if (st != UCS_OK) {
      munmap(addr, mapped);
      return std::string("ucp_mem_map failed: ") + ucs_status_string(st);
    }
    // ucp_memh_pack() only packs exported handles (UCX 1.22), so the rkey
    // goes through the compat packer.
    st = ucp_rkey_pack(ctx_, memh, rkey, rkey_len);
    if (st != UCS_OK) {
      ucp_mem_unmap(ctx_, memh);
      munmap(addr, mapped);
      return std::string("ucp_rkey_pack failed: ") + ucs_status_string(st);
    }
    Region r{next_region_++, conn, addr, bytes, mapped, memh};
    regions_[r.id] = r;
    lent_ += bytes;
    rep->region_id = r.id;
    rep->addr = reinterpret_cast<uint64_t>(addr);
    rep->bytes = bytes;
    return "";
  }

  // Populates the pages so a shortage fails here instead of at access time.
  static std::string prefault(void* addr, size_t len) {
#ifndef MADV_POPULATE_WRITE
  #define MADV_POPULATE_WRITE 23
#endif
    if (madvise(addr, len, MADV_POPULATE_WRITE) == 0) return "";
    if (errno != EINVAL)
      return std::string("cannot populate memory: ") + std::strerror(errno);
    auto* p = static_cast<volatile char*>(addr);
    for (size_t off = 0; off < len; off += 4096) p[off] = 0;
    return "";
  }

  void free_region(const Region& r) {
    ucp_mem_unmap(ctx_, r.memh);
    munmap(r.addr, r.mapped);
    lent_ -= r.bytes;
  }

  void send(ucp_ep_h ep, std::vector<char> payload) {
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER;
    void* req = ucp_am_send_nbx(ep, ple_mem::kAmReplyId, nullptr, 0,
                                payload.data(), payload.size(), &p);
    if (UCS_PTR_IS_ERR(req)) {
      log_msg("reply send failed: %s", ucs_status_string(UCS_PTR_STATUS(req)));
    } else if (req != nullptr) {
      sends_.push_back(PendingSend{req, std::move(payload)});
    }
  }

  void reap_requests() {
    for (auto it = sends_.begin(); it != sends_.end();) {
      const ucs_status_t st = ucp_request_check_status(it->request);
      if (st == UCS_INPROGRESS) {
        ++it;
        continue;
      }
      if (st != UCS_OK) log_msg("reply send failed: %s", ucs_status_string(st));
      ucp_request_free(it->request);
      it = sends_.erase(it);
    }
    for (auto it = closing_.begin(); it != closing_.end();) {
      if (ucp_request_check_status(it->request) == UCS_INPROGRESS) {
        ++it;
        continue;
      }
      ucp_request_free(it->request);
      release_conn_regions(it->conn);
      it = closing_.erase(it);
    }
  }

  void log_totals() {
    log_msg("%zu clients, %zu regions, %s lent", conns_.size(), regions_.size(),
            ple_mem::human_bytes(lent_).c_str());
  }

  void shutdown() {
    if (worker_ != nullptr) {
      if (!conns_.empty())
        log_msg("shutting down, closing %zu clients", conns_.size());
      std::vector<ucp_ep_h> eps;
      for (auto& kv : conns_) eps.push_back(kv.first);
      for (ucp_ep_h ep : eps) drop_conn(ep, UCS_ERR_CANCELED);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while ((!closing_.empty() || !sends_.empty()) &&
             std::chrono::steady_clock::now() < deadline) {
        ucp_worker_progress(worker_);
        reap_requests();
      }
      for (auto& kv : regions_) free_region(kv.second);
      regions_.clear();
      if (listener_ != nullptr) ucp_listener_destroy(listener_);
      ucp_worker_destroy(worker_);
    }
    if (ctx_ != nullptr) ucp_cleanup(ctx_);
    log_msg("stopped");
  }

  Options opt_;
  ucp_context_h ctx_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  ucp_listener_h listener_ = nullptr;
  int efd_ = -1;
  std::deque<Event> events_;
  std::unordered_map<ucp_ep_h, Conn> conns_;
  std::unordered_map<uint64_t, Region> regions_;
  std::vector<PendingSend> sends_;
  std::vector<Closing> closing_;
  uint64_t next_conn_ = 1;
  uint64_t next_region_ = 1;
  uint64_t lent_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    usage(argv[0]);
    return 2;
  }
  struct sigaction sa{};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);
  Server server(opt);
  return server.run();
}
