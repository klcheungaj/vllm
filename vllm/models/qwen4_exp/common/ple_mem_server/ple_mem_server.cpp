// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PLE memory server: lends RAM to PLE row readers over UCX.
//
// Clients connect with a sockaddr endpoint and talk the request / response
// protocol of ple_mem_protocol.h: they borrow regions (ALLOC_REQ), fill them
// (WRITE_REQ), read rows back in batches (READ_REQ) and release them
// (FREE_REQ, or by disconnecting). Regions are plain anonymous memory, never
// registered with UCX, and only this server's handlers touch them after
// checking that the region belongs to the requesting connection and that the
// access is in bounds. The server only answers requests; it never initiates
// a transfer to a client. The context has active messages only (no RMA), and
// UCX's tcp transport is excluded unless --allow-tcp. Build with the Makefile
// next to this file (see README.md).

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
  if (ss.ss_family == AF_INET) {
    const auto* a = reinterpret_cast<const sockaddr_in*>(&ss);
    inet_ntop(AF_INET, &a->sin_addr, host, sizeof(host));
    return std::string(host) + ":" + std::to_string(ntohs(a->sin_port));
  }
  if (ss.ss_family == AF_INET6) {
    const auto* a = reinterpret_cast<const sockaddr_in6*>(&ss);
    inet_ntop(AF_INET6, &a->sin6_addr, host, sizeof(host));
    return "[" + std::string(host) + "]:" + std::to_string(ntohs(a->sin6_port));
  }
  return "?";
}

enum class Fault { kNone, kShortReply, kUnsolicited };

struct Options {
  std::string host;
  std::string port = "18515";
  uint64_t max_memory = 0;  // 0 = auto
  uint64_t max_regions = 16384;
  uint64_t max_client_regions = 4096;
  bool event_wait = false;
  bool hugepages = false;
  bool verbose = false;
  bool allow_tcp = false;
  std::string token_file;
  Fault fault = Fault::kNone;
};

void usage(const char* prog) {
  std::fprintf(
      stderr,
      "usage: %s --host ADDR [options]\n"
      "  --host ADDR            listen address, required: the address of the\n"
      "                         RDMA / storage interface the vLLM hosts reach\n"
      "                         (0.0.0.0 listens on every interface)\n"
      "  --port PORT            listen port (default 18515, 0 picks one)\n"
      "  --max-memory BYTES     cap on lent memory (mapped bytes), K/M/G/T\n"
      "                         suffixes, or auto = 90%% of MemAvailable at\n"
      "                         startup (default auto)\n"
      "  --max-regions N        regions lent in total (default 16384)\n"
      "  --max-client-regions N regions per client (default 4096)\n"
      "  --token-file PATH      require this shared secret from clients\n"
      "  --allow-tcp            allow UCX's tcp transport (insecure: it lets\n"
      "                         any peer write this process's memory)\n"
      "  --event-wait           sleep on UCX wakeup events between requests\n"
      "                         (saves a core, adds wakeup latency; default\n"
      "                         busy-polls)\n"
      "  --hugepages            madvise(MADV_HUGEPAGE) on lent regions\n"
      "  --verbose              log every allocation and release\n"
      "  --inject-fault MODE    testing only: short-reply | unsolicited\n",
      prog);
}

bool parse_count(const char* v, uint64_t* out) {
  char* end = nullptr;
  errno = 0;
  const unsigned long long n = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || *end != '\0' || n == 0) return false;
  *out = n;
  return true;
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
      if (std::string(v) == "auto") {
        o->max_memory = 0;
      } else if (!ple_mem::parse_size(v, &o->max_memory) ||
                 o->max_memory == 0) {
        log_msg("invalid --max-memory %s (a size > 0 or auto)", v);
        return false;
      }
    } else if (a == "--max-regions" && value(&v)) {
      if (!parse_count(v, &o->max_regions)) {
        log_msg("invalid --max-regions %s", v);
        return false;
      }
    } else if (a == "--max-client-regions" && value(&v)) {
      if (!parse_count(v, &o->max_client_regions)) {
        log_msg("invalid --max-client-regions %s", v);
        return false;
      }
    } else if (a == "--token-file" && value(&v)) {
      o->token_file = v;
    } else if (a == "--allow-tcp") {
      o->allow_tcp = true;
    } else if (a == "--event-wait") {
      o->event_wait = true;
    } else if (a == "--hugepages") {
      o->hugepages = true;
    } else if (a == "--verbose") {
      o->verbose = true;
    } else if (a == "--inject-fault" && value(&v)) {
      const std::string m = v;
      if (m == "short-reply") {
        o->fault = Fault::kShortReply;
      } else if (m == "unsolicited") {
        o->fault = Fault::kUnsolicited;
      } else {
        log_msg("invalid --inject-fault %s", v);
        return false;
      }
    } else {
      if (a != "-h" && a != "--help") log_msg("invalid argument %s", a.c_str());
      return false;
    }
  }
  if (o->host.empty()) {
    log_msg(
        "--host is required: pass the address of the RDMA / storage network "
        "interface the vLLM hosts connect to (e.g. --host 10.0.0.2), or "
        "0.0.0.0 to listen on every interface");
    return false;
  }
  return true;
}

struct Region {
  uint64_t id;
  uint64_t conn;
  uint8_t* addr;
  uint64_t bytes;
  uint64_t mapped;
  uint32_t receives = 0;  // rendezvous writes landing in the region
  bool released = false;  // freed while receives were in flight
};

struct Conn {
  uint64_t id;
  std::string peer;
  uint64_t regions = 0;
  uint64_t mapped = 0;
  bool authed = false;
  bool checked = false;  // transport lanes verified
  bool closing = false;
};

struct Event {
  enum Kind { kConnect, kEpError, kDrop } kind;
  ucp_conn_request_h conn_request = nullptr;
  ucp_ep_h ep = nullptr;
  ucs_status_t status = UCS_OK;
  bool graceful = false;
};

// A reply waiting to be sent after the progress call that produced it.
struct Reply {
  ucp_ep_h ep;
  unsigned char header[48];
  size_t header_length;
  int buffer;  // index into the buffer pool, -1 for none
  size_t data_length;
};

// A rendezvous WRITE_REQ whose data is received straight into its region.
struct Receive {
  class Server* server;
  ucp_ep_h ep;
  uint64_t conn;  // the ep handle may be reused once the connection closed
  void* desc;
  uint64_t region;
  ple_mem::WriteReq req;
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
    if (!opt_.token_file.empty()) {
      std::string err;
      if (!ple_mem::read_token_file(opt_.token_file, ple_mem::kMaxTokenBytes,
                                    &token_, &err)) {
        log_msg("%s", err.c_str());
        return false;
      }
    }
    if (opt_.max_memory == 0) {
      const uint64_t avail = ple_mem::mem_available();
      if (avail == 0) {
        log_msg("cannot read MemAvailable; pass --max-memory explicitly");
        return false;
      }
      opt_.max_memory = avail / 10 * 9;
    }
    ucp_config_t* config = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) return fail("ucp_config_read", st);
    std::string err;
    if (!ple_mem::exclude_tcp_transport(config, opt_.allow_tcp, &err)) {
      ucp_config_release(config);
      log_msg("%s", err.c_str());
      return false;
    }
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_AM;
    if (opt_.event_wait) params.features |= UCP_FEATURE_WAKEUP;
    st = ucp_init(&params, config, &ctx_);
    ucp_config_release(config);
    if (st != UCS_OK) return fail("ucp_init", st);

    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(ctx_, &wp, &worker_);
    if (st != UCS_OK) return fail("ucp_worker_create", st);
    if (opt_.event_wait) {
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

    std::vector<sockaddr_storage> addrs;
    std::vector<socklen_t> lens;
    const std::string where = opt_.host.find(':') != std::string::npos
                                  ? "[" + opt_.host + "]:" + opt_.port
                                  : opt_.host + ":" + opt_.port;
    if (!ple_mem::resolve_addresses(where, true, &addrs, &lens, &err)) {
      log_msg("%s", err.c_str());
      return false;
    }
    for (size_t i = 0; i < addrs.size() && listener_ == nullptr; ++i) {
      ucp_listener_params_t lp{};
      lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                      UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
      lp.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addrs[i]);
      lp.sockaddr.addrlen = lens[i];
      lp.conn_handler.cb = &Server::conn_cb;
      lp.conn_handler.arg = this;
      st = ucp_listener_create(worker_, &lp, &listener_);
      if (st != UCS_OK) {
        log_msg("cannot listen on %s: %s", sockaddr_str(addrs[i]).c_str(),
                ucs_status_string(st));
        listener_ = nullptr;
      }
    }
    if (listener_ == nullptr) return false;

    ucp_listener_attr_t la{};
    la.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    st = ucp_listener_query(listener_, &la);
    if (st != UCS_OK) return fail("ucp_listener_query", st);
    log_msg("listening on %s (max memory %s, %" PRIu64 " regions, %" PRIu64
            " per client, %s%s%s)",
            sockaddr_str(la.sockaddr).c_str(),
            ple_mem::human_bytes(opt_.max_memory).c_str(), opt_.max_regions,
            opt_.max_client_regions,
            opt_.event_wait ? "event wait" : "busy poll",
            token_.empty() ? "" : ", token required",
            opt_.allow_tcp ? ", tcp allowed" : "");
    return true;
  }

  bool fail(const char* what, ucs_status_t st) {
    log_msg("%s failed: %s", what, ucs_status_string(st));
    return false;
  }

  void loop() {
    while (!g_stop) {
      unsigned progressed = 0;
      while (ucp_worker_progress(worker_) != 0) ++progressed;
      const bool did_work = progressed || !replies_.empty() ||
                            !events_.empty() || !receives_.empty();
      start_receives();
      flush_replies();
      handle_events();
      reap_closing();
      if (did_work || !opt_.event_wait) continue;
      const ucs_status_t st = ucp_worker_arm(worker_);
      if (st == UCS_ERR_BUSY) continue;
      if (st != UCS_OK) {
        log_msg("ucp_worker_arm failed: %s", ucs_status_string(st));
        break;
      }
      pollfd pfd{efd_, POLLIN, 0};
      poll(&pfd, 1, closing_.empty() && sends_ == 0 ? 100 : 1);
    }
  }

  static void conn_cb(ucp_conn_request_h req, void* arg) {
    Event e;
    e.kind = Event::kConnect;
    e.conn_request = req;
    static_cast<Server*>(arg)->events_.push_back(e);
  }

  static void ep_err_cb(void* arg, ucp_ep_h ep, ucs_status_t status) {
    Event e;
    e.kind = Event::kEpError;
    e.ep = ep;
    e.status = status;
    static_cast<Server*>(arg)->events_.push_back(e);
  }

  // Closes the connection of `ep` once the queued replies went out.
  void drop_later(ucp_ep_h ep, Conn& c, bool graceful, const char* why) {
    if (c.closing) return;
    c.closing = true;
    log_msg("client %" PRIu64 " (%s): %s; closing the connection", c.id,
            c.peer.c_str(), why);
    Event e;
    e.kind = Event::kDrop;
    e.ep = ep;
    e.status = UCS_ERR_REJECTED;
    e.graceful = graceful;
    events_.push_back(e);
  }

  static ucs_status_t am_cb(void* arg, const void* header, size_t header_length,
                            void* data, size_t length,
                            const ucp_am_recv_param_t* param) {
    auto* self = static_cast<Server*>(arg);
    ucp_ep_h ep = (param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP)
                      ? param->reply_ep
                      : nullptr;
    auto it = ep == nullptr ? self->conns_.end() : self->conns_.find(ep);
    if (it == self->conns_.end()) {
      log_msg("dropping a message from an unknown endpoint");
      return UCS_OK;
    }
    Conn& c = it->second;
    if (c.closing) return UCS_OK;
    if (!c.checked) {
      c.checked = true;
      bool tcp = false;
      if (!self->opt_.allow_tcp &&
          (ple_mem::ep_uses_tcp(ep, &tcp) != UCS_OK || tcp)) {
        self->drop_later(ep, c, false, "connection uses the tcp transport");
        return UCS_OK;
      }
    }
    const bool rndv = param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;
    const uint16_t type = ple_mem::message_type(header, header_length);
    if (!self->token_.empty() && !c.authed && type != ple_mem::kAllocReq) {
      self->drop_later(ep, c, false, "request before authentication");
      return UCS_OK;
    }
    if (rndv && type != ple_mem::kWriteReq) {
      self->drop_later(ep, c, false, "unexpected rendezvous message");
      return UCS_OK;
    }
    const auto* d = static_cast<const uint8_t*>(data);
    switch (type) {
      case ple_mem::kAllocReq:
        self->on_alloc(ep, c, header, header_length, d, length);
        return UCS_OK;
      case ple_mem::kFreeReq:
        self->on_free(ep, c, header, header_length, length);
        return UCS_OK;
      case ple_mem::kWriteReq:
        return self->on_write(ep, c, header, header_length, data, length, rndv);
      case ple_mem::kReadReq:
        self->on_read(ep, c, header, header_length, d, length);
        return UCS_OK;
      default:
        self->drop_later(ep, c, false, "invalid message");
        return UCS_OK;
    }
  }

  // Region `id` when it belongs to `c`, else null.
  Region* owned(const Conn& c, uint64_t id) {
    auto r = regions_.find(id);
    if (r == regions_.end() || r->second.conn != c.id || r->second.released)
      return nullptr;
    return &r->second;
  }

  void on_alloc(ucp_ep_h ep, Conn& c, const void* h, size_t hlen,
                const uint8_t* data, size_t length) {
    ple_mem::AllocReq req;
    if (!ple_mem::parse(h, hlen, &req) || req.token_len != length ||
        length > ple_mem::kMaxTokenBytes) {
      drop_later(ep, c, false, "malformed allocation request");
      return;
    }
    ple_mem::AllocRep rep =
        ple_mem::make<ple_mem::AllocRep>(ple_mem::kAllocRep);
    rep.cookie = req.cookie;
    if (!token_.empty() && !c.authed) {
      if (!ple_mem::constant_time_equal(
              token_, reinterpret_cast<const char*>(data), length)) {
        rep.status = ple_mem::kErrAuth;
        queue_reply(ep, &rep, sizeof(rep), "wrong or missing token");
        drop_later(ep, c, true, "authentication failed");
        return;
      }
      c.authed = true;
    }
    std::string err;
    rep.status = try_alloc(c, req.bytes, &rep.region_id, &err);
    if (rep.status == ple_mem::kOk) {
      rep.bytes = req.bytes;
      if (opt_.verbose)
        log_msg("client %" PRIu64 " borrowed region %" PRIu64 " (%" PRIu64
                " bytes); client total %s",
                c.id, rep.region_id, req.bytes,
                ple_mem::human_bytes(c.mapped).c_str());
      queue_reply(ep, &rep, sizeof(rep), "");
      return;
    }
    if (failed_allocs_++ < 8 || opt_.verbose)
      log_msg("client %" PRIu64 " allocation of %" PRIu64 " bytes refused: %s",
              c.id, req.bytes, err.c_str());
    queue_reply(ep, &rep, sizeof(rep), err);
  }

  int32_t try_alloc(Conn& c, uint64_t bytes, uint64_t* id, std::string* err) {
    const uint64_t page = opt_.hugepages ? (uint64_t{2} << 20) : 4096;
    if (bytes == 0) {
      *err = "zero-byte region";
      return ple_mem::kErrInvalid;
    }
    if (bytes > UINT64_MAX - page) {
      *err = "region too large";
      return ple_mem::kErrLimit;
    }
    if (c.regions >= opt_.max_client_regions ||
        regions_.size() >= opt_.max_regions) {
      *err = "region limit reached (--max-client-regions " +
             std::to_string(opt_.max_client_regions) + ", --max-regions " +
             std::to_string(opt_.max_regions) + ")";
      return ple_mem::kErrLimit;
    }
    const uint64_t mapped = (bytes + page - 1) / page * page;
    if (mapped > opt_.max_memory - lent_) {
      *err = "exceeds --max-memory (" + ple_mem::human_bytes(lent_) +
             " lent of " + ple_mem::human_bytes(opt_.max_memory) + ")";
      return ple_mem::kErrLimit;
    }
    void* addr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (addr == MAP_FAILED) {
      *err = std::string("mmap failed: ") + std::strerror(errno);
      return ple_mem::kErrSystem;
    }
    if (opt_.hugepages) madvise(addr, mapped, MADV_HUGEPAGE);
    *err = prefault(addr, mapped);
    if (!err->empty()) {
      munmap(addr, mapped);
      return ple_mem::kErrSystem;
    }
    Region r{next_region_++, c.id, static_cast<uint8_t*>(addr), bytes, mapped};
    regions_[r.id] = r;
    lent_ += mapped;
    c.regions += 1;
    c.mapped += mapped;
    *id = r.id;
    return ple_mem::kOk;
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

  void on_free(ucp_ep_h ep, Conn& c, const void* h, size_t hlen,
               size_t length) {
    ple_mem::FreeReq req;
    if (!ple_mem::parse(h, hlen, &req) || length != 0) {
      drop_later(ep, c, false, "malformed free request");
      return;
    }
    Region* r = owned(c, req.region_id);
    if (r == nullptr) {
      log_msg("client %" PRIu64 " freed unknown region %" PRIu64, c.id,
              req.region_id);
      return;
    }
    if (opt_.verbose)
      log_msg("client %" PRIu64 " freed region %" PRIu64, c.id, r->id);
    release_region(c, *r);
  }

  ucs_status_t on_write(ucp_ep_h ep, Conn& c, const void* h, size_t hlen,
                        void* data, size_t length, bool rndv) {
    ple_mem::WriteReq req;
    if (!ple_mem::parse(h, hlen, &req)) {
      drop_later(ep, c, false, "malformed write request");
      return UCS_OK;
    }
    int32_t status = ple_mem::kOk;
    Region* r = nullptr;
    if (req.length != length || length > ple_mem::kMaxWriteBytes) {
      status = ple_mem::kErrInvalid;
    } else if ((r = owned(c, req.region_id)) == nullptr) {
      status = ple_mem::kErrNoRegion;
    } else if (!ple_mem::in_bounds(req.offset, req.length, r->bytes)) {
      status = ple_mem::kErrBounds;
    }
    if (status != ple_mem::kOk) {
      reject(c, "write", status);
      write_ack(ep, req.seq, status);
      return UCS_OK;
    }
    if (!rndv) {
      std::memcpy(r->addr + req.offset, data, length);
      write_ack(ep, req.seq, ple_mem::kOk);
      return UCS_OK;
    }
    ++r->receives;
    receives_.push_back(Receive{this, ep, c.id, data, r->id, req});
    return UCS_INPROGRESS;
  }

  void write_ack(ucp_ep_h ep, uint64_t seq, int32_t status) {
    ple_mem::WriteAck ack =
        ple_mem::make<ple_mem::WriteAck>(ple_mem::kWriteAck);
    ack.seq = seq;
    ack.status = status;
    queue_reply(ep, &ack, sizeof(ack), "");
  }

  // Receives the queued rendezvous writes straight into their regions.
  void start_receives() {
    while (!receives_.empty()) {
      auto* rc = new Receive(receives_.front());
      receives_.pop_front();
      Region& r = regions_.at(rc->region);
      ucp_request_param_t p{};
      p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
      p.cb.recv_am = &Server::received_cb;
      p.user_data = rc;
      ++sends_;
      void* req = ucp_am_recv_data_nbx(
          worker_, rc->desc, r.addr + rc->req.offset, rc->req.length, &p);
      if (UCS_PTR_IS_PTR(req)) continue;
      received(rc, UCS_PTR_IS_ERR(req) ? UCS_PTR_STATUS(req) : UCS_OK);
    }
  }

  static void received_cb(void* request, ucs_status_t status, size_t,
                          void* user) {
    ucp_request_free(request);
    auto* rc = static_cast<Receive*>(user);
    rc->server->received(rc, status);
  }

  void received(Receive* rc, ucs_status_t status) {
    --sends_;
    Region& r = regions_.at(rc->region);
    --r.receives;
    if (r.released) {
      if (r.receives == 0) unmap_region(rc->region);
    } else if (auto it = conns_.find(rc->ep);
               it != conns_.end() && it->second.id == rc->conn) {
      write_ack(rc->ep, rc->req.seq,
                status == UCS_OK ? ple_mem::kOk : ple_mem::kErrSystem);
    }
    delete rc;
  }

  void on_read(ucp_ep_h ep, Conn& c, const void* h, size_t hlen,
               const uint8_t* data, size_t length) {
    ple_mem::ReadReq req;
    if (!ple_mem::parse(h, hlen, &req)) {
      drop_later(ep, c, false, "malformed read request");
      return;
    }
    ple_mem::ReadRep rep = ple_mem::make<ple_mem::ReadRep>(ple_mem::kReadRep);
    rep.req_id = req.req_id;
    const uint64_t out = uint64_t{req.count} * req.row_bytes;
    if (req.row_bytes == 0 || req.row_bytes > ple_mem::kMaxRowBytes ||
        req.count == 0 || req.count > ple_mem::kMaxReadRows ||
        out > ple_mem::kMaxReadReplyBytes ||
        length != uint64_t{req.count} * sizeof(ple_mem::ReadEntry)) {
      rep.status = ple_mem::kErrInvalid;
    }
    int buf = -1;
    if (rep.status == ple_mem::kOk) {
      buf = take_buffer(out);
      uint8_t* dst = buffers_[buf].data();
      const Region* r = nullptr;
      for (uint32_t i = 0; i < req.count; ++i) {
        ple_mem::ReadEntry e;
        std::memcpy(&e, data + size_t{i} * sizeof(e), sizeof(e));
        if (r == nullptr || r->id != e.region_id) r = owned(c, e.region_id);
        if (r == nullptr) {
          rep.status = ple_mem::kErrNoRegion;
          break;
        }
        if (!ple_mem::in_bounds(e.offset, req.row_bytes, r->bytes)) {
          rep.status = ple_mem::kErrBounds;
          break;
        }
        std::memcpy(dst + size_t{i} * req.row_bytes, r->addr + e.offset,
                    req.row_bytes);
      }
    }
    if (rep.status != ple_mem::kOk) {
      if (buf >= 0) free_buffers_.push_back(buf);
      reject(c, "read", rep.status);
      queue_reply(ep, &rep, sizeof(rep), "");
      return;
    }
    rep.count = req.count;
    queue(ep, &rep, sizeof(rep), buf,
          opt_.fault == Fault::kShortReply ? out - 1 : out);
    if (opt_.fault == Fault::kUnsolicited) write_ack(ep, ~0ull, ple_mem::kOk);
  }

  void reject(const Conn& c, const char* what, int32_t status) {
    if (rejects_++ < 8 || opt_.verbose)
      log_msg("client %" PRIu64 " %s rejected: %s", c.id, what,
              ple_mem::status_name(status));
  }

  int take_buffer(size_t bytes) {
    int b;
    if (free_buffers_.empty()) {
      buffers_.emplace_back();
      b = static_cast<int>(buffers_.size() - 1);
    } else {
      b = free_buffers_.back();
      free_buffers_.pop_back();
    }
    if (buffers_[b].size() < bytes) buffers_[b].resize(bytes);
    return b;
  }

  void queue_reply(ucp_ep_h ep, const void* header, size_t header_length,
                   const std::string& text) {
    int buf = -1;
    size_t n = std::min<size_t>(text.size(), ple_mem::kMaxErrorBytes);
    if (n > 0) {
      buf = take_buffer(n);
      std::memcpy(buffers_[buf].data(), text.data(), n);
    }
    queue(ep, header, header_length, buf, n);
  }

  void queue(ucp_ep_h ep, const void* header, size_t header_length, int buf,
             size_t data_length) {
    Reply r;
    r.ep = ep;
    std::memcpy(r.header, header, header_length);
    r.header_length = header_length;
    r.buffer = buf;
    r.data_length = data_length;
    replies_.push_back(r);
  }

  struct SendCtx {
    Server* server;
    int buffer;
  };

  void flush_replies() {
    for (const Reply& r : replies_) {
      if (conns_.count(r.ep) == 0) {
        if (r.buffer >= 0) free_buffers_.push_back(r.buffer);
        continue;
      }
      ucp_request_param_t p{};
      p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA;
      p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_COPY_HEADER;
      p.cb.send = &Server::sent_cb;
      p.user_data = new SendCtx{this, r.buffer};
      const void* data = r.buffer >= 0 ? buffers_[r.buffer].data() : nullptr;
      void* req = ucp_am_send_nbx(r.ep, ple_mem::kAmReplyId, r.header,
                                  r.header_length, data, r.data_length, &p);
      if (UCS_PTR_IS_PTR(req)) {
        ++sends_;
        continue;
      }
      delete static_cast<SendCtx*>(p.user_data);
      if (UCS_PTR_IS_ERR(req))
        log_msg("reply send failed: %s",
                ucs_status_string(UCS_PTR_STATUS(req)));
      if (r.buffer >= 0) free_buffers_.push_back(r.buffer);
    }
    replies_.clear();
  }

  static void sent_cb(void* request, ucs_status_t status, void* user) {
    auto* ctx = static_cast<SendCtx*>(user);
    Server* self = ctx->server;
    ucp_request_free(request);
    if (status != UCS_OK && status != UCS_ERR_CANCELED)
      log_msg("reply send failed: %s", ucs_status_string(status));
    if (ctx->buffer >= 0) self->free_buffers_.push_back(ctx->buffer);
    --self->sends_;
    delete ctx;
  }

  void handle_events() {
    while (!events_.empty()) {
      Event e = events_.front();
      events_.pop_front();
      switch (e.kind) {
        case Event::kConnect:
          accept(e.conn_request);
          break;
        case Event::kEpError:
        case Event::kDrop:
          drop_conn(e.ep, e.status, e.graceful);
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
    c.authed = token_.empty();
    log_msg("client %" PRIu64 " connected from %s", c.id, peer.c_str());
    conns_[h] = std::move(c);
  }

  void drop_conn(ucp_ep_h ep, ucs_status_t status, bool graceful) {
    auto it = conns_.find(ep);
    if (it == conns_.end()) return;
    const Conn c = std::move(it->second);
    conns_.erase(it);
    log_msg("client %" PRIu64 " (%s) disconnected (%s); releasing %" PRIu64
            " regions, %s",
            c.id, c.peer.c_str(), ucs_status_string(status), c.regions,
            ple_mem::human_bytes(c.mapped).c_str());
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = graceful ? 0 : UCP_EP_CLOSE_FLAG_FORCE;
    void* req = ucp_ep_close_nbx(ep, &p);
    if (UCS_PTR_IS_PTR(req)) {
      closing_.push_back(Closing{req, c.id});
    } else {
      release_conn_regions(c.id);
    }
  }

  void release_region(Conn& c, Region& r) {
    c.regions -= 1;
    c.mapped -= r.mapped;
    lent_ -= r.mapped;
    r.released = true;
    if (r.receives == 0) unmap_region(r.id);
  }

  void unmap_region(uint64_t id) {
    auto it = regions_.find(id);
    munmap(it->second.addr, it->second.mapped);
    regions_.erase(it);
  }

  void release_conn_regions(uint64_t conn) {
    std::vector<uint64_t> ids;
    for (auto& kv : regions_)
      if (kv.second.conn == conn && !kv.second.released)
        ids.push_back(kv.first);
    for (uint64_t id : ids) {
      Region& r = regions_.at(id);
      lent_ -= r.mapped;
      r.released = true;
      if (r.receives == 0) unmap_region(id);
    }
    log_totals();
  }

  void reap_closing() {
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
      for (ucp_ep_h ep : eps) drop_conn(ep, UCS_ERR_CANCELED, false);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while ((!closing_.empty() || sends_ > 0) &&
             std::chrono::steady_clock::now() < deadline) {
        ucp_worker_progress(worker_);
        reap_closing();
      }
      if (listener_ != nullptr) ucp_listener_destroy(listener_);
      ucp_worker_destroy(worker_);
      for (auto& kv : regions_) munmap(kv.second.addr, kv.second.mapped);
      regions_.clear();
    }
    if (ctx_ != nullptr) ucp_cleanup(ctx_);
    log_msg("stopped");
  }

  Options opt_;
  std::string token_;
  ucp_context_h ctx_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  ucp_listener_h listener_ = nullptr;
  int efd_ = -1;
  std::deque<Event> events_;
  std::deque<Receive> receives_;
  std::vector<Reply> replies_;
  std::vector<std::vector<uint8_t>> buffers_;
  std::vector<int> free_buffers_;
  std::unordered_map<ucp_ep_h, Conn> conns_;
  std::unordered_map<uint64_t, Region> regions_;
  std::vector<Closing> closing_;
  uint64_t sends_ = 0;
  uint64_t next_conn_ = 1;
  uint64_t next_region_ = 1;
  uint64_t lent_ = 0;
  uint64_t failed_allocs_ = 0;
  uint64_t rejects_ = 0;
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
