// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Client of the PLE memory server protocol (ple_mem_protocol.h), shared by
// the PLE row reader's UCX transport and ple_mem_bench.
//
// The client exposes none of its memory: its UCX context has active messages
// only (no RMA / AMO, so UCX's software RMA handlers, which write raw peer
// addresses, are never registered), it registers no buffer for remote access
// and sends every message eagerly, so it never discloses a remote key, and
// UCX's tcp transport is excluded unless explicitly allowed. Reply bytes are
// copied into the caller's buffers only after the reply matched an
// outstanding request of this connection exactly (type, id, size); anything
// else fails the client.
//
// Single-threaded: every call, including the callbacks run from progress(),
// happens on the caller's thread. Every posted read op and every write call
// reports exactly one Completion: by progress() once answered, or right away
// with the failure status when the client fails (also from close()).

#pragma once

#include <poll.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "ple_mem_protocol.h"
#include "ple_mem_util.h"

namespace ple_mem {

struct ClientOptions {
  std::string address;
  std::string token;  // shared secret, empty for none
  bool allow_tcp = false;
  uint32_t read_tokens = 1;  // read op tokens are in [0, read_tokens)
  uint32_t max_batch_rows = 256;
  uint32_t write_window = 8;  // WRITE_REQs in flight
};

// Copy `bytes` at `offset` of borrowed region `region` into `dst`.
struct ReadOp {
  uint32_t token;
  void* dst;
  uint32_t region;
  uint64_t offset;
  uint32_t bytes;
};

struct Completion {
  uint32_t token;
  ucs_status_t status;
};

class Client {
 public:
  static constexpr uint32_t kMaxReadTokens = 1u << 20;
  static constexpr uint32_t kMaxWriteWindow = 1u << 12;

  Client() = default;
  ~Client() { close(); }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  // Creates the UCX context and worker.
  std::string open(const ClientOptions& o) {
    opt_ = o;
    if (opt_.read_tokens == 0 || opt_.read_tokens > kMaxReadTokens)
      return "read tokens must be in [1, " + std::to_string(kMaxReadTokens) +
             "]";
    if (opt_.max_batch_rows == 0 || opt_.max_batch_rows > kMaxReadRows)
      return "max_batch_rows must be in [1, " + std::to_string(kMaxReadRows) +
             "]";
    if (opt_.write_window == 0 || opt_.write_window > kMaxWriteWindow)
      return "write window must be in [1, " + std::to_string(kMaxWriteWindow) +
             "]";
    if (opt_.token.size() > kMaxTokenBytes) return "token too long";
    std::string err;
    if (!resolve_addresses(opt_.address, false, &addrs_, &addr_lens_, &err))
      return err;

    ucp_config_t* config = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) return ucs_error("ucp_config_read", st);
    if (!exclude_tcp_transport(config, opt_.allow_tcp, &err)) {
      ucp_config_release(config);
      return err;
    }
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_AM | UCP_FEATURE_WAKEUP;
    st = ucp_init(&params, config, &ctx_);
    ucp_config_release(config);
    if (st != UCS_OK) return ucs_error("ucp_init", st);
    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(ctx_, &wp, &worker_);
    if (st != UCS_OK) return ucs_error("ucp_worker_create", st);
    st = ucp_worker_get_efd(worker_, &efd_);
    if (st != UCS_OK) return ucs_error("ucp_worker_get_efd", st);
    ucp_am_handler_param_t am{};
    am.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am.id = kAmReplyId;
    am.flags = UCP_AM_FLAG_WHOLE_MSG;
    am.cb = &Client::reply_cb;
    am.arg = this;
    st = ucp_worker_set_am_recv_handler(worker_, &am);
    if (st != UCS_OK) return ucs_error("ucp_worker_set_am_recv_handler", st);

    for (uint32_t s = 0; s < opt_.read_tokens; ++s) add_request();
    for (uint32_t s = opt_.read_tokens; s-- > 0;) free_reqs_.push_back(s);
    wslots_.resize(opt_.write_window);
    for (uint32_t s = 0; s < opt_.write_window; ++s) wslots_[s].client = this;
    for (uint32_t s = opt_.write_window; s-- > 0;) free_wslots_.push_back(s);
    done_.reserve(opt_.read_tokens + opt_.write_window);
    return "";
  }

  // Connects (trying every resolved address) and borrows one region per
  // entry of `region_bytes`, all granted at exactly that size. `timeout_s`
  // bounds the wait for each answer (the server populates every region before
  // granting it), not the whole borrow; `granted` counts the answers.
  std::string borrow(const std::vector<uint64_t>& region_bytes,
                     double timeout_s, const std::atomic<bool>* stop,
                     std::atomic<uint64_t>* granted = nullptr) {
    const uint64_t n = region_bytes.size();
    const auto step = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_s));
    auto deadline = std::chrono::steady_clock::now() + step;
    for (size_t a = 0; a < addrs_.size(); ++a) {
      requested_ = region_bytes;
      regions_.assign(n, Region{});
      grants_ = 0;
      refusal_.clear();
      failed_ = false;
      status_ = UCS_OK;
      reason_.clear();
      std::string err = connect(addrs_[a], addr_lens_[a]);
      if (!err.empty()) return err;
      borrowing_ = true;
      for (uint64_t i = 0; i < n && !failed_; ++i) {
        AllocReq req = make<AllocReq>(kAllocReq);
        req.token_len = static_cast<uint32_t>(opt_.token.size());
        req.cookie = i;
        req.bytes = region_bytes[i];
        control_send(&req, sizeof(req), opt_.token.data(), opt_.token.size());
      }
      for (uint64_t seen = 0; grants_ < n && refusal_.empty() && !failed_ &&
                              !(stop && stop->load()) &&
                              std::chrono::steady_clock::now() < deadline;) {
        progress();
        wait(-1, 10);
        if (grants_ == seen) continue;
        if (granted != nullptr) granted->fetch_add(grants_ - seen);
        seen = grants_;
        deadline = std::chrono::steady_clock::now() + step;
      }
      borrowing_ = false;
      if (failed_ && grants_ == 0 && a + 1 < addrs_.size() &&
          std::chrono::steady_clock::now() < deadline) {
        close_ep(false);
        continue;
      }
      break;
    }
    if (stop && stop->load()) return "closed during setup";
    if (!refusal_.empty())
      return opt_.address + " refused " + std::to_string(n) +
             " regions: " + refusal_;
    if (failed_) return reason_;
    if (grants_ < n)
      return "timed out after " + std::to_string(timeout_s) +
             " s without an answer from " + opt_.address + " (" +
             std::to_string(grants_) + " of " + std::to_string(n) +
             " regions answered)";
    bool tcp = false;
    const ucs_status_t st = ep_uses_tcp(ep_, &tcp);
    if (st != UCS_OK) return ucs_error("ucp_ep_query", st);
    if (tcp && !opt_.allow_tcp) {
      fail(UCS_ERR_UNSUPPORTED, "the connection to " + opt_.address +
                                    " uses UCX's tcp transport, which is "
                                    "refused without allow_tcp");
      return reason_;
    }
    return "";
  }

  // Posts `n` reads, batched into READ_REQs of at most max_batch_rows rows of
  // equal size. At most read_tokens ops may be outstanding.
  void read(const ReadOp* ops, uint32_t n) {
    ReadRequest* r = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
      const ReadOp& op = ops[i];
      if (!failed_ && (op.region >= regions_.size() ||
                       !in_bounds(op.offset, op.bytes, requested_[op.region])))
        fail(UCS_ERR_INVALID_PARAM, "read outside the borrowed regions");
      if (failed_) {
        if (r != nullptr) drop_unsent(r);
        r = nullptr;
        done_.push_back({op.token, status_});
        continue;
      }
      if (r != nullptr &&
          (r->count == opt_.max_batch_rows || r->row_bytes != op.bytes)) {
        send_read(r);
        r = nullptr;
      }
      if (r == nullptr) r = begin_read(op.bytes);
      add_entry(r, op.token, op.dst, regions_[op.region].id, op.offset);
    }
    if (r != nullptr) send_read(r);
  }

  // Writes `bytes` from `src` (valid until the completion) at `offset` of
  // borrowed region `region`; completes once the data is in the region.
  void write(uint32_t token, const void* src, uint32_t region, uint64_t offset,
             uint64_t bytes) {
    if (!failed_ && (region >= regions_.size() ||
                     !in_bounds(offset, bytes, requested_[region])))
      fail(UCS_ERR_INVALID_PARAM, "write outside the borrowed regions");
    if (failed_) {
      done_.push_back({token, status_});
      return;
    }
    write_raw(token, src, regions_[region].id, offset, bytes);
  }

  // Unchecked variants addressing a server region id, for ple_mem_bench's
  // server probes.
  void read_raw(uint32_t token, void* dst, uint64_t region_id, uint64_t offset,
                uint32_t bytes) {
    if (failed_) {
      done_.push_back({token, status_});
      return;
    }
    ReadRequest* r = begin_read(bytes);
    add_entry(r, token, dst, region_id, offset);
    send_read(r);
  }
  void write_raw(uint32_t token, const void* src, uint64_t region_id,
                 uint64_t offset, uint64_t bytes) {
    if (failed_) {
      done_.push_back({token, status_});
      return;
    }
    ++outstanding_;
    if (bytes == 0) {
      --outstanding_;
      done_.push_back({token, UCS_OK});
      return;
    }
    writes_.push_back(WriteCall{token, static_cast<const uint8_t*>(src),
                                region_id, offset, bytes});
    pump_writes();
  }

  void progress() {
    if (worker_ == nullptr) return;
    while (ucp_worker_progress(worker_) != 0) {
    }
    pump_writes();
  }

  // Sleeps until the worker has events, `extra_fd` is readable, or
  // `timeout_ms` passed.
  void wait(int extra_fd, int timeout_ms) {
    if (worker_ == nullptr) return;
    const ucs_status_t st = ucp_worker_arm(worker_);
    if (st == UCS_ERR_BUSY) return;
    if (st != UCS_OK) {
      fail(st, ucs_error("ucp_worker_arm", st));
      return;
    }
    pollfd fds[2] = {{efd_, POLLIN, 0}, {extra_fd, POLLIN, 0}};
    poll(fds, extra_fd >= 0 ? 2 : 1, timeout_ms);
  }

  uint64_t outstanding() const { return outstanding_; }
  uint64_t requests_sent() const { return requests_sent_; }
  std::vector<Completion>& completions() { return done_; }
  bool failed() const { return failed_; }
  ucs_status_t failure() const { return status_; }
  const std::string& failure_reason() const { return reason_; }
  uint32_t num_regions() const {
    return static_cast<uint32_t>(regions_.size());
  }
  uint64_t region_id(uint32_t region) const { return regions_[region].id; }

  // Fails the client: every outstanding op completes with `status` and no
  // later reply is accepted.
  void fail(ucs_status_t status, const std::string& reason) {
    if (!failed_) {
      failed_ = true;
      status_ = status == UCS_OK ? UCS_ERR_CANCELED : status;
      reason_ = reason;
    }
    for (auto& r : reqs_) {
      if (!r.active) continue;
      r.active = false;
      for (uint32_t i = 0; i < r.count; ++i)
        done_.push_back({r.tokens[i], status_});
      outstanding_ -= r.count;
      if (!r.sending) free_reqs_.push_back(r.slot);
    }
    for (auto& w : wslots_) {
      w.call = nullptr;
      if (!w.active) continue;
      w.active = false;
      if (!w.sending) free_wslots_.push_back(wslot_of(w));
    }
    for (auto& c : writes_) {
      if (c.done) continue;
      c.done = true;
      done_.push_back({c.token, status_});
      --outstanding_;
    }
    writes_.clear();
    next_unsent_ = 0;
  }

  // Returns the regions (when healthy and idle) and releases every resource.
  // Idempotent.
  void close() {
    if (worker_ != nullptr) {
      const bool graceful = ep_ != nullptr && !failed_ && outstanding_ == 0;
      if (graceful) {
        for (const Region& reg : regions_) {
          if (!reg.granted) continue;
          FreeReq req = make<FreeReq>(kFreeReq);
          req.region_id = reg.id;
          control_send(&req, sizeof(req), nullptr, 0);
        }
      }
      if (!failed_) fail(UCS_ERR_CANCELED, "closed");
      close_ep(graceful);
      ucp_worker_destroy(worker_);
      worker_ = nullptr;
    }
    if (ctx_ != nullptr) ucp_cleanup(ctx_);
    ctx_ = nullptr;
  }

 private:
  struct Region {
    uint64_t id = 0;
    bool granted = false;
  };

  struct ReadRequest {
    Client* client = nullptr;
    uint32_t slot = 0;
    ReadReq hdr{};
    uint32_t row_bytes = 0;
    uint32_t count = 0;
    bool active = false;   // awaiting its reply
    bool sending = false;  // UCX still holds hdr / entries
    std::vector<ReadEntry> entries;
    std::vector<uint32_t> tokens;
    std::vector<void*> dsts;
  };

  struct WriteCall {
    uint32_t token;
    const uint8_t* src;
    uint64_t region_id;
    uint64_t offset;
    uint64_t bytes;
    uint64_t sent = 0;
    uint64_t acked = 0;
    uint32_t sending = 0;  // pieces UCX may still read from src
    bool done = false;
  };

  struct WriteSlot {
    Client* client = nullptr;
    WriteReq hdr{};
    WriteCall* call = nullptr;
    bool active = false;
    bool sending = false;
  };

  static std::string ucs_error(const char* what, ucs_status_t st) {
    return std::string(what) + ": " + ucs_status_string(st);
  }

  // Request slots outlive their reply while UCX still sends them, so the
  // pool grows past read_tokens when needed (addresses stay stable).
  void add_request() {
    ReadRequest& r = reqs_.emplace_back();
    r.client = this;
    r.slot = static_cast<uint32_t>(reqs_.size() - 1);
    r.entries.resize(opt_.max_batch_rows);
    r.tokens.resize(opt_.max_batch_rows);
    r.dsts.resize(opt_.max_batch_rows);
  }
  uint32_t wslot_of(const WriteSlot& w) const {
    return static_cast<uint32_t>(&w - wslots_.data());
  }

  std::string connect(const sockaddr_storage& addr, socklen_t len) {
    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
    ep.sockaddr.addrlen = len;
    ep.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep.err_handler.cb = &Client::ep_err_cb;
    ep.err_handler.arg = this;
    const ucs_status_t st = ucp_ep_create(worker_, &ep, &ep_);
    if (st != UCS_OK) {
      ep_ = nullptr;
      return "cannot connect to " + opt_.address + ": " + ucs_status_string(st);
    }
    return "";
  }

  // Closes the endpoint (flushed when `graceful`), then progresses until UCX
  // released every send buffer or 2 s passed.
  void close_ep(bool graceful) {
    if (ep_ != nullptr) {
      ucp_request_param_t p{};
      p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
      p.flags = graceful ? 0 : UCP_EP_CLOSE_FLAG_FORCE;
      void* req = ucp_ep_close_nbx(ep_, &p);
      ep_ = nullptr;
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (UCS_PTR_IS_PTR(req) &&
             ucp_request_check_status(req) == UCS_INPROGRESS &&
             std::chrono::steady_clock::now() < deadline)
        ucp_worker_progress(worker_);
      if (UCS_PTR_IS_PTR(req)) ucp_request_free(req);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((sending_ > 0 || control_pending_ > 0) &&
           std::chrono::steady_clock::now() < deadline)
      ucp_worker_progress(worker_);
  }

  static void ep_err_cb(void* arg, ucp_ep_h, ucs_status_t status) {
    auto* c = static_cast<Client*>(arg);
    c->fail(status, "connection to " + c->opt_.address +
                        " failed: " + ucs_status_string(status));
  }

  void violation(const std::string& what) {
    fail(UCS_ERR_IO_ERROR,
         "protocol violation by " + opt_.address + ": " + what);
  }

  static ucs_status_t reply_cb(void* arg, const void* header,
                               size_t header_length, void* data, size_t length,
                               const ucp_am_recv_param_t* p) {
    auto* c = static_cast<Client*>(arg);
    if (c->failed_) return UCS_OK;
    if (p->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      c->violation("rendezvous reply");
      return UCS_OK;
    }
    const auto* d = static_cast<const uint8_t*>(data);
    switch (message_type(header, header_length)) {
      case kReadRep:
        c->on_read_rep(header, header_length, d, length);
        break;
      case kWriteAck:
        c->on_write_ack(header, header_length, length);
        break;
      case kAllocRep:
        c->on_alloc_rep(header, header_length, d, length);
        break;
      default:
        c->violation("unexpected message");
    }
    return UCS_OK;
  }

  void on_read_rep(const void* h, size_t hlen, const uint8_t* data,
                   size_t length) {
    ReadRep rep;
    if (!parse(h, hlen, &rep)) return violation("malformed read reply");
    const uint64_t slot = rep.req_id & (kMaxReadTokens - 1);
    if (slot >= reqs_.size() || !reqs_[slot].active ||
        reqs_[slot].hdr.req_id != rep.req_id)
      return violation("unsolicited read reply");
    ReadRequest& r = reqs_[slot];
    if (const char* e = read_rep_error(rep, length, r.count, r.row_bytes))
      return violation(std::string("read reply: ") + e);
    if (rep.status != kOk)
      return fail(UCS_ERR_REJECTED, opt_.address + " rejected a read: " +
                                        status_name(rep.status));
    for (uint32_t i = 0; i < r.count; ++i) {
      std::memcpy(r.dsts[i], data + size_t{i} * r.row_bytes, r.row_bytes);
      done_.push_back({r.tokens[i], UCS_OK});
    }
    outstanding_ -= r.count;
    r.active = false;
    if (!r.sending) free_reqs_.push_back(static_cast<uint32_t>(slot));
  }

  void on_write_ack(const void* h, size_t hlen, size_t length) {
    WriteAck ack;
    if (!parse(h, hlen, &ack) || length != 0)
      return violation("malformed write ack");
    const uint64_t slot = ack.seq & (kMaxWriteWindow - 1);
    if (slot >= wslots_.size() || !wslots_[slot].active ||
        wslots_[slot].hdr.seq != ack.seq)
      return violation("unsolicited write ack");
    if (ack.status != kOk)
      return fail(UCS_ERR_REJECTED, opt_.address + " rejected a write: " +
                                        status_name(ack.status));
    WriteSlot& w = wslots_[slot];
    WriteCall& call = *w.call;
    call.acked += w.hdr.length;
    w.active = false;
    if (!w.sending) {
      w.call = nullptr;
      free_wslots_.push_back(static_cast<uint32_t>(slot));
    }
    complete_write(call);
  }

  // A write completes once every piece is acknowledged (the data is in the
  // region) and UCX no longer reads its source buffer.
  void complete_write(WriteCall& call) {
    if (call.done || call.acked != call.bytes || call.sending != 0) return;
    call.done = true;
    done_.push_back({call.token, UCS_OK});
    --outstanding_;
    while (!writes_.empty() && writes_.front().done) {
      writes_.pop_front();
      --next_unsent_;
    }
  }

  void on_alloc_rep(const void* h, size_t hlen, const uint8_t* data,
                    size_t length) {
    AllocRep rep;
    if (!borrowing_ || !parse(h, hlen, &rep) || length > kMaxErrorBytes ||
        rep.cookie >= regions_.size() || regions_[rep.cookie].granted)
      return violation("unexpected allocation reply");
    if (rep.status == kOk) {
      if (length != 0 || rep.bytes != requested_[rep.cookie])
        return violation("allocation granted with a wrong size");
      regions_[rep.cookie] = Region{rep.region_id, true};
    } else if (refusal_.empty()) {
      std::string text(reinterpret_cast<const char*>(data), length);
      text = text.substr(0, text.find('\0'));
      for (char& ch : text)
        if (ch < 0x20 || ch > 0x7e) ch = '?';
      refusal_ = std::string(status_name(rep.status)) +
                 (text.empty() ? "" : " (" + text + ")");
    }
    ++grants_;
  }

  ReadRequest* begin_read(uint32_t row_bytes) {
    if (free_reqs_.empty()) {
      if (reqs_.size() >= kMaxReadTokens) std::abort();
      add_request();
      free_reqs_.push_back(reqs_.back().slot);
    }
    ReadRequest& r = reqs_[free_reqs_.back()];
    free_reqs_.pop_back();
    r.row_bytes = row_bytes;
    r.count = 0;
    return &r;
  }

  void add_entry(ReadRequest* r, uint32_t token, void* dst, uint64_t region_id,
                 uint64_t offset) {
    r->entries[r->count] = ReadEntry{region_id, offset};
    r->tokens[r->count] = token;
    r->dsts[r->count] = dst;
    ++r->count;
  }

  void drop_unsent(ReadRequest* r) {
    for (uint32_t i = 0; i < r->count; ++i)
      done_.push_back({r->tokens[i], status_});
    free_reqs_.push_back(r->slot);
  }

  void send_read(ReadRequest* r) {
    r->hdr = make<ReadReq>(kReadReq);
    r->hdr.row_bytes = r->row_bytes;
    r->hdr.count = r->count;
    r->hdr.req_id = (next_id_++ * kMaxReadTokens) | r->slot;
    r->active = true;
    outstanding_ += r->count;
    ++requests_sent_;
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY;
    p.cb.send = &Client::read_sent_cb;
    p.user_data = r;
    void* req =
        ucp_am_send_nbx(ep_, kAmRequestId, &r->hdr, sizeof(r->hdr),
                        r->entries.data(), r->count * sizeof(ReadEntry), &p);
    if (UCS_PTR_IS_PTR(req)) {
      r->sending = true;
      ++sending_;
    } else if (UCS_PTR_IS_ERR(req)) {
      fail(UCS_PTR_STATUS(req),
           ucs_error("read request send", UCS_PTR_STATUS(req)));
    }
  }

  static void read_sent_cb(void* request, ucs_status_t status, void* user) {
    auto* r = static_cast<ReadRequest*>(user);
    Client* c = r->client;
    ucp_request_free(request);
    r->sending = false;
    --c->sending_;
    if (status != UCS_OK && !c->failed_)
      c->fail(status, ucs_error("read request send", status));
    if (!r->active) c->free_reqs_.push_back(r->slot);
  }

  void pump_writes() {
    while (!failed_ && !free_wslots_.empty() && next_unsent_ < writes_.size()) {
      WriteCall& call = writes_[next_unsent_];
      const uint64_t piece = std::min(kMaxWriteBytes, call.bytes - call.sent);
      const uint32_t slot = free_wslots_.back();
      free_wslots_.pop_back();
      WriteSlot& w = wslots_[slot];
      w.hdr = make<WriteReq>(kWriteReq);
      w.hdr.seq = (next_id_++ * kMaxWriteWindow) | slot;
      w.hdr.region_id = call.region_id;
      w.hdr.offset = call.offset + call.sent;
      w.hdr.length = piece;
      w.call = &call;
      w.active = true;
      const uint8_t* src = call.src + call.sent;
      call.sent += piece;
      if (call.sent == call.bytes) ++next_unsent_;
      ucp_request_param_t p{};
      p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_FLAGS;
      // Large pieces may go by rendezvous: the server then receives straight
      // into its region from this (UCX-registered) source buffer.
      p.flags = UCP_AM_SEND_FLAG_REPLY;
      p.cb.send = &Client::write_sent_cb;
      p.user_data = &w;
      void* req = ucp_am_send_nbx(ep_, kAmRequestId, &w.hdr, sizeof(w.hdr), src,
                                  piece, &p);
      if (UCS_PTR_IS_PTR(req)) {
        w.sending = true;
        ++call.sending;
        ++sending_;
      } else if (UCS_PTR_IS_ERR(req)) {
        fail(UCS_PTR_STATUS(req),
             ucs_error("write request send", UCS_PTR_STATUS(req)));
      }
    }
  }

  static void write_sent_cb(void* request, ucs_status_t status, void* user) {
    auto* w = static_cast<WriteSlot*>(user);
    Client* c = w->client;
    WriteCall* call = w->call;
    ucp_request_free(request);
    w->sending = false;
    --c->sending_;
    if (call != nullptr) --call->sending;
    if (!w->active) {
      w->call = nullptr;
      c->free_wslots_.push_back(c->wslot_of(*w));
    }
    if (status != UCS_OK && !c->failed_) {
      c->fail(status, ucs_error("write request send", status));
      return;
    }
    if (call != nullptr && !c->failed_) c->complete_write(*call);
  }

  void control_send(const void* header, size_t header_length, const void* data,
                    size_t length) {
    if (ep_ == nullptr) return;
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY |
              UCP_AM_SEND_FLAG_COPY_HEADER;
    p.cb.send = &Client::control_sent_cb;
    p.user_data = this;
    void* req = ucp_am_send_nbx(ep_, kAmRequestId, header, header_length, data,
                                length, &p);
    if (UCS_PTR_IS_PTR(req)) {
      ++control_pending_;
    } else if (UCS_PTR_IS_ERR(req) && !failed_) {
      fail(UCS_PTR_STATUS(req), "cannot reach " + opt_.address + ": " +
                                    ucs_status_string(UCS_PTR_STATUS(req)));
    }
  }

  static void control_sent_cb(void* request, ucs_status_t status, void* user) {
    auto* c = static_cast<Client*>(user);
    ucp_request_free(request);
    --c->control_pending_;
    if (status != UCS_OK && !c->failed_)
      c->fail(status, "cannot reach " + c->opt_.address + ": " +
                          ucs_status_string(status));
  }

  ClientOptions opt_;
  std::vector<sockaddr_storage> addrs_;
  std::vector<socklen_t> addr_lens_;
  ucp_context_h ctx_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  ucp_ep_h ep_ = nullptr;
  int efd_ = -1;
  bool failed_ = false;
  ucs_status_t status_ = UCS_OK;
  std::string reason_;

  bool borrowing_ = false;
  uint64_t grants_ = 0;
  std::string refusal_;
  std::vector<uint64_t> requested_;
  std::vector<Region> regions_;

  std::deque<ReadRequest> reqs_;
  std::vector<uint32_t> free_reqs_;
  std::vector<WriteSlot> wslots_;
  std::vector<uint32_t> free_wslots_;
  std::deque<WriteCall> writes_;
  size_t next_unsent_ = 0;
  uint64_t next_id_ = 1;
  uint64_t outstanding_ = 0;
  uint64_t requests_sent_ = 0;
  uint64_t sending_ = 0;
  uint64_t control_pending_ = 0;
  std::vector<Completion> done_;
};

}  // namespace ple_mem
