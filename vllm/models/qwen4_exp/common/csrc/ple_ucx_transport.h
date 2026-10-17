// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// UCX transport of the PLE remote memory source (see the Transport concept in
// ple_remote_source.h). Regions are borrowed from a PLE memory server
// (../ple_mem_server/) with its active-message protocol and rows move with
// one-sided RMA (ucp_get_nbx / ucp_put_nbx) on a single-threaded worker.

#pragma once

#include <sys/socket.h>
#include <ucp/api/ucp.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../ple_mem_server/ple_mem_protocol.h"
#include "../ple_mem_server/ple_mem_util.h"
#include "ple_remote_source.h"

namespace ple_remote {

class UcxTransport {
 public:
  using Status = ucs_status_t;
  using Mem = ucp_mem_h;
  static constexpr Status kOk = UCS_OK;
  static constexpr Status kCanceled = UCS_ERR_CANCELED;
  static constexpr const char* kName = "ucx";

  explicit UcxTransport(std::string address) : address_(std::move(address)) {}
  UcxTransport(const UcxTransport&) = delete;
  UcxTransport& operator=(const UcxTransport&) = delete;

  static const char* status_string(Status s) { return ucs_status_string(s); }

  // Creates the context, worker and endpoint.
  std::string open(uint32_t num_tokens) {
    sockaddr_storage addr{};
    socklen_t addrlen = 0;
    std::string err;
    if (!ple_mem::resolve_address(address_, &addr, &addrlen, &err)) return err;

    ucp_config_t* config = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) return ucs_error("ucp_config_read", st);
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_AM;
    st = ucp_init(&params, config, &ctx_);
    ucp_config_release(config);
    if (st != UCS_OK) return ucs_error("ucp_init", st);
    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(ctx_, &wp, &worker_);
    if (st != UCS_OK) return ucs_error("ucp_worker_create", st);
    ucp_am_handler_param_t am{};
    am.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am.id = ple_mem::kAmReplyId;
    am.flags = UCP_AM_FLAG_WHOLE_MSG;
    am.cb = &UcxTransport::reply_cb;
    am.arg = this;
    st = ucp_worker_set_am_recv_handler(worker_, &am);
    if (st != UCS_OK) return ucs_error("ucp_worker_set_am_recv_handler", st);

    ops_.resize(num_tokens);
    for (uint32_t t = 0; t < num_tokens; ++t) ops_[t] = OpCtx{this, t};
    done_.reserve(num_tokens);

    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
    ep.sockaddr.addrlen = addrlen;
    ep.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep.err_handler.cb = &UcxTransport::ep_err_cb;
    ep.err_handler.arg = this;
    st = ucp_ep_create(worker_, &ep, &ep_);
    if (st != UCS_OK) {
      ep_ = nullptr;
      return "cannot connect to " + address_ + ": " + ucs_status_string(st);
    }
    return "";
  }

  std::string register_memory(void* addr, size_t bytes, Mem* mem) {
    ucp_mem_map_params_t mp{};
    mp.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = addr;
    mp.length = bytes;
    const ucs_status_t st = ucp_mem_map(ctx_, &mp, mem);
    if (st != UCS_OK) return ucs_error("ucp_mem_map", st);
    mems_.push_back(*mem);
    return "";
  }

  // Requests every region at once and waits for all grants.
  std::string borrow(const std::vector<uint64_t>& region_bytes,
                     double timeout_s, const std::atomic<bool>& stop) {
    const uint64_t n = region_bytes.size();
    regions_.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
      ple_mem::AllocReq req{};
      req.hdr = ple_mem::make_header(ple_mem::kAllocReq);
      req.cookie = i;
      req.bytes = region_bytes[i];
      std::vector<char> payload(sizeof(req));
      std::memcpy(payload.data(), &req, sizeof(req));
      am_send(std::move(payload));
    }
    const double deadline = now_us() + timeout_s * 1e6;
    while (replies_.size() < n && !failed_ && !stop.load()) {
      if (now_us() > deadline)
        return "timed out after " + std::to_string(timeout_s) +
               " s connecting to " + address_ + " (" +
               std::to_string(replies_.size()) + " of " + std::to_string(n) +
               " regions granted)";
      ucp_worker_progress(worker_);
      reap_sends();
    }
    if (failed_)
      return "cannot reach " + address_ + ": " + ucs_status_string(status_);
    if (stop.load()) return "reader closed during setup";
    std::vector<char> granted(n, 0);
    for (const auto& m : replies_) {
      ple_mem::AllocRep rep;
      if (ple_mem::message_type(m.data(), m.size()) != ple_mem::kAllocRep ||
          m.size() < sizeof(rep))
        return "invalid reply from " + address_;
      std::memcpy(&rep, m.data(), sizeof(rep));
      if (rep.cookie >= n || granted[rep.cookie])
        return "unexpected reply from " + address_;
      if (rep.status != 0) {
        rep.error[sizeof(rep.error) - 1] = '\0';
        return address_ + " refused " + std::to_string(n) +
               " regions: " + rep.error;
      }
      if (m.size() < sizeof(rep) + rep.rkey_len)
        return "truncated reply from " + address_;
      Region& reg = regions_[rep.cookie];
      reg.id = rep.region_id;
      reg.addr = rep.addr;
      granted[rep.cookie] = 1;
      const ucs_status_t st =
          ucp_ep_rkey_unpack(ep_, m.data() + sizeof(rep), &reg.rkey);
      if (st != UCS_OK) {
        reg.rkey = nullptr;
        return ucs_error("ucp_ep_rkey_unpack", st);
      }
    }
    replies_.clear();
    return "";
  }

  void get(uint32_t token, void* dst, Mem mem, uint32_t region, uint64_t offset,
           uint32_t bytes) {
    const Region& reg = regions_[region];
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_MEMH;
    p.cb.send = &UcxTransport::op_cb;
    p.user_data = &ops_[token];
    p.memh = mem;
    track(token, ucp_get_nbx(ep_, dst, bytes, reg.addr + offset, reg.rkey, &p));
  }

  void put(uint32_t token, const void* src, Mem mem, uint32_t region,
           uint64_t offset, uint64_t bytes) {
    const Region& reg = regions_[region];
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_MEMH;
    p.cb.send = &UcxTransport::op_cb;
    p.user_data = &ops_[token];
    p.memh = mem;
    track(token, ucp_put_nbx(ep_, src, bytes, reg.addr + offset, reg.rkey, &p));
  }

  void flush(uint32_t token) {
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.cb.send = &UcxTransport::op_cb;
    p.user_data = &ops_[token];
    track(token, ucp_ep_flush_nbx(ep_, &p));
  }

  void progress() {
    ucp_worker_progress(worker_);
    if (!sends_.empty()) reap_sends();
  }

  uint64_t outstanding() const { return outstanding_; }
  std::vector<Completion<Status>>& completions() { return done_; }
  bool failed() const { return failed_; }
  Status failure() const { return status_; }
  void fail(Status status) {
    if (!failed_) status_ = status;
    failed_ = true;
  }

  // Frees the regions and flushes when healthy and idle, then closes the
  // endpoint (forced unless flushed) and releases every UCX resource.
  void close() {
    const bool graceful = ep_ != nullptr && !failed_ && outstanding_ == 0;
    if (graceful) {
      for (const auto& reg : regions_) {
        if (reg.rkey == nullptr) continue;
        ple_mem::FreeReq req{};
        req.hdr = ple_mem::make_header(ple_mem::kFreeReq);
        req.region_id = reg.id;
        std::vector<char> payload(sizeof(req));
        std::memcpy(payload.data(), &req, sizeof(req));
        am_send(std::move(payload));
      }
    }
    bool flushed = false;
    if (graceful) {
      ucp_request_param_t fp{};
      flushed =
          wait_request(ucp_ep_flush_nbx(ep_, &fp), now_us() + 2e6) == UCS_OK;
      reap_sends();
    }
    if (flushed) destroy_rkeys();
    if (ep_ != nullptr) {
      ucp_request_param_t cp{};
      cp.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
      cp.flags = flushed ? 0 : UCP_EP_CLOSE_FLAG_FORCE;
      wait_request(ucp_ep_close_nbx(ep_, &cp), now_us() + 2e6);
      ep_ = nullptr;
    }
    for (auto& s : sends_) ucp_request_free(s.request);
    sends_.clear();
    destroy_rkeys();
    for (ucp_mem_h m : mems_) ucp_mem_unmap(ctx_, m);
    mems_.clear();
    if (worker_ != nullptr) ucp_worker_destroy(worker_);
    worker_ = nullptr;
    if (ctx_ != nullptr) ucp_cleanup(ctx_);
    ctx_ = nullptr;
  }

 private:
  struct Region {
    uint64_t id = 0;
    uint64_t addr = 0;
    ucp_rkey_h rkey = nullptr;
  };

  struct OpCtx {
    UcxTransport* transport;
    uint32_t token;
  };

  struct AmSend {
    void* request;
    std::vector<char> payload;
  };

  static std::string ucs_error(const char* what, ucs_status_t st) {
    return std::string(what) + ": " + ucs_status_string(st);
  }

  static void ep_err_cb(void* arg, ucp_ep_h, ucs_status_t status) {
    static_cast<UcxTransport*>(arg)->fail(status);
  }

  static ucs_status_t reply_cb(void* arg, const void*, size_t, void* data,
                               size_t length, const ucp_am_recv_param_t* p) {
    if (p->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) return UCS_OK;
    const char* d = static_cast<const char*>(data);
    static_cast<UcxTransport*>(arg)->replies_.emplace_back(d, d + length);
    return UCS_OK;
  }

  static void op_cb(void* request, ucs_status_t status, void* user_data) {
    const auto* op = static_cast<const OpCtx*>(user_data);
    UcxTransport* t = op->transport;
    --t->outstanding_;
    t->done_.push_back({op->token, status});
    ucp_request_free(request);
  }

  void track(uint32_t token, void* req) {
    if (UCS_PTR_IS_PTR(req)) {
      ++outstanding_;
      return;
    }
    done_.push_back({token, UCS_PTR_STATUS(req)});
  }

  void am_send(std::vector<char> payload) {
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY;
    void* req = ucp_am_send_nbx(ep_, ple_mem::kAmRequestId, nullptr, 0,
                                payload.data(), payload.size(), &p);
    if (UCS_PTR_IS_PTR(req)) sends_.push_back(AmSend{req, std::move(payload)});
  }

  void reap_sends() {
    for (auto it = sends_.begin(); it != sends_.end();) {
      if (ucp_request_check_status(it->request) == UCS_INPROGRESS) {
        ++it;
        continue;
      }
      ucp_request_free(it->request);
      it = sends_.erase(it);
    }
  }

  // Progresses until `req` completes or `deadline_us`; frees it either way.
  ucs_status_t wait_request(void* req, double deadline_us) {
    if (req == nullptr) return UCS_OK;
    if (UCS_PTR_IS_ERR(req)) return UCS_PTR_STATUS(req);
    ucs_status_t st;
    while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS &&
           now_us() < deadline_us)
      ucp_worker_progress(worker_);
    ucp_request_free(req);
    return st;
  }

  void destroy_rkeys() {
    for (auto& reg : regions_) {
      if (reg.rkey != nullptr) ucp_rkey_destroy(reg.rkey);
      reg.rkey = nullptr;
    }
  }

  const std::string address_;
  ucp_context_h ctx_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  ucp_ep_h ep_ = nullptr;
  bool failed_ = false;
  ucs_status_t status_ = UCS_OK;
  uint64_t outstanding_ = 0;
  std::vector<Region> regions_;
  std::vector<ucp_mem_h> mems_;
  std::vector<OpCtx> ops_;
  std::vector<Completion<Status>> done_;
  std::vector<std::vector<char>> replies_;
  std::vector<AmSend> sends_;
};

}  // namespace ple_remote
