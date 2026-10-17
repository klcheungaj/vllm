// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// In-process transport of the PLE remote memory source, for tests (see the
// Transport concept in ple_remote_source.h). Regions are heap buffers and
// operations complete in order on the second progress() after they were
// posted, so the asynchronous paths of the source are exercised.
//
// Address: "loopback" or "loopback?key=value&...", keys:
//   fail_after=N  the transport fails once more than N GETs and puts were
//                 posted; pending and later operations complete with an error
//   refuse=1      borrowing the regions fails
//   delay_us=D    operations complete no earlier than D us after posting

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ple_remote_source.h"

namespace ple_remote {

class LoopbackTransport {
 public:
  using Status = int;
  struct Mem {};
  static constexpr Status kOk = 0;
  static constexpr Status kCanceled = 1;
  static constexpr Status kInjected = 2;
  static constexpr const char* kName = "loopback";

  explicit LoopbackTransport(std::string address)
      : address_(std::move(address)) {}
  LoopbackTransport(const LoopbackTransport&) = delete;
  LoopbackTransport& operator=(const LoopbackTransport&) = delete;

  static const char* status_string(Status s) {
    switch (s) {
      case kOk:
        return "Success";
      case kCanceled:
        return "Operation canceled";
      case kInjected:
        return "injected failure";
    }
    return "unknown status";
  }

  std::string open(uint32_t num_tokens) {
    const size_t q = address_.find('?');
    if (address_.compare(0, q, kName) != 0)
      return "invalid loopback address " + address_;
    for (size_t at = q; at != std::string::npos && at < address_.size();) {
      const size_t end = address_.find('&', at + 1);
      const std::string kv = address_.substr(at + 1, end - at - 1);
      const size_t eq = kv.find('=');
      const std::string key = kv.substr(0, eq);
      const char* value = eq == std::string::npos ? "" : kv.c_str() + eq + 1;
      char* stop = nullptr;
      const double v = std::strtod(value, &stop);
      if (*value == '\0' || *stop != '\0' || v < 0)
        return "invalid loopback address " + address_;
      if (key == "fail_after") {
        fail_after_ = static_cast<int64_t>(v);
      } else if (key == "refuse") {
        refuse_ = v != 0;
      } else if (key == "delay_us") {
        delay_us_ = v;
      } else {
        return "invalid loopback address " + address_;
      }
      at = end;
    }
    done_.reserve(num_tokens);
    return "";
  }

  std::string register_memory(void*, size_t, Mem*) { return ""; }

  std::string borrow(const std::vector<uint64_t>& region_bytes, double,
                     const std::atomic<bool>&) {
    if (refuse_)
      return address_ + " refused " + std::to_string(region_bytes.size()) +
             " regions";
    for (uint64_t bytes : region_bytes)
      regions_.emplace_back(new uint8_t[bytes]);
    return "";
  }

  void get(uint32_t token, void* dst, Mem, uint32_t region, uint64_t offset,
           uint32_t bytes) {
    post(Op{token, dst, regions_[region].get() + offset, bytes, true});
  }

  void put(uint32_t token, const void* src, Mem, uint32_t region,
           uint64_t offset, uint64_t bytes) {
    post(Op{token, regions_[region].get() + offset, src, bytes, true});
  }

  void flush(uint32_t token) { post(Op{token, nullptr, nullptr, 0, false}); }

  void progress() {
    const uint64_t gen = gen_++;
    const double now = delay_us_ > 0 ? now_us() : 0;
    while (!pending_.empty()) {
      const Op& op = pending_.front();
      if (op.gen >= gen || op.due > now) break;
      const Status st = failed_ ? status_ : kOk;
      if (st == kOk && op.bytes > 0) std::memcpy(op.dst, op.src, op.bytes);
      done_.push_back({op.token, st});
      pending_.pop_front();
    }
  }

  uint64_t outstanding() const { return pending_.size(); }
  std::vector<Completion<Status>>& completions() { return done_; }
  bool failed() const { return failed_; }
  Status failure() const { return status_; }
  void fail(Status status) {
    if (!failed_) status_ = status;
    failed_ = true;
  }

  void close() {
    for (const Op& op : pending_) done_.push_back({op.token, kCanceled});
    pending_.clear();
    regions_.clear();
  }

 private:
  struct Op {
    uint32_t token;
    void* dst;
    const void* src;
    uint64_t bytes;
    bool data;
    uint64_t gen = 0;
    double due = 0;
  };

  void post(Op op) {
    if (op.data && fail_after_ >= 0 && ++data_ops_ > fail_after_)
      fail(kInjected);
    op.gen = gen_;
    op.due = delay_us_ > 0 ? now_us() + delay_us_ : 0;
    pending_.push_back(op);
  }

  const std::string address_;
  int64_t fail_after_ = -1;
  bool refuse_ = false;
  double delay_us_ = 0;
  int64_t data_ops_ = 0;
  bool failed_ = false;
  Status status_ = kOk;
  uint64_t gen_ = 0;
  std::deque<Op> pending_;
  std::vector<std::unique_ptr<uint8_t[]>> regions_;
  std::vector<Completion<Status>> done_;
};

}  // namespace ple_remote
