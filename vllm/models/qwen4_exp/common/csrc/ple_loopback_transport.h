// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// In-process transport of the PLE remote memory source, for tests (see the
// Transport concept in ple_remote_source.h). Regions are heap buffers. Reads
// are batched into requests like the UCX client's, answered with a reply
// that goes through the same validation, and every request or write
// completes in order on the second progress() after it was posted, so the
// asynchronous paths of the source are exercised.
//
// Address: "loopback" or "loopback?key=value&...", keys:
//   fail_after=N  the transport fails once more than N read rows and writes
//                 were posted; pending and later operations complete with an
//                 error
//   refuse=1      borrowing the regions fails
//   delay_us=D    operations complete no earlier than D us after posting
//   corrupt=M     the first read reply is malformed: 1 = one byte short,
//                 2 = answers an unknown request id

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

#include "../ple_mem_server/ple_mem_protocol.h"
#include "ple_remote_source.h"

namespace ple_remote {

class LoopbackTransport {
 public:
  using Status = int;
  struct ReadOp {
    uint32_t token;
    void* dst;
    uint32_t region;
    uint64_t offset;
    uint32_t bytes;
  };
  struct Completion {
    uint32_t token;
    Status status;
  };
  static constexpr Status kOk = 0;
  static constexpr Status kCanceled = 1;
  static constexpr Status kInjected = 2;
  static constexpr Status kViolation = 3;
  static constexpr const char* kName = "loopback";

  explicit LoopbackTransport(const Params& p) : params_(p) {}
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
      case kViolation:
        return "protocol violation";
    }
    return "unknown status";
  }

  std::string open(uint32_t read_tokens) {
    const std::string& address = params_.address;
    const size_t q = address.find('?');
    if (address.compare(0, q, kName) != 0)
      return "invalid loopback address " + address;
    for (size_t at = q; at != std::string::npos && at < address.size();) {
      const size_t end = address.find('&', at + 1);
      const std::string kv = address.substr(at + 1, end - at - 1);
      const size_t eq = kv.find('=');
      const std::string key = kv.substr(0, eq);
      const char* value = eq == std::string::npos ? "" : kv.c_str() + eq + 1;
      char* stop = nullptr;
      const double v = std::strtod(value, &stop);
      if (*value == '\0' || *stop != '\0' || v < 0)
        return "invalid loopback address " + address;
      if (key == "fail_after") {
        fail_after_ = static_cast<int64_t>(v);
      } else if (key == "refuse") {
        refuse_ = v != 0;
      } else if (key == "delay_us") {
        delay_us_ = v;
      } else if (key == "corrupt") {
        corrupt_ = static_cast<int>(v);
      } else {
        return "invalid loopback address " + address;
      }
      at = end;
    }
    done_.reserve(read_tokens);
    return "";
  }

  std::string borrow(const std::vector<uint64_t>& region_bytes, double,
                     const std::atomic<bool>&) {
    if (refuse_)
      return params_.address + " refused " +
             std::to_string(region_bytes.size()) + " regions";
    for (uint64_t bytes : region_bytes) {
      regions_.emplace_back(new uint8_t[bytes]);
      sizes_.push_back(bytes);
    }
    return "";
  }

  void read(const ReadOp* ops, uint32_t n) {
    for (uint32_t i = 0; i < n;) {
      Pending p = make_pending();
      p.req_id = next_id_++;
      while (i < n && p.reads.size() < params_.max_batch_rows &&
             (p.reads.empty() || p.reads[0].bytes == ops[i].bytes)) {
        p.reads.push_back(ops[i++]);
        count_op();
      }
      ++requests_;
      pending_.push_back(std::move(p));
    }
  }

  void write(uint32_t token, const void* src, uint32_t region, uint64_t offset,
             uint64_t bytes) {
    Pending p = make_pending();
    p.write = ReadOp{token, const_cast<void*>(src), region, offset,
                     static_cast<uint32_t>(0)};
    p.write_bytes = bytes;
    count_op();
    pending_.push_back(std::move(p));
  }

  void progress() {
    const uint64_t gen = gen_++;
    const double now = delay_us_ > 0 ? now_us() : 0;
    while (!pending_.empty()) {
      Pending& p = pending_.front();
      if (p.gen >= gen || p.due > now) break;
      if (failed_)
        finish(p, status_);
      else if (p.reads.empty())
        do_write(p);
      else
        answer(p);
      pending_.pop_front();
    }
  }

  void wait(int, int) {}
  uint64_t outstanding() const {
    uint64_t n = 0;
    for (const auto& p : pending_) n += p.reads.empty() ? 1 : p.reads.size();
    return n;
  }
  uint64_t requests() const { return requests_; }
  std::vector<Completion>& completions() { return done_; }
  bool failed() const { return failed_; }
  const std::string& failure_reason() const { return reason_; }
  void fail(Status status, const std::string& reason) {
    if (failed_) return;
    failed_ = true;
    status_ = status;
    reason_ = reason.empty() ? status_string(status) : reason;
  }

  void close() {
    for (Pending& p : pending_) finish(p, kCanceled);
    pending_.clear();
    regions_.clear();
  }

 private:
  struct Pending {
    uint64_t req_id = 0;
    std::vector<ReadOp> reads;  // empty for a write
    ReadOp write{};
    uint64_t write_bytes = 0;
    uint64_t gen = 0;
    double due = 0;
  };

  Pending make_pending() {
    Pending p;
    p.gen = gen_;
    p.due = delay_us_ > 0 ? now_us() + delay_us_ : 0;
    return p;
  }

  void count_op() {
    if (fail_after_ >= 0 && ++data_ops_ > fail_after_)
      fail(kInjected, "injected failure");
  }

  void finish(const Pending& p, Status st) {
    if (p.reads.empty()) {
      done_.push_back({p.write.token, st});
      return;
    }
    for (const ReadOp& op : p.reads) done_.push_back({op.token, st});
  }

  void do_write(const Pending& p) {
    const ReadOp& w = p.write;
    if (w.region >= regions_.size() ||
        !ple_mem::in_bounds(w.offset, p.write_bytes, sizes_[w.region])) {
      fail(kViolation, "write outside the borrowed regions");
      return finish(p, status_);
    }
    std::memcpy(regions_[w.region].get() + w.offset, w.dst, p.write_bytes);
    finish(p, kOk);
  }

  // Builds the reply like the server would, then accepts it like the UCX
  // client does.
  void answer(const Pending& p) {
    const uint32_t rb = p.reads[0].bytes;
    const uint32_t count = static_cast<uint32_t>(p.reads.size());
    reply_.resize(size_t{count} * rb);
    ple_mem::ReadRep rep = ple_mem::make<ple_mem::ReadRep>(ple_mem::kReadRep);
    rep.req_id = p.req_id;
    rep.count = count;
    for (uint32_t i = 0; i < count; ++i) {
      const ReadOp& op = p.reads[i];
      if (op.region >= regions_.size() ||
          !ple_mem::in_bounds(op.offset, rb, sizes_[op.region])) {
        rep.status = ple_mem::kErrBounds;
        break;
      }
      std::memcpy(&reply_[size_t{i} * rb],
                  regions_[op.region].get() + op.offset, rb);
    }
    size_t length = rep.status == ple_mem::kOk ? reply_.size() : 0;
    if (corrupt_ == 1) length -= 1;
    if (corrupt_ == 2) rep.req_id += 1000;
    corrupt_ = 0;
    if (rep.req_id != p.req_id) {
      fail(kViolation,
           "protocol violation by loopback: unsolicited read reply");
      return finish(p, status_);
    }
    if (const char* e = ple_mem::read_rep_error(rep, length, count, rb)) {
      fail(kViolation,
           std::string("protocol violation by loopback: read reply: ") + e);
      return finish(p, status_);
    }
    if (rep.status != ple_mem::kOk) {
      fail(kViolation, std::string("loopback rejected a read: ") +
                           ple_mem::status_name(rep.status));
      return finish(p, status_);
    }
    for (uint32_t i = 0; i < count; ++i)
      std::memcpy(p.reads[i].dst, &reply_[size_t{i} * rb], rb);
    finish(p, kOk);
  }

  const Params& params_;
  int64_t fail_after_ = -1;
  bool refuse_ = false;
  double delay_us_ = 0;
  int corrupt_ = 0;
  int64_t data_ops_ = 0;
  bool failed_ = false;
  Status status_ = kOk;
  std::string reason_;
  uint64_t gen_ = 0;
  uint64_t next_id_ = 1;
  uint64_t requests_ = 0;
  std::deque<Pending> pending_;
  std::vector<std::unique_ptr<uint8_t[]>> regions_;
  std::vector<uint64_t> sizes_;
  std::vector<uint8_t> reply_;
  std::vector<Completion> done_;
};

}  // namespace ple_remote
