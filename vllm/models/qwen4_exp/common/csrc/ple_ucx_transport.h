// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// UCX transport of the PLE remote memory source (see the Transport concept in
// ple_remote_source.h): the request / response client of the PLE memory
// server (../ple_mem_server/ple_mem_client.h). The client exposes no memory
// to the server and validates every reply before copying it out; UCX's tcp
// transport is excluded unless `allow_insecure_transport`.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "../ple_mem_server/ple_mem_client.h"
#include "../ple_mem_server/ple_mem_util.h"
#include "ple_remote_source.h"

namespace ple_remote {

class UcxTransport {
 public:
  using Status = ucs_status_t;
  using ReadOp = ple_mem::ReadOp;
  using Completion = ple_mem::Completion;
  static constexpr Status kOk = UCS_OK;
  static constexpr Status kCanceled = UCS_ERR_CANCELED;
  static constexpr const char* kName = "ucx";

  explicit UcxTransport(const Params& p) : params_(p) {}
  UcxTransport(const UcxTransport&) = delete;
  UcxTransport& operator=(const UcxTransport&) = delete;

  static const char* status_string(Status s) { return ucs_status_string(s); }

  std::string open(uint32_t read_tokens) {
    ple_mem::ClientOptions o;
    o.address = params_.address;
    o.allow_tcp = params_.allow_insecure_transport;
    o.read_tokens = read_tokens;
    o.max_batch_rows = params_.max_batch_rows;
    if (!params_.token_file.empty()) {
      std::string err;
      if (!ple_mem::read_token_file(params_.token_file, ple_mem::kMaxTokenBytes,
                                    &o.token, &err))
        return err;
    }
    return client_.open(o);
  }

  std::string borrow(const std::vector<uint64_t>& region_bytes,
                     double timeout_s, const std::atomic<bool>& stop,
                     std::atomic<uint64_t>& granted) {
    return client_.borrow(region_bytes, timeout_s, &stop, &granted);
  }

  void read(const ReadOp* ops, uint32_t n) { client_.read(ops, n); }
  void write(uint32_t token, const void* src, uint32_t region, uint64_t offset,
             uint64_t bytes) {
    client_.write(token, src, region, offset, bytes);
  }
  void progress() { client_.progress(); }
  void wait(int wake_fd, int timeout_ms) { client_.wait(wake_fd, timeout_ms); }
  uint64_t outstanding() const { return client_.outstanding(); }
  uint64_t requests() const { return client_.requests_sent(); }
  std::vector<Completion>& completions() { return client_.completions(); }
  bool failed() const { return client_.failed(); }
  const std::string& failure_reason() const { return client_.failure_reason(); }
  void fail(Status status, const std::string& reason) {
    client_.fail(status, reason);
  }
  void close() { client_.close(); }

 private:
  const Params& params_;
  ple_mem::Client client_;
};

}  // namespace ple_remote
