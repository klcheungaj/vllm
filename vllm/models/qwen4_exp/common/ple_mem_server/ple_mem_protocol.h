// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Wire protocol between the PLE memory server (ple_mem_server.cpp) and the
// remote source of the PLE row reader (ple_disk_reader.cpp).
//
// Control messages are UCP active messages with an empty header and one of
// the POD payloads below. Clients send requests with UCP_AM_SEND_FLAG_REPLY
// so the server knows the owning endpoint; regions live until FREE_REQ or
// until that endpoint disconnects. Row data moves with one-sided RMA.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ple_mem {

constexpr uint32_t kMagic = 0x4D454C50;  // "PLEM"
constexpr uint16_t kVersion = 1;

// Requests are handled by the server, replies by the client.
constexpr unsigned kAmRequestId = 0x504C;
constexpr unsigned kAmReplyId = 0x504D;

enum MsgType : uint16_t {
  kAllocReq = 1,
  kAllocRep = 2,
  kFreeReq = 3,
};

struct MsgHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
};

struct AllocReq {
  MsgHeader hdr;
  uint64_t cookie;
  uint64_t bytes;
};

// Followed by `rkey_len` bytes of packed rkey when `status` is 0.
struct AllocRep {
  MsgHeader hdr;
  uint64_t cookie;
  int32_t status;
  uint64_t region_id;
  uint64_t addr;
  uint64_t bytes;
  uint32_t rkey_len;
  char error[128];
};

struct FreeReq {
  MsgHeader hdr;
  uint64_t region_id;
};

static_assert(std::is_trivially_copyable<AllocReq>::value, "POD");
static_assert(std::is_trivially_copyable<AllocRep>::value, "POD");
static_assert(std::is_trivially_copyable<FreeReq>::value, "POD");

inline MsgHeader make_header(MsgType type) {
  return MsgHeader{kMagic, kVersion, static_cast<uint16_t>(type)};
}

// Returns the message type, or 0 when `data` is not a valid message.
inline uint16_t message_type(const void* data, size_t length) {
  MsgHeader h;
  if (data == nullptr || length < sizeof(h)) return 0;
  std::memcpy(&h, data, sizeof(h));
  if (h.magic != kMagic || h.version != kVersion) return 0;
  return h.type;
}

}  // namespace ple_mem
