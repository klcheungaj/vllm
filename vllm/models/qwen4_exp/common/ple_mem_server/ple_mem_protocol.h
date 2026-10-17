// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Wire protocol between the PLE memory server (ple_mem_server.cpp) and its
// clients (ple_mem_client.h: the PLE row reader and ple_mem_bench).
//
// Request / response over UCP active messages only. Each message is a small
// fixed struct sent as the AM user header, plus a variable payload as the AM
// data. Clients send requests to kAmRequestId with UCP_AM_SEND_FLAG_REPLY (the
// server keys connections by the reply endpoint); the server answers on
// kAmReplyId. Every message is sent eagerly (UCP_AM_SEND_FLAG_EAGER): no
// rendezvous, so neither side ever discloses a remote key. The server's
// regions are plain memory that only its own request handlers touch, and the
// server only ever sends replies to requests, so it cannot write a client's
// memory: the client copies reply bytes into its buffers after validating
// them against its own outstanding requests.
//
//   ALLOC_REQ {cookie, bytes, token_len} + token
//     -> ALLOC_REP {cookie, status, region_id, bytes} + error text
//   FREE_REQ {region_id}                                     (no reply)
//   WRITE_REQ {seq, region_id, offset, length} + data
//     -> WRITE_ACK {seq, status}   (sent once the data is in the region)
//   READ_REQ {req_id, row_bytes, count} + ReadEntry[count]
//     -> READ_REP {req_id, status, count} + count * row_bytes (entry order;
//        no data unless status is kOk)
//
// Structs are native-endian with explicit padding; both ends must share the
// byte order (a version mismatch or a wrong magic is a protocol error).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ple_mem {

constexpr uint32_t kMagic = 0x4D454C50;  // "PLEM"
constexpr uint16_t kVersion = 2;

constexpr unsigned kAmRequestId = 0x504C;
constexpr unsigned kAmReplyId = 0x504D;

constexpr uint32_t kMaxTokenBytes = 256;
constexpr uint32_t kMaxErrorBytes = 128;
constexpr uint64_t kMaxWriteBytes = uint64_t{4} << 20;
constexpr uint32_t kMaxRowBytes = 4096;
constexpr uint32_t kMaxReadRows = 1024;
constexpr uint64_t kMaxReadReplyBytes = uint64_t{4} << 20;

enum MsgType : uint16_t {
  kAllocReq = 1,
  kAllocRep = 2,
  kFreeReq = 3,
  kWriteReq = 4,
  kWriteAck = 5,
  kReadReq = 6,
  kReadRep = 7,
};

enum Status : int32_t {
  kOk = 0,
  kErrInvalid = 1,   // malformed request
  kErrLimit = 2,     // memory or region limit reached
  kErrNoRegion = 3,  // region unknown or owned by another client
  kErrBounds = 4,    // access outside the region
  kErrAuth = 5,      // missing or wrong token
  kErrSystem = 6,    // mmap / prefault failure on the server
};

inline const char* status_name(int32_t s) {
  switch (s) {
    case kOk:
      return "ok";
    case kErrInvalid:
      return "invalid request";
    case kErrLimit:
      return "limit reached";
    case kErrNoRegion:
      return "unknown region";
    case kErrBounds:
      return "out of bounds";
    case kErrAuth:
      return "authentication failed";
    case kErrSystem:
      return "server system error";
  }
  return "unknown status";
}

struct MsgHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
};

struct AllocReq {
  MsgHeader hdr;
  uint32_t token_len;
  uint32_t pad;
  uint64_t cookie;
  uint64_t bytes;
};

struct AllocRep {
  MsgHeader hdr;
  int32_t status;
  uint32_t pad;
  uint64_t cookie;
  uint64_t region_id;
  uint64_t bytes;
};

struct FreeReq {
  MsgHeader hdr;
  uint64_t region_id;
};

struct WriteReq {
  MsgHeader hdr;
  uint64_t seq;
  uint64_t region_id;
  uint64_t offset;
  uint64_t length;
};

struct WriteAck {
  MsgHeader hdr;
  int32_t status;
  uint32_t pad;
  uint64_t seq;
};

struct ReadReq {
  MsgHeader hdr;
  uint32_t row_bytes;
  uint32_t count;
  uint64_t req_id;
};

struct ReadEntry {
  uint64_t region_id;
  uint64_t offset;
};

struct ReadRep {
  MsgHeader hdr;
  int32_t status;
  uint32_t count;
  uint64_t req_id;
};

template <class T>
constexpr bool kIsMessage =
    std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value;
static_assert(kIsMessage<AllocReq> && kIsMessage<AllocRep>, "POD");
static_assert(kIsMessage<FreeReq> && kIsMessage<WriteReq>, "POD");
static_assert(kIsMessage<WriteAck> && kIsMessage<ReadReq>, "POD");
static_assert(kIsMessage<ReadEntry> && kIsMessage<ReadRep>, "POD");

template <class T>
T make(MsgType type) {
  T m{};
  m.hdr = MsgHeader{kMagic, kVersion, static_cast<uint16_t>(type)};
  return m;
}

// Returns the message type of an AM user header, or 0 when it is not a valid
// header of this protocol version.
inline uint16_t message_type(const void* header, size_t length) {
  MsgHeader h;
  if (header == nullptr || length < sizeof(h)) return 0;
  std::memcpy(&h, header, sizeof(h));
  if (h.magic != kMagic || h.version != kVersion) return 0;
  return h.type;
}

// Copies the AM user header into `out` when it is exactly a T.
template <class T>
bool parse(const void* header, size_t length, T* out) {
  if (length != sizeof(T)) return false;
  std::memcpy(out, header, sizeof(T));
  return true;
}

// Returns null when a READ_REP header plus `data_length` payload bytes is a
// well-formed answer to a request for `count` rows of `row_bytes`, or why not.
inline const char* read_rep_error(const ReadRep& rep, size_t data_length,
                                  uint32_t count, uint32_t row_bytes) {
  if (rep.status != kOk) return data_length == 0 ? nullptr : "error with data";
  if (rep.count != count) return "wrong row count";
  if (data_length != uint64_t{count} * row_bytes) return "wrong data length";
  return nullptr;
}

// True when [offset, offset + length) lies within `size` bytes.
inline bool in_bounds(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size - offset;
}

}  // namespace ple_mem
