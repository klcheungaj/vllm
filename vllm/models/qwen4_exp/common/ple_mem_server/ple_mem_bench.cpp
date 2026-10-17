// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PLE memory server benchmark: borrows memory from a running ple_mem_server
// and measures what the PLE row reader relies on: upload bandwidth (large
// PUTs + flush), large GET bandwidth, and random small GETs (the row reads)
// at several queue depths with per-op latency percentiles.
//
// Needs only UCX; build with the Makefile next to this file.

#include <ucp/api/ucp.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ple_mem_protocol.h"
#include "ple_mem_util.h"

namespace {

double now_us() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

struct Options {
  std::string server;
  uint64_t bytes = 1ull << 30;
  uint64_t region_bytes = 1ull << 30;
  uint64_t chunk_bytes = 64ull << 20;
  uint64_t row_bytes = 160;
  std::vector<uint32_t> depths{1, 16, 64, 256};
  uint32_t upload_depth = 4;
  uint64_t iters = 100000;
  double seconds = 0;
  bool iters_set = false;
  bool verify = false;
  bool json = false;
  double timeout_s = 30;
  uint64_t seed = 1;
};

void usage(const char* prog) {
  std::fprintf(
      stderr,
      "usage: %s --server HOST:PORT [options]\n"
      "  --server HOST:PORT    memory server address ([v6]:port for IPv6)\n"
      "  --bytes N             memory to borrow (default 1G; K/M/G/T)\n"
      "  --region-bytes N      size of each borrowed region (default 1G)\n"
      "  --chunk-bytes N       PUT / large GET size (default 64M)\n"
      "  --upload-depth N      large PUTs / GETs in flight (default 4)\n"
      "  --row-bytes N         small GET size (default 160)\n"
      "  --depth D[,D...]      small GETs in flight (default 1,16,64,256)\n"
      "  --iters N             small GETs per depth (default 100000)\n"
      "  --seconds S           time per depth instead of --iters\n"
      "  --verify              fill with an offset pattern, check every GET\n"
      "  --json                print one JSON object instead of tables\n"
      "  --timeout S           fail if an operation stalls (default 30)\n"
      "  --seed N              RNG seed for small GET offsets (default 1)\n",
      prog);
}

bool parse_args(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
    auto size = [&](uint64_t* out) {
      if (v == nullptr || !ple_mem::parse_size(v, out)) return false;
      ++i;
      return true;
    };
    auto number = [&](double* out) {
      if (v == nullptr) return false;
      char* end = nullptr;
      *out = std::strtod(v, &end);
      if (*end != '\0') return false;
      ++i;
      return true;
    };
    uint64_t n = 0;
    bool ok = true;
    if (a == "--server" && v != nullptr) {
      o->server = v;
      ++i;
    } else if (a == "--bytes") {
      ok = size(&o->bytes);
    } else if (a == "--region-bytes") {
      ok = size(&o->region_bytes);
    } else if (a == "--chunk-bytes") {
      ok = size(&o->chunk_bytes);
    } else if (a == "--row-bytes") {
      ok = size(&o->row_bytes);
    } else if (a == "--upload-depth") {
      ok = size(&n) && n > 0;
      o->upload_depth = static_cast<uint32_t>(n);
    } else if (a == "--depth" && v != nullptr) {
      o->depths.clear();
      std::string list = v;
      ++i;
      for (size_t p = 0; ok && p <= list.size();) {
        const size_t c = std::min(list.find(',', p), list.size());
        ok = ple_mem::parse_size(list.substr(p, c - p).c_str(), &n) && n > 0;
        o->depths.push_back(static_cast<uint32_t>(n));
        p = c + 1;
      }
    } else if (a == "--iters") {
      ok = size(&o->iters);
      o->iters_set = true;
    } else if (a == "--seconds") {
      ok = number(&o->seconds) && o->seconds > 0;
    } else if (a == "--timeout") {
      ok = number(&o->timeout_s) && o->timeout_s > 0;
    } else if (a == "--seed") {
      ok = size(&o->seed);
    } else if (a == "--verify") {
      o->verify = true;
    } else if (a == "--json") {
      o->json = true;
    } else {
      if (a != "-h" && a != "--help")
        std::fprintf(stderr, "ple_mem_bench: bad argument %s\n", a.c_str());
      return false;
    }
    if (!ok) {
      std::fprintf(stderr, "ple_mem_bench: bad value for %s\n", a.c_str());
      return false;
    }
  }
  if (o->seconds > 0 && !o->iters_set) o->iters = 0;
  if (o->server.empty()) {
    std::fprintf(stderr, "ple_mem_bench: --server is required\n");
    return false;
  }
  if (o->bytes == 0 || o->region_bytes < o->row_bytes || o->row_bytes == 0 ||
      o->chunk_bytes == 0 || o->bytes < o->row_bytes) {
    std::fprintf(stderr,
                 "ple_mem_bench: need bytes >= region-bytes >= row-bytes > 0 "
                 "and chunk-bytes > 0\n");
    return false;
  }
  return true;
}

// Byte at global offset `off` of the verify pattern (little-endian words).
uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
uint8_t pattern(uint64_t off) {
  return static_cast<uint8_t>(mix(off >> 3) >> ((off & 7) * 8));
}
void fill(uint8_t* p, uint64_t off, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = pattern(off + i);
}
uint64_t mismatches(const uint8_t* p, uint64_t off, size_t n) {
  uint64_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += p[i] != pattern(off + i);
  return bad;
}

struct Region {
  uint64_t id = 0;
  uint64_t addr = 0;
  uint64_t bytes = 0;
  ucp_rkey_h rkey = nullptr;
};

class Bench;

// One in-flight transfer; may span several regions (several requests).
struct Op {
  Bench* bench = nullptr;
  uint32_t slot = 0;
  uint64_t off = 0;
  uint64_t len = 0;
  double t0 = 0;
  int pending = 0;
  ucs_status_t status = UCS_OK;
};

struct Result {
  uint32_t depth;
  uint64_t ops;
  double seconds;
  double p50, p99, p999, max;
};

class Bench {
 public:
  explicit Bench(const Options& o) : opt_(o) {}
  ~Bench() { shutdown(); }

  int run() {
    connect();
    borrow();
    if (!opt_.json)
      std::printf("server %s: borrowed %s in %zu regions\n",
                  opt_.server.c_str(), ple_mem::human_bytes(opt_.bytes).c_str(),
                  regions_.size());
    up_ = bulk(true);
    down_ = bulk(false);
    for (uint32_t d : opt_.depths) small_.push_back(small_gets(d));
    release();
    report();
    if (mismatched_ != 0)
      std::fprintf(stderr, "ple_mem_bench: %" PRIu64 " mismatched bytes\n",
                   mismatched_);
    return mismatched_ == 0 ? 0 : 1;
  }

 private:
  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
  }
  void check(ucs_status_t st, const char* what) {
    if (st != UCS_OK) fail(std::string(what) + ": " + ucs_status_string(st));
  }

  void connect() {
    sockaddr_storage addr{};
    socklen_t addrlen = 0;
    std::string err;
    if (!ple_mem::resolve_address(opt_.server, &addr, &addrlen, &err))
      fail(err);
    ucp_config_t* config = nullptr;
    check(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_AM;
    const ucs_status_t st = ucp_init(&params, config, &ctx_);
    ucp_config_release(config);
    check(st, "ucp_init");
    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    check(ucp_worker_create(ctx_, &wp, &worker_), "ucp_worker_create");
    ucp_am_handler_param_t am{};
    am.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am.id = ple_mem::kAmReplyId;
    am.flags = UCP_AM_FLAG_WHOLE_MSG;
    am.cb = &Bench::reply_cb;
    am.arg = this;
    check(ucp_worker_set_am_recv_handler(worker_, &am), "am handler");
    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
    ep.sockaddr.addrlen = addrlen;
    ep.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep.err_handler.cb = &Bench::ep_err_cb;
    ep.err_handler.arg = this;
    check(ucp_ep_create(worker_, &ep, &ep_), "ucp_ep_create");
  }

  static void ep_err_cb(void* arg, ucp_ep_h, ucs_status_t status) {
    auto* b = static_cast<Bench*>(arg);
    if (b->ep_status_ == UCS_OK) b->ep_status_ = status;
  }

  static ucs_status_t reply_cb(void* arg, const void*, size_t, void* data,
                               size_t length, const ucp_am_recv_param_t* p) {
    if (p->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) return UCS_OK;
    const char* d = static_cast<const char*>(data);
    static_cast<Bench*>(arg)->replies_.emplace_back(d, d + length);
    return UCS_OK;
  }

  static void op_cb(void* request, ucs_status_t status, void* user_data) {
    auto* op = static_cast<Op*>(user_data);
    if (status != UCS_OK) op->status = status;
    if (--op->pending == 0) op->bench->done_.push_back(op->slot);
    ucp_request_free(request);
  }

  void progress() {
    ucp_worker_progress(worker_);
    if (ep_status_ != UCS_OK)
      fail(std::string("connection failed: ") + ucs_status_string(ep_status_));
  }

  template <typename Pred>
  void wait_until(Pred done, const char* what) {
    const double deadline = now_us() + opt_.timeout_s * 1e6;
    while (!done()) {
      progress();
      if (now_us() > deadline)
        fail(std::string("timed out waiting for ") + what);
    }
  }

  void wait_request(void* req, const char* what) {
    if (req == nullptr) return;
    if (UCS_PTR_IS_ERR(req)) check(UCS_PTR_STATUS(req), what);
    wait_until([&] { return ucp_request_check_status(req) != UCS_INPROGRESS; },
               what);
    const ucs_status_t st = ucp_request_check_status(req);
    ucp_request_free(req);
    check(st, what);
  }

  void am_send(const void* msg, size_t len) {
    sends_.emplace_back(static_cast<const char*>(msg),
                        static_cast<const char*>(msg) + len);
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY;
    void* req = ucp_am_send_nbx(ep_, ple_mem::kAmRequestId, nullptr, 0,
                                sends_.back().data(), len, &p);
    if (UCS_PTR_IS_ERR(req)) check(UCS_PTR_STATUS(req), "am send");
    if (req != nullptr) pending_sends_.push_back(req);
  }

  void flush_sends(const char* what) {
    for (void* r : pending_sends_) wait_request(r, what);
    pending_sends_.clear();
    sends_.clear();
  }

  void borrow() {
    const uint64_t n = (opt_.bytes + opt_.region_bytes - 1) / opt_.region_bytes;
    regions_.resize(n);
    sends_.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      ple_mem::AllocReq req{};
      req.hdr = ple_mem::make_header(ple_mem::kAllocReq);
      req.cookie = i;
      req.bytes =
          std::min(opt_.region_bytes, opt_.bytes - i * opt_.region_bytes);
      am_send(&req, sizeof(req));
    }
    wait_until([&] { return replies_.size() >= n; }, "region replies");
    flush_sends("alloc request");
    std::string refused;
    for (const auto& m : replies_) {
      ple_mem::AllocRep rep;
      if (ple_mem::message_type(m.data(), m.size()) != ple_mem::kAllocRep ||
          m.size() < sizeof(rep))
        fail("invalid reply from the server");
      std::memcpy(&rep, m.data(), sizeof(rep));
      if (rep.cookie >= n) fail("unexpected reply from the server");
      if (rep.status != 0) {
        rep.error[sizeof(rep.error) - 1] = '\0';
        if (refused.empty()) refused = rep.error;
        continue;
      }
      if (m.size() < sizeof(rep) + rep.rkey_len) fail("truncated reply");
      Region& r = regions_[rep.cookie];
      r.id = rep.region_id;
      r.addr = rep.addr;
      r.bytes = rep.bytes;
      check(ucp_ep_rkey_unpack(ep_, m.data() + sizeof(rep), &r.rkey),
            "ucp_ep_rkey_unpack");
    }
    replies_.clear();
    if (!refused.empty()) fail("server refused a region: " + refused);
  }

  ucp_mem_h map(void* addr, size_t len) {
    ucp_mem_map_params_t mp{};
    mp.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = addr;
    mp.length = len;
    ucp_mem_h memh = nullptr;
    check(ucp_mem_map(ctx_, &mp, &memh), "ucp_mem_map");
    memhs_.push_back(memh);
    return memh;
  }

  uint8_t* alloc(size_t len) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, std::max<size_t>(len, 1)) != 0)
      fail("out of memory");
    buffers_.push_back(p);
    std::memset(p, 0, len);
    return static_cast<uint8_t*>(p);
  }

  // Posts a PUT or GET of [off, off + len) of the borrowed space, split at
  // region boundaries, completing `op` once every piece is done.
  void post(bool put, uint8_t* local, ucp_mem_h memh, Op& op) {
    op.pending = 1;
    op.status = UCS_OK;
    op.t0 = now_us();
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_MEMH;
    p.cb.send = &Bench::op_cb;
    p.user_data = &op;
    p.memh = memh;
    for (uint64_t pos = op.off; pos < op.off + op.len;) {
      const Region& r = regions_[pos / opt_.region_bytes];
      const uint64_t in = pos % opt_.region_bytes;
      const uint64_t take = std::min(op.off + op.len - pos, r.bytes - in);
      uint8_t* buf = local + (pos - op.off);
      void* req = put ? ucp_put_nbx(ep_, buf, take, r.addr + in, r.rkey, &p)
                      : ucp_get_nbx(ep_, buf, take, r.addr + in, r.rkey, &p);
      if (UCS_PTR_IS_ERR(req)) op.status = UCS_PTR_STATUS(req);
      if (UCS_PTR_IS_PTR(req)) ++op.pending;
      pos += take;
    }
    if (--op.pending == 0) done_.push_back(op.slot);
  }

  struct Bulk {
    double seconds = 0;
    uint64_t bytes = 0;
  };

  // Streams the whole borrowed space in chunk-sized PUTs (then a flush) or
  // GETs with upload_depth transfers in flight.
  Bulk bulk(bool put) {
    const uint32_t depth = opt_.upload_depth;
    if (bulk_bufs_.empty())
      for (uint32_t s = 0; s < depth; ++s) {
        bulk_bufs_.push_back(alloc(opt_.chunk_bytes));
        bulk_memh_.push_back(map(bulk_bufs_.back(), opt_.chunk_bytes));
      }
    std::vector<Op> ops(depth);
    std::vector<uint32_t> free_slots;
    for (uint32_t s = 0; s < depth; ++s) {
      ops[s].bench = this;
      ops[s].slot = s;
      free_slots.push_back(s);
    }
    uint64_t next = 0;
    const double t0 = now_us();
    double last = t0;
    while (next < opt_.bytes || free_slots.size() < depth) {
      while (!free_slots.empty() && next < opt_.bytes) {
        const uint32_t s = free_slots.back();
        free_slots.pop_back();
        Op& op = ops[s];
        op.off = next;
        op.len = std::min(opt_.chunk_bytes, opt_.bytes - next);
        if (put && opt_.verify) fill(bulk_bufs_[s], op.off, op.len);
        post(put, bulk_bufs_[s], bulk_memh_[s], op);
        next += op.len;
      }
      progress();
      for (uint32_t s : done_) {
        Op& op = ops[s];
        check(op.status, put ? "PUT" : "GET");
        if (!put && opt_.verify)
          mismatched_ += mismatches(bulk_bufs_[s], op.off, op.len);
        free_slots.push_back(s);
        last = now_us();
      }
      done_.clear();
      if (now_us() - last > opt_.timeout_s * 1e6)
        fail(put ? "upload stalled" : "large GETs stalled");
    }
    if (put) {
      ucp_request_param_t fp{};
      wait_request(ucp_ep_flush_nbx(ep_, &fp), "flush");
    }
    return Bulk{(now_us() - t0) / 1e6, opt_.bytes};
  }

  Result small_gets(uint32_t depth) {
    const uint64_t rb = opt_.row_bytes;
    uint8_t* slab = alloc(size_t{depth} * rb);
    ucp_mem_h memh = map(slab, size_t{depth} * rb);
    std::vector<Op> ops(depth);
    std::mt19937_64 rng(opt_.seed + depth);
    const uint64_t rows = opt_.bytes / rb;
    auto pick = [&]() {
      uint64_t off = (rng() % rows) * rb;
      const uint64_t region_end = std::min(
          opt_.bytes, (off / opt_.region_bytes + 1) * opt_.region_bytes);
      return std::min(off, region_end - rb);
    };
    std::vector<double> lat;
    lat.reserve(opt_.iters ? opt_.iters : 1 << 20);
    uint64_t issued = 0, inflight = 0;
    const double t0 = now_us();
    const double end = opt_.seconds > 0 ? t0 + opt_.seconds * 1e6 : 0;
    auto more = [&]() {
      if (opt_.iters && issued >= opt_.iters) return false;
      return end == 0 || now_us() < end;
    };
    auto issue = [&](uint32_t s) {
      Op& op = ops[s];
      op.bench = this;
      op.slot = s;
      op.off = pick();
      op.len = rb;
      post(false, slab + size_t{s} * rb, memh, op);
      ++issued;
      ++inflight;
    };
    for (uint32_t s = 0; s < depth && more(); ++s) issue(s);
    double last = now_us();
    while (inflight > 0) {
      progress();
      const double now = now_us();
      std::vector<uint32_t> done;
      done.swap(done_);
      for (uint32_t s : done) {
        Op& op = ops[s];
        check(op.status, "small GET");
        lat.push_back(now - op.t0);
        if (opt_.verify)
          mismatched_ += mismatches(slab + size_t{s} * rb, op.off, rb);
        --inflight;
        last = now;
        if (more()) issue(s);
      }
      if (now - last > opt_.timeout_s * 1e6) fail("small GETs stalled");
    }
    const double secs = (now_us() - t0) / 1e6;
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double q) {
      if (lat.empty()) return 0.0;
      return lat[std::min(lat.size() - 1,
                          static_cast<size_t>(q * (lat.size() - 1) + 0.5))];
    };
    return Result{depth,
                  lat.size(),
                  secs,
                  pct(0.5),
                  pct(0.99),
                  pct(0.999),
                  lat.empty() ? 0.0 : lat.back()};
  }

  void release() {
    for (const Region& r : regions_) {
      if (r.rkey == nullptr) continue;
      ple_mem::FreeReq req{};
      req.hdr = ple_mem::make_header(ple_mem::kFreeReq);
      req.region_id = r.id;
      am_send(&req, sizeof(req));
    }
    flush_sends("free request");
    ucp_request_param_t fp{};
    wait_request(ucp_ep_flush_nbx(ep_, &fp), "flush");
    destroy_rkeys();
    ucp_request_param_t cp{};
    void* req = ucp_ep_close_nbx(ep_, &cp);
    ep_ = nullptr;
    wait_request(req, "endpoint close");
  }

  void destroy_rkeys() {
    for (Region& r : regions_) {
      if (r.rkey != nullptr) ucp_rkey_destroy(r.rkey);
      r.rkey = nullptr;
    }
  }

  // Best-effort cleanup, also after a failure.
  void shutdown() {
    if (worker_ != nullptr && ep_ != nullptr) {
      ucp_request_param_t cp{};
      cp.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
      cp.flags = UCP_EP_CLOSE_FLAG_FORCE;
      void* req = ucp_ep_close_nbx(ep_, &cp);
      ep_ = nullptr;
      const double deadline = now_us() + 2e6;
      while (UCS_PTR_IS_PTR(req) &&
             ucp_request_check_status(req) == UCS_INPROGRESS &&
             now_us() < deadline)
        ucp_worker_progress(worker_);
      if (UCS_PTR_IS_PTR(req)) ucp_request_free(req);
    }
    destroy_rkeys();
    for (ucp_mem_h m : memhs_) ucp_mem_unmap(ctx_, m);
    memhs_.clear();
    if (worker_ != nullptr) ucp_worker_destroy(worker_);
    worker_ = nullptr;
    if (ctx_ != nullptr) ucp_cleanup(ctx_);
    ctx_ = nullptr;
    for (void* p : buffers_) std::free(p);
    buffers_.clear();
  }

  void report() const {
    auto gbps = [](const Bulk& b) {
      return b.seconds > 0 ? b.bytes / b.seconds / 1e9 : 0.0;
    };
    if (opt_.json) {
      std::printf("{\"server\": \"%s\", \"bytes\": %" PRIu64
                  ", \"regions\": %zu, "
                  "\"chunk_bytes\": %" PRIu64
                  ", \"upload_depth\": %u, "
                  "\"row_bytes\": %" PRIu64
                  ", \"verify\": %s, \"mismatched_bytes\": "
                  "%" PRIu64
                  ",\n \"upload\": {\"seconds\": %.6f, \"GBps\": %.4f},\n"
                  " \"large_get\": {\"seconds\": %.6f, \"GBps\": %.4f},\n"
                  " \"small_get\": [",
                  opt_.server.c_str(), opt_.bytes, regions_.size(),
                  opt_.chunk_bytes, opt_.upload_depth, opt_.row_bytes,
                  opt_.verify ? "true" : "false", mismatched_, up_.seconds,
                  gbps(up_), down_.seconds, gbps(down_));
      for (size_t i = 0; i < small_.size(); ++i) {
        const Result& r = small_[i];
        const double ops = r.seconds > 0 ? r.ops / r.seconds : 0;
        std::printf("%s\n  {\"depth\": %u, \"ops\": %" PRIu64
                    ", \"seconds\": %.6f, \"ops_per_s\": %.1f, \"MBps\": %.3f, "
                    "\"p50_us\": %.2f, \"p99_us\": %.2f, \"p999_us\": %.2f, "
                    "\"max_us\": %.2f}",
                    i ? "," : "", r.depth, r.ops, r.seconds, ops,
                    ops * opt_.row_bytes / 1e6, r.p50, r.p99, r.p999, r.max);
      }
      std::printf("]}\n");
      return;
    }
    std::printf("upload     %s chunks x %u: %8.3f GB/s (%.3f s)\n",
                ple_mem::human_bytes(opt_.chunk_bytes).c_str(),
                opt_.upload_depth, gbps(up_), up_.seconds);
    std::printf("large GET  %s chunks x %u: %8.3f GB/s (%.3f s)\n",
                ple_mem::human_bytes(opt_.chunk_bytes).c_str(),
                opt_.upload_depth, gbps(down_), down_.seconds);
    std::printf("small GETs of %" PRIu64 " B%s:\n", opt_.row_bytes,
                opt_.verify ? " (verified)" : "");
    std::printf("%7s %10s %12s %9s %9s %9s %9s %9s\n", "depth", "ops", "ops/s",
                "MB/s", "p50 us", "p99 us", "p99.9 us", "max us");
    for (const Result& r : small_) {
      const double ops = r.seconds > 0 ? r.ops / r.seconds : 0;
      std::printf("%7u %10" PRIu64 " %12.0f %9.2f %9.2f %9.2f %9.2f %9.2f\n",
                  r.depth, r.ops, ops, ops * opt_.row_bytes / 1e6, r.p50, r.p99,
                  r.p999, r.max);
    }
  }

  Options opt_;
  ucp_context_h ctx_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  ucp_ep_h ep_ = nullptr;
  ucs_status_t ep_status_ = UCS_OK;
  std::vector<Region> regions_;
  std::vector<std::vector<char>> replies_;
  std::vector<std::vector<char>> sends_;
  std::vector<void*> pending_sends_;
  std::vector<uint32_t> done_;
  std::vector<ucp_mem_h> memhs_;
  std::vector<void*> buffers_;
  std::vector<uint8_t*> bulk_bufs_;
  std::vector<ucp_mem_h> bulk_memh_;
  Bulk up_, down_;
  std::vector<Result> small_;
  uint64_t mismatched_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    usage(argv[0]);
    return 2;
  }
  try {
    Bench bench(opt);
    return bench.run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ple_mem_bench: error: %s\n", e.what());
    return 1;
  }
}
