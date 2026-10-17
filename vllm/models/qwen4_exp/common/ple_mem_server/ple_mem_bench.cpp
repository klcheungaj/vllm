// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PLE memory server benchmark: borrows memory from a running ple_mem_server
// with the same client as the PLE row reader (ple_mem_client.h) and measures
// what the reader relies on: upload bandwidth (WRITE_REQ stream), the idle
// round-trip latency of a one-row read (ping), the latency of fetching one
// step's rows the way the reader does (N random rows as parallel READ_REQs of
// at most --step-batch rows, timed until the last row arrives), and batched
// reads at several batch sizes and depths with latency percentiles per
// request and per row. `--probe` checks that the server rejects
// out-of-bounds and foreign-region accesses.
//
// Needs only UCX; build with the Makefile next to this file.

#include <ucp/api/ucp.h>

#include <algorithm>
#include <atomic>
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

#include "ple_mem_client.h"
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
  uint32_t upload_depth = 4;
  uint32_t write_window = 8;
  std::vector<uint32_t> batch_rows{1, 16, 64, 256, 1024};
  std::vector<uint32_t> depths{1, 4, 16};
  uint64_t ping_iters = 10000;
  std::vector<uint32_t> step_rows{16, 128, 256, 1024, 4096};
  uint32_t step_batch = 256;
  uint64_t step_iters = 2000;
  bool sweep = true;
  uint64_t iters = 20000;
  double seconds = 0;
  bool iters_set = false;
  bool verify = false;
  bool json = false;
  bool probe = false;
  bool allow_tcp = false;
  std::string token_file;
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
      "  --chunk-bytes N       bytes per upload write (default 64M)\n"
      "  --upload-depth N      upload writes in flight (default 4)\n"
      "  --write-window N      WRITE_REQ messages in flight (default 8)\n"
      "  --row-bytes N         row size (default 160)\n"
      "  --batch-rows B[,B..]  rows per READ_REQ (default 1,16,64,256,1024)\n"
      "  --depth D[,D...]      READ_REQs in flight (default 1,4,16)\n"
      "  --iters N             READ_REQs per point (default 20000)\n"
      "  --seconds S           time per point instead of --iters\n"
      "  --no-sweep            skip the batch-rows x depth sweep\n"
      "  --ping-iters N        one-row round trips to time (default 10000;\n"
      "                        0 skips the ping)\n"
      "  --step-rows N[,N..]   rows per simulated step (default\n"
      "                        16,128,256,1024,4096)\n"
      "  --step-batch B        rows per READ_REQ within a step (default 256,\n"
      "                        like remote_max_batch_rows)\n"
      "  --step-iters N        steps to time per size (default 2000; 0 skips)\n"
      "  --verify              fill with an offset pattern, check every row\n"
      "  --probe               check that the server rejects out-of-bounds\n"
      "                        and foreign-region accesses\n"
      "  --token-file PATH     shared secret expected by the server\n"
      "  --allow-tcp           allow UCX's tcp transport (insecure)\n"
      "  --json                print one JSON object instead of tables\n"
      "  --timeout S           fail if an operation stalls (default 30)\n"
      "  --seed N              RNG seed for row offsets (default 1)\n",
      prog);
}

bool parse_list(const char* v, std::vector<uint32_t>* out) {
  out->clear();
  const std::string list = v;
  for (size_t p = 0; p <= list.size();) {
    const size_t c = std::min(list.find(',', p), list.size());
    uint64_t n = 0;
    if (!ple_mem::parse_size(list.substr(p, c - p).c_str(), &n) || n == 0 ||
        n > (1u << 20))
      return false;
    out->push_back(static_cast<uint32_t>(n));
    p = c + 1;
  }
  return !out->empty();
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
      if (end == v || *end != '\0') return false;
      ++i;
      return true;
    };
    auto list = [&](std::vector<uint32_t>* out) {
      if (v == nullptr || !parse_list(v, out)) return false;
      ++i;
      return true;
    };
    uint64_t n = 0;
    bool ok = true;
    if (a == "--server" && v != nullptr) {
      o->server = v;
      ++i;
    } else if (a == "--token-file" && v != nullptr) {
      o->token_file = v;
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
      ok = size(&n) && n > 0 && n <= 1024;
      o->upload_depth = static_cast<uint32_t>(n);
    } else if (a == "--write-window") {
      ok = size(&n) && n > 0 && n <= ple_mem::Client::kMaxWriteWindow;
      o->write_window = static_cast<uint32_t>(n);
    } else if (a == "--batch-rows") {
      ok = list(&o->batch_rows);
    } else if (a == "--depth") {
      ok = list(&o->depths);
    } else if (a == "--ping-iters") {
      ok = size(&o->ping_iters);
    } else if (a == "--step-rows") {
      ok = list(&o->step_rows);
    } else if (a == "--step-batch") {
      ok = size(&n) && n > 0 && n <= ple_mem::kMaxReadRows;
      o->step_batch = static_cast<uint32_t>(n);
    } else if (a == "--step-iters") {
      ok = size(&o->step_iters);
    } else if (a == "--no-sweep") {
      o->sweep = false;
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
    } else if (a == "--probe") {
      o->probe = true;
    } else if (a == "--allow-tcp") {
      o->allow_tcp = true;
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
  uint32_t max_batch = 0;
  for (uint32_t b : o->batch_rows) max_batch = std::max(max_batch, b);
  if (o->row_bytes == 0 || o->row_bytes > ple_mem::kMaxRowBytes ||
      o->region_bytes < o->row_bytes || o->bytes < o->region_bytes ||
      o->chunk_bytes == 0 || max_batch > ple_mem::kMaxReadRows ||
      max_batch * o->row_bytes > ple_mem::kMaxReadReplyBytes ||
      o->step_batch * o->row_bytes > ple_mem::kMaxReadReplyBytes) {
    std::fprintf(stderr,
                 "ple_mem_bench: need bytes >= region-bytes >= row-bytes, "
                 "row-bytes <= %u, chunk-bytes > 0 and batch-rows and "
                 "step-batch <= %u "
                 "(and <= 4 MiB per reply)\n",
                 ple_mem::kMaxRowBytes, ple_mem::kMaxReadRows);
    return false;
  }
  return true;
}

// Byte at global offset `off` of the verify pattern.
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

struct Percentiles {
  double min = 0, mean = 0, p50 = 0, p90 = 0, p99 = 0, p999 = 0, max = 0;
};

Percentiles percentiles(std::vector<double>& v) {
  Percentiles p;
  if (v.empty()) return p;
  std::sort(v.begin(), v.end());
  auto at = [&](double q) {
    return v[std::min(v.size() - 1,
                      static_cast<size_t>(q * (v.size() - 1) + 0.5))];
  };
  double sum = 0;
  for (double x : v) sum += x;
  p.min = v.front();
  p.mean = sum / static_cast<double>(v.size());
  p.p50 = at(0.5);
  p.p90 = at(0.9);
  p.p99 = at(0.99);
  p.p999 = at(0.999);
  p.max = v.back();
  return p;
}

// Latency of fetching `rows` random rows as one burst of READ_REQs.
struct Latency {
  uint32_t rows;
  uint32_t batch;
  uint64_t samples;
  Percentiles us;
};

struct Point {
  uint32_t batch;
  uint32_t depth;
  uint64_t requests;
  double seconds;
  Percentiles req, row;
};

class Bench {
 public:
  explicit Bench(const Options& o) : opt_(o) {
    for (uint32_t b : opt_.batch_rows) max_batch_ = std::max(max_batch_, b);
    for (uint32_t d : opt_.depths) max_depth_ = std::max(max_depth_, d);
    for (uint32_t n : opt_.step_rows) max_step_ = std::max(max_step_, n);
    if (!opt_.token_file.empty()) {
      std::string err;
      if (!ple_mem::read_token_file(opt_.token_file, ple_mem::kMaxTokenBytes,
                                    &token_, &err))
        throw std::runtime_error(err);
    }
  }

  int run() {
    connect(client_, opt_.bytes,
            std::max(max_batch_ * max_depth_,
                     opt_.step_iters ? max_step_ : uint32_t{1}));
    if (!opt_.json)
      std::printf("server %s: borrowed %s in %u regions\n", opt_.server.c_str(),
                  ple_mem::human_bytes(opt_.bytes).c_str(),
                  client_.num_regions());
    upload();
    if (opt_.ping_iters) ping_ = latency(1, 1, opt_.ping_iters);
    if (opt_.step_iters)
      for (uint32_t n : opt_.step_rows)
        steps_.push_back(latency(n, opt_.step_batch, opt_.step_iters));
    if (opt_.sweep)
      for (uint32_t b : opt_.batch_rows)
        for (uint32_t d : opt_.depths) points_.push_back(reads(b, d));
    int probe_failures = opt_.probe ? probe() : 0;
    client_.close();
    report();
    if (mismatched_ != 0)
      std::fprintf(stderr, "ple_mem_bench: %" PRIu64 " mismatched bytes\n",
                   mismatched_);
    return mismatched_ == 0 && probe_failures == 0 ? 0 : 1;
  }

 private:
  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
  }

  void connect(ple_mem::Client& c, uint64_t bytes, uint32_t read_tokens) {
    ple_mem::ClientOptions co;
    co.address = opt_.server;
    co.token = token_;
    co.allow_tcp = opt_.allow_tcp;
    co.read_tokens = std::max<uint32_t>(read_tokens, 1);
    co.max_batch_rows = std::max(max_batch_, opt_.step_batch);
    co.write_window = opt_.write_window;
    std::string err = c.open(co);
    if (!err.empty()) fail(err);
    std::vector<uint64_t> regions;
    for (uint64_t off = 0; off < bytes; off += opt_.region_bytes)
      regions.push_back(std::min(opt_.region_bytes, bytes - off));
    err = c.borrow(regions, opt_.timeout_s, nullptr);
    if (!err.empty()) fail(err);
  }

  // Progresses until `done()` holds; fails on a client error or a stall.
  template <typename Pred>
  void drive(ple_mem::Client& c, Pred done, const char* what) {
    const double start = now_us();
    while (!done()) {
      c.progress();
      if (c.failed() && c.completions().empty()) fail(c.failure_reason());
      if (now_us() - start > opt_.timeout_s * 1e6)
        fail(std::string(what) + " stalled");
    }
  }

  void upload() {
    const uint32_t depth = opt_.upload_depth;
    bufs_.assign(depth, std::vector<uint8_t>(opt_.chunk_bytes));
    std::vector<uint64_t> free_bufs;
    for (uint32_t b = depth; b-- > 0;) free_bufs.push_back(b);
    uint64_t next = 0, inflight = 0;
    const double t0 = now_us();
    while (next < opt_.bytes || inflight > 0) {
      while (!free_bufs.empty() && next < opt_.bytes) {
        const uint32_t b = static_cast<uint32_t>(free_bufs.back());
        free_bufs.pop_back();
        const uint64_t len = std::min(opt_.chunk_bytes, opt_.bytes - next);
        if (opt_.verify) fill(bufs_[b].data(), next, len);
        for (uint64_t pos = next; pos < next + len;) {
          const uint32_t region =
              static_cast<uint32_t>(pos / opt_.region_bytes);
          const uint64_t in = pos % opt_.region_bytes;
          const uint64_t take =
              std::min(next + len - pos,
                       std::min(opt_.region_bytes,
                                opt_.bytes - region * opt_.region_bytes) -
                           in);
          ++pending_[b];
          ++inflight;
          client_.write(b, bufs_[b].data() + (pos - next), region, in, take);
          pos += take;
        }
        next += len;
      }
      drive(client_, [&] { return !client_.completions().empty(); }, "upload");
      for (const auto& c : client_.completions()) {
        if (c.status != UCS_OK) fail(client_.failure_reason());
        --inflight;
        if (--pending_[c.token] == 0) free_bufs.push_back(c.token);
      }
      client_.completions().clear();
    }
    up_seconds_ = (now_us() - t0) / 1e6;
  }

  uint64_t total_rows() const {
    const uint64_t rows_per_region = opt_.region_bytes / opt_.row_bytes;
    return (opt_.bytes / opt_.region_bytes) * rows_per_region +
           (opt_.bytes % opt_.region_bytes) / opt_.row_bytes;
  }

  // Times `iters` closed-loop bursts: `rows` random rows posted at once as
  // READ_REQs of at most `batch` rows, each timed until its last row
  // arrives. The first few bursts warm up and are not recorded.
  Latency latency(uint32_t rows, uint32_t batch, uint64_t iters) {
    const uint64_t rb = opt_.row_bytes;
    const uint64_t rows_per_region = opt_.region_bytes / rb;
    const uint64_t total = total_rows();
    const uint64_t warmup = std::min<uint64_t>(100, iters / 10);
    slab_.assign(size_t{rows} * rb, 0);
    ops_.resize(rows);
    offsets_.assign(rows, 0);
    std::mt19937_64 rng(opt_.seed * 7919 + rows);
    std::vector<double> lat;
    lat.reserve(iters);
    for (uint64_t it = 0; it < warmup + iters; ++it) {
      for (uint32_t i = 0; i < rows; ++i) {
        const uint64_t row = rng() % total;
        const uint32_t region = static_cast<uint32_t>(row / rows_per_region);
        const uint64_t in = (row % rows_per_region) * rb;
        offsets_[i] = region * opt_.region_bytes + in;
        ops_[i] = ple_mem::ReadOp{i, slab_.data() + size_t{i} * rb, region, in,
                                  static_cast<uint32_t>(rb)};
      }
      const double t0 = now_us();
      for (uint32_t i = 0; i < rows; i += batch)
        client_.read(ops_.data() + i, std::min(batch, rows - i));
      uint32_t left = rows;
      while (left > 0) {
        drive(
            client_, [&] { return !client_.completions().empty(); }, "latency");
        for (const auto& c : client_.completions()) {
          if (c.status != UCS_OK) fail(client_.failure_reason());
          if (opt_.verify)
            mismatched_ += mismatches(slab_.data() + size_t{c.token} * rb,
                                      offsets_[c.token], rb);
          --left;
        }
        client_.completions().clear();
      }
      if (it >= warmup) lat.push_back(now_us() - t0);
    }
    return Latency{rows, batch, lat.size(), percentiles(lat)};
  }

  Point reads(uint32_t batch, uint32_t depth) {
    const uint64_t rb = opt_.row_bytes;
    const uint64_t rows_per_region = opt_.region_bytes / rb;
    const uint64_t total_rows = this->total_rows();
    slab_.assign(size_t{depth} * batch * rb, 0);
    ops_.resize(batch);
    offsets_.assign(size_t{depth} * batch, 0);
    std::vector<uint32_t> pending(depth, 0);
    std::vector<double> t0(depth, 0);
    std::mt19937_64 rng(opt_.seed + batch * 131 + depth);
    std::vector<double> lat;
    lat.reserve(opt_.iters ? opt_.iters : 1 << 20);
    uint64_t issued = 0;
    uint32_t inflight = 0;
    const double start = now_us();
    const double end = opt_.seconds > 0 ? start + opt_.seconds * 1e6 : 0;
    auto more = [&]() {
      if (opt_.iters && issued >= opt_.iters) return false;
      return end == 0 || now_us() < end;
    };
    auto issue = [&](uint32_t s) {
      for (uint32_t i = 0; i < batch; ++i) {
        const uint64_t row = rng() % total_rows;
        const uint32_t region = static_cast<uint32_t>(row / rows_per_region);
        const uint64_t in = (row % rows_per_region) * rb;
        const uint32_t token = s * batch + i;
        offsets_[token] = region * opt_.region_bytes + in;
        ops_[i] = ple_mem::ReadOp{token, slab_.data() + size_t{token} * rb,
                                  region, in, static_cast<uint32_t>(rb)};
      }
      pending[s] = batch;
      t0[s] = now_us();
      client_.read(ops_.data(), batch);
      ++issued;
      ++inflight;
    };
    for (uint32_t s = 0; s < depth && more(); ++s) issue(s);
    while (inflight > 0) {
      drive(client_, [&] { return !client_.completions().empty(); }, "reads");
      const double now = now_us();
      for (const auto& c : client_.completions()) {
        if (c.status != UCS_OK) fail(client_.failure_reason());
        if (opt_.verify)
          mismatched_ += mismatches(slab_.data() + size_t{c.token} * rb,
                                    offsets_[c.token], rb);
        const uint32_t s = c.token / batch;
        if (--pending[s] != 0) continue;
        lat.push_back(now - t0[s]);
        --inflight;
        if (more()) issue(s);
      }
      client_.completions().clear();
    }
    Point p{batch, depth, lat.size(), (now_us() - start) / 1e6, {}, {}};
    std::vector<double> per_row(lat);
    for (double& v : per_row) v /= batch;
    p.req = percentiles(lat);
    p.row = percentiles(per_row);
    return p;
  }

  // Each probe uses a fresh connection, since a rejected request fails it.
  int probe() {
    const uint64_t rb = opt_.row_bytes;
    const uint64_t region = std::max<uint64_t>(4096, rb);
    ple_mem::Client other;
    connect(other, region, 1);
    const uint64_t foreign = other.region_id(0);
    std::vector<uint8_t> buf(std::max<uint64_t>(rb, 2));
    struct Case {
      const char* name;
      bool write;
      bool foreign;
      uint64_t offset;
      uint64_t bytes;
      const char* expect;
    };
    const Case cases[] = {
        {"read past the region end", false, false, region - rb + 1, rb,
         "out of bounds"},
        {"read at offset 2^64-1", false, false, ~0ull, rb, "out of bounds"},
        {"write past the region end", true, false, region - 1, 2,
         "out of bounds"},
        {"read of another client's region", false, true, 0, rb,
         "unknown region"},
        {"write to another client's region", true, true, 0, rb,
         "unknown region"},
    };
    int failures = 0;
    for (const Case& k : cases) {
      ple_mem::Client c;
      connect(c, region, 1);
      const uint64_t id = k.foreign ? foreign : c.region_id(0);
      if (k.write)
        c.write_raw(0, buf.data(), id, k.offset, k.bytes);
      else
        c.read_raw(0, buf.data(), id, k.offset, static_cast<uint32_t>(k.bytes));
      const double deadline = now_us() + opt_.timeout_s * 1e6;
      while (c.completions().empty() && now_us() < deadline) c.progress();
      const bool rejected =
          c.failed() && c.failure_reason().find(k.expect) != std::string::npos;
      failures += !rejected;
      if (!opt_.json)
        std::printf("probe %-34s %s (%s)\n", k.name,
                    rejected ? "rejected" : "NOT REJECTED",
                    c.failed() ? c.failure_reason().c_str() : "accepted");
    }
    return failures;
  }

  void report() const {
    const double gbps = up_seconds_ > 0 ? opt_.bytes / up_seconds_ / 1e9 : 0;
    auto rows_per_s = [](const Point& p) {
      return p.seconds > 0 ? p.requests * double(p.batch) / p.seconds : 0.0;
    };
    if (opt_.json) {
      std::printf("{\"server\": \"%s\", \"bytes\": %" PRIu64
                  ", \"regions\": %u, \"chunk_bytes\": %" PRIu64
                  ", \"row_bytes\": %" PRIu64
                  ", \"verify\": %s, \"mismatched_bytes\": %" PRIu64
                  ",\n \"upload\": {\"seconds\": %.6f, \"GBps\": %.4f},\n",
                  opt_.server.c_str(), opt_.bytes, client_.num_regions(),
                  opt_.chunk_bytes, opt_.row_bytes,
                  opt_.verify ? "true" : "false", mismatched_, up_seconds_,
                  gbps);
      auto lat_json = [](const Latency& l) {
        std::printf("{\"rows\": %u, \"batch_rows\": %u, \"samples\": %" PRIu64
                    ", \"min_us\": %.2f, \"mean_us\": %.2f, \"p50_us\": "
                    "%.2f, \"p90_us\": %.2f, \"p99_us\": %.2f, "
                    "\"p999_us\": %.2f, \"max_us\": %.2f}",
                    l.rows, l.batch, l.samples, l.us.min, l.us.mean, l.us.p50,
                    l.us.p90, l.us.p99, l.us.p999, l.us.max);
      };
      std::printf(" \"ping\": ");
      if (opt_.ping_iters)
        lat_json(ping_);
      else
        std::printf("null");
      std::printf(",\n \"steps\": [");
      for (size_t i = 0; i < steps_.size(); ++i) {
        std::printf("%s\n  ", i ? "," : "");
        lat_json(steps_[i]);
      }
      std::printf("],\n \"reads\": [");
      for (size_t i = 0; i < points_.size(); ++i) {
        const Point& p = points_[i];
        const double r = rows_per_s(p);
        std::printf(
            "%s\n  {\"batch_rows\": %u, \"depth\": %u, \"requests\": %" PRIu64
            ", \"rows_per_s\": %.1f, \"MBps\": %.3f, \"req_p50_us\": %.2f, "
            "\"req_p99_us\": %.2f, \"req_p999_us\": %.2f, \"req_max_us\": "
            "%.2f, \"row_p50_us\": %.4f, \"row_p99_us\": %.4f, "
            "\"row_p999_us\": %.4f}",
            i ? "," : "", p.batch, p.depth, p.requests, r,
            r * opt_.row_bytes / 1e6, p.req.p50, p.req.p99, p.req.p999,
            p.req.max, p.row.p50, p.row.p99, p.row.p999);
      }
      std::printf("]}\n");
      return;
    }
    std::printf("upload %s writes x %u: %.3f GB/s (%.3f s)\n",
                ple_mem::human_bytes(opt_.chunk_bytes).c_str(),
                opt_.upload_depth, gbps, up_seconds_);
    auto lat_row = [](const char* label, const Latency& l) {
      std::printf("%-8s %6u %6u %8" PRIu64
                  " %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f %9.2f\n",
                  label, l.rows, l.batch, l.samples, l.us.min, l.us.mean,
                  l.us.p50, l.us.p90, l.us.p99, l.us.p999, l.us.max);
    };
    if (opt_.ping_iters || !steps_.empty()) {
      std::printf("latency of %" PRIu64
                  " B row reads, posted at once and timed until the last row "
                  "arrives (us):\n",
                  opt_.row_bytes);
      std::printf("%-8s %6s %6s %8s %8s %8s %8s %8s %8s %9s %9s\n", "", "rows",
                  "batch", "samples", "min", "mean", "p50", "p90", "p99",
                  "p99.9", "max");
      if (opt_.ping_iters) lat_row("ping", ping_);
      for (const Latency& l : steps_) lat_row("step", l);
    }
    if (points_.empty()) return;
    std::printf("reads of %" PRIu64
                " B rows%s (request latency; us/row = "
                "request latency / rows):\n",
                opt_.row_bytes, opt_.verify ? " (verified)" : "");
    std::printf("%6s %6s %9s %12s %9s %9s %9s %9s %9s %8s %8s %8s\n", "batch",
                "depth", "requests", "rows/s", "MB/s", "p50 us", "p99 us",
                "p99.9 us", "max us", "us/row", "p99/row", "p999/row");
    for (const Point& p : points_) {
      const double r = rows_per_s(p);
      std::printf("%6u %6u %9" PRIu64
                  " %12.0f %9.2f %9.2f %9.2f %9.2f %9.2f %8.3f "
                  "%8.3f %8.3f\n",
                  p.batch, p.depth, p.requests, r, r * opt_.row_bytes / 1e6,
                  p.req.p50, p.req.p99, p.req.p999, p.req.max, p.row.p50,
                  p.row.p99, p.row.p999);
    }
  }

  Options opt_;
  std::string token_;
  uint32_t max_batch_ = 1;
  uint32_t max_depth_ = 1;
  uint32_t max_step_ = 1;
  // Buffers outlive client_ (declared first, destroyed last): a failed
  // run unwinds through ~Client, which may still complete operations.
  std::vector<std::vector<uint8_t>> bufs_;
  std::vector<uint8_t> slab_;
  std::vector<uint64_t> offsets_;
  std::vector<ple_mem::ReadOp> ops_;
  uint32_t pending_[1024] = {};
  ple_mem::Client client_;
  double up_seconds_ = 0;
  std::vector<Point> points_;
  Latency ping_{};
  std::vector<Latency> steps_;
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
