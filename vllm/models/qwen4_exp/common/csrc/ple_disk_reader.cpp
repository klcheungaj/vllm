// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// On-demand reader for the Qwen4Exp PLE n-gram embedding table.
//
// Rows are read straight from the SSD with page-aligned preads (O_DIRECT when
// available) by a pool of blocking I/O threads. A set-associative row cache
// keeps rows already fetched. Demand jobs come from `read_rows` (blocking) or
// from CUDA host functions (`launch_notify` / `launch_wait`) acting on bound
// pinned buffers, which makes the path CUDA-graph capturable. Hints prefetch
// into the cache at low priority.
//
// Built with -DPLE_WITH_UCX, the table can also be uploaded to RAM borrowed
// from a PLE memory server (../ple_mem_server/) and read back with one-sided
// UCX GETs; the SSD stays the source for rows not uploaded yet and the
// fallback when the server fails.
//
// No CUDA headers: `cuLaunchHostFunc` is resolved from libcuda at runtime.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef PLE_WITH_UCX
  #include <netdb.h>
  #include <ucp/api/ucp.h>

  #include "../ple_mem_server/ple_mem_protocol.h"
  #include "../ple_mem_server/ple_mem_util.h"
#endif

namespace py = pybind11;

namespace {

constexpr uint64_t kPage = 4096;
constexpr uint32_t kWays = 8;
constexpr uint64_t kEmptyKey = ~0ull;
constexpr uint64_t kRemoteKey = 1ull << 63;
constexpr size_t kLatencyRing = 65536;
constexpr size_t kCacheStripes = 1024;
constexpr size_t kHintInboxRows = size_t{1} << 20;
constexpr size_t kHintQueueReads = size_t{1} << 16;

double now_us() {
  using namespace std::chrono;
  return static_cast<double>(
             duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

// 8-way set-associative row cache with CLOCK (second chance) replacement:
// a hit marks the way referenced, an insert sweeps the set's hand clearing
// marks and takes the first unreferenced way. New rows start unreferenced,
// so one-off rows and hints do not push out rows that keep getting hit.
// Lock-striped by set so completions and lookups rarely contend.
class RowCache {
 public:
  void init(uint64_t rows, uint32_t row_bytes) {
    rb_ = row_bytes;
    sets_ = rows == 0 ? 0 : std::max<uint64_t>(1, rows / kWays);
    keys_.assign(sets_ * kWays, kEmptyKey);
    data_.assign(sets_ * kWays * rb_, 0);
    ref_.assign(sets_ * kWays, 0);
    hand_.assign(sets_, 0);
    data_.shrink_to_fit();
    keys_.shrink_to_fit();
  }
  bool enabled() const { return sets_ > 0; }
  uint64_t capacity() const { return sets_ * kWays; }

  // Copies the row into `dst` (when not null) and returns true on a hit.
  bool lookup(uint64_t row, uint8_t* dst) {
    if (sets_ == 0) return false;
    const uint64_t s = set_of(row);
    std::lock_guard<std::mutex> lk(locks_[s % kCacheStripes]);
    const int w = find_way(s, row);
    if (w < 0) return false;
    ref_[s * kWays + w] = 1;
    if (dst != nullptr) std::memcpy(dst, slot(s, w), rb_);
    return true;
  }

  void insert(uint64_t row, const uint8_t* bytes) {
    if (sets_ == 0) return;
    const uint64_t s = set_of(row);
    std::lock_guard<std::mutex> lk(locks_[s % kCacheStripes]);
    if (find_way(s, row) >= 0) return;
    uint32_t w;
    for (;;) {
      w = hand_[s];
      hand_[s] = static_cast<uint8_t>((w + 1) % kWays);
      uint8_t& ref = ref_[s * kWays + w];
      if (keys_[s * kWays + w] == kEmptyKey || ref == 0) break;
      ref = 0;
    }
    keys_[s * kWays + w] = row;
    ref_[s * kWays + w] = 0;
    std::memcpy(slot(s, w), bytes, rb_);
  }

 private:
  static uint64_t mix(uint64_t r) {
    uint64_t x = r * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
  }
  uint64_t set_of(uint64_t row) const { return mix(row) % sets_; }
  int find_way(uint64_t s, uint64_t row) const {
    for (uint32_t w = 0; w < kWays; ++w)
      if (keys_[s * kWays + w] == row) return static_cast<int>(w);
    return -1;
  }
  uint8_t* slot(uint64_t s, uint32_t w) {
    return &data_[(s * kWays + w) * rb_];
  }

  uint64_t sets_ = 0;
  uint32_t rb_ = 0;
  std::vector<uint64_t> keys_;
  std::vector<uint8_t> data_;
  std::vector<uint8_t> ref_;
  std::vector<uint8_t> hand_;
  std::unique_ptr<std::mutex[]> locks_{new std::mutex[kCacheStripes]};
};

// A batch of rows whose bytes land in `dst` (row i at dst + i * row_bytes).
// `pending` counts the planning step plus every row still being read.
struct Job {
  const int64_t* ids = nullptr;
  size_t n = 0;
  uint8_t* dst = nullptr;
  uint64_t pending = 1;
  int writers = 0;
  bool cancelled = false;
  bool failed = false;
  bool waited = false;
};
using JobPtr = std::shared_ptr<Job>;

struct Use {
  uint64_t row;
  uint32_t in_page;
  uint8_t* dst;
  JobPtr job;  // null for hint-only uses
};

// A page read from the SSD or, when `remote`, a one-row GET from the memory
// server (`offset` then holds the row).
struct PageRead {
  uint64_t key = 0;
  uint32_t file = 0;
  uint64_t offset = 0;
  uint32_t length = 0;
  bool remote = false;
  bool started = false;
  bool finished = false;
  bool demand = false;
  std::vector<Use> uses;
};
using ReadPtr = std::shared_ptr<PageRead>;

struct Miss {
  uint64_t key;
  uint64_t page_key;
  uint64_t row;
  uint64_t offset;  // aligned page offset
  uint32_t in_page;
  uint32_t length;
  uint32_t file;
  bool remote;
  uint8_t* dst;
};

struct Stats {
  uint64_t requests = 0;
  uint64_t cache_hits = 0;
  uint64_t dedup_rows = 0;
  uint64_t reads = 0;
  uint64_t bytes = 0;
  uint64_t hint_rows = 0;
  uint64_t hint_reads = 0;
  uint64_t hint_dropped = 0;
  uint64_t demand_jobs = 0;
  uint64_t timeouts = 0;
  uint64_t errors = 0;
  uint64_t remote_reads = 0;
  uint64_t remote_bytes = 0;
  uint64_t remote_requeued = 0;
  double demand_wait_us = 0;
};

struct LatencyRing {
  std::vector<float> v;
  size_t pos = 0;
  void add(double us) {
    if (v.size() < kLatencyRing)
      v.push_back(static_cast<float>(us));
    else
      v[pos++ % kLatencyRing] = static_cast<float>(us);
  }
  void clear() {
    v.clear();
    pos = 0;
  }
};

enum class RemoteState { kDisabled, kConnecting, kUploading, kReady, kFailed };

const char* remote_state_name(RemoteState s) {
  switch (s) {
    case RemoteState::kDisabled:
      return "disabled";
    case RemoteState::kConnecting:
      return "connecting";
    case RemoteState::kUploading:
      return "uploading";
    case RemoteState::kReady:
      return "ready";
    case RemoteState::kFailed:
      return "failed";
  }
  return "unknown";
}

struct RemoteParams {
  std::string server;
  uint32_t max_inflight = 0;
  uint64_t region_rows = 0;
  uint64_t chunk_rows = 0;
  double connect_timeout_s = 0;
  bool fallback_to_disk = true;
};

using LaunchHostFunc = int (*)(void*, void (*)(void*), void*);

LaunchHostFunc get_launch_host_func() {
  static LaunchHostFunc fn = []() -> LaunchHostFunc {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (lib == nullptr) return nullptr;
    return reinterpret_cast<LaunchHostFunc>(dlsym(lib, "cuLaunchHostFunc"));
  }();
  return fn;
}

#ifdef PLE_WITH_UCX
constexpr uint32_t kUploadBuffers = 4;

struct RemoteRegion {
  uint64_t id = 0;
  uint64_t addr = 0;
  uint64_t bytes = 0;
  ucp_rkey_h rkey = nullptr;
};

struct Ucx;

struct GetOp {
  Ucx* ucx = nullptr;
  uint32_t slot = 0;
  ReadPtr read;
  double t0 = 0;
  ucs_status_t status = UCS_OK;
};

// Puts of one staged upload chunk plus the flush making them visible.
struct ChunkOp {
  Ucx* ucx = nullptr;
  uint32_t buf = 0;
  uint64_t seq = 0;
  uint64_t row_end = 0;
  uint64_t bytes = 0;
  int outstanding = 0;
  ucs_status_t status = UCS_OK;
};

struct AmSend {
  void* request;
  std::vector<char> payload;
};

// State owned by the UCX thread.
struct Ucx {
  ucp_context_h ctx = nullptr;
  ucp_worker_h worker = nullptr;
  ucp_ep_h ep = nullptr;
  bool ep_failed = false;
  ucs_status_t ep_status = UCS_OK;
  std::vector<RemoteRegion> regions;
  std::vector<std::vector<char>> replies;
  std::vector<AmSend> sends;

  uint8_t* get_slab = nullptr;
  ucp_mem_h get_memh = nullptr;
  std::vector<GetOp> gets;
  std::vector<uint32_t> get_free;
  std::vector<uint32_t> get_done;

  std::vector<uint8_t*> stage;
  std::vector<ucp_mem_h> stage_memh;
  std::vector<ChunkOp> chunks;
  std::vector<uint32_t> chunk_done;
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> visible;
  uint64_t next_seq = 0;

  uint64_t outstanding = 0;
  uint64_t seen_kick = 0;
  bool backlog = false;
};

struct UploadChunk {
  uint64_t seq;
  uint32_t buf;
  uint32_t shift;
  uint64_t row;
  uint64_t rows;
};
#endif

class Core {
 public:
  Core(const std::vector<std::string>& paths,
       const std::vector<int64_t>& seg_file,
       const std::vector<int64_t>& seg_offset,
       const std::vector<int64_t>& seg_row_start,
       const std::vector<int64_t>& seg_num_rows, int64_t row_bytes,
       int64_t cache_rows, int64_t io_threads, int64_t max_inflight,
       bool direct_io, double wait_timeout_s, const std::string& remote_server,
       int64_t remote_max_inflight, int64_t remote_region_bytes,
       int64_t remote_upload_chunk_bytes, double remote_connect_timeout_s,
       bool remote_fallback_to_disk)
      : rb_(static_cast<uint32_t>(row_bytes)),
        timeout_us_(wait_timeout_s * 1e6) {
    if (row_bytes <= 0 || row_bytes > static_cast<int64_t>(kPage))
      throw std::invalid_argument("row_bytes must be in [1, 4096]");
    if (io_threads <= 0 || max_inflight <= 0 || cache_rows < 0)
      throw std::invalid_argument(
          "io_threads and max_inflight must be > 0, cache_rows >= 0");
    if (wait_timeout_s <= 0)
      throw std::invalid_argument("wait_timeout_s must be > 0");
    if (!remote_server.empty()) {
#ifndef PLE_WITH_UCX
      throw std::invalid_argument(
          "remote_server needs the reader built with UCX support");
#endif
      if (remote_max_inflight <= 0 || remote_region_bytes <= 0 ||
          remote_upload_chunk_bytes <= 0 || remote_connect_timeout_s <= 0)
        throw std::invalid_argument(
            "remote max_inflight, region_bytes, upload_chunk_bytes and "
            "connect_timeout_s must be > 0");
      remote_.server = remote_server;
      remote_.max_inflight = static_cast<uint32_t>(remote_max_inflight);
      remote_.region_rows =
          std::max<uint64_t>(1, uint64_t(remote_region_bytes) / rb_);
      remote_.chunk_rows =
          std::max<uint64_t>(1, uint64_t(remote_upload_chunk_bytes) / rb_);
      remote_.connect_timeout_s = remote_connect_timeout_s;
      remote_.fallback_to_disk = remote_fallback_to_disk;
    }
    const size_t ns = seg_file.size();
    if (seg_offset.size() != ns || seg_row_start.size() != ns ||
        seg_num_rows.size() != ns)
      throw std::invalid_argument("segment arrays differ in length");
    std::vector<size_t> order(ns);
    for (size_t i = 0; i < ns; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return seg_row_start[a] < seg_row_start[b];
    });
    for (size_t i : order) {
      if (seg_file[i] < 0 || seg_file[i] >= (int64_t)paths.size() ||
          seg_offset[i] < 0 || seg_row_start[i] < 0 || seg_num_rows[i] < 0)
        throw std::invalid_argument("invalid segment");
      if (!seg_start_.empty() &&
          (uint64_t)seg_row_start[i] < seg_start_.back() + seg_rows_.back())
        throw std::invalid_argument("segments overlap");
      seg_file_.push_back(static_cast<uint32_t>(seg_file[i]));
      seg_offset_.push_back(static_cast<uint64_t>(seg_offset[i]));
      seg_start_.push_back(static_cast<uint64_t>(seg_row_start[i]));
      seg_rows_.push_back(static_cast<uint64_t>(seg_num_rows[i]));
      remote_total_bytes_ += uint64_t(seg_num_rows[i]) * rb_;
    }
    paths_ = paths;
    direct_ = direct_io;
    direct_atomic_.store(direct_io);
    for (const auto& p : paths) {
      int fd = direct_ ? ::open(p.c_str(), O_RDONLY | O_DIRECT) : -1;
      if (fd < 0 && direct_) {
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd >= 0) disable_direct_locked();
      } else if (fd < 0) {
        fd = ::open(p.c_str(), O_RDONLY);
      }
      if (fd < 0) {
        const std::string msg = "cannot open " + p + ": " + strerror(errno);
        close_fds();
        throw std::runtime_error(msg);
      }
      if (!direct_) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
      fds_.push_back(fd);
    }
    slots_ = static_cast<uint32_t>(max_inflight);
    void* slab = nullptr;
    if (posix_memalign(&slab, kPage, size_t{slots_} * 2 * kPage) != 0) {
      close_fds();
      throw std::runtime_error("cannot allocate the read slab");
    }
    slab_ = static_cast<uint8_t*>(slab);
    for (uint32_t s = slots_; s-- > 0;) free_slots_.push_back(s);
    cache_.init(static_cast<uint64_t>(cache_rows), rb_);
#ifdef PLE_WITH_UCX
    if (!remote_.server.empty()) start_remote();
#endif
    planner_ = std::thread([this] { planner_loop(); });
    num_workers_ = static_cast<size_t>(io_threads);
    for (int64_t i = 0; i < io_threads; ++i)
      workers_.emplace_back([this] { worker_loop(); });
  }

  ~Core() { close(); }

  void close() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_) return;
      closed_ = true;
    }
    stop_remote_.store(true);
    cv_work_.notify_all();
    cv_plan_.notify_all();
    cv_remote_.notify_all();
    cv_upload_.notify_all();
    if (planner_.joinable()) planner_.join();
    for (auto& t : workers_) t.join();
    workers_.clear();
    if (upload_thread_.joinable()) upload_thread_.join();
    if (ucx_thread_.joinable()) ucx_thread_.join();
#ifdef PLE_WITH_UCX
    free_stage_buffers();
#endif
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto* q : {&demand_q_, &hint_q_, &rdemand_q_, &rhint_q_}) {
        for (auto& r : *q) {
          if (r->finished) continue;
          r->finished = true;
          for (auto& u : r->uses) {
            if (!u.job) continue;
            u.job->failed = true;
            --u.job->pending;
          }
          r->uses.clear();
        }
        q->clear();
      }
      for (auto& j : plan_q_) {
        j->failed = true;
        --j->pending;
      }
      plan_q_.clear();
      hint_inbox_.clear();
      inflight_.clear();
      bound_latest_.reset();
      ready_rows_.store(0);
      if (error_.empty()) error_ = "PLE disk reader closed";
    }
    cv_done_.notify_all();
    close_fds();
    std::free(slab_);
    slab_ = nullptr;
    cache_.init(0, rb_);
  }

  void read_rows(uint64_t ids, int64_t n, uint64_t out) {
    auto job = std::make_shared<Job>();
    job->ids = reinterpret_cast<const int64_t*>(ids);
    job->n = static_cast<size_t>(n);
    job->dst = reinterpret_cast<uint8_t*>(out);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_) throw std::runtime_error("PLE disk reader is closed");
      ++stats_.demand_jobs;
    }
    plan(job.get(), job);
    std::unique_lock<std::mutex> lk(mu_);
    if (!wait_job(job, lk))
      throw std::runtime_error("PLE disk read timed out: " + error_);
    if (job->failed)
      throw std::runtime_error("PLE disk read failed: " + error_);
  }

  void hint(uint64_t ids, int64_t n) {
    if (n <= 0 || !cache_.enabled()) return;
    const int64_t* p = reinterpret_cast<const int64_t*>(ids);
    std::vector<int64_t> v(p, p + n);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_) return;
      stats_.hint_rows += static_cast<uint64_t>(n);
      if (hint_inbox_rows_ + v.size() > kHintInboxRows) {
        stats_.hint_dropped += static_cast<uint64_t>(n);
        return;
      }
      hint_inbox_rows_ += v.size();
      hint_inbox_.push_back(std::move(v));
    }
    cv_plan_.notify_one();
  }

  void bind(uint64_t ids, uint64_t num_tokens, uint64_t out, int64_t max_tokens,
            int64_t heads) {
    std::lock_guard<std::mutex> lk(mu_);
    bound_ids_ = reinterpret_cast<const int64_t*>(ids);
    bound_num_tokens_ = reinterpret_cast<const int64_t*>(num_tokens);
    bound_out_ = reinterpret_cast<uint8_t*>(out);
    bound_max_tokens_ = max_tokens;
    bound_heads_ = heads;
  }

  void launch(uint64_t stream, bool wait) {
    LaunchHostFunc fn = get_launch_host_func();
    if (fn == nullptr)
      throw std::runtime_error(
          "cuLaunchHostFunc not found (cannot load libcuda.so.1)");
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_) throw std::runtime_error("PLE disk reader is closed");
      if (bound_ids_ == nullptr)
        throw std::runtime_error("bind_demand() must be called first");
    }
    host_funcs_launched_ = true;
    const int rc = fn(reinterpret_cast<void*>(stream),
                      wait ? &Core::wait_cb : &Core::notify_cb, this);
    if (rc != 0)
      throw std::runtime_error("cuLaunchHostFunc failed with CUresult " +
                               std::to_string(rc));
  }

  std::string error() {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
  }

  py::dict stats() {
    Stats s;
    std::vector<float> lat, rlat;
    bool direct;
    {
      std::lock_guard<std::mutex> lk(mu_);
      s = stats_;
      lat = latency_.v;
      rlat = remote_latency_.v;
      direct = direct_;
    }
    py::dict d;
    d["requests"] = s.requests;
    d["cache_hits"] = s.cache_hits;
    d["dedup_rows"] = s.dedup_rows;
    d["reads"] = s.reads;
    d["bytes"] = s.bytes;
    d["hint_rows"] = s.hint_rows;
    d["hint_reads"] = s.hint_reads;
    d["hint_dropped"] = s.hint_dropped;
    d["demand_jobs"] = s.demand_jobs;
    d["timeouts"] = s.timeouts;
    d["errors"] = s.errors;
    d["demand_wait_us"] = s.demand_wait_us;
    d["read_us_p50"] = percentile(lat, 0.5);
    d["read_us_p99"] = percentile(lat, 0.99);
    d["cache_capacity"] = cache_.capacity();
    d["direct_io"] = direct;
    d["remote_reads"] = s.remote_reads;
    d["remote_bytes"] = s.remote_bytes;
    d["remote_read_us_p50"] = percentile(rlat, 0.5);
    d["remote_read_us_p99"] = percentile(rlat, 0.99);
    d["remote_requeued"] = s.remote_requeued;
    return d;
  }

  void reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = Stats{};
    latency_.clear();
    remote_latency_.clear();
  }

  py::dict remote_status() {
    RemoteState state;
    uint64_t uploaded, regions;
    std::string error;
    {
      std::lock_guard<std::mutex> lk(mu_);
      state = remote_state_;
      uploaded = remote_uploaded_bytes_;
      regions = remote_regions_;
      error = remote_error_;
    }
    py::dict d;
    d["state"] = remote_state_name(state);
    d["uploaded_bytes"] = uploaded;
    d["total_bytes"] =
        state == RemoteState::kDisabled ? 0 : remote_total_bytes_;
    d["regions"] = regions;
    d["error"] = error;
    return d;
  }

  bool direct_io() {
    std::lock_guard<std::mutex> lk(mu_);
    return direct_;
  }

  bool host_funcs_launched() const { return host_funcs_launched_; }

 private:
  static void notify_cb(void* p) {
    try {
      static_cast<Core*>(p)->on_notify();
    } catch (...) {
    }
  }
  static void wait_cb(void* p) {
    try {
      static_cast<Core*>(p)->on_wait();
    } catch (...) {
    }
  }

  void on_notify() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_ || bound_ids_ == nullptr) return;
      const int64_t n =
          std::clamp<int64_t>(*bound_num_tokens_, 0, bound_max_tokens_);
      auto job = std::make_shared<Job>();
      job->ids = bound_ids_;
      job->n = static_cast<size_t>(n * bound_heads_);
      job->dst = bound_out_;
      bound_latest_ = job;
      ++stats_.demand_jobs;
      plan_q_.push_back(std::move(job));
    }
    cv_plan_.notify_one();
  }

  void on_wait() {
    std::unique_lock<std::mutex> lk(mu_);
    JobPtr job = bound_latest_;
    if (!job || job->waited) return;
    job->waited = true;
    wait_job(job, lk);
  }

  // Waits (lock held) for `job` to finish; on timeout records an error,
  // cancels the job so late reads no longer touch its buffer, returns false.
  bool wait_job(const JobPtr& job, std::unique_lock<std::mutex>& lk) {
    const double t0 = now_us();
    const bool ok = cv_done_.wait_for(
        lk, std::chrono::microseconds(static_cast<int64_t>(timeout_us_)),
        [&] { return job->pending == 0 || closed_; });
    stats_.demand_wait_us += now_us() - t0;
    if (ok && job->pending == 0) return true;
    if (ok) {
      job->failed = true;
      return true;
    }
    ++stats_.timeouts;
    set_error_locked("PLE disk read timed out after " +
                     std::to_string(timeout_us_ / 1e6) + " s");
    job->cancelled = true;
    job->failed = true;
    cv_done_.wait(lk, [&] { return job->writers == 0; });
    return false;
  }

  void set_error_locked(const std::string& msg) {
    ++stats_.errors;
    if (error_.empty()) error_ = msg;
  }

  void planner_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_plan_.wait(lk, [&] {
        return closed_ || !plan_q_.empty() || !hint_inbox_.empty();
      });
      if (closed_) return;
      if (!plan_q_.empty()) {
        JobPtr job = std::move(plan_q_.front());
        plan_q_.pop_front();
        lk.unlock();
        plan(job.get(), job);
      } else {
        std::vector<int64_t> ids = std::move(hint_inbox_.front());
        hint_inbox_.pop_front();
        hint_inbox_rows_ -= ids.size();
        lk.unlock();
        Job hint_job;
        hint_job.ids = ids.data();
        hint_job.n = ids.size();
        plan(&hint_job, nullptr);
      }
      lk.lock();
    }
  }

  // Resolves cache hits and out-of-range rows, then attaches the misses to
  // reads (new or already queued / in flight): remote GETs for rows already
  // uploaded, SSD page reads otherwise. `owner` is null for hints.
  void plan(Job* job, const JobPtr& owner) {
    const bool demand = owner != nullptr;
    if (demand) {
      std::lock_guard<std::mutex> lk(mu_);
      if (job->cancelled || closed_) {
        job->failed = true;
        finish_pending(job, 1);
        cv_done_.notify_all();
        return;
      }
      ++job->writers;
    }
    const uint64_t ready = ready_rows_.load(std::memory_order_acquire);
    std::vector<Miss> misses;
    uint64_t hits = 0;
    bool any_remote = false;
    for (size_t i = 0; i < job->n; ++i) {
      const int64_t id = job->ids[i];
      uint8_t* dst = demand ? job->dst + i * rb_ : nullptr;
      size_t seg;
      if (!locate(id, &seg)) {
        if (dst != nullptr) std::memset(dst, 0, rb_);
        continue;
      }
      const uint64_t row = static_cast<uint64_t>(id);
      if (cache_.lookup(row, dst)) {
        ++hits;
        continue;
      }
      misses.push_back(disk_miss(row, seg, dst));
      if (row < ready) {
        Miss& m = misses.back();
        m.remote = true;
        m.key = kRemoteKey | row;
        m.in_page = 0;
        m.length = rb_;
        any_remote = true;
      }
    }
    auto by_key = [](const Miss& a, const Miss& b) { return a.key < b.key; };
    std::sort(misses.begin(), misses.end(), by_key);

    size_t queued = 0, queued_remote = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (demand) {
        --job->writers;
        stats_.requests += job->n;
        stats_.cache_hits += hits;
      }
      if (closed_) {
        if (demand) {
          job->failed = true;
          finish_pending(job, 1);
          cv_done_.notify_all();
        }
        return;
      }
      if (any_remote && !remote_usable_locked()) {
        for (auto& m : misses) {
          if (!m.remote) continue;
          size_t seg;
          locate(static_cast<int64_t>(m.row), &seg);
          m = disk_miss(m.row, seg, m.dst);
        }
        std::sort(misses.begin(), misses.end(), by_key);
      }
      for (size_t i = 0; i < misses.size();) {
        size_t j = i;
        uint32_t need = 0;
        while (j < misses.size() && misses[j].key == misses[i].key)
          need = std::max(need, misses[j++].length);
        const size_t group = j - i;
        bool fresh = false;
        ReadPtr r = attach_locked(misses[i], need, demand, group, &fresh);
        if (r) {
          if (fresh) ++(r->remote ? queued_remote : queued);
          for (size_t k = i; k < j; ++k)
            r->uses.push_back(
                Use{misses[k].row, misses[k].in_page, misses[k].dst, owner});
          if (demand) job->pending += group;
        }
        i = j;
      }
      if (demand && finish_pending(job, 1)) cv_done_.notify_all();
      if (demand && job->writers == 0 && job->cancelled) cv_done_.notify_all();
    }
    if (queued_remote) cv_remote_.notify_one();
    notify_workers(queued);
  }

  void notify_workers(size_t queued) {
    if (queued >= num_workers_) {
      cv_work_.notify_all();
    } else {
      for (size_t k = 0; k < queued; ++k) cv_work_.notify_one();
    }
  }

  Miss disk_miss(uint64_t row, size_t seg, uint8_t* dst) const {
    const uint64_t at = seg_offset_[seg] + (row - seg_start_[seg]) * rb_;
    const uint64_t first = at & ~(kPage - 1);
    const uint64_t last = (at + rb_ - 1) & ~(kPage - 1);
    const uint32_t file = seg_file_[seg];
    const uint64_t key = (uint64_t{file} << 52) | (first >> 12);
    return Miss{key,
                key,
                row,
                first,
                static_cast<uint32_t>(at - first),
                static_cast<uint32_t>(last - first + kPage),
                file,
                false,
                dst};
  }

  // Lock held. Returns the read serving `group` misses keyed like `m` (an
  // existing one when possible), or null when a hint has to be dropped.
  // `fresh` is set when a read was (re)queued.
  ReadPtr attach_locked(const Miss& m, uint32_t need, bool demand, size_t group,
                        bool* fresh) {
    auto& dq = m.remote ? rdemand_q_ : demand_q_;
    auto& hq = m.remote ? rhint_q_ : hint_q_;
    ReadPtr r;
    auto it = inflight_.find(m.key);
    if (it != inflight_.end() && !it->second->finished &&
        (!it->second->started || it->second->length >= need)) {
      r = it->second;
      if (!r->started) r->length = std::max(r->length, need);
      if (demand && !r->demand && !r->started) {
        r->demand = true;
        dq.push_back(r);
        *fresh = true;
      }
      stats_.dedup_rows += group;
    } else {
      if (!demand && hq.size() >= kHintQueueReads) {
        stats_.hint_dropped += group;
        return nullptr;
      }
      r = std::make_shared<PageRead>();
      r->key = m.key;
      r->file = m.file;
      r->offset = m.remote ? m.row : m.offset;
      r->length = need;
      r->remote = m.remote;
      r->demand = demand;
      inflight_[r->key] = r;
      (demand ? dq : hq).push_back(r);
      if (!demand) ++stats_.hint_reads;
      stats_.dedup_rows += group - 1;
      *fresh = true;
    }
    if (*fresh && m.remote) remote_kick_.fetch_add(1);
    return r;
  }

  bool remote_usable_locked() const {
    return remote_state_ == RemoteState::kUploading ||
           remote_state_ == RemoteState::kReady;
  }

  // Returns true when the job just completed.
  static bool finish_pending(Job* job, uint64_t k) {
    job->pending -= k;
    return job->pending == 0;
  }

  bool locate(int64_t id, size_t* seg) const {
    if (id < 0 || seg_start_.empty()) return false;
    const uint64_t row = static_cast<uint64_t>(id);
    auto it = std::upper_bound(seg_start_.begin(), seg_start_.end(), row);
    if (it == seg_start_.begin()) return false;
    const size_t s = static_cast<size_t>(it - seg_start_.begin()) - 1;
    if (row - seg_start_[s] >= seg_rows_[s]) return false;
    *seg = s;
    return true;
  }

  static ReadPtr pop_read_locked(std::deque<ReadPtr>& dq,
                                 std::deque<ReadPtr>& hq) {
    for (auto* q : {&dq, &hq}) {
      while (!q->empty()) {
        ReadPtr r = std::move(q->front());
        q->pop_front();
        if (!r->started && !r->finished) return r;
      }
    }
    return nullptr;
  }

  void worker_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_work_.wait(lk, [&] {
        return closed_ || (!free_slots_.empty() &&
                           (!demand_q_.empty() || !hint_q_.empty()));
      });
      if (closed_) return;
      ReadPtr r = pop_read_locked(demand_q_, hint_q_);
      if (!r) continue;
      r->started = true;
      const uint32_t slot = free_slots_.back();
      free_slots_.pop_back();
      uint8_t* buf = slab_ + size_t{slot} * 2 * kPage;
      const int fd = fds_[r->file];
      const uint64_t offset = r->offset;
      const uint32_t length = r->length;
      lk.unlock();

      const double t0 = now_us();
      int err = 0;
      const size_t got = do_pread(fd, buf, length, offset, &err);
      const double t1 = now_us();

      finish_read(r, buf, got, err, t1 - t0, static_cast<int>(slot));
      lk.lock();
    }
  }

  // Delivers a finished read (`got` bytes at `buf`) to its uses: copies to
  // live jobs, fills the cache, updates stats and pending counts. Takes mu_.
  // `slot` is the disk slab slot to return, or -1 for a remote read.
  void finish_read(const ReadPtr& r, const uint8_t* buf, size_t got, int err,
                   double us, int slot) {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = inflight_.find(r->key);
    if (it != inflight_.end() && it->second == r) inflight_.erase(it);
    r->finished = true;
    std::vector<Use> uses = std::move(r->uses);
    std::vector<char> live(uses.size(), 0);
    for (size_t i = 0; i < uses.size(); ++i) {
      if (uses[i].job && !uses[i].job->cancelled) {
        ++uses[i].job->writers;
        live[i] = 1;
      }
    }
    lk.unlock();

    bool short_row = false;
    std::vector<char> bad(uses.size(), 0);
    for (size_t i = 0; i < uses.size(); ++i) {
      const Use& u = uses[i];
      const bool ok = err == 0 && u.in_page + rb_ <= got;
      if (!ok) {
        bad[i] = 1;
        short_row = true;
        if (live[i]) std::memset(u.dst, 0, rb_);
        continue;
      }
      if (live[i]) std::memcpy(u.dst, buf + u.in_page, rb_);
      cache_.insert(u.row, buf + u.in_page);
    }

    lk.lock();
    if (r->remote) {
      ++stats_.remote_reads;
      stats_.remote_bytes += got;
      remote_latency_.add(us);
    } else {
      free_slots_.push_back(static_cast<uint32_t>(slot));
      ++stats_.reads;
      stats_.bytes += got;
      latency_.add(us);
      if (err != 0) {
        set_error_locked("pread failed at offset " + std::to_string(r->offset) +
                         ": " + strerror(err));
      } else if (short_row) {
        set_error_locked("short read at offset " + std::to_string(r->offset) +
                         " (" + std::to_string(got) + " of " +
                         std::to_string(r->length) +
                         " bytes): file truncated?");
      }
    }
    bool notify = false;
    for (size_t i = 0; i < uses.size(); ++i) {
      Job* job = uses[i].job.get();
      if (job == nullptr) continue;
      if (live[i] && --job->writers == 0 && job->cancelled) notify = true;
      if (bad[i]) job->failed = true;
      if (finish_pending(job, 1)) notify = true;
    }
    if (notify) cv_done_.notify_all();
    if (!r->remote && (!demand_q_.empty() || !hint_q_.empty()))
      cv_work_.notify_one();
  }

  // Reads up to `length` bytes, stopping at EOF. Falls back to buffered I/O
  // when O_DIRECT is rejected with EINVAL.
  size_t do_pread(int fd, uint8_t* buf, uint32_t length, uint64_t offset,
                  int* err) {
    size_t total = 0;
    while (total < length) {
      const ssize_t g = ::pread(fd, buf + total, length - total,
                                static_cast<off_t>(offset + total));
      if (g < 0) {
        if (errno == EINTR) continue;
        if (errno == EINVAL && direct_atomic_.load()) {
          std::lock_guard<std::mutex> lk(mu_);
          disable_direct_locked();
          continue;
        }
        *err = errno;
        return total;
      }
      if (g == 0) break;
      total += static_cast<size_t>(g);
      if (direct_atomic_.load() && total % kPage != 0) break;
    }
    return total;
  }

  void disable_direct_locked() {
    if (!direct_) return;
    direct_ = false;
    direct_atomic_.store(false);
    for (int fd : fds_) {
      const int fl = fcntl(fd, F_GETFL);
      if (fl >= 0) fcntl(fd, F_SETFL, fl & ~O_DIRECT);
      posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
  }

  static double percentile(std::vector<float>& v, double q) {
    if (v.empty()) return 0.0;
    const size_t k = std::min(
        v.size() - 1, static_cast<size_t>(q * (double)(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + static_cast<ptrdiff_t>(k), v.end());
    return v[k];
  }

  void close_fds() {
    for (int fd : fds_) ::close(fd);
    fds_.clear();
  }

  // Lock held. Moves the uses of a remote read that will not complete to SSD
  // page reads (fallback) or fails their jobs. Returns true when a job
  // completed; `queued` counts newly queued disk reads.
  bool requeue_remote_locked(const ReadPtr& r, size_t* queued) {
    if (r->finished) return false;
    auto it = inflight_.find(r->key);
    if (it != inflight_.end() && it->second == r) inflight_.erase(it);
    r->finished = true;
    std::vector<Use> uses = std::move(r->uses);
    const bool fallback = remote_.fallback_to_disk && !closed_;
    if (fallback) ++stats_.remote_requeued;
    bool done = false;
    for (Use& u : uses) {
      size_t seg;
      if (fallback && locate(static_cast<int64_t>(u.row), &seg)) {
        const Miss m = disk_miss(u.row, seg, u.dst);
        bool fresh = false;
        ReadPtr d = attach_locked(m, m.length, u.job != nullptr, 1, &fresh);
        if (fresh) ++*queued;
        if (d) d->uses.push_back(Use{u.row, m.in_page, u.dst, u.job});
        continue;
      }
      if (!u.job) continue;
      u.job->failed = true;
      if (finish_pending(u.job.get(), 1)) done = true;
    }
    return done;
  }

#ifdef PLE_WITH_UCX
  void start_remote() {
    remote_state_ = RemoteState::kConnecting;
    ucx_thread_ = std::thread([this] { ucx_main(); });
    const auto guard =
        std::chrono::duration<double>(remote_.connect_timeout_s + 30.0);
    std::unique_lock<std::mutex> lk(mu_);
    if (!cv_setup_.wait_for(lk, guard, [&] { return remote_setup_done_; })) {
      remote_state_ = RemoteState::kFailed;
      remote_error_ = "timed out setting up the remote memory source";
    }
    const bool failed = remote_state_ == RemoteState::kFailed;
    if (failed && !remote_.fallback_to_disk) {
      const std::string msg =
          "PLE remote memory setup failed: " + remote_error_;
      lk.unlock();
      stop_remote_.store(true);
      cv_remote_.notify_all();
      ucx_thread_.join();
      free_stage_buffers();
      close_fds();
      std::free(slab_);
      slab_ = nullptr;
      throw std::runtime_error(msg);
    }
    if (!failed) upload_thread_ = std::thread([this] { upload_main(); });
  }

  static void ep_err_cb(void* arg, ucp_ep_h, ucs_status_t status) {
    auto* u = static_cast<Ucx*>(arg);
    if (!u->ep_failed) u->ep_status = status;
    u->ep_failed = true;
  }

  static ucs_status_t reply_cb(void* arg, const void*, size_t, void* data,
                               size_t length, const ucp_am_recv_param_t* p) {
    if (p->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) return UCS_OK;
    const char* d = static_cast<const char*>(data);
    static_cast<Ucx*>(arg)->replies.emplace_back(d, d + length);
    return UCS_OK;
  }

  static void get_cb(void* request, ucs_status_t status, void* user_data) {
    auto* op = static_cast<GetOp*>(user_data);
    op->status = status;
    --op->ucx->outstanding;
    op->ucx->get_done.push_back(op->slot);
    ucp_request_free(request);
  }

  static void chunk_cb(void* request, ucs_status_t status, void* user_data) {
    auto* op = static_cast<ChunkOp*>(user_data);
    if (status != UCS_OK) op->status = status;
    --op->ucx->outstanding;
    if (--op->outstanding == 0) op->ucx->chunk_done.push_back(op->buf);
    ucp_request_free(request);
  }

  static std::string ucs_error(const char* what, ucs_status_t st) {
    return std::string(what) + ": " + ucs_status_string(st);
  }

  ucs_status_t mem_map(Ucx& u, void* addr, size_t len, ucp_mem_h* memh) {
    ucp_mem_map_params_t mp{};
    mp.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = addr;
    mp.length = len;
    return ucp_mem_map(u.ctx, &mp, memh);
  }

  size_t stage_bytes() const {
    const uint64_t data = remote_.chunk_rows * rb_;
    return (data + kPage - 1) / kPage * kPage + 2 * kPage;
  }

  void am_send(Ucx& u, std::vector<char> payload) {
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY;
    void* req = ucp_am_send_nbx(u.ep, ple_mem::kAmRequestId, nullptr, 0,
                                payload.data(), payload.size(), &p);
    if (UCS_PTR_IS_PTR(req)) u.sends.push_back(AmSend{req, std::move(payload)});
  }

  void reap_sends(Ucx& u) {
    for (auto it = u.sends.begin(); it != u.sends.end();) {
      if (ucp_request_check_status(it->request) == UCS_INPROGRESS) {
        ++it;
        continue;
      }
      ucp_request_free(it->request);
      it = u.sends.erase(it);
    }
  }

  // Progresses until `req` completes or `deadline`; frees it either way.
  ucs_status_t wait_request(Ucx& u, void* req, double deadline_us) {
    if (req == nullptr) return UCS_OK;
    if (UCS_PTR_IS_ERR(req)) return UCS_PTR_STATUS(req);
    ucs_status_t st;
    while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS &&
           now_us() < deadline_us)
      ucp_worker_progress(u.worker);
    ucp_request_free(req);
    return st;
  }

  // Creates the context, worker and endpoint, borrows every region and
  // registers the local buffers. Returns an error message on failure.
  std::string ucx_setup(Ucx& u, uint64_t num_rows) {
    sockaddr_storage addr{};
    socklen_t addrlen = 0;
    std::string err;
    if (!ple_mem::resolve_address(remote_.server, &addr, &addrlen, &err))
      return err;

    ucp_config_t* config = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) return ucs_error("ucp_config_read", st);
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_AM;
    st = ucp_init(&params, config, &u.ctx);
    ucp_config_release(config);
    if (st != UCS_OK) return ucs_error("ucp_init", st);
    ucp_worker_params_t wp{};
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(u.ctx, &wp, &u.worker);
    if (st != UCS_OK) return ucs_error("ucp_worker_create", st);
    ucp_am_handler_param_t am{};
    am.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am.id = ple_mem::kAmReplyId;
    am.flags = UCP_AM_FLAG_WHOLE_MSG;
    am.cb = &Core::reply_cb;
    am.arg = &u;
    st = ucp_worker_set_am_recv_handler(u.worker, &am);
    if (st != UCS_OK) return ucs_error("ucp_worker_set_am_recv_handler", st);

    const uint32_t slots = remote_.max_inflight;
    const size_t slab_bytes = (size_t{slots} * rb_ + kPage - 1) / kPage * kPage;
    void* p = nullptr;
    if (posix_memalign(&p, kPage, slab_bytes) != 0)
      return "cannot allocate the remote read slab";
    u.get_slab = static_cast<uint8_t*>(p);
    st = mem_map(u, u.get_slab, slab_bytes, &u.get_memh);
    if (st != UCS_OK) return ucs_error("ucp_mem_map", st);
    u.gets.resize(slots);
    for (uint32_t s = 0; s < slots; ++s) {
      u.gets[s].ucx = &u;
      u.gets[s].slot = s;
    }
    for (uint32_t s = slots; s-- > 0;) u.get_free.push_back(s);
    u.chunks.resize(kUploadBuffers);
    for (uint32_t b = 0; b < kUploadBuffers; ++b) {
      if (posix_memalign(&p, kPage, stage_bytes()) != 0)
        return "cannot allocate the upload staging buffers";
      u.stage.push_back(static_cast<uint8_t*>(p));
      ucp_mem_h memh = nullptr;
      st = mem_map(u, p, stage_bytes(), &memh);
      if (st != UCS_OK) return ucs_error("ucp_mem_map", st);
      u.stage_memh.push_back(memh);
      u.chunks[b].ucx = &u;
      u.chunks[b].buf = b;
    }

    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
    ep.sockaddr.addrlen = addrlen;
    ep.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep.err_handler.cb = &Core::ep_err_cb;
    ep.err_handler.arg = &u;
    st = ucp_ep_create(u.worker, &ep, &u.ep);
    if (st != UCS_OK) {
      u.ep = nullptr;
      return "cannot connect to " + remote_.server + ": " +
             ucs_status_string(st);
    }

    const uint64_t n =
        (num_rows + remote_.region_rows - 1) / remote_.region_rows;
    u.regions.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
      const uint64_t rows =
          std::min(remote_.region_rows, num_rows - i * remote_.region_rows);
      ple_mem::AllocReq req{};
      req.hdr = ple_mem::make_header(ple_mem::kAllocReq);
      req.cookie = i;
      req.bytes = rows * rb_;
      std::vector<char> payload(sizeof(req));
      std::memcpy(payload.data(), &req, sizeof(req));
      am_send(u, std::move(payload));
    }
    const double deadline = now_us() + remote_.connect_timeout_s * 1e6;
    while (u.replies.size() < n && !u.ep_failed && !stop_remote_.load()) {
      if (now_us() > deadline)
        return "timed out after " + std::to_string(remote_.connect_timeout_s) +
               " s connecting to " + remote_.server + " (" +
               std::to_string(u.replies.size()) + " of " + std::to_string(n) +
               " regions granted)";
      ucp_worker_progress(u.worker);
      reap_sends(u);
    }
    if (u.ep_failed)
      return "cannot reach " + remote_.server + ": " +
             ucs_status_string(u.ep_status);
    if (stop_remote_.load()) return "reader closed during setup";
    std::vector<char> granted(n, 0);
    for (const auto& m : u.replies) {
      ple_mem::AllocRep rep;
      if (ple_mem::message_type(m.data(), m.size()) != ple_mem::kAllocRep ||
          m.size() < sizeof(rep))
        return "invalid reply from " + remote_.server;
      std::memcpy(&rep, m.data(), sizeof(rep));
      if (rep.cookie >= n || granted[rep.cookie])
        return "unexpected reply from " + remote_.server;
      if (rep.status != 0) {
        rep.error[sizeof(rep.error) - 1] = '\0';
        return remote_.server + " refused " + std::to_string(n) +
               " regions: " + rep.error;
      }
      if (m.size() < sizeof(rep) + rep.rkey_len)
        return "truncated reply from " + remote_.server;
      RemoteRegion& reg = u.regions[rep.cookie];
      reg.id = rep.region_id;
      reg.addr = rep.addr;
      reg.bytes = rep.bytes;
      granted[rep.cookie] = 1;
      st = ucp_ep_rkey_unpack(u.ep, m.data() + sizeof(rep), &reg.rkey);
      if (st != UCS_OK) {
        reg.rkey = nullptr;
        return ucs_error("ucp_ep_rkey_unpack", st);
      }
    }
    u.replies.clear();
    return "";
  }

  void ucx_main() {
    Ucx u;
    const uint64_t num_rows =
        seg_start_.empty() ? 0 : seg_start_.back() + seg_rows_.back();
    std::string err = ucx_setup(u, num_rows);
    {
      std::lock_guard<std::mutex> lk(mu_);
      stage_bufs_ = u.stage;
      if (remote_state_ == RemoteState::kFailed) err = remote_error_;
      if (err.empty()) {
        remote_state_ = RemoteState::kUploading;
        remote_regions_ = u.regions.size();
        for (uint32_t b = 0; b < kUploadBuffers; ++b) upload_free_.push_back(b);
      } else {
        remote_state_ = RemoteState::kFailed;
        remote_error_ = err;
      }
      remote_setup_done_ = true;
    }
    cv_setup_.notify_all();
    if (err.empty()) ucx_loop(u);
    ucx_teardown(u);
  }

  void ucx_loop(Ucx& u) {
    std::vector<ReadPtr> reads;
    std::vector<UploadChunk> ups;
    for (;;) {
      const uint64_t kick = remote_kick_.load();
      const bool need_lock = u.outstanding == 0 || kick != u.seen_kick ||
                             stop_remote_.load() ||
                             (u.backlog && !u.get_free.empty());
      if (need_lock) {
        std::unique_lock<std::mutex> lk(mu_);
        if (u.outstanding == 0)
          cv_remote_.wait_for(lk, std::chrono::milliseconds(1), [&] {
            return closed_ || remote_kick_.load() != u.seen_kick;
          });
        if (closed_) return;
        u.seen_kick = remote_kick_.load();
        if (!upload_error_.empty()) {
          const std::string msg = upload_error_;
          lk.unlock();
          remote_fail(u, msg);
          return;
        }
        while (reads.size() < u.get_free.size()) {
          ReadPtr r = pop_read_locked(rdemand_q_, rhint_q_);
          if (!r) break;
          r->started = true;
          reads.push_back(std::move(r));
        }
        u.backlog = !rdemand_q_.empty() || !rhint_q_.empty();
        while (!upload_q_.empty()) {
          ups.push_back(upload_q_.front());
          upload_q_.pop_front();
        }
        if (upload_done_ && u.next_seq == upload_chunks_ &&
            remote_state_ == RemoteState::kUploading)
          remote_state_ = RemoteState::kReady;
      }
      for (const auto& c : ups) post_chunk(u, c);
      ups.clear();
      for (auto& r : reads) post_get(u, std::move(r));
      reads.clear();
      ucp_worker_progress(u.worker);
      reap_sends(u);
      complete_gets(u);
      const std::string chunk_err = complete_chunks(u);
      if (u.ep_failed) {
        remote_fail(u, "connection to " + remote_.server +
                           " failed: " + ucs_status_string(u.ep_status));
        return;
      }
      if (!chunk_err.empty()) {
        remote_fail(u, chunk_err);
        return;
      }
    }
  }

  void post_get(Ucx& u, ReadPtr r) {
    const uint32_t slot = u.get_free.back();
    u.get_free.pop_back();
    GetOp& op = u.gets[slot];
    const uint64_t row = r->offset;
    op.read = std::move(r);
    op.status = UCS_OK;
    op.t0 = now_us();
    const RemoteRegion& reg = u.regions[row / remote_.region_rows];
    const uint64_t raddr = reg.addr + (row % remote_.region_rows) * rb_;
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_MEMH;
    p.cb.send = &Core::get_cb;
    p.user_data = &op;
    p.memh = u.get_memh;
    void* req = ucp_get_nbx(u.ep, u.get_slab + size_t{slot} * rb_, rb_, raddr,
                            reg.rkey, &p);
    if (UCS_PTR_IS_PTR(req)) {
      ++u.outstanding;
      return;
    }
    op.status = UCS_PTR_STATUS(req);
    u.get_done.push_back(slot);
  }

  // Delivers completed GETs; failed ones are requeued (or failed) after the
  // remote source is marked failed.
  void complete_gets(Ucx& u) {
    if (u.get_done.empty()) return;
    std::vector<uint32_t> done;
    done.swap(u.get_done);
    std::vector<ReadPtr> failed;
    ucs_status_t first_err = UCS_OK;
    for (uint32_t slot : done) {
      GetOp& op = u.gets[slot];
      ReadPtr r = std::move(op.read);
      if (op.status == UCS_OK) {
        finish_read(r, u.get_slab + size_t{slot} * rb_, rb_, 0,
                    now_us() - op.t0, -1);
      } else {
        if (first_err == UCS_OK) first_err = op.status;
        failed.push_back(std::move(r));
      }
      u.get_free.push_back(slot);
    }
    if (failed.empty()) return;
    if (!u.ep_failed) {
      u.ep_failed = true;
      u.ep_status = first_err;
    }
    mark_failed(std::string("remote read failed: ") +
                ucs_status_string(first_err));
    requeue(failed);
  }

  void post_chunk(Ucx& u, const UploadChunk& c) {
    ChunkOp& op = u.chunks[c.buf];
    op.seq = c.seq;
    op.row_end = c.row + c.rows;
    op.bytes = c.rows * rb_;
    op.status = UCS_OK;
    op.outstanding = 1;
    const uint8_t* src = u.stage[c.buf] + c.shift;
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                     UCP_OP_ATTR_FIELD_MEMH;
    p.cb.send = &Core::chunk_cb;
    p.user_data = &op;
    p.memh = u.stage_memh[c.buf];
    auto track = [&](void* req) {
      if (UCS_PTR_IS_PTR(req)) {
        ++op.outstanding;
        ++u.outstanding;
      } else if (UCS_PTR_IS_ERR(req)) {
        op.status = UCS_PTR_STATUS(req);
      }
    };
    for (uint64_t row = c.row; row < op.row_end;) {
      const RemoteRegion& reg = u.regions[row / remote_.region_rows];
      const uint64_t in = row % remote_.region_rows;
      const uint64_t take = std::min(op.row_end - row, reg.bytes / rb_ - in);
      track(ucp_put_nbx(u.ep, src + (row - c.row) * rb_, take * rb_,
                        reg.addr + in * rb_, reg.rkey, &p));
      row += take;
    }
    ucp_request_param_t fp{};
    fp.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    fp.cb.send = &Core::chunk_cb;
    fp.user_data = &op;
    track(ucp_ep_flush_nbx(u.ep, &fp));
    if (--op.outstanding == 0) u.chunk_done.push_back(c.buf);
  }

  // Recycles flushed staging buffers and advances the ready watermark over
  // the contiguous prefix of visible chunks. Returns an error message when a
  // put or flush failed.
  std::string complete_chunks(Ucx& u) {
    if (u.chunk_done.empty()) return "";
    std::vector<uint32_t> done;
    done.swap(u.chunk_done);
    std::string err;
    for (uint32_t b : done) {
      const ChunkOp& op = u.chunks[b];
      if (op.status != UCS_OK && err.empty())
        err = ucs_error("remote upload failed", op.status);
      u.visible[op.seq] = {op.row_end, op.bytes};
    }
    uint64_t ready = 0, bytes = 0;
    for (auto it = u.visible.begin();
         it != u.visible.end() && it->first == u.next_seq;
         it = u.visible.erase(it)) {
      ready = it->second.first;
      bytes += it->second.second;
      ++u.next_seq;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (uint32_t b : done) upload_free_.push_back(b);
      if (err.empty() && remote_usable_locked()) {
        remote_uploaded_bytes_ += bytes;
        if (ready > 0) ready_rows_.store(ready, std::memory_order_release);
        if (upload_done_ && u.next_seq == upload_chunks_)
          remote_state_ = RemoteState::kReady;
      }
    }
    cv_upload_.notify_one();
    return err;
  }

  // Marks the remote source failed: no new remote reads are planned and the
  // upload stops. Queued remote reads go back to the SSD (or fail).
  void mark_failed(const std::string& reason) {
    size_t queued = 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (remote_state_ != RemoteState::kFailed && !closed_) {
        remote_state_ = RemoteState::kFailed;
        remote_error_ = reason;
        if (!remote_.fallback_to_disk)
          set_error_locked("PLE remote memory failed: " + reason);
      }
      ready_rows_.store(0);
      upload_stop_ = true;
      for (auto* q : {&rdemand_q_, &rhint_q_}) {
        for (auto& r : *q)
          if (!r->started) done |= requeue_remote_locked(r, &queued);
        q->clear();
      }
    }
    cv_upload_.notify_all();
    if (done) cv_done_.notify_all();
    notify_workers(queued);
  }

  void requeue(const std::vector<ReadPtr>& reads) {
    size_t queued = 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& r : reads) done |= requeue_remote_locked(r, &queued);
    }
    if (done) cv_done_.notify_all();
    notify_workers(queued);
  }

  void remote_fail(Ucx& u, const std::string& reason) {
    if (!u.ep_failed) {
      u.ep_failed = true;
      u.ep_status = UCS_ERR_CANCELED;
    }
    mark_failed(reason);
  }

  // Drains outstanding operations, returns the regions and releases every
  // UCX resource. Reads still in flight are requeued or failed.
  void ucx_teardown(Ucx& u) {
    if (u.ep != nullptr && !u.ep_failed) {
      const double drain = now_us() + 5e6;
      while (u.outstanding > 0 && now_us() < drain) {
        ucp_worker_progress(u.worker);
        complete_gets(u);
        complete_chunks(u);
      }
    }
    const bool graceful = u.ep != nullptr && !u.ep_failed && u.outstanding == 0;
    if (graceful) {
      for (const auto& reg : u.regions) {
        if (reg.rkey == nullptr) continue;
        ple_mem::FreeReq req{};
        req.hdr = ple_mem::make_header(ple_mem::kFreeReq);
        req.region_id = reg.id;
        std::vector<char> payload(sizeof(req));
        std::memcpy(payload.data(), &req, sizeof(req));
        am_send(u, std::move(payload));
      }
    }
    bool flushed = false;
    if (graceful) {
      ucp_request_param_t fp{};
      flushed = wait_request(u, ucp_ep_flush_nbx(u.ep, &fp), now_us() + 2e6) ==
                UCS_OK;
      reap_sends(u);
    }
    if (flushed) destroy_rkeys(u);
    if (u.ep != nullptr) {
      ucp_request_param_t cp{};
      cp.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
      cp.flags = flushed ? 0 : UCP_EP_CLOSE_FLAG_FORCE;
      wait_request(u, ucp_ep_close_nbx(u.ep, &cp), now_us() + 2e6);
      u.ep = nullptr;
    }
    complete_gets(u);
    complete_chunks(u);
    std::vector<ReadPtr> stuck;
    for (auto& op : u.gets)
      if (op.read) stuck.push_back(std::move(op.read));
    requeue(stuck);
    for (auto& s : u.sends) ucp_request_free(s.request);
    u.sends.clear();
    destroy_rkeys(u);
    if (u.get_memh != nullptr) ucp_mem_unmap(u.ctx, u.get_memh);
    for (ucp_mem_h m : u.stage_memh) ucp_mem_unmap(u.ctx, m);
    if (u.worker != nullptr) ucp_worker_destroy(u.worker);
    if (u.ctx != nullptr) ucp_cleanup(u.ctx);
    {
      std::lock_guard<std::mutex> lk(mu_);
      upload_stop_ = true;
    }
    cv_upload_.notify_all();
    std::free(u.get_slab);
  }

  // The upload thread may still read into a staging buffer after the UCX
  // thread is gone, so they are freed once both threads were joined.
  void free_stage_buffers() {
    for (uint8_t* b : stage_bufs_) std::free(b);
    stage_bufs_.clear();
  }

  void destroy_rkeys(Ucx& u) {
    for (auto& reg : u.regions) {
      if (reg.rkey != nullptr) ucp_rkey_destroy(reg.rkey);
      reg.rkey = nullptr;
    }
  }

  // Reads the segments in row order into the staging buffers and hands each
  // filled buffer to the UCX thread.
  void upload_main() {
    std::vector<int> fds;
    bool direct = direct_atomic_.load();
    std::string err;
    for (const auto& path : paths_) {
      int fd = direct ? ::open(path.c_str(), O_RDONLY | O_DIRECT) : -1;
      if (fd < 0) {
        direct = false;
        fd = ::open(path.c_str(), O_RDONLY);
      }
      if (fd < 0) {
        err = "upload cannot open " + path + ": " + strerror(errno);
        break;
      }
      fds.push_back(fd);
    }
    uint64_t seq = 0;
    for (size_t s = 0; err.empty() && s < seg_start_.size(); ++s) {
      for (uint64_t r = 0; r < seg_rows_[s];) {
        const uint64_t n = std::min(remote_.chunk_rows, seg_rows_[s] - r);
        uint32_t b;
        uint8_t* buf;
        {
          std::unique_lock<std::mutex> lk(mu_);
          cv_upload_.wait(lk, [&] {
            return upload_stop_ || closed_ || !upload_free_.empty();
          });
          if (upload_stop_ || closed_) {
            for (int fd : fds) ::close(fd);
            return;
          }
          b = upload_free_.back();
          upload_free_.pop_back();
          buf = stage_bufs_[b];
        }
        const uint64_t off = seg_offset_[s] + r * rb_;
        const uint64_t start = off & ~(kPage - 1);
        const uint64_t end = (off + n * rb_ + kPage - 1) & ~(kPage - 1);
        const int fd = fds[seg_file_[s]];
        size_t got = 0;
        while (got < end - start) {
          const ssize_t g = ::pread(fd, buf + got, end - start - got,
                                    static_cast<off_t>(start + got));
          if (g < 0 && errno == EINTR) continue;
          if (g < 0 && errno == EINVAL && direct) {
            direct = false;
            for (int f : fds) {
              const int fl = fcntl(f, F_GETFL);
              if (fl >= 0) fcntl(f, F_SETFL, fl & ~O_DIRECT);
            }
            continue;
          }
          if (g < 0) {
            err = "upload pread failed at offset " +
                  std::to_string(start + got) + ": " + strerror(errno);
            break;
          }
          if (g == 0) break;
          got += static_cast<size_t>(g);
        }
        if (err.empty() && got < off - start + n * rb_)
          err = "upload short read at offset " + std::to_string(off) +
                ": file truncated?";
        if (!direct)
          posix_fadvise(fd, static_cast<off_t>(start),
                        static_cast<off_t>(end - start), POSIX_FADV_DONTNEED);
        {
          std::lock_guard<std::mutex> lk(mu_);
          if (!err.empty()) {
            upload_free_.push_back(b);
            break;
          }
          upload_q_.push_back(UploadChunk{seq++, b,
                                          static_cast<uint32_t>(off - start),
                                          seg_start_[s] + r, n});
          remote_kick_.fetch_add(1);
        }
        cv_remote_.notify_one();
        r += n;
      }
    }
    for (int fd : fds) ::close(fd);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (err.empty()) {
        upload_done_ = true;
        upload_chunks_ = seq;
      } else {
        upload_error_ = err;
      }
      remote_kick_.fetch_add(1);
    }
    cv_remote_.notify_one();
  }

  std::vector<uint8_t*> stage_bufs_;
  std::deque<UploadChunk> upload_q_;
  std::vector<uint32_t> upload_free_;
#endif

  const uint32_t rb_;
  const double timeout_us_;
  std::vector<std::string> paths_;
  std::vector<uint32_t> seg_file_;
  std::vector<uint64_t> seg_offset_, seg_start_, seg_rows_;
  std::vector<int> fds_;
  bool direct_ = false;
  std::atomic<bool> direct_atomic_{false};
  uint8_t* slab_ = nullptr;
  uint32_t slots_ = 0;
  RowCache cache_;

  std::mutex mu_;
  std::condition_variable cv_work_, cv_plan_, cv_done_;
  bool closed_ = false;
  std::vector<uint32_t> free_slots_;
  std::deque<ReadPtr> demand_q_, hint_q_;
  std::unordered_map<uint64_t, ReadPtr> inflight_;
  std::deque<JobPtr> plan_q_;
  std::deque<std::vector<int64_t>> hint_inbox_;
  size_t hint_inbox_rows_ = 0;
  Stats stats_;
  LatencyRing latency_, remote_latency_;
  std::string error_;

  // Remote source. Fields below are guarded by mu_ unless atomic.
  RemoteParams remote_;
  RemoteState remote_state_ = RemoteState::kDisabled;
  std::string remote_error_;
  uint64_t remote_total_bytes_ = 0;
  uint64_t remote_uploaded_bytes_ = 0;
  uint64_t remote_regions_ = 0;
  std::atomic<uint64_t> ready_rows_{0};
  std::atomic<uint64_t> remote_kick_{0};
  std::atomic<bool> stop_remote_{false};
  std::deque<ReadPtr> rdemand_q_, rhint_q_;
  std::condition_variable cv_remote_, cv_upload_, cv_setup_;
  bool remote_setup_done_ = false;
  bool upload_stop_ = false;
  bool upload_done_ = false;
  uint64_t upload_chunks_ = 0;
  std::string upload_error_;
  std::thread ucx_thread_, upload_thread_;

  const int64_t* bound_ids_ = nullptr;
  const int64_t* bound_num_tokens_ = nullptr;
  uint8_t* bound_out_ = nullptr;
  int64_t bound_max_tokens_ = 0;
  int64_t bound_heads_ = 0;
  JobPtr bound_latest_;
  std::atomic<bool> host_funcs_launched_{false};

  size_t num_workers_ = 0;
  std::thread planner_;
  std::vector<std::thread> workers_;
};

// Python handle. Once a host function referencing the core was enqueued, a
// captured CUDA graph may still call into it, so the (closed, small) core is
// intentionally leaked instead of freed.
class Reader {
 public:
  template <typename... Args>
  explicit Reader(Args&&... args)
      : core_(new Core(std::forward<Args>(args)...)) {}
  ~Reader() { release(); }

  void close() {
    if (core_ != nullptr) core_->close();
  }
  Core& core() {
    if (core_ == nullptr) throw std::runtime_error("reader released");
    return *core_;
  }

 private:
  void release() {
    if (core_ == nullptr) return;
    core_->close();
    if (!core_->host_funcs_launched()) delete core_;
    core_ = nullptr;
  }
  Core* core_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using Release = py::call_guard<py::gil_scoped_release>;
  py::class_<Reader>(m, "Reader")
      .def(
          py::init<const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<int64_t>&, const std::vector<int64_t>&,
                   const std::vector<int64_t>&, int64_t, int64_t, int64_t,
                   int64_t, bool, double, const std::string&, int64_t, int64_t,
                   int64_t, double, bool>(),
          Release())
      .def("close", &Reader::close, Release())
      .def(
          "read_rows",
          [](Reader& r, uint64_t ids, int64_t n, uint64_t out) {
            r.core().read_rows(ids, n, out);
          },
          Release())
      .def(
          "hint",
          [](Reader& r, uint64_t ids, int64_t n) { r.core().hint(ids, n); },
          Release())
      .def(
          "bind",
          [](Reader& r, uint64_t ids, uint64_t num_tokens, uint64_t out,
             int64_t max_tokens, int64_t heads) {
            r.core().bind(ids, num_tokens, out, max_tokens, heads);
          },
          Release())
      .def(
          "launch",
          [](Reader& r, uint64_t stream, bool wait) {
            r.core().launch(stream, wait);
          },
          Release())
      .def("error", [](Reader& r) { return r.core().error(); })
      .def("direct_io", [](Reader& r) { return r.core().direct_io(); })
      .def("stats", [](Reader& r) { return r.core().stats(); })
      .def("remote_status", [](Reader& r) { return r.core().remote_status(); })
      .def("reset_stats", [](Reader& r) { r.core().reset_stats(); })
      .def_property_readonly_static("with_ucx", [](py::object) {
#ifdef PLE_WITH_UCX
        return true;
#else
        return false;
#endif
      });
}
