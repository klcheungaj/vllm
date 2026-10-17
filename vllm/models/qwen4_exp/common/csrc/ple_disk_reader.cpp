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
// Built with remote memory backends (-DPLE_WITH_UCX, -DPLE_WITH_LOOPBACK for
// in-process tests; any subset), the table can also be uploaded to RAM
// borrowed from a memory server and read back with one-row GETs
// (ple_remote_source.h); the SSD stays the source for rows not uploaded yet
// and the fallback when the server fails. The backend is chosen by name once,
// when the reader starts.
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

#include "ple_remote_source.h"

#ifdef PLE_WITH_UCX
  #include "ple_ucx_transport.h"
#endif
#ifdef PLE_WITH_LOOPBACK
  #include "ple_loopback_transport.h"
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

using ple_remote::now_us;

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

class Core;

using RemoteFactory = std::unique_ptr<ple_remote::RemoteSourceBase> (*)(
    Core&, ple_remote::Layout, bool);

// Factory of the compiled-in backend `name`, or null.
RemoteFactory find_remote_backend(const std::string& name);
std::vector<std::string> remote_backend_names();

class Core {
 public:
  Core(const std::vector<std::string>& paths,
       const std::vector<int64_t>& seg_file,
       const std::vector<int64_t>& seg_offset,
       const std::vector<int64_t>& seg_row_start,
       const std::vector<int64_t>& seg_num_rows, int64_t row_bytes,
       int64_t cache_rows, int64_t io_threads, int64_t max_inflight,
       bool direct_io, double wait_timeout_s, const std::string& remote_server,
       const std::string& remote_backend, int64_t remote_max_inflight,
       int64_t remote_region_bytes, int64_t remote_upload_chunk_bytes,
       double remote_connect_timeout_s, bool remote_fallback_to_disk)
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
      remote_make_ = find_remote_backend(remote_backend);
      if (remote_make_ == nullptr) {
        std::string names;
        for (const auto& n : remote_backend_names())
          names += (names.empty() ? "" : ", ") + n;
        throw std::invalid_argument(
            "PLE remote memory backend '" + remote_backend +
            "' is not compiled into this reader (compiled in: " +
            (names.empty() ? "none" : names) + ")");
      }
      if (remote_max_inflight <= 0 || remote_region_bytes <= 0 ||
          remote_upload_chunk_bytes <= 0 || remote_connect_timeout_s <= 0)
        throw std::invalid_argument(
            "remote max_inflight, region_bytes, upload_chunk_bytes and "
            "connect_timeout_s must be > 0");
      ple_remote::Params& p = remote_.params;
      p.address = remote_server;
      p.max_inflight = static_cast<uint32_t>(remote_max_inflight);
      p.region_rows =
          std::max<uint64_t>(1, uint64_t(remote_region_bytes) / rb_);
      p.chunk_rows =
          std::max<uint64_t>(1, uint64_t(remote_upload_chunk_bytes) / rb_);
      p.connect_timeout_s = remote_connect_timeout_s;
      p.fallback_to_disk = remote_fallback_to_disk;
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
      remote_.total_bytes += uint64_t(seg_num_rows[i]) * rb_;
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
    if (remote_make_ != nullptr) start_remote();
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
    cv_work_.notify_all();
    cv_plan_.notify_all();
    remote_.cv.notify_all();
    if (planner_.joinable()) planner_.join();
    for (auto& t : workers_) t.join();
    workers_.clear();
    if (remote_src_) remote_src_->stop();
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
      remote_.ready_rows.store(0);
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
    ple_remote::State state;
    uint64_t uploaded, regions;
    std::string error;
    {
      std::lock_guard<std::mutex> lk(mu_);
      state = remote_.state;
      uploaded = remote_.uploaded_bytes;
      regions = remote_.regions;
      error = remote_.error;
    }
    py::dict d;
    d["state"] = ple_remote::state_name(state);
    d["uploaded_bytes"] = uploaded;
    d["total_bytes"] =
        state == ple_remote::State::kDisabled ? 0 : remote_.total_bytes;
    d["regions"] = regions;
    d["error"] = error;
    return d;
  }

  bool direct_io() {
    std::lock_guard<std::mutex> lk(mu_);
    return direct_;
  }

  bool host_funcs_launched() const { return host_funcs_launched_; }

  // Host interface of the remote source (ple_remote_source.h).
  using RemoteRead = ReadPtr;

  std::mutex& remote_mutex() { return mu_; }
  ple_remote::Link& remote_link() { return remote_; }
  bool closed_locked() const { return closed_; }

  ReadPtr pop_remote_read_locked() {
    ReadPtr r = pop_read_locked(rdemand_q_, rhint_q_);
    if (r) r->started = true;
    return r;
  }

  bool remote_reads_queued_locked() const {
    return !rdemand_q_.empty() || !rhint_q_.empty();
  }

  void complete_remote_read(const ReadPtr& r, const uint8_t* row, double us) {
    finish_read(r, row, rb_, 0, us, -1);
  }

  void requeue_remote_reads(const std::vector<ReadPtr>& reads) {
    size_t queued = 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& r : reads) done |= requeue_remote_locked(r, &queued);
    }
    if (done) cv_done_.notify_all();
    notify_workers(queued);
  }

  // No new remote reads are planned; queued ones go back to the SSD (or
  // fail).
  void fail_remote(const std::string& reason) {
    size_t queued = 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (remote_.state != ple_remote::State::kFailed && !closed_) {
        remote_.state = ple_remote::State::kFailed;
        remote_.error = reason;
        if (!remote_.params.fallback_to_disk)
          set_error_locked("PLE remote memory failed: " + reason);
      }
      remote_.ready_rows.store(0);
      for (auto* q : {&rdemand_q_, &rhint_q_}) {
        for (auto& r : *q)
          if (!r->started) done |= requeue_remote_locked(r, &queued);
        q->clear();
      }
    }
    if (done) cv_done_.notify_all();
    notify_workers(queued);
  }

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
    const uint64_t ready = remote_.ready_rows.load(std::memory_order_acquire);
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
      if (any_remote && !remote_.usable()) {
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
    if (queued_remote) remote_.cv.notify_one();
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
    if (*fresh && m.remote) remote_.kick.fetch_add(1);
    return r;
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
    const bool fallback = remote_.params.fallback_to_disk && !closed_;
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

  void start_remote();

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

  // Remote source state, guarded by mu_ like the remote read queues.
  ple_remote::Link remote_;
  std::deque<ReadPtr> rdemand_q_, rhint_q_;

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
  RemoteFactory remote_make_ = nullptr;
  // Last, so it is destroyed before the state it refers to.
  std::unique_ptr<ple_remote::RemoteSourceBase> remote_src_;
};

template <class Transport>
std::unique_ptr<ple_remote::RemoteSourceBase> make_remote_source(
    Core& core, ple_remote::Layout layout, bool direct_io) {
  return std::make_unique<ple_remote::RemoteSource<Core, Transport>>(
      core, std::move(layout), direct_io);
}

struct RemoteBackend {
  const char* name;
  RemoteFactory make;
};

// Backends compiled into this variant, null-terminated. One is picked by name
// when the reader starts; its RemoteSource<Transport> then serves every read.
constexpr RemoteBackend kRemoteBackends[] = {
#ifdef PLE_WITH_UCX
    {ple_remote::UcxTransport::kName,
     &make_remote_source<ple_remote::UcxTransport>},
#endif
#ifdef PLE_WITH_LOOPBACK
    {ple_remote::LoopbackTransport::kName,
     &make_remote_source<ple_remote::LoopbackTransport>},
#endif
    {nullptr, nullptr},
};

RemoteFactory find_remote_backend(const std::string& name) {
  for (const RemoteBackend* b = kRemoteBackends; b->name != nullptr; ++b)
    if (name == b->name) return b->make;
  return nullptr;
}

std::vector<std::string> remote_backend_names() {
  std::vector<std::string> names;
  for (const RemoteBackend* b = kRemoteBackends; b->name != nullptr; ++b)
    names.emplace_back(b->name);
  return names;
}

void Core::start_remote() {
  ple_remote::Layout layout{rb_,         paths_,     seg_file_,
                            seg_offset_, seg_start_, seg_rows_};
  remote_src_ = remote_make_(*this, std::move(layout), direct_);
  const std::string err = remote_src_->start();
  if (err.empty()) return;
  remote_src_.reset();
  close_fds();
  std::free(slab_);
  slab_ = nullptr;
  throw std::runtime_error("PLE remote memory setup failed: " + err);
}

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
      .def(py::init<const std::vector<std::string>&,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    int64_t, int64_t, int64_t, int64_t, bool, double,
                    const std::string&, const std::string&, int64_t, int64_t,
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
      .def("reset_stats", [](Reader& r) { r.core().reset_stats(); });
  m.attr("REMOTE_BACKENDS") = py::tuple(py::cast(remote_backend_names()));
}
