// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Remote memory source of the PLE row reader (ple_disk_reader.cpp), generic
// over the transport that reaches the memory server.
//
// The table is uploaded in row order into regions borrowed from the server
// (region i holds rows [i * region_rows, (i + 1) * region_rows)); rows below
// the contiguous uploaded prefix (`Link::ready_rows`) are then read back with
// one-row GETs. A remote thread owns the transport: it pops the remote reads
// queued by the host, posts GETs and staged upload chunks, progresses the
// transport and delivers completions through the host. An upload thread fills
// the staging buffers from the table files.
//
// Transport concept. One instance per source; every member function is called
// from the remote thread only, so a transport needs no locking.
//   using Status;                 operation status
//   static constexpr Status kOk, kCanceled;
//   static const char* status_string(Status);
//   using Mem;                    registration of a local buffer, cheap to copy
//   static constexpr const char* kName;     backend name
//   explicit T(std::string address);        no side effects
//  Control path (cold), in this order; each returns "" or an error message:
//   std::string open(uint32_t num_tokens);  connects; op tokens < num_tokens
//   std::string register_memory(void* addr, size_t bytes, Mem* mem);
//   std::string borrow(const std::vector<uint64_t>& region_bytes,
//                      double timeout_s, const std::atomic<bool>& stop);
//   void close();  returns the regions (gracefully when healthy with nothing
//                  outstanding, forced otherwise) and releases everything;
//                  also safe after a failed open() or borrow().
//  Data path (hot; inline, non-blocking, no allocation once warm):
//   void get(uint32_t token, void* dst, Mem dst_mem, uint32_t region,
//            uint64_t offset, uint32_t bytes);
//   void put(uint32_t token, const void* src, Mem src_mem, uint32_t region,
//            uint64_t offset, uint64_t bytes);
//   void flush(uint32_t token);   completes once every earlier put is visible
//   void progress();
//   uint64_t outstanding() const; posted ops whose completion is pending
//   std::vector<Completion<Status>>& completions();
//   bool failed() const; Status failure() const; void fail(Status);
// Every posted op reports exactly one Completion{token, status}: appended to
// completions() by the post call itself when it completes or is rejected
// immediately, otherwise by a later progress() or close(). The source swaps
// the vector out and drains it in batches. Once failed, a transport stays
// failed and reports `failure()` through failed().
//
// Host concept (the reader core):
//   using RemoteRead;   shared pointer to a queued read; `->offset` is its row
//   std::mutex& remote_mutex();   guards the Link, the host's remote queues
//                                 and this source's upload state
//   Link& remote_link();
//   bool closed_locked() const;
//   RemoteRead pop_remote_read_locked();  next queued read, marked started
//   bool remote_reads_queued_locked() const;
//   void complete_remote_read(const RemoteRead&, const uint8_t* row,
//                             double us);
//   void requeue_remote_reads(const std::vector<RemoteRead>&);  to the SSD
//                                 or failed, per fallback_to_disk
//   void fail_remote(const std::string& reason);  marks the Link failed,
//                                 clears ready_rows, requeues queued reads
// The three functions without `_locked` take the mutex themselves.
//
// The host owns the source through RemoteSourceBase, whose virtual functions
// are the cold start / stop only; once started, the remote thread runs the
// statically typed loop of RemoteSource<Host, Transport>, so posting,
// progress and completion never go through virtual calls.

#pragma once

#include <fcntl.h>
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
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ple_remote {

constexpr uint64_t kPage = 4096;

inline double now_us() {
  using namespace std::chrono;
  return static_cast<double>(
             duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

enum class State { kDisabled, kConnecting, kUploading, kReady, kFailed };

inline const char* state_name(State s) {
  switch (s) {
    case State::kDisabled:
      return "disabled";
    case State::kConnecting:
      return "connecting";
    case State::kUploading:
      return "uploading";
    case State::kReady:
      return "ready";
    case State::kFailed:
      return "failed";
  }
  return "unknown";
}

template <class Status>
struct Completion {
  uint32_t token;
  Status status;
};

struct Params {
  std::string address;
  uint32_t max_inflight = 0;
  uint64_t region_rows = 0;
  uint64_t chunk_rows = 0;
  double connect_timeout_s = 0;
  bool fallback_to_disk = true;
};

// Table files and segments (sorted by row_start, non-overlapping).
struct Layout {
  uint32_t row_bytes = 0;
  std::vector<std::string> paths;
  std::vector<uint32_t> seg_file;
  std::vector<uint64_t> seg_offset, seg_start, seg_rows;

  uint64_t num_rows() const {
    return seg_start.empty() ? 0 : seg_start.back() + seg_rows.back();
  }
};

// State shared by the host and the source; guarded by the host mutex unless
// atomic.
struct Link {
  Params params;
  State state = State::kDisabled;
  std::string error;
  uint64_t total_bytes = 0;
  uint64_t uploaded_bytes = 0;
  uint64_t regions = 0;
  std::atomic<uint64_t> ready_rows{0};  // rows below are readable remotely
  std::atomic<uint64_t> kick{0};        // bumped whenever remote work queues
  std::condition_variable cv;           // wakes the remote thread

  bool usable() const {
    return state == State::kUploading || state == State::kReady;
  }
};

class RemoteSourceBase {
 public:
  virtual ~RemoteSourceBase() = default;
  // Connects and borrows every region (bounded by the connect timeout), then
  // starts the upload. Returns the setup error when there is no fallback.
  virtual std::string start() = 0;
  // Joins the remote and upload threads. Idempotent.
  virtual void stop() = 0;
};

template <class Host, class Transport>
class RemoteSource final : public RemoteSourceBase {
 public:
  using Read = typename Host::RemoteRead;
  using Status = typename Transport::Status;
  using Mem = typename Transport::Mem;

  RemoteSource(Host& host, Layout layout, bool direct_io)
      : host_(host),
        link_(host.remote_link()),
        params_(link_.params),
        layout_(std::move(layout)),
        rb_(layout_.row_bytes),
        num_rows_(layout_.num_rows()),
        direct_(direct_io),
        t_(params_.address) {}
  ~RemoteSource() override { stop(); }
  RemoteSource(const RemoteSource&) = delete;
  RemoteSource& operator=(const RemoteSource&) = delete;

  std::string start() override {
    link_.state = State::kConnecting;
    remote_thread_ = std::thread([this] { run(); });
    const auto guard =
        std::chrono::duration<double>(params_.connect_timeout_s + 30.0);
    std::unique_lock<std::mutex> lk(host_.remote_mutex());
    if (!cv_setup_.wait_for(lk, guard, [&] { return setup_done_; })) {
      link_.state = State::kFailed;
      link_.error = "timed out setting up the remote memory source";
    }
    if (link_.state == State::kFailed)
      return params_.fallback_to_disk ? "" : link_.error;
    upload_thread_ = std::thread([this] { upload_main(); });
    return "";
  }

  // The upload thread may still read into a staging buffer after the remote
  // thread exited, so the buffers are freed last.
  void stop() override {
    stop_.store(true);
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      upload_stop_ = true;
    }
    link_.cv.notify_all();
    cv_upload_.notify_all();
    if (upload_thread_.joinable()) upload_thread_.join();
    if (remote_thread_.joinable()) remote_thread_.join();
    for (uint8_t* b : stage_) std::free(b);
    stage_.clear();
  }

 private:
  static constexpr uint32_t kUploadBuffers = 4;

  struct GetSlot {
    Read read;
    double t0 = 0;
  };

  // Puts of one staged upload chunk plus the flush making them visible.
  struct ChunkOp {
    uint64_t seq = 0;
    uint64_t row_end = 0;
    uint64_t bytes = 0;
    uint32_t pending = 0;
    Status status = Transport::kOk;
  };

  struct UploadChunk {
    uint64_t seq;
    uint32_t buf;
    uint32_t shift;
    uint64_t row;
    uint64_t rows;
  };

  void run() {
    std::string err = setup();
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      if (link_.state == State::kFailed) err = link_.error;
      if (err.empty()) {
        link_.state = State::kUploading;
        link_.regions = num_regions_;
        for (uint32_t b = 0; b < kUploadBuffers; ++b) upload_free_.push_back(b);
      } else {
        link_.state = State::kFailed;
        link_.error = err;
      }
      setup_done_ = true;
    }
    cv_setup_.notify_all();
    if (err.empty()) loop();
    teardown();
  }

  size_t stage_bytes() const {
    const uint64_t data = params_.chunk_rows * rb_;
    return (data + kPage - 1) / kPage * kPage + 2 * kPage;
  }

  uint64_t region_rows(uint64_t region) const {
    return std::min(params_.region_rows,
                    num_rows_ - region * params_.region_rows);
  }

  // Connects, registers the local buffers and borrows every region. GET slot
  // s uses token s, staging buffer b token max_inflight + b.
  std::string setup() {
    const uint32_t slots = params_.max_inflight;
    std::string err = t_.open(slots + kUploadBuffers);
    if (!err.empty()) return err;
    const size_t slab_bytes = (size_t{slots} * rb_ + kPage - 1) / kPage * kPage;
    void* p = nullptr;
    if (posix_memalign(&p, kPage, slab_bytes) != 0)
      return "cannot allocate the remote read slab";
    get_slab_ = static_cast<uint8_t*>(p);
    err = t_.register_memory(get_slab_, slab_bytes, &get_mem_);
    if (!err.empty()) return err;
    gets_.resize(slots);
    for (uint32_t s = slots; s-- > 0;) get_free_.push_back(s);
    chunks_.resize(kUploadBuffers);
    stage_mem_.resize(kUploadBuffers);
    for (uint32_t b = 0; b < kUploadBuffers; ++b) {
      if (posix_memalign(&p, kPage, stage_bytes()) != 0)
        return "cannot allocate the upload staging buffers";
      stage_.push_back(static_cast<uint8_t*>(p));
      err = t_.register_memory(p, stage_bytes(), &stage_mem_[b]);
      if (!err.empty()) return err;
    }
    done_.reserve(slots + kUploadBuffers);
    chunk_done_.reserve(kUploadBuffers);
    num_regions_ = (num_rows_ + params_.region_rows - 1) / params_.region_rows;
    std::vector<uint64_t> bytes(num_regions_);
    for (uint64_t i = 0; i < num_regions_; ++i) bytes[i] = region_rows(i) * rb_;
    return t_.borrow(bytes, params_.connect_timeout_s, stop_);
  }

  void loop() {
    std::vector<Read> reads;
    std::vector<UploadChunk> ups;
    for (;;) {
      const uint64_t kick = link_.kick.load();
      const bool need_lock = t_.outstanding() == 0 || kick != seen_kick_ ||
                             stop_.load() || (backlog_ && !get_free_.empty());
      if (need_lock) {
        std::unique_lock<std::mutex> lk(host_.remote_mutex());
        if (t_.outstanding() == 0)
          link_.cv.wait_for(lk, std::chrono::milliseconds(1), [&] {
            return host_.closed_locked() || link_.kick.load() != seen_kick_;
          });
        if (host_.closed_locked() || stop_.load()) return;
        seen_kick_ = link_.kick.load();
        if (!upload_error_.empty()) {
          const std::string msg = upload_error_;
          lk.unlock();
          remote_fail(msg);
          return;
        }
        while (reads.size() < get_free_.size()) {
          Read r = host_.pop_remote_read_locked();
          if (!r) break;
          reads.push_back(std::move(r));
        }
        backlog_ = host_.remote_reads_queued_locked();
        while (!upload_q_.empty()) {
          ups.push_back(upload_q_.front());
          upload_q_.pop_front();
        }
        if (upload_done_ && next_seq_ == upload_chunks_ &&
            link_.state == State::kUploading)
          link_.state = State::kReady;
      }
      for (const auto& c : ups) post_chunk(c);
      ups.clear();
      for (auto& r : reads) post_get(std::move(r));
      reads.clear();
      t_.progress();
      const std::string chunk_err = complete();
      if (t_.failed()) {
        remote_fail("connection to " + params_.address +
                    " failed: " + Transport::status_string(t_.failure()));
        return;
      }
      if (!chunk_err.empty()) {
        remote_fail(chunk_err);
        return;
      }
    }
  }

  void post_get(Read r) {
    const uint32_t slot = get_free_.back();
    get_free_.pop_back();
    GetSlot& g = gets_[slot];
    const uint64_t row = r->offset;
    g.read = std::move(r);
    g.t0 = now_us();
    t_.get(slot, get_slab_ + size_t{slot} * rb_, get_mem_,
           static_cast<uint32_t>(row / params_.region_rows),
           (row % params_.region_rows) * rb_, rb_);
  }

  // Puts the chunk split at region boundaries, then flushes.
  void post_chunk(const UploadChunk& c) {
    ChunkOp& op = chunks_[c.buf];
    op.seq = c.seq;
    op.row_end = c.row + c.rows;
    op.bytes = c.rows * rb_;
    op.status = Transport::kOk;
    op.pending = 0;
    const uint32_t token = params_.max_inflight + c.buf;
    const uint8_t* src = stage_[c.buf] + c.shift;
    for (uint64_t row = c.row; row < op.row_end;) {
      const uint64_t region = row / params_.region_rows;
      const uint64_t in = row % params_.region_rows;
      const uint64_t take =
          std::min(op.row_end - row, region_rows(region) - in);
      ++op.pending;
      t_.put(token, src + (row - c.row) * rb_, stage_mem_[c.buf],
             static_cast<uint32_t>(region), in * rb_, take * rb_);
      row += take;
    }
    ++op.pending;
    t_.flush(token);
  }

  // Delivers completed GETs (failed ones fail the source and are requeued)
  // and finishes upload chunks. Returns an upload error, "" if none.
  std::string complete() {
    std::vector<Completion<Status>>& fresh = t_.completions();
    if (fresh.empty()) return "";
    done_.swap(fresh);
    const uint32_t slots = params_.max_inflight;
    Status first_err = Transport::kOk;
    for (const auto& c : done_) {
      if (c.token >= slots) {
        ChunkOp& op = chunks_[c.token - slots];
        if (c.status != Transport::kOk) op.status = c.status;
        if (--op.pending == 0) chunk_done_.push_back(c.token - slots);
        continue;
      }
      GetSlot& g = gets_[c.token];
      Read r = std::move(g.read);
      if (c.status == Transport::kOk) {
        host_.complete_remote_read(r, get_slab_ + size_t{c.token} * rb_,
                                   now_us() - g.t0);
      } else {
        if (failed_.empty()) first_err = c.status;
        failed_.push_back(std::move(r));
      }
      get_free_.push_back(c.token);
    }
    done_.clear();
    if (!failed_.empty()) {
      if (!t_.failed()) t_.fail(first_err);
      mark_failed(std::string("remote read failed: ") +
                  Transport::status_string(first_err));
      host_.requeue_remote_reads(failed_);
      failed_.clear();
    }
    return complete_chunks();
  }

  // Recycles flushed staging buffers and advances the ready watermark over
  // the contiguous prefix of visible chunks.
  std::string complete_chunks() {
    if (chunk_done_.empty()) return "";
    std::string err;
    for (uint32_t b : chunk_done_) {
      const ChunkOp& op = chunks_[b];
      if (op.status != Transport::kOk && err.empty())
        err = std::string("remote upload failed: ") +
              Transport::status_string(op.status);
      visible_[op.seq] = {op.row_end, op.bytes};
    }
    uint64_t ready = 0, bytes = 0;
    for (auto it = visible_.begin();
         it != visible_.end() && it->first == next_seq_;
         it = visible_.erase(it)) {
      ready = it->second.first;
      bytes += it->second.second;
      ++next_seq_;
    }
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      for (uint32_t b : chunk_done_) upload_free_.push_back(b);
      if (err.empty() && link_.usable()) {
        link_.uploaded_bytes += bytes;
        if (ready > 0) link_.ready_rows.store(ready, std::memory_order_release);
        if (upload_done_ && next_seq_ == upload_chunks_)
          link_.state = State::kReady;
      }
    }
    chunk_done_.clear();
    cv_upload_.notify_one();
    return err;
  }

  // No new remote reads are planned and the upload stops; queued remote
  // reads go back to the SSD (or fail).
  void mark_failed(const std::string& reason) {
    host_.fail_remote(reason);
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      upload_stop_ = true;
    }
    cv_upload_.notify_all();
  }

  void remote_fail(const std::string& reason) {
    if (!t_.failed()) t_.fail(Transport::kCanceled);
    mark_failed(reason);
  }

  // Drains outstanding operations, closes the transport and requeues (or
  // fails) the reads that never completed.
  void teardown() {
    if (!t_.failed()) {
      const double drain = now_us() + 5e6;
      while (t_.outstanding() > 0 && now_us() < drain) {
        t_.progress();
        complete();
      }
    }
    t_.close();
    complete();
    std::vector<Read> stuck;
    for (auto& g : gets_)
      if (g.read) stuck.push_back(std::move(g.read));
    host_.requeue_remote_reads(stuck);
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      upload_stop_ = true;
    }
    cv_upload_.notify_all();
    std::free(get_slab_);
    get_slab_ = nullptr;
  }

  // Reads the segments in row order into the staging buffers and hands each
  // filled buffer to the remote thread.
  void upload_main() {
    std::vector<int> fds;
    bool direct = direct_;
    std::string err;
    for (const auto& path : layout_.paths) {
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
    const Layout& l = layout_;
    uint64_t seq = 0;
    for (size_t s = 0; err.empty() && s < l.seg_start.size(); ++s) {
      for (uint64_t r = 0; r < l.seg_rows[s];) {
        const uint64_t n = std::min(params_.chunk_rows, l.seg_rows[s] - r);
        uint32_t b;
        uint8_t* buf;
        {
          std::unique_lock<std::mutex> lk(host_.remote_mutex());
          cv_upload_.wait(lk, [&] {
            return upload_stop_ || host_.closed_locked() ||
                   !upload_free_.empty();
          });
          if (upload_stop_ || host_.closed_locked()) {
            for (int fd : fds) ::close(fd);
            return;
          }
          b = upload_free_.back();
          upload_free_.pop_back();
          buf = stage_[b];
        }
        const uint64_t off = l.seg_offset[s] + r * rb_;
        const uint64_t start = off & ~(kPage - 1);
        const uint64_t end = (off + n * rb_ + kPage - 1) & ~(kPage - 1);
        const int fd = fds[l.seg_file[s]];
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
          std::lock_guard<std::mutex> lk(host_.remote_mutex());
          if (!err.empty()) {
            upload_free_.push_back(b);
            break;
          }
          upload_q_.push_back(UploadChunk{seq++, b,
                                          static_cast<uint32_t>(off - start),
                                          l.seg_start[s] + r, n});
          link_.kick.fetch_add(1);
        }
        link_.cv.notify_one();
        r += n;
      }
    }
    for (int fd : fds) ::close(fd);
    {
      std::lock_guard<std::mutex> lk(host_.remote_mutex());
      if (err.empty()) {
        upload_done_ = true;
        upload_chunks_ = seq;
      } else {
        upload_error_ = err;
      }
      link_.kick.fetch_add(1);
    }
    link_.cv.notify_one();
  }

  Host& host_;
  Link& link_;
  const Params& params_;
  const Layout layout_;
  const uint32_t rb_;
  const uint64_t num_rows_;
  const bool direct_;
  std::atomic<bool> stop_{false};
  std::thread remote_thread_, upload_thread_;
  std::condition_variable cv_setup_, cv_upload_;

  // Remote thread only.
  Transport t_;
  uint64_t num_regions_ = 0;
  uint8_t* get_slab_ = nullptr;
  Mem get_mem_{};
  std::vector<GetSlot> gets_;
  std::vector<uint32_t> get_free_;
  std::vector<Mem> stage_mem_;
  std::vector<ChunkOp> chunks_;
  std::vector<Completion<Status>> done_;
  std::vector<uint32_t> chunk_done_;
  std::vector<Read> failed_;
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> visible_;
  uint64_t next_seq_ = 0;
  uint64_t seen_kick_ = 0;
  bool backlog_ = false;

  // Written by the remote thread during setup, read by the upload thread.
  std::vector<uint8_t*> stage_;

  // Guarded by the host mutex.
  bool setup_done_ = false;
  bool upload_stop_ = false;
  bool upload_done_ = false;
  uint64_t upload_chunks_ = 0;
  std::string upload_error_;
  std::deque<UploadChunk> upload_q_;
  std::vector<uint32_t> upload_free_;
};

}  // namespace ple_remote
