/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cxxabi.h>
#include <execinfo.h>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <regex>

#include "framework/core/caching_allocator.h"
#include "utils/python_interface.h"
#include "framework/core/device.h"
#include "framework/core/mlu_guard.h"
#include "framework/core/caching_notifier.h"

#include <c10/util/llvmMathExtras.h>
#if USE_PROFILE
#include <c10/util/ThreadLocalDebugInfo.h>
#endif

#define MASK_WORDS 1

static bool is_native_memory_strategy_ = true;

namespace torch_mlu {
C10_DEFINE_REGISTRY(FreeMluMemoryCallbacksRegistry, FreeMemoryCallback);

using queue_set = ska::flat_hash_set<torch_mlu::Queue>;

// the constant parameters for chunk
constexpr size_t minimum_round_size =
    512;  // all chunks are rounded at least 512 bytes
constexpr size_t small_allocation_size =
    1048576;  // maximum for "small" allocation is 1 Mib
constexpr size_t small_buffer_size =
    2097152;  // "small" allocations are in 2 Mibs chunks
constexpr size_t large_allocation_size =
    10485760;  // allocation sizes from 1 Mib to 10 Mibs use larger chunks
constexpr size_t large_buffer_size =
    20971520;  // "large" allocations may be in 20 Mibs chunks
constexpr size_t maximum_round_size =
    2097152;  // all chunks are rounded at most 2 Mibs

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

// // DEBUG_MODE: the mask size
// constexpr size_t mask_bytes = MASK_WORDS * sizeof(int64_t);
// 
// // DEBUG_MODE: Debugging Flag
// static bool debug_mode = false;
// 
// // DEBUG_MODE: backtrace layers
// constexpr int layer_num = 10;
//
// // DEBUG_MODE: newMask
// std::shared_ptr<int64_t> newMask(int64_t magic_word) {
//   std::shared_ptr<int64_t> m (new int64_t[MASK_WORDS], std::default_delete<int64_t[]>());
//   for (int i = 0; i < MASK_WORDS; ++i) {
//     m.get()[i] = magic_word;
//   }
//   return m;
// }

// // DEBUG_MODE: Caculate amount for mask
// int64_t mask_amount(int64_t amount) {
//   if (amount > 0) {
//     amount -= 2 * mask_bytes;
//   } else if (amount < 0) {
//     amount += 2 * mask_bytes;
//   } else {
//   }
//   return amount;
// }
// 
// void DebugStats::update_allocated_bytes(int64_t amount, const StatTypes& stat_types) {
//   // if amount > 0 means that it is allocating else if amount < 0 means that it is deallocating
//   amount = mask_amount(amount);
//   update_stat_array(allocated_bytes, amount, stat_types);
// }
// 
// void DebugStats::update_reserved_bytes(int64_t amount, const StatTypes& stat_types) {
//   amount = mask_amount(amount);
//   update_stat_array(reserved_bytes, amount, stat_types);
// }
// 
// void DebugStats::update_active_bytes(int64_t amount, const StatTypes& stat_types) {
//   amount = mask_amount(amount);
//   update_stat_array(active_bytes, amount, stat_types);
// }
// 
// std::shared_ptr<int64_t> header_mask = newMask(0x4c4955595558494e);
// std::shared_ptr<int64_t> footer_mask = newMask(0x48574a4341544348);

bool is_native_memory_strategy() { return is_native_memory_strategy_; }

bool isCatchMemoryStrategy() {
  auto c_str = getenv("PYTORCH_MLU_MEMORY_STRATEGY");
  if (c_str == nullptr) return false;
  auto env_str = std::string(c_str);
  return env_str != "0" && env_str != "NO" && env_str != "OFF";
}

void set_memory_strategy(bool ms) {
  is_native_memory_strategy_ = ms;
}

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(stat.current >= 0,
                        "Negative tracked stat in MLU allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(StatArray &stat_array, int64_t amount,
                       const StatTypes &stat_types) {
  for_each_selected_stat_type(
     stat_types, [&stat_array, amount](size_t stat_type) {
       update_stat(stat_array[stat_type], amount);
     });
}

// ChunkPool is a sorted list of Chunk, using pointer for comparing
struct Chunk;
typedef bool (*Comparison)(const Chunk*, const Chunk*);

struct ChunkPool {
  ChunkPool(Comparison comparator, bool small) : chunks(comparator), is_small(small) {}
  std::set<Chunk*, Comparison> chunks;
  const bool is_small;
};

struct Chunk {
  int device;      // mlu device id
  cnrtQueue_t queue;  // allocation queue
  queue_set queue_in_use;  // queues on which the chunk was used
  size_t size;         // chunk size in bytes
  ChunkPool* pool;     // owning memory pool
  void* ptr;           // memory address
  bool allocated;      // is_allocated flag
  Chunk* prev;         // prev chunk if split from a larger allocation
  Chunk* next;         // next chunk if split from a larger allocation
  int notifier_count;  // number of outstanding MLU notifiers.
  int gc_count;        // counter for prioritizing older / less useful chunks for
                       // garbage collection
  std::unique_ptr<CachingAllocatorHistory> history;
  CachingAllocatorHistory* history_last;

  Chunk(int device, cnrtQueue_t queue, size_t size, ChunkPool* pool,
        void* ptr)
      : device(device),
        queue(queue),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        queue_in_use(),
        notifier_count(0),
        gc_count(0) {}

  // constructor for search key
  Chunk(int device, cnrtQueue_t queue, size_t size)
      : device(device),
        queue(queue),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        queue_in_use(),
        notifier_count(0),
        gc_count(0) {}

    bool is_split() const {
      return (prev != nullptr) || (next != nullptr);
    }
};

static bool ChunkComparator(const Chunk* a, const Chunk* b) {
  if (a->queue != b->queue) {
    return (uintptr_t)a->queue < (uintptr_t)b->queue;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

// format size(byte) in string
static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cnrtQueue_t queue,
      ChunkPool* pool,
      size_t alloc_size,
      MemoryStats& stats)
      : search_key(device, queue, size),
        pool(pool),
        alloc_size(alloc_size),
        chunk(nullptr),
        err(cnrtSuccess) {}

  int device() const {
    return search_key.device;
  }
  cnrtQueue_t queue() const {
    return search_key.queue;
  }
  size_t size() const {
    return search_key.size;
  }

  Chunk search_key;
  ChunkPool* pool;
  size_t alloc_size;
  Chunk* chunk;
  StatTypes stat_types = {false};
  cnrtRet_t err;
};

int trimHistoryBefore(Chunk* chunk, void* point) {
  int n = 0;
  while (chunk->history && chunk->history->addr < point) {
    chunk->history = std::move(chunk->history->next);
    ++n;
  }
  if (!chunk->history) {
    chunk->history_last = nullptr;
  }
  return n;
}

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_MLU_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }
  static size_t roundup_bypass_threshold() {
    return instance().m_roundup_bypass_threshold;
  }

  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_MLU_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }
  // for more info: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
  void parseArgs(const char* env) {
    // If empty, set the default values
    m_max_split_size = std::numeric_limits<size_t>::max();
    m_roundup_power2_divisions = 0;
    m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();
    m_garbage_collection_threshold = 0;
    if (isCatchMemoryStrategy()) is_native_memory_strategy_ = false;

    if (env == nullptr || !is_native_memory_strategy()) {
      return;
    }

    const std::string config(env);

    std::regex exp("[\\s,]+");
    std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> options(it, end);

    for (auto option : options) {
      std::regex exp2("[:]+");
      std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
      std::sregex_token_iterator end2;
      std::vector<std::string> kv(it2, end2);
      if (kv.size() >= 2) {
        /* Maximum split size in MB.  Limited to large size chunks */
        if (kv[0].compare("max_split_size_mb") == 0) {
          size_t val2 = stoi(kv[1]);
          TORCH_CHECK(
              val2 > large_buffer_size / (1024 * 1024),
              "CachingAllocator option max_split_size_mb too small, must be > ",
              large_buffer_size / (1024 * 1024),
              "");
          val2 = std::max(val2, large_buffer_size / (1024 * 1024));
          val2 = std::min(
              val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
          m_max_split_size = val2 * 1024 * 1024;
        } else if (kv[0].compare("roundup_power2_divisions") == 0) {
          size_t val2 = stoi(kv[1]);
          TORCH_CHECK(
              c10::llvm::isPowerOf2_64(val2),
              "For roundups, the divisons has to be power of 2 ",
              "");
          m_roundup_power2_divisions = val2;
        } else if (kv[0].compare("roundup_bypass_threshold_mb") == 0) {
          size_t val2 = stoi(kv[1]);
          m_roundup_bypass_threshold = val2 * 1024 * 1024;
        } else if (kv[0].compare("garbage_collection_threshold") == 0) {
          /*
           * Perform garbage collection of MLU memory chunks to avoid
           * triggering expensive sync-and-reclaim-all operation. Upon setting
           * the threshold (e.g., 0.8), the allocator will start reclaiming
           * chunks if MLU memory capacity usage exceeds the threshold (i.e.,
           * 80% of total memory).
           * Values 0.0 and 1.0 are not allowed as they are less meaningful.
           */
          double val2 = stod(kv[1]);
          TORCH_CHECK(
              val2 > 0,
              "garbage_collect_threshold too small, set it 0.0~1.0",
              "");
          TORCH_CHECK(
              val2 < 1.0,
              "garbage_collect_threshold too big, set it 0.0~1.0",
              "");
          m_garbage_collection_threshold = val2;
        } else {
          TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
        }
      }
    }
  }

 private:
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_roundup_power2_divisions(0),
        m_garbage_collection_threshold(0) {}
  std::atomic<size_t> m_max_split_size;
  std::atomic<size_t> m_roundup_power2_divisions;
  std::atomic<size_t> m_roundup_bypass_threshold;
  std::atomic<double> m_garbage_collection_threshold;
};


class DeviceCachingAllocator {
 protected:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // Memory statistics
  MemoryStats stats;

  // cached chunks are larger than 1 MB
  ChunkPool large_chunks;

  // cached chunks are 1 MB or smaller
  ChunkPool small_chunks;

  // allocated or in use by a queue.
  ska::flat_hash_set<Chunk*> active_chunks;

  // outstanding mlu notifiers
  ska::flat_hash_map<
    torch_mlu::Queue,
    std::deque<std::pair<std::shared_ptr<Notifier>, Chunk*>>> 
    notifiers;

  // record used memory
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

  std::atomic<CreateContextFn> context_recorder_;

 public:
  DeviceCachingAllocator()
      : large_chunks(ChunkComparator, /*is_small=*/false),
        small_chunks(ChunkComparator, /*is_small*/true) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  void setContextRecorder(CreateContextFn c) {
    context_recorder_.store(c);
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Chunk* malloc(int device, size_t orig_size, cnrtQueue_t queue) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    CreateContextFn context_recorder = context_recorder_.load();
    std::unique_ptr<CachingAllocatorContext> context =
        context_recorder ? context_recorder() : nullptr;

    std::unique_lock<std::recursive_mutex> lock(mutex);

    process_notifiers();

    size_t size = roundUpSize(orig_size);  
    auto& pool = getChunkPool(size);
    size_t alloc_size = getAllocationSize(size);
    AllocParams params(device, size, queue, &pool, alloc_size, stats);

    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    // First, try to get a chunk from the existing pool.
    bool chunk_found = 
        // search pool
        getFreeChunk(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && getFreeChunk(params));

    // Can't reuse existing chunk, try to get a new one.
    if (!chunk_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction && 
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_chunks();
      }
      // Get allocation size again but with getAllocationSizeWithMluStrategy, for that
      // we can implement the MLU 1/4 memory strategy without performance degradation.
      alloc_size = getAllocationSizeWithMluStrategy(size); 
      params.alloc_size = alloc_size;
      // Attempt allocate
      chunk_found = alloc_chunk(params, false)
            // Free enough available cached chunks to satisfy alloc and retry
            // alloc.
            || (free_available_cached_chunks(params) && 
                alloc_chunk(params, false))
            || (free_cached_chunks() && alloc_chunk(params, true));
    }

    if (!chunk_found) {
      // For any error code other than cnrtErrorNoMem,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cnrtErrorNoMem);

      size_t device_free;
      size_t device_total;
      TORCH_CNRT_CHECK(cnrtMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
        size,
        stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current,
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current,
        c10::Device(c10::DeviceType::MLU, static_cast<DeviceIndex>(device)));
      // "total capacity": total global memory on MLU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CNRT API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
        OutOfMemoryError,
        false,
        "MLU out of memory. Tried to allocate ", format_size(alloc_size),
        " (MLU ", device, "; ",
        format_size(device_total), " total capacity; ",
        format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " already allocated; ",
        format_size(device_free), " free; ",
        allowed_info,
        format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " reserved in total by PyTorch)",
        " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
        " fragmentation.  See documentation for Memory Management and PYTORCH_MLU_ALLOC_CONF",
        "");
    }

    TORCH_INTERNAL_ASSERT(
        params.err == cnrtSuccess && params.chunk != nullptr &&
        params.chunk->ptr != nullptr);
    Chunk* chunk = params.chunk;
    Chunk* remain_chunk = nullptr;
    const bool already_split = chunk->is_split();

    //If a chunk needs to be split, we create a new chunk from old, and update stats 
    if (shouldSplit(chunk, size)) {
      remain_chunk = chunk;
      // create a new chunk from old chunk 
      chunk = new Chunk(device, queue, size, &pool, chunk->ptr);
      chunk->prev = remain_chunk->prev;
      if (chunk->prev) {
        chunk->prev->next = chunk;
      }
      chunk->next = remain_chunk;

      remain_chunk->prev = chunk;
      remain_chunk->ptr = static_cast<char*>(remain_chunk->ptr) + size;
      remain_chunk->size -= size;
      // carveMasks(chunk, remain_chunk);
      bool inserted = pool.chunks.insert(remain_chunk).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (context) {
        trimHistoryBefore(remain_chunk, (char*)chunk->ptr + size);
      }

      if (already_split) {
        // An already-split inactive chunk is being shrunk by size bytes.
        update_stat_array(
            stats.inactive_split_bytes, -chunk->size, params.stat_types);
      } else {
        // A new split inactive chunk is being created from a previously unsplit chunk
        // size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(stats.inactive_split_bytes[stat_type], remain_chunk->size);
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }
    } else if (already_split) {
      // An already-split chunk is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split_bytes[stat_type], -chunk->size);
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    chunk->allocated = true;
    if (context) {
      trimHistoryBefore(chunk, (char*)chunk->ptr + size);
      chunk->history = std::make_unique<CachingAllocatorHistory>(CachingAllocatorHistory{
          chunk->ptr,
          orig_size,
          std::move(context),
          std::move(chunk->history)});
      if (!chunk->history_last) {
        chunk->history_last = chunk->history.get();
      }
    }
    bool inserted = active_chunks.insert(chunk).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(stats.allocated_bytes[stat_type], chunk->size);
      update_stat(stats.active[stat_type], 1);
      update_stat(stats.active_bytes[stat_type], chunk->size);
    });
    if (chunk->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    // recordBacktrace(chunk);

    c10::reportMemoryUsageToProfiler(
        chunk->ptr,
        chunk->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::MLU, device));

    return chunk;
  }

  void free(Chunk* chunk) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    chunk->allocated = false;

    // following logic might modifying underlaying Chunk, causing the size
    // changed. We store ahead for reporting
    auto orig_chunk_ptr = chunk->ptr;
    auto orig_chunk_size = chunk->size;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(chunk->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(stats.allocated_bytes[stat_type], -chunk->size);
    });
    if (chunk->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!chunk->queue_in_use.empty()) {
      insert_notifier(chunk);
    } else {
      freeChunk(chunk);
    }

    c10::reportMemoryUsageToProfiler(
        orig_chunk_ptr,
        -orig_chunk_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::MLU, chunk->device));
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (c10::llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = c10::llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  size_t roundUpSize(size_t size) {
    if (size < minimum_round_size) {
      return minimum_round_size;
    } else if (size > CachingAllocatorConfig::roundup_bypass_threshold()) {
      return minimum_round_size *
             ((size + minimum_round_size - 1) / minimum_round_size);
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions();
      if (divisions > 0 && size > (minimum_round_size * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return minimum_round_size *
              ((size + minimum_round_size - 1) / minimum_round_size);
      }
    }
  }

  StatType get_stat_type_for_pool(const ChunkPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  void* getBaseAllocation(Chunk* chunk, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (chunk->prev) {
      chunk = chunk->prev;
    }
    void *basePtr = chunk->ptr;
    if (outSize) {
      size_t size = 0;
      while (chunk) {
        size += chunk->size;
        chunk = chunk->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordQueue(Chunk* chunk, Queue queue) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (queue.queue() == chunk->queue) {
      // ignore uses on the allocation queue, since those don't require any
      // special synchronization
      return;
    }
    chunk->queue_in_use.insert(queue);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    TORCH_CNRT_CHECK(cnrtMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached chunks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // synchronize_and_free_notifier();
    free_cached_chunks();
  }
  /** Retrieves info (total size + largest chunk) of the memory cache **/
  void cacheInfo(size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest == 0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      TORCH_CNRT_CHECK(cnrtMemGetInfo(
            largest,
            &tmp_bytes)); // Use free memory as an optimistic initial guess of *largest
    }
    cache_info_aux(large_chunks, total, largest);
    cache_info_aux(small_chunks, total, largest);
  }

  /** Returns a copy of the memory allocator stats for the device **/
  MemoryStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType : c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<SegmentInfo> result;
    const auto all_chunks = get_all_chunks();

    for (const Chunk* const head_chunk : all_chunks) {
      if (head_chunk->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_chunk->device;
      segment_info.address = reinterpret_cast<int64_t>(head_chunk->ptr);
      segment_info.queue = head_chunk->queue;
      segment_info.is_large = (!head_chunk->pool->is_small);

      const Chunk* chunk = head_chunk;
      while (chunk != nullptr) {
        segment_info.chunks.emplace_back();
        ChunkInfo& chunk_info = segment_info.chunks.back();

        chunk_info.size = chunk->size;
        chunk_info.allocated = chunk->allocated;
        chunk_info.active = chunk->allocated || (chunk->notifier_count > 0) ||
            !chunk->queue_in_use.empty();

        segment_info.total_size += chunk_info.size;
        if (chunk_info.allocated) {
          segment_info.allocated_size += chunk_info.size;
        }
        if (chunk_info.active) {
          segment_info.active_size += chunk_info.size;
        }
        chunk_info.history = chunk->history.get();
        chunk = chunk->next;
      }
    }

    std::sort(result.begin(), result.end(),
              [](const SegmentInfo &a, const SegmentInfo &b) {
                if (a.device != b.device) {
                  return a.device < b.device;
                }
                return a.address < b.address;
              });

    return result;
  }

  private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Chunk*> get_all_chunks() const {
    std::vector<const Chunk*> chunks;
    chunks.insert(chunks.end(), small_chunks.chunks.begin(), small_chunks.chunks.end());
    chunks.insert(chunks.end(), large_chunks.chunks.begin(), large_chunks.chunks.end());
    chunks.insert(chunks.end(), active_chunks.begin(), active_chunks.end());
    return chunks;
  }

  /** moves a block into a pool of cached free blocks */
  void freeChunk(Chunk* chunk) {
    TORCH_INTERNAL_ASSERT(!chunk->allocated && chunk->notifier_count == 0 &&
        chunk->queue_in_use.empty());

    size_t original_chunk_size = chunk->size;

    auto& pool = *chunk->pool;
    int64_t net_change_inactive_split_chunks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Chunk*, 2> merge_candidates = {chunk->prev, chunk->next};
    for (Chunk* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = mergeChunks(chunk, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_chunks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_chunks.erase(chunk);

    bool inserted = pool.chunks.insert(chunk).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (chunk->is_split()) {
      net_change_inactive_split_chunks += 1;
      net_change_inactive_split_size += chunk->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(chunk->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(
          stats.inactive_split[stat_type], net_change_inactive_split_chunks);
      update_stat(
          stats.inactive_split_bytes[stat_type],
          net_change_inactive_split_size);
      update_stat(stats.active[stat_type], -1);
      update_stat(stats.active_bytes[stat_type], -original_chunk_size);
    });
  }

  /** combine previously split chunks. returns the size of the subsumed chunk, or 0 on failure. */
  size_t mergeChunks(Chunk* dst, Chunk* src, ChunkPool& pool) {
    if (!src || src->allocated || src->notifier_count > 0 || 
        !src->queue_in_use.empty()) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        src->history_last->next = std::move(dst->history);
        dst->history = std::move(src->history);
      }
      src->history_last = nullptr;
    } else { // [dst src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        dst->history_last->next = std::move(src->history);
        dst->history_last = src->history_last;
      }
      src->history_last = nullptr;
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.chunks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  // get chunk pool
  ChunkPool& getChunkPool(size_t size) {
    if (size <= small_allocation_size) {
      return small_chunks;
    } else {
      return large_chunks;
    }
  }

  bool shouldSplit(Chunk* chunk, size_t size) {
    size_t remaining = chunk->size - size;
    if (chunk->pool->is_small) {
      return remaining >= minimum_round_size;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > small_allocation_size);
    }
  }

  // get allocation size
  size_t getAllocationSize(size_t size) {
    if (size <= small_allocation_size) {
      return small_buffer_size;
    } else {
      if (size < large_allocation_size) {
        return large_buffer_size;
      } else {
        return maximum_round_size *
               ((size + maximum_round_size - 1) / maximum_round_size);
      }
    }
  }

  // get allocation size with MLU 1/4 memory Strategy
  size_t getAllocationSizeWithMluStrategy(size_t size) {
    size_t malloc_size = size;
    // get a quarter of free memory size(byte)
    if(!is_native_memory_strategy()) {
      size_t free = 0;
      size_t _total = 0;
      // get free memory size(Bytes)
      TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &_total));
      free = free / 4;
      malloc_size = std::max(size, free);
    }

    if (size <= small_allocation_size) {
      return small_buffer_size;
    } else {
      if (malloc_size < large_allocation_size) {
        return large_buffer_size;
      } else {
        return maximum_round_size *
               ((malloc_size + maximum_round_size - 1) / maximum_round_size);
      }
    }
  }

  bool getFreeChunk(AllocParams& p) {
    ChunkPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.chunks) {
        ++b->gc_count;
      }
    }
    auto it = pool.chunks.lower_bound(&p.search_key);
    if (it == pool.chunks.end() || (*it)->queue != p.queue())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + large_buffer_size))
      return false;
    p.chunk = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.chunks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeMluMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeMluMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_chunks() {
    // Free unused cached chunks to reclaim MLU memory.
    // Unlike free_cached_chunks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able chunks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_chunk_count = 0;
    for (auto& b : large_chunks.chunks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_chunk_count;
      }
    }
    // No free-able chunks?
    if (freeable_chunk_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool chunk_freed = true;
    while (gc_reclaimed < target_size && chunk_freed == true &&
           freeable_chunk_count > 0) {
      // Free chunks exceeding this age threshold first.
      double age_threshold = total_age / freeable_chunk_count;
      // Stop iteration if we can no longer free a chunk.
      chunk_freed = false;

      // Free chunks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_chunks.chunks.begin();
      while (it != large_chunks.chunks.end()) {
        Chunk* chunk = *it;
        ++it;
        if (!chunk->is_split() && chunk->gc_count >= age_threshold) {
          chunk_freed = true;
          gc_reclaimed += chunk->size;
          total_age -= chunk->gc_count; // Decrement the age
          freeable_chunk_count--; // One less chunk that can be freed
          releaseChunk(chunk);
        }
      }
    }
  }

  bool alloc_chunk(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CNRT error state.
    TORCH_CNRT_CHECK(cnrtGetLastError());
    size_t alloc_size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + alloc_size > allowed_memory_maximum) {
      p.err = cnrtErrorNoMem;
      return false;
    } else {
      p.err = cnrtMalloc(&ptr, alloc_size); 
      if (p.err != cnrtSuccess) {
        if ((p.err == cnrtErrorNoMem) || (p.err == cnrtErrorCndrvFuncCall)) {
          // when oom happens, the alloc_chunk will return nullptr, handle it outside
          // this function.
          cnrtGetLastError();  // clear MLU error 
        }else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          TORCH_CNRT_CHECK(p.err);
        }
        return false;
      }
    }

    total_allocated_memory += alloc_size;
    p.chunk = new Chunk(p.device(), p.queue(), alloc_size, p.pool, (char*)ptr);
    //carve mask in debug mode
    // carveHeader(p.chunk);
    // carveFooter(p.chunk);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], alloc_size);
    });
    if (alloc_size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new cnrtMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.chunk != nullptr && p.chunk->ptr != nullptr);
    return true;
  }

  /* Free one or more oversize blocks to the system allocator. But only enough to 
   * satisfy the target size
   * */
  bool free_available_cached_chunks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    auto& pool = *p.pool;

    Chunk key(p.search_key.device,
              p.search_key.queue,
              p.search_key.size,
              p.search_key.pool,
              p.search_key.ptr);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.chunks.lower_bound(&key);
    if (it == pool.chunks.end() || (*it)->queue != p.queue()) {
      // no single chunk is large enough; free multiple oversize chunks,
      // starting with largest
      if (it == pool.chunks.begin())
        return false;
      size_t total_freed = 0;
      --it; // backup one item。 Now on the largest chunk for the correct queue.
      while ((total_freed < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->queue == p.queue())) {
        auto cur = it;
        total_freed += (*it)->size;
        if (it != pool.chunks.begin()) {
          --it;
          releaseChunk(*cur);
        } else {
          releaseChunk(*cur);
          break;
        }
      }
      if (total_freed < key.size)
        return false;
    } else {
      releaseChunk(*it);
    }
    return true;
  }

  bool free_cached_chunks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding notifiers are returned to the pool.
    synchronize_and_free_notifier();

    releaseChunks(large_chunks);
    releaseChunks(small_chunks);

    return true;
  }

  void releaseChunk(Chunk* chunk) {
    // cudaFree is implicit synchronizes the device before freeing the memory.
    // cnrtFree doesn't support this function, so sync device before call cnrtFree.
    TORCH_CNRT_CHECK(cnrtSyncDevice());
    // std::lock_guard<std::mutex> lock(mlu_free_mutex);
    TORCH_CNRT_CHECK(cnrtFree((void*)chunk->ptr));
    total_allocated_memory -= chunk->size;
    auto* pool = chunk->pool;
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(chunk->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(stats.reserved_bytes[stat_type], -chunk->size);
    });
    if (chunk->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    pool->chunks.erase(chunk);
    delete chunk;
  }

  void releaseChunks(ChunkPool& pool) {
    // Frees all non-split chunks
    auto it = pool.chunks.begin();
    while (it != pool.chunks.end()) {
      Chunk* chunk = *it;
      ++it;
      if (!chunk->prev && !chunk->next) {
        releaseChunk(chunk);
      }
    }
  }

  void synchronize_and_free_notifier() {
    for (auto& q : notifiers) {
      for (auto& n : q.second) {
        auto notifier_sptr = n.first;
        Chunk* chunk = n.second;
        notifier_sptr->synchronize();
        NotifierPool_Manager.give_back_notifier(notifier_sptr);
        chunk->notifier_count--;
        if (chunk->notifier_count == 0) {
          freeChunk(chunk);
        }
      }
    }

    notifiers.clear();
  }

  void insert_notifier(Chunk* chunk) {
    int prev_device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&prev_device));

    queue_set queues(std::move(chunk->queue_in_use));
    AT_ASSERT(chunk->queue_in_use.empty());
    for (auto& queue : queues) {
      TORCH_CNRT_CHECK(cnrtSetDevice(queue.device_index()));

      c10::DeviceIndex device_id = static_cast<c10::DeviceIndex>(queue.device_index());
      auto notifier_sptr = NotifierPool_Manager.alloc_notifier(device_id);
      notifier_sptr->place(queue);
      chunk->notifier_count++;
      notifiers[queue].emplace_back(notifier_sptr, chunk);
    }

    TORCH_CNRT_CHECK(cnrtSetDevice(prev_device));
  }

  void process_notifiers() {
    // Process outstanding MLU notifiers. Notifiers that are completed are
    // removed from the queue, and the 'notifier_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of notifiers per queue to avoid head-of-line delays if one
    // or more queues has long-running operations.

    // Iterate over different queues.
    for (auto it = notifiers.begin(); it != notifiers.end();) {
      while (!it->second.empty()) {
        auto& n = it->second.front();
        auto notifier_sptr = n.first;
        Chunk* chunk = n.second;
        torch_mlu::mlu::MLUGuard guard(notifier_sptr->device_index());
        const bool ret = notifier_sptr->query();
        if (ret == false) {
          // ignore and clear the error if not ready
          cnrtGetLastError();
          break;
        }
        NotifierPool_Manager.give_back_notifier(notifier_sptr);
        chunk->notifier_count--;
        if (chunk->notifier_count == 0) {
          freeChunk(chunk);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = notifiers.erase(it);
      } else {
        it++;
      }
    }
  }

  // Accumulates sizes of all memory chunks for given device in given pool
  void cache_info_aux(const ChunkPool& pool, size_t* total, size_t* largest) {
    for (const auto& chunk : pool.chunks) {
      size_t chunksize = chunk->size;
      *total += chunksize;
      if (chunksize > *largest) {
        *largest = chunksize;
      }
    }
  }
};

class THMCachingAllocator {
 private:
  std::mutex mutex;

  // allocated chunks by device pointer
  ska::flat_hash_map<void*, Chunk*> allocated_chunks;

  // lock around calls to cnrtFree (to prevent deadlocks with CNCL)
  mutable std::mutex mlu_free_mutex;

  void add_allocated_chunk(Chunk* chunk) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_chunks[chunk->ptr] = chunk;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  std::mutex* getMluFreeMutex() const {
    return &mlu_free_mutex;
  }

  Chunk* find_allocated_chunk(void *ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      return nullptr;
    }
    Chunk* chunk = it->second;
    if (remove) {
      allocated_chunks.erase(it);
    }
    return chunk;
  }

  void init(int device_count) {
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  /** allocates a chunk which is safe to use from the provided queue */
  void malloc(void** devPtr, int device, size_t size, cnrtQueue_t queue) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Chunk* chunk = device_allocator[device]->malloc(device, size, queue);
    add_allocated_chunk(chunk);
    *devPtr = (void*)chunk->ptr;
    // TODO: trace mlu memory allocation
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Chunk* chunk = find_allocated_chunk(ptr, true /*remove*/);
    if (!chunk) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    // TODO: trace mlu memory deallocation
    device_allocator[chunk->device]->free(chunk);
  }

  void setMemoryFraction(double fraction, int device) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&activated_device));
    if (activated_device != device) {
      TORCH_CNRT_CHECK(cnrtSetDevice(device));
    }
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void setContextRecorder(CreateContextFn recorder) {
    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    device_allocator[device]->setContextRecorder(std::move(recorder));
  }

  void emptyCache() {
    for (auto& da : device_allocator) {
      da->emptyCache();
    }
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    Chunk* chunk = find_allocated_chunk(ptr);
    if (!chunk) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[chunk->device]->getBaseAllocation(chunk, outSize);
  }

  void recordQueue(const c10::DataPtr& ptr, torch_mlu::Queue queue) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // chunks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when MLU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &raw_delete)
      return;

    Chunk* chunk = find_allocated_chunk(ptr.get());
    TORCH_INTERNAL_ASSERT(chunk != nullptr, "No allocated block can be found");
    device_allocator[chunk->device]->recordQueue(chunk, queue);
  }

  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;
    for (auto& da : device_allocator) {
      auto snap = da->snapshot();
      result.insert(result.end(), snap.begin(), snap.end());
    }

    return result;
  }
};

// allocator using for memory management
THMCachingAllocator caching_allocator;

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cnrtMalloc.  This setting is useful when debugging MLU memory
// errors, since the caching allocator foils cnrt-memcheck.
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_MLU_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  // TODO: trace mlu memory deallocation
  TORCH_CNRT_CHECK(cnrtFree(ptr));
}

// class DebugAllocator : public CachingAllocator {
//   public:
//   // DEBUG MODE: Memory statistics
//   std::vector<DebugStats> debug_memory_stats;
// 
//   // DEBUG MODE: saved backtrace;
//   std::unordered_map<Chunk*, std::pair<int, char**>> saved_backtrace;
// 
//   DebugStats &get_memory_stats_for_device(int device) override {
//     auto dev_count = device_count();
//     auto cur_device = current_device();
//     device = device == -1 ? cur_device : device;
//     if (device >=0 && device < dev_count) {
//       if ((size_t) device >= debug_memory_stats.size()) {
//         debug_memory_stats.resize(device + 1);
//       }
//       return debug_memory_stats.at(device);
//     } else {
//       LOG(FATAL) << "Debug Allocator: wrong device!";
//     }
//   }
// 
//   void recordBacktrace(Chunk* chunk) override {
//     void *buffer[layer_num];
//     char **backtrace_str = nullptr;
//     int size = backtrace(buffer, layer_num);
//     backtrace_str = backtrace_symbols(buffer, size);
//     if (backtrace_str) {
//       saved_backtrace[chunk] = std::make_pair(size, backtrace_str);
//     } else {
//       return;
//     }
//   }
// 
//   void dumpBacktrace(Chunk* chunk) {
//     auto iter = saved_backtrace.find(chunk);
//     if (iter != saved_backtrace.end()) {
//       int layer_num = iter->second.first;
//       char** backtrace_str  = iter->second.second;
//       for (int i = 0; i < layer_num; i++) {
//         std::string sys_str(backtrace_str[i]);
//         size_t start = sys_str.find('(') + 1;
//         size_t end = sys_str.find('+');
//         std::string sub = sys_str.substr(start, end - start);
// 
//         int status;
//         const char* func_name = abi::__cxa_demangle(sub.c_str(), nullptr, nullptr, &status);
//         LOG(INFO) << "stack[" << i << "] : " << func_name;
//       }
//     }
//   }
// 
//   void carveHeader(Chunk* chunk) override {
//     void* ptr = chunk->ptr;
//     auto& queue = chunk->queue;
//     auto size = chunk->size;
//     TORCH_CNRT_CHECK(cnrtMemcpyAsync(ptr, header_mask.get(),
//           mask_bytes, queue, CNRT_MEM_TRANS_DIR_HOST2DEV));
//     TORCH_CNRT_CHECK(cnrtQueueSync(queue));
//   }
// 
//   void carveFooter(Chunk* chunk) override {
//     void* ptr = chunk->ptr;
//     auto& queue = chunk->queue;
//     auto size = chunk->size;
//     TORCH_CNRT_CHECK(cnrtMemcpyAsync(static_cast<char*>(ptr) + (size - mask_bytes),
//           footer_mask.get(), mask_bytes, queue, CNRT_MEM_TRANS_DIR_HOST2DEV));
//     TORCH_CNRT_CHECK(cnrtQueueSync(queue));
//   }
// 
//   // carve masks on the memory
//   void carveMasks(Chunk* chunk, Chunk* remain_chunk) override {
//       carveHeader(remain_chunk);
//       carveFooter(chunk);
//   }
// 
//   // round up masked size
//   size_t roundUpSize(size_t size) override {
//     size += 2 * mask_bytes;
//     if (size < minimum_round_size) {
//       return minimum_round_size;
//     } else {
//       return minimum_round_size * ((size + minimum_round_size - 1) / minimum_round_size);
//     }
//   }
// 
//   // check the ptr if in the chunks map
//   void checkChunks(void* ptr) {
//     char* chunk_ptr = reinterpret_cast<char*>(ptr) - mask_bytes;
//     auto it = allocated_chunks.find(reinterpret_cast<void*>(chunk_ptr));
//     if (it == allocated_chunks.end()) {
//       throw ManageException();
//     }
//   }
// 
//   template<class T>
//   Chunk* getC(T c) {
//     return c;
//   }
// 
// 
//   // check mask of a chunk
//   bool checkMask(Chunk* chunk) override {
//     int64_t header[MASK_WORDS];
//     int64_t footer[MASK_WORDS];
//     bool no_error = true;
//     void* ptr = chunk->ptr;
//     size_t size = chunk->size;
//     auto& queue = chunk->queue;
//     TORCH_CNRT_CHECK(cnrtMemcpyAsync(header, ptr, mask_bytes, queue, CNRT_MEM_TRANS_DIR_DEV2HOST));
//     TORCH_CNRT_CHECK(cnrtMemcpyAsync(footer, static_cast<char*>(ptr) + (size - mask_bytes),
//           mask_bytes, queue, CNRT_MEM_TRANS_DIR_DEV2HOST));
//     TORCH_CNRT_CHECK(cnrtQueueSync(queue));
//     for (int i = 0; i < MASK_WORDS; ++i) {
//       no_error &= (header[i] == header_mask.get()[i]);
//       no_error &= (footer[i] == footer_mask.get()[i]);
//       if (!no_error) {
//         LOG(INFO) << "The memory is out of bound ! mask index = " << i
//                    << " ;\n origin header mask = " << header_mask.get()[i]
//                    << " , now header mask = " << header[i]
//                    << " ;\n origin footer mask = " << footer_mask.get()[i]
//                    << " , now footer mask = " << footer[i] << std::endl;
//         dumpBacktrace(chunk);
//       }
//     }
//     return no_error;
//   }
// 
//   std::vector<SegmentInfo> snapshot() const {
//     std::lock_guard<std::recursive_mutex> lock(base_mutex);
// 
//     std::vector<SegmentInfo> result;
//     const auto all_chunks = get_all_chunks();
// 
//     for (const Chunk* const head_chunk : all_chunks) {
//       if (head_chunk->prev != nullptr) {
//         continue;
//       }
//       result.emplace_back();
//       SegmentInfo& segment_info = result.back();
//       segment_info.device = head_chunk->device_id;
//       segment_info.address = reinterpret_cast<int64_t>(head_chunk->ptr);
//       segment_info.is_large = (head_chunk->pool == &large_chunks);
// 
//       const Chunk* chunk = head_chunk;
//       while (chunk != nullptr) {
//         segment_info.chunks.emplace_back();
//         ChunkInfo& chunk_info = segment_info.chunks.back();
// 
//         chunk_info.allocated = chunk->allocated;
// 
//         // The allocated chunk size should remove the size of mask bytes
//         chunk_info.size = chunk->allocated ? (chunk->size - 2 * mask_bytes) : chunk->size;
//         chunk_info.active = chunk->allocated || (chunk->notifier_count > 0);
// 
//         segment_info.total_size += chunk->size;
//         if (chunk_info.allocated) {
//           segment_info.allocated_size += chunk_info.size;
//         }
//         if (chunk_info.active) {
//           segment_info.active_size += chunk_info.size;
//         }
// 
//         chunk = chunk->next;
//       }
//       // In debug mode, each segment reserved by malloc was set to minus two mask bytes
//       segment_info.total_size -= 2 * mask_bytes;
//     }
// 
//     std::sort(result.begin(), result.end(), [](const SegmentInfo& a, const SegmentInfo& b) {
//       if (a.device != b.device) {
//         return a.device < b.device;
//       }
//       return a.address < b.address;
//     });
// 
//     return result;
//   }
// 
//   template<class T>
//   bool checkPoolMask(T pool) {
//     int64_t header[MASK_WORDS];
//     int64_t footer[MASK_WORDS];
//     bool no_error = true;
//     for (auto c : pool) {
//       // c is the copy of iterated elments of pool
//       Chunk* chunk = nullptr;
//       chunk = DebugAllocator::getC<decltype(c)>(c);
//       no_error &= checkMask(chunk);
//     }
//     return no_error;
//   }
// 
//   void checkMasks() {
//     bool no_error = true;
//     no_error &= checkPoolMask<decltype(allocated_chunks)>(allocated_chunks);
//     no_error &= checkPoolMask<decltype(large_chunks)>(large_chunks);
//     no_error &= checkPoolMask<decltype(small_chunks)>(small_chunks);
//     if (!no_error) {
//       throw  BoundException();
//     }
//   }
// };
// 
// typedef std::pair<void* const, torch_mlu::Chunk*> Ck;
// template<>
// Chunk* DebugAllocator::getC<Ck>(Ck c) {
//   return c.second;
// }

// allocator using for memory debugging
// DebugAllocator debugging_allocator;
// 
// inline void retriveDebugFlag() {
//   char* env = std::getenv("ENABLE_CATCH_MEMORY_DEBUG");
//   if (env != NULL) {
//     debug_mode = (*env == '1');
//   } else {
//     debug_mode = false;
//   }
// }
// 
// CachingAllocator& get_allocator_by_mode() {
//   retriveDebugFlag();
//   if (debug_mode) {
//     return debugging_allocator; 
//   } else {
//     return caching_allocator;
//   }
// }
// The library provides a recordQueue() function to help insert the correct
// synchronization when allocations are used on multiple queues. This will
// ensure that the chunk is not reused before each recorded queue completes
// work.

// void recordQueue(const c10::DataPtr& data_ptr, Queue queue) {
//   get_allocator_by_mode().recordQueue(data_ptr, queue);
// }

struct MLUCachingAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "MLU out of memory. Tried to allocate more than 1EB memory.");
    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    void* data = nullptr;
    if (forceUncachedAllocator()) {
      TORCH_CNRT_CHECK(cnrtMalloc(&data, size));
      // TODO: trace mlu memory allocation
      return {data, data, &uncached_delete, c10::Device(c10::DeviceType::MLU, device)};
    }

    // retriveDebugFlag();
    if (size != 0) {
      caching_allocator.malloc(&data,
                               device,
                               size,
                               torch_mlu::getCurQueue(device));
    }
    return {data, data, &raw_delete,
            c10::Device(c10::DeviceType::MLU, device)};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &raw_delete;
    }
  }
};

MLUCachingAllocator device_allocator;

c10::Allocator* get(void) {
  return &device_allocator;
}

C10_API c10::Allocator* getMLUCachingAllocator(void) {
  return &device_allocator;
}

void init(int device_count) {
  caching_allocator.init(device_count);
}

// Functions for python api
void setMemoryFraction(double fraction, int device) {
  caching_allocator.setMemoryFraction(fraction, device);
}

void setContextRecorder(CreateContextFn recorder) {
  caching_allocator.setContextRecorder(std::move(recorder));
}

void setAllocatorSettings(const std::string& env) {
  if (isCatchMemoryStrategy()) {
    TORCH_WARN_ONCE("PYTORCH_MLU_MEMORY_STRATEGY is set, setAllocatorSettings may not work correctly. Please unset it.");
  }
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

void emptyCache() {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestChunk) {
  caching_allocator.device_allocator[dev_id]->cacheInfo(cachedAndFree, largestChunk);
}

void* getBaseAllocation(void *ptr, size_t *size) {
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordQueue(const c10::DataPtr& ptr, torch_mlu::Queue queue) {
  caching_allocator.recordQueue(ptr, queue);
}

std::mutex* getFreeMutex() {
  return caching_allocator.getMluFreeMutex();
}

static inline void assertValidDevice(int device) {
  const auto device_num = caching_allocator.device_allocator.size();
  TORCH_CHECK(
      0 <= device && device < static_cast<int64_t>(device_num),
      "Invalid device argument ",
      device,
      ": did you call init?");
}

MemoryStats getMemoryStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->getStats();
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->resetAccumulatedStats();
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->resetPeakStats();
}


std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}

std::pair<size_t, size_t> MemGetInfo(int device) {
  torch_mlu::mlu::MLUGuard guard(device);
  size_t device_free = 0;
  size_t device_total = 0;
  cnrtMemGetInfo(&device_free, &device_total);
  return {device_free, device_total};
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  TORCH_CNRT_CHECK(cnrtGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(
      &r, device, nbytes, torch_mlu::getCurrentQueue(device));
  return r;
}

void* raw_alloc_with_queue(size_t nbytes, cnrtQueue_t queue) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  TORCH_CNRT_CHECK(cnrtGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, queue);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

// C++ interface for printing edge memory stats
std::map<std::string, int64_t> mlu_memory_stats(int device) {
  const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
      "all", "small_pool", "large_pool"
    };
  const std::array<const char*, 4> statNames = {
      "current", "peak", "allocated", "freed"
    };

  const auto statToDict = [](const Stat& stat) {
    std::vector<int64_t> dict(4, 0);
    dict[0] = stat.current;
    dict[1] = stat.peak;
    dict[2] = stat.allocated;
    dict[3] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    std::vector<std::vector<int64_t>> dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict.push_back(statToDict(statArray[i]));
    }
    return dict;
  };

  const MemoryStats stats = torch_mlu::getMemoryStats(device);
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>  result;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"].push_back(statToDict(stats.oversize_allocations));
  result["oversize_segments"].push_back(statToDict(stats.oversize_segments));

  std::map<std::string, int64_t> res;
  for (const auto &r : result) {
    for (int i = 0; i < statTypeNames.size(); ++i) {
      if (r.first != "oversize_allocations" && r.first != "oversize_segments") {
        const auto& d = r.second[i];
        std::string out(r.first + "." + statTypeNames[i] + ".");
        for (int j = 0; j < statNames.size(); ++j) {
          res[out + statNames[j]] = d[j];
        }
      } else {
        const auto& d = r.second[0];
        std::string out(r.first + ".");
        for (int j = 0; j < statNames.size(); ++j) {
          res[out + statNames[j]] = d[j];
        }
        continue;
      }
    }
  }
  res["num_alloc_retries"] = stats.num_alloc_retries;
  res["num_ooms"] = stats.num_ooms;

  return res;
}

// return the current memory allocated on MLU
uint64_t currentMemoryAllocated(int dev) {
  TORCH_CHECK(dev == -1 || dev >= 0,
    "Device index must be -1 or non-negative, got ", dev);
  dev = dev == -1 ? current_device() : dev;
  // retriveDebugFlag();
  return mlu_memory_stats(dev)["allocated_bytes.all.current"];
}

// return the current memory cached on MLU
uint64_t currentMemoryCached(int dev) {
  TORCH_CHECK(dev == -1 || dev >= 0,
    "Device index must be -1 or non-negative, got ", dev);

  dev = dev == -1 ? current_device() : dev;
  // retriveDebugFlag();
  return mlu_memory_stats(dev)["reserved_bytes.all.current"];
}

// return the max memory allocated on MLU
uint64_t maxMemoryAllocated(int dev) {
  TORCH_CHECK(dev == -1 || dev >= 0,
    "Device index must be -1 or non-negative, got ", dev);

  dev = dev == -1 ? current_device() : dev;
  // retriveDebugFlag();
  return mlu_memory_stats(dev)["allocated_bytes.all.peak"];
}

// return the max memory cached on MLU
uint64_t maxMemoryCached(int dev) {
  TORCH_CHECK(dev == -1 || dev >= 0,
    "Device index must be -1 or non-negative, got ", dev);

  dev = dev == -1 ? current_device() : dev;
  // retriveDebugFlag();
  return mlu_memory_stats(dev)["reserved_bytes.all.peak"];
}

// set debug env value (only for gtest)
// void setDebugEnv(char* flag) {
//   char* env = std::getenv("ENABLE_CATCH_MEMORY_DEBUG");
//   int overwrite = 0;
//   if (env == NULL) {
//     overwrite = 1;
//   } else {
//     overwrite = (*env == *flag) ? 0 : 1;
//   }
//   int status = setenv("ENABLE_CATCH_MEMORY_DEBUG", flag , overwrite);
//   if (status != 0) {
//       AT_ERROR("set env value failed : ENABLE_CATCH_MEMORY_DEBUG");
//   }
// }

// memory debugging
// void memoryDebug(c10::DataPtr* data) {
//   if (data->device().type() != c10::DeviceType::MLU) {
//     LOG(INFO) << "storage of non-MLU type can not debugged by allocator!!";
//     return;
//   }
//   LOG(INFO) << "===================== Checking Memory Out of Bound ...  =====================";
//   // debugging_allocator.checkMasks();
//   LOG(INFO) << "===================== No Memory Out of Bound !!! =====================";
//   // debugging_allocator.checkChunks(data->get());
//   LOG(INFO) << "===================== Storage is managed by allocator !!! =====================";
// }
// 
// void memoryDebug(const c10::DataPtr* data) {
//   memoryDebug(const_cast<c10::DataPtr*>(data));
// }
// 
// void memoryDebug() {
//   LOG(INFO) << "===================== Checking Memory Out of Bound ...  =====================";
//   // debugging_allocator.checkMasks();
// }

} // namespace torch_mlu
