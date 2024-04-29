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

#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include "aten/utils/exceptions.h"
#include "framework/core/queue.h"
#include <bitset>


namespace torch_mlu {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMluMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMluMemoryCallbacksRegistry, name, __VA_ARGS__);

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3,    // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct MemoryStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from cnrtMalloc().
  StatArray segment;
  // COUNT: number of active memory chunks (allocated or used by queue)
  StatArray active;
  // COUNT: number of inactive. split memory chunks (unallocated but can't be released via cnrtFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes requested by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory chunks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory chunks
  StatArray inactive_split_bytes;

  // COUNT: total number of failed calls to MLU malloc necessitating cache flushed.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to MLU after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;

};

struct CachingAllocatorContext {
  virtual ~CachingAllocatorContext() {}
};

typedef std::unique_ptr<CachingAllocatorContext> (*CreateContextFn)(void);

struct CachingAllocatorHistory {
  void* addr;
  size_t real_size; // unrounded, actually requested size
  std::unique_ptr<CachingAllocatorContext> context; // per-watcher context
  std::unique_ptr<CachingAllocatorHistory> next; // when blocks are merged we keep records of
                                 // what used to be in the block
};

//struct DebugStats : public MemoryStats {
//  void update_allocated_bytes(int64_t amount, const StatTypes& stat_types);
//  void update_reserved_bytes(int64_t amount, const StatTypes& stat_types);
//  void update_active_bytes(int64_t amount, const StatTypes& stat_types);
//};

// Struct containing info of an allocation chunk (i.e. a fractional part of a cnrtMalloc)..
struct ChunkInfo {
  int64_t size = 0;
  int32_t gc_count = 0;
  bool allocated = false;
  bool active = false;
  CachingAllocatorHistory* history = nullptr; // borrowed reference because it is owned by the allocator.
};

// Struct containing info of a memory segment (i.e. one contiguous cnrtMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  cnrtQueue_t queue = 0;
  bool is_large = false;
  std::vector<ChunkInfo> chunks;
};

// struct BoundException : public std::exception {
//   const char * what() const throw() {
//     return "MLU memory out of bounds!";
//   }
// };
// 
// struct ManageException : public std::exception {
//   const char * what() const throw() {
//     return "MLU memory out of allocator!";
//   }
// };


void* raw_alloc(size_t nbytes);
void* raw_alloc_with_queue(size_t nbytes, cnrtQueue_t queue);
void raw_delete(void* ptr);

c10::Allocator* getMLUCachingAllocator(void);
c10::Allocator* get();
void init(int device_count);
void setMemoryFraction(double fraction, int device);
void setAllocatorSettings(const std::string& env);
void emptyCache();
void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestChunk);
void* getBaseAllocation(void* ptr, size_t* size);
void recordQueue(const c10::DataPtr& ptr, Queue queue);
MemoryStats getMemoryStats(int device);
void resetAccumulatedStats(int device);
void resetPeakStats(int device);
std::vector<SegmentInfo> snapshot();
std::pair<size_t, size_t> MemGetInfo(int device);

std::mutex* getFreeMutex();
void setContextRecorder(CreateContextFn recorder);
bool is_native_memory_strategy();
void set_memory_strategy(bool ms);

// C++ API
std::map<std::string, int64_t> mlu_memory_stats(int device);
uint64_t currentMemoryAllocated(int device_id);
uint64_t currentMemoryCached(int device_id);
uint64_t maxMemoryAllocated(int device_id);
uint64_t maxMemoryCached(int device_id);

//void setDebugEnv(char* flag);
//void memoryDebug(c10::DataPtr* data);
//void memoryDebug(const c10::DataPtr* data);
//void memoryDebug();

} // namespace torch_mlu
