#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "framework/core/queue.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

const int iterations = 100;
const size_t size = 10 * 1024;
const size_t free_size = 100 * 1024 * 1024;  // 100 Mibs
const size_t large_buffer_size = 36 * 1024 * 1024;  // 36 Mibs


TEST(MLUCachingAllocatorTest, allocate) {
  torch_mlu::init(device_count());
  auto allocator = torch_mlu::getMLUCachingAllocator();
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = allocator->allocate(size);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, emptyCache) {
  for (int s = 0; s < iterations; ++s) {
    auto allocator = torch_mlu::getMLUCachingAllocator();
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = allocator->allocate(size * size);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
    emptyCache();
  }
}

void thread_func() {
  auto allocator = torch_mlu::getMLUCachingAllocator();
  int16_t device = current_device();
  for (int i = 0; i < iterations; i++) {
    auto data_ptr = allocator->allocate(size);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiThread) {
  for (int i = 0; i < 100; ++i) {
    std::thread t{thread_func};
    t.join();
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiDevice) {
  auto allocator = torch_mlu::getMLUCachingAllocator();
  for (int d = 0; d < device_count(); ++d) {
    setDevice(d);
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = allocator->allocate(size);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
  }
}

TEST(MLUCachingAllocatorTest, recordQueue) {
  auto allocator = torch_mlu::getMLUCachingAllocator();
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = allocator->allocate(size);
    recordQueue(data_ptr, getQueueFromPool());
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, getAllocationSize) {
  auto allocator = torch_mlu::getMLUCachingAllocator();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  size_t malloc_size = free - free_size;
  auto data_ptr0 = allocator->allocate(malloc_size);
  size_t free_size0 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size0, &total));
  auto data_ptr1 = allocator->allocate(large_buffer_size);
  size_t free_size1 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size1, &total));
  size_t diff = free_size0 - free_size1;
  TORCH_CHECK(diff == large_buffer_size, "diff not equal large_buffer_size!");
}

TEST(MLUCachingAllocatorTest, MemoryStrategyFalseTest) {
  set_memory_strategy(false);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == false);
  // Get MLUCachingAllocator
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case.
  emptyCache();
  int16_t device = current_device();

  size_t free = 0;
  size_t total = 0;
  // Get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  // Allocate memory less than 1/4 free size
  size_t malloc_size = free - (free * 4 / 5);

  auto reserved_before = currentMemoryCached(device);
  auto data_ptr0 = allocator->allocate(malloc_size);
  auto reserved_after = currentMemoryCached(device);

  // If strategy is false, reserved memory will be max(malloc_size, free / 4)
  size_t expected_reserved_size = free / 4;
  ASSERT_GE(reserved_after - reserved_before, expected_reserved_size);
}

TEST(MLUCachingAllocatorTest, MemoryStrategyTrueTest) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  // Get MLUCachingAllocator
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case
  emptyCache();
  int16_t device = current_device();

  size_t free = 0;
  size_t total = 0;
  // Get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  // Allocate memory less than 1/4 free size
  size_t malloc_size = free - (free * 4 / 5);
  auto reserved_before = currentMemoryCached(device);
  auto data_ptr0 = allocator->allocate(malloc_size);
  auto reserved_after = currentMemoryCached(device);

  // If strategy is True, reserved memory will be malloc_size
  ASSERT_GE(reserved_after - reserved_before, malloc_size);
}

TEST(MLUCachingAllocatorTest, free_available_cached_chunks_case1) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case
  emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  auto data_ptr1 = allocator->allocate(free_size * 2); // chunk1 -> 200 MB
  auto data_ptr2 = allocator->allocate(free_size * 3); // chunk2 -> 300 MB
  auto data_ptr3 = allocator->allocate(free - free_size * 5.5); // chunk3

  // data_ptr is a instance of c10:DataPtr (a wrapper of unique_ptr)
  // this will call the `MLUCachingDeleter`, so the chunk will be cached.
  data_ptr1.clear(); // chunk1 cached
  data_ptr2.clear(); // chunk2 cached

  // try to allocate 300 MB, this will free available cached chunks which is chunk2.
  // since chunk1 is less than 300 MB.
  auto data_ptr4 = allocator->allocate(free_size * 3);
  auto cached = free_size * 2; // chunk1 cached in chunk pool, total 200 MB.
  auto stats = mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, cached);
}

TEST(MLUCachingAllocatorTest, free_available_cached_chunks_case2) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case
  emptyCache();
  size_t max_split_size = 200;
  std::string env = "max_split_size_mb:" + std::to_string(max_split_size);
  torch_mlu::setAllocatorSettings(env);
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  auto data_ptr1 = allocator->allocate(free_size); // chunk1 -> 100 MB
  auto data_ptr2 = allocator->allocate(free_size * 2); // chunk2 -> 200 MB
  auto data_ptr3 = allocator->allocate(free_size * 2); // chunk3 -> 200 MB
  auto data_ptr4 = allocator->allocate(free - free_size * 5.5); // chunk4

  // data_ptr is a instance of c10:DataPtr (a wrapper of unique_ptr)
  // this will call the `MLUCachingDeleter`, so the chunk will be cached.
  data_ptr1.clear(); // chunk1 cached
  data_ptr2.clear(); // chunk2 cached
  data_ptr3.clear(); // chunk3 cached

  // try to allocate 250 MB, this will free oversized chunks until to fit the size requested,
  // so chunk2 and chunk3 will be freed since they are greater than max_split_size (200 MB)
  auto data_ptr5 = allocator->allocate(free_size * 2.5);
  auto cached = free_size; // chunk1 cached in chunk pool, total 100 MB.
  auto stats = mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, cached);
  // set to default value, do not affect other test case.
  torch_mlu::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, roundup_power2_divisions) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  size_t roundup_bypass_threshold_mb = 1280;
  size_t roundup_power2_divisions = 4;
  std::string env = "roundup_bypass_threshold_mb:" + std::to_string(roundup_bypass_threshold_mb);
  env += ",roundup_power2_divisions:" + std::to_string(roundup_power2_divisions);
  torch_mlu::setAllocatorSettings(env);
  int16_t device = current_device();
  auto allocator = torch_mlu::getMLUCachingAllocator();
  auto data_ptr0 = allocator->allocate(free_size * 11); // 1100 MB -> 1280 MB [1024, 1280, 1536, 1792, 2048]
  auto stats = mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  size_t gt = 1280 * 1024 * 1024; // roundup to 1280 MB.

  ASSERT_GE(reserved, gt);

  // Empty reserved memory in other case
  emptyCache();
  auto data_ptr1 = allocator->allocate(free_size * 15); // 1500 MB -> 1500 MB (1500 > threshold 1280)
  stats = mlu_memory_stats(device);
  reserved = stats["reserved_bytes.all.allocated"];
  gt = 1500 * 1024 * 1024;

  ASSERT_GE(reserved, gt);
  // set to default value, do not affect other test case.
  torch_mlu::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, garbage_collection) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  double garbage_collection_threshold = 0.8;
  std::string env = "garbage_collection_threshold:" + std::to_string(garbage_collection_threshold);
  torch_mlu::setAllocatorSettings(env);
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case
  emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  double fraction = 0.8;
  setMemoryFraction(fraction, device);
  auto data_ptr1 = allocator->allocate(free_size); // chunk1 -> 100 MB
  auto data_ptr2 = allocator->allocate(free_size); // chunk2 -> 100 MB
  auto data_ptr3 = allocator->allocate(free_size * 2); // chunk3 -> 200 MB
  auto data_ptr4 = allocator->allocate(free_size * 2); // chunk4 -> 200 MB
  auto data_ptr5 = allocator->allocate(free * 0.8 - free_size * 6); // chunk5

  data_ptr1.clear(); // chunk1 cached
  data_ptr3.clear(); // chunk3 cached

  // try to allocate 300 MB chunk, this will free chunk1 and chun3
  // so there are no chunks cached.
  auto data_ptr6 = allocator->allocate(free_size * 3);
  auto stats = mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, 0);
  // set to default value, do not affect other test case.
  torch_mlu::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, set_memory_fraction) {
  set_memory_strategy(true);
  auto strategy = is_native_memory_strategy();
  ASSERT_TRUE(strategy == true);
  auto allocator = torch_mlu::getMLUCachingAllocator();
  // Empty reserved memory in other case
  emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  double fraction = 0.6;
  setMemoryFraction(fraction, device);
  bool exception_flag = true;
  // case 1 allocate more than 0.6 * free size, throw exception
  try {
    auto data_ptr = allocator->allocate(free * 0.8);
    exception_flag = false;
  } catch (c10::Error) {
    ASSERT_GE(1, 0); // True
  }
  if (!exception_flag) {
    ASSERT_GE(0, 1); // False
  }
  // case 2: allocate less than 0.6 * free size, success.
  try {
    auto data_ptr1 = allocator->allocate(free * 0.5);
    exception_flag = false;
  } catch (c10::Error) {
    ASSERT_GE(0, 1); // False
  }
  if (!exception_flag) {
    ASSERT_GE(1, 0); // True
  }
}

}  // namespace torch_mlu
