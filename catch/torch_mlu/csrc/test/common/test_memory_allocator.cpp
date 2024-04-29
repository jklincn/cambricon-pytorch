#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include "caffe2/core/logging.h"
#include "framework/core/memory_allocator.h"

namespace torch_mlu {

const int iterations = 1000;
const size_t size = 10 * 1024 * 1024;

// Pageable memory only can be used in sync preocss.
TEST(MemoryAllocatorTest, allocateGlobalBuffer) {
  float* buffer = allocPageableBuffer<float>(size);
  TORCH_CHECK_NE(buffer, nullptr);
  memset(buffer, 0, size);
  freePageableBuffer(buffer);
}

}  // namespace torch_mlu
