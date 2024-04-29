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

#include <iostream>
#include <memory>
#include "c10/util/Exception.h"
#include "utils/common.h"
#include "framework/core/queue.h"
#include "framework/hooks/MLUHooks.h"

namespace torch_mlu {

// Normal pageable memory. Can't using this memory for async.
namespace memory {
namespace {
// Allocate one buffer
template <typename T>
inline T* allocatePageableMemory(int64_t size) {
  T* tmp = static_cast<T*>(malloc(size * sizeof(T)));
  TORCH_CHECK(tmp != nullptr, "Fail to allocate memory!!!");
  return tmp;
}

template <>
inline void* allocatePageableMemory(int64_t size) {
  void* tmp = static_cast<void*>(malloc(size));
  TORCH_CHECK(tmp != nullptr, "Fail to allocate memory!!!");
  return tmp;
}

// Deallocate one buffer
template <typename T>
inline void freePageableMemory(T* buffer) {
  if (buffer) {
    free(buffer);
  }
}
}  // anonymous namespace
}  // end of namespace memory.

// This function is designed to allocate buffers, and this function only
// can be used in sync process. And you need to call freePageableBuffer
// to free this pageable memory in scope.
template <typename T>
T* allocPageableBuffer(int64_t size) {
  return memory::allocatePageableMemory<T>(size);
}

template <typename T>
void freePageableBuffer(T* buffer) {
  memory::freePageableMemory<T>(buffer);
}

/**
 * Note [HostMemoryAllocator]
 * ~~~~~~~~~~~~~~~~
 * A host caching allocator is to hold MLU host page-locked memory.
 * Which is designed for re-uses freed pinned (page-locked) memory,
 * and avoid too many time-used api call. Like cnrtHostMalloc, cnrtFreeHost.
 *
 * Also Caching allocator tries to avoid allocating and freeing memory for each use
 * for performance reasons. Resources only be freed by explicitly clearing the cache or
 * at the teardown of process.
 * https://discuss.pytorch.org/t/why-dont-explicit-free-cpu-resource-in-cachinghostallocator/189714
 * 
 * Also can get more details in [CUDAHostAllocator design]
 * https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/cuda/CachingHostAllocator.cpp#L116
 *
 * Note1: To ensure correct behavior, MLUCachingHostAllocator_recordEvent must be
 * called anytime a pointer from this allocator is used.
 * Example:
 *   {
 *     at::DataPtr ptr = getMLUCachingHostAllocator()->allocate(size);
 *     // do something
 *     MLUCachingHostAllocator_recordEvent(ptr.get(), ptr.get_context(), queue);
 *   }
 * 
 * Note2: when you add new public function in this class, you may
 * need add a lock guard protection.
 *
 * Note3: that this allocator does not split larger allocations into smaller
 * blocks, unlike the caching device allocator.
 *
 */

bool MLUCachingHostAllocator_recordEvent(void* ptr, void* ctx, torch_mlu::Queue queue);

void MLUCachingHostAllocator_emptyCache();

// To get MLUCachingHostAllocator
at::Allocator* getMLUCachingHostAllocator();

// Not using now, but aligned with pytorch gpu host allocator.
inline at::DataPtr HostAlloc(size_t size) {
  return getMLUCachingHostAllocator()->allocate(size);
}

}  // namespace torch_mlu
