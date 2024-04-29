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

#include <mutex>
#include <deque>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>

#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "c10/core/Storage.h"

#include "framework/core/caching_notifier.h"
#include "framework/core/memory_allocator.h"

namespace torch_mlu {
namespace memory {

namespace {

struct HostChunkSize {
  size_t size_{0};        // allocation size
  void* ptr_{nullptr};    // host memory pointer
};

// Cached host memory, which is malloc or free by cnrt interface.
struct HostChunk {
  size_t size_{0};        // allocation size
  void* ptr_{nullptr};    // host memory pointer
  bool allocated_{false};
  int notifier_count_{0};
  std::mutex mutex_;
  std::unordered_set<torch_mlu::Queue> queues_;

  explicit HostChunk(size_t size, void* ptr = nullptr, bool allocated = false) :
      size_(size), ptr_(ptr), allocated_(allocated),
      notifier_count_(0), queues_() {}
};

// functor for HostChunk compare.
struct ChunkComparator {
  // is_transparemt is used in c++/7/bits/stl_tree.h:1251,
  // and this is C++14 feature.
  using is_transparent = void;
  bool operator()(const HostChunk* a, const HostChunk* b) const {
    // sort by size, break ties with pointer
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  bool operator()(const HostChunk* a, const HostChunkSize& b) const {
    // sort by size, break ties with pointer
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }

  bool operator()(const HostChunkSize& a, const HostChunk* b) const {
    // sort by size, break ties with pointer
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

class HostMemoryAllocator {
  public:
    // Allocate pinned host chunk.
    std::pair<void*, void*> allocate(size_t size) {
      if (size == 0) return {nullptr, nullptr};

      processNotifiers();

      // search for the smallest block which can hold this allocation.
      {
        std::lock_guard<std::mutex> lock(available_list_mutex_);
        // C++14 support this interface.
        auto it = available_list_.lower_bound(HostChunkSize{size, nullptr});
        if (it != available_list_.end()) {
          HostChunk* chunk = *it;
          chunk->allocated_ = true;
          available_list_.erase(it);
          return {chunk->ptr_, reinterpret_cast<void*>(chunk)};
        }
      }
      // Pinned memory pointers allocated by any device can be directly used by any
      // other device, regardless of the current device at the time of allocation,
      // unified addressing.
      // So we grab any existing primary context, if available.
      // See pytorch/pytorch#21081.
      at::OptionalDeviceGuard device_guard;
      auto shared_context_device_index = torch_mlu::getDevceIndexWithSharedContext();
      if (shared_context_device_index.has_value()) {
        device_guard.reset_device(at::Device(at::kMLU, *shared_context_device_index));
      }
      // cnrtHostMalloc is different with cudaHostAlloc when size is 0.
      // cnrtHostMalloc will return nullptr when size is 0.
      void* ptr = nullptr;
      size_t align_size = c10::llvm::PowerOf2Ceil(size);
      TORCH_CNRT_CHECK(cnrtHostMalloc(&ptr, align_size));
      // create a new host chunk.
      HostChunk* chunk = new HostChunk(align_size, ptr, true);
      {
        std::lock_guard<std::mutex> lock(chunks_mutex_);
        chunks_.insert(chunk);
        pinned_chunks_.insert({ptr, chunk});
      }
      return {ptr, reinterpret_cast<void*>(chunk)};
    }

    // Free the pinned host chunk
    void deallocate(void* ctx) {
      if (!ctx) return;

      HostChunk* chunk = reinterpret_cast<HostChunk*>(ctx);
      AT_ASSERT(chunk->allocated_, "The chunk should be allocated!");
      // update chunk info
      c10::optional<std::vector<std::shared_ptr<Notifier>>> notifiers_optional_;
      {
        std::lock_guard<std::mutex> lock(chunk->mutex_);
        chunk->allocated_ = false;
        if (chunk->queues_.size() == 0) {
          TORCH_INTERNAL_ASSERT(chunk->notifier_count_ == 0);
        } else {
          // Insert notify to check whether this cpu memory is
          // using in different queues.
          notifiers_optional_ = std::vector<std::shared_ptr<Notifier>>();
          notifiers_optional_->reserve(chunk->queues_.size());
          for (auto& iter : chunk->queues_) {
            auto notifier_sptr = NotifierPool_Manager.alloc_notifier(iter.device_index());
            notifier_sptr->place(iter);
            chunk->notifier_count_++;
            notifiers_optional_->emplace_back(std::move(notifier_sptr));
          }
          chunk->queues_.clear();
        }
      }
      if (notifiers_optional_.has_value()) {
        // push notifier to notifier list.
        std::lock_guard<std::mutex> lock(notifiers_list_mutex_);
        for (auto&& iter : *notifiers_optional_) {
          notifiers_list_.emplace_back(std::move(iter), chunk);
        }
      } else {
        std::lock_guard<std::mutex> lock(available_list_mutex_);
        available_list_.insert(chunk);
      }
    }

    // place a notifier tag with the current corresponding pointer
    bool recordNotifier(void* ptr, void* ctx, torch_mlu::Queue queue) {
      HostChunk* chunk = reinterpret_cast<HostChunk*>(ctx);
      {
        std::lock_guard<std::mutex> lock(chunks_mutex_);
        // Find in chunks.
        if (chunks_.find(chunk) != chunks_.end()) {
          std::lock_guard<std::mutex> lock(chunk->mutex_);
          TORCH_INTERNAL_ASSERT(chunk->allocated_);
          chunk->queues_.insert(queue);
          return true;
        }
        // Find in ptr_chunks.
        auto it = pinned_chunks_.find(ptr);
        if (it != pinned_chunks_.end()) {
          chunk = it->second;
          std::lock_guard<std::mutex> lock(chunk->mutex_);
          TORCH_INTERNAL_ASSERT(chunk->allocated_);
          chunk->queues_.insert(queue);
          return true;
        }
      }
      return false;
    }

    // Only free available list.
    // Here is a little with pytorch gpu side, CUDAHostAllocator also clean up
    // Notifier pool, but we don't need to do that.
    void emptyCache() {
      processNotifiers();

      // chunks ptr is stored in three container, so we need to clean
      // up all of them.
      std::lock(available_list_mutex_, chunks_mutex_);
      std::lock_guard<std::mutex> lock1(available_list_mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> lock2(chunks_mutex_, std::adopt_lock);
      std::vector<HostChunk*> need_to_clean_up_(available_list_.begin(),
                                                available_list_.end());

      available_list_.clear();
      // Clean each chunk in need_to_clean_up_.
      for (auto* chunk : need_to_clean_up_) {
        chunks_.erase(chunk);
        pinned_chunks_.erase(chunk->ptr_);
        TORCH_CNRT_CHECK(cnrtFreeHost(chunk->ptr_));
        delete chunk;
      }
    }

  private:
    alignas(64) std::mutex chunks_mutex_;
    alignas(64) std::mutex available_list_mutex_;
    alignas(64) std::mutex notifiers_list_mutex_;

    std::unordered_set<HostChunk*> chunks_;

    std::unordered_map<void*, HostChunk*> pinned_chunks_;

    std::set<HostChunk*, ChunkComparator> available_list_;

    // // outstanding mlu notifiers
    std::deque<std::pair<std::shared_ptr<Notifier>, HostChunk*>> notifiers_list_;

    // Process outstanding notifiers.
    // When the marked notifier is completed the notifier is removed and
    // the corresponding cpu pointer return back to available list.
    void processNotifiers() {
      while (true) {
        c10::optional<std::pair<std::shared_ptr<Notifier>, HostChunk*>> notifier_;
        {
          std::lock_guard<std::mutex> lock(notifiers_list_mutex_);
          if (!notifiers_list_.empty()) {
            notifier_ = std::move(notifiers_list_.front());
            notifiers_list_.pop_front();
          }
        }
        if (!notifier_) return;

        // Check Notify status.
        const bool ret = notifier_->first->query();
        if (ret == false) {
          std::lock_guard<std::mutex> lock(notifiers_list_mutex_);
          notifiers_list_.push_front(std::move(*notifier_));
          return;
        }
        // Give back notifier to notifier pool and pop notifier
        // output of notifier-ptr deque.
        NotifierPool_Manager.give_back_notifier(notifier_->first);

        // Process the notifier.
        TORCH_INTERNAL_ASSERT(notifier_);
        HostChunk* chunk = notifier_->second;
        bool available_ = false;
        {
          std::lock_guard<std::mutex> lock(chunk->mutex_);
          TORCH_INTERNAL_ASSERT(!chunk->allocated_);
          chunk->notifier_count_--;
          if (chunk->notifier_count_ == 0) {
            available_ = true;
          }
        }
        if (available_) {
          std::lock_guard<std::mutex> lock(available_list_mutex_);
          available_list_.insert(chunk);
        }
      }
    }
};

}  // anonymous namespace

static HostMemoryAllocator allocator;

}  // end of namespace memory

bool MLUCachingHostAllocator_recordEvent(void* ptr,
                                         void* ctx,
                                         torch_mlu::Queue queue) {
  return torch_mlu::memory::allocator.recordNotifier(ptr, ctx, queue);
}

void MLUCachingHostAllocator_emptyCache() {
  torch_mlu::memory::allocator.emptyCache();
}

// Not real free memory, push a notify to queue, and try to push ptr
// to avaiable list in host cached memory.
static void MLUCachingHostDeleter(void* ctx) {
  torch_mlu::memory::allocator.deallocate(ctx);
}

// Malloc a pin memory through cnrt interface.
struct MLUCachingHostAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto ptr_and_ctx = torch_mlu::memory::allocator.allocate(size);
    return {ptr_and_ctx.first,
            ptr_and_ctx.second,
            &MLUCachingHostDeleter,
            at::DeviceType::CPU};
  }
};

static MLUCachingHostAllocator mlu_caching_host_allocator;

at::Allocator* getMLUCachingHostAllocator() {
  return &mlu_caching_host_allocator;
}

}  // namespace torch_mlu
