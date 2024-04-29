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

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/core/Stream.h>
#include "framework/core/device.h"
#include "utils/cnlog.h"
#include "cnrt.h"  // NOLINT

using c10::DeviceType;
using c10::DeviceIndex;
using c10::Device;
using QueueIndex = int16_t;

namespace torch_mlu {

class Queue;

/**
 * Get a new Queue from the MLU Queue pool.  You can think of this
 * as "creating" a new Queue, but no such creation actually happens;
 * instead, Queues are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a Queue from the high priority pool by setting
 * isHighPriority to true, or a Queue for a specific device by setting device
 * (defaulting to the current MLU Queue.)
 */
Queue getQueueFromPool(const bool isHighPriority = false, DeviceIndex device_index = -1);

/**
 * Get a Queue from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
Queue getQueueFromExternal(cnrtQueue_t ext_queue, DeviceIndex device_index);

/**
 * Get the default MLU Queue, for the passed MLU device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
Queue getDefaultQueue(DeviceIndex device_index = -1);

/**
 * Get the current MLU Queue, for the passed MLU device, or for the
 * current device if no device index is passed.  The current MLU Queue
 * will usually be the default MLU Queue for the device, but it may
 * be different if someone called 'setCurrentQueue' or used 'StreamGuard'
 * or 'MLUQueueGuard'.
 */
Queue getCurrentQueue(DeviceIndex device_index = -1);

/**
 * Get the current cnrtQueue_t from getCurrentQueue.
 */
cnrtQueue_t getCurQueue(DeviceIndex device_index = -1);

/**
 * Set the current Queue on the device of the passed in Queue to be
 * the passed in Queue.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * Queue of the device of the passed Queue.
 *
 * Confused?  Avoid using this function; prefer using 'MLUQueueGuard' instead
 * (which will switch both your current device and current Queue in the way you
 * expect, and reset it back to its original state afterwards).
 */
void setCurrentQueue(Queue queue);

std::ostream& operator<<(std::ostream& stream, const Queue& queue);

inline int mlu2Torch_Priority( int priority) {
  // Currently Catch support 1 and 0, and transfer -1 and 0 to follow native Pytorch.
  // external stream may come from third lib, it's priority can't be confirmed.
  // external Stream return real mlu priority, which it's priority out of [0, 1].
  if (priority == 0 || priority == 1)
    return  priority - 1;
  return priority;
}

/**
 * Represents an abstract object of the MLU Queue, which comes with some
 * encapsulation and additional functionalityfor cnrtQueue_t.
 * cnrtQueue_t lifecycle is created and destory by the internal abstract
 * class MLUQueueInternals.
 */
class Queue {
 public:
  enum Unchecked { UNCHECKED };
  // Queue() {}
  explicit Queue(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::MLU);
  }

  explicit Queue(Unchecked, c10::Stream stream) : stream_(stream) {}

  bool operator==(const Queue& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const Queue& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to cnrtQueue_t.
  operator cnrtQueue_t() const {
    return queue();
  }

  /// Implicit conversion to Stream (a.k.a., forget that the stream is a
  /// cnrtQueue).
  operator c10::Stream() const {
    return unwrap();
  }

  /// Get the MLU device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a MLU device.
  c10::Device device() const {
    return c10::Device(c10::DeviceType::MLU, device_index());
  }

  /// Return the stream ID corresponding to this particular stream.
  c10::StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    c10::DeviceGuard guard{stream_.device()};
    cnrtRet_t err = cnrtQueueQuery(queue());
    if (err == CNRT_RET_SUCCESS) {
      return true;
    }  else if (err != cnrtErrorNotReady) {
      TORCH_CNRT_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)cnrtGetLastError();
    }
    return false;
  }

  void synchronize() const {
    c10::DeviceGuard guard{stream_.device()};
    TORCH_CNRT_CHECK(cnrtQueueSync(queue()));
  }

  int priority() const {
    // TORCH_INTERNAL_ASSERT(0, "now, priority() function is not supported.");
    c10::DeviceGuard guard{stream_.device()};
    int priority = 0;
    TORCH_CNRT_CHECK(cnrtQueueGetPriority(queue(), &priority));
    priority = mlu2Torch_Priority(priority);
    return priority;
  }

  // Explicit conversion to cnrtQueue_t.
  cnrtQueue_t queue() const;

  // Explicit conversion to Stream.
  c10::Stream unwrap() const {
    return stream_;
  }

  uint64_t pack() const noexcept {
    return stream_.pack();
  }

  static Queue unpack(uint64_t bits) {
    return Queue(c10::Stream::unpack(bits));
  }

  static std::tuple<int, int> priority_range() {
    // Note: this returns the range of priority **supported by PyTorch**, not
    // the range of priority **supported by MLU** is [7, 0]. The former is a subset of
    // the latter. Currently Catch supports 1 and 0, and tansfer to -1 and 0, which are "low" and
    // "high" priority.
    int least_priority, greatest_priority;
    TORCH_CNRT_CHECK(
        cnrtDeviceGetQueuePriorityRange(&least_priority, &greatest_priority));
    TORCH_INTERNAL_ASSERT(
        least_priority == 7, "Unexpected MLU queue priority range");
    TORCH_INTERNAL_ASSERT(
        greatest_priority == 0, "Unexpected MLU queue priority range");
    // Get a subset of priority range is [-1, 0] to follow native Pytorch.
    return std::make_tuple(-1, 0);
  }

 private:
  c10::Stream stream_;
};
}  // namespace torch_mlu

namespace std {
template <>
struct hash<torch_mlu::Queue> {
  size_t operator()(torch_mlu::Queue s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
}  // namespace std
