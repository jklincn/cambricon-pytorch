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

#include "framework/core/queue.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include "utils/python_interface.h"

namespace torch_mlu {

// Global stream state and constants
static c10::once_flag init_flag;
static c10::DeviceIndex num_mlus = -1;
static constexpr int kQueuesPerPoolBits = 5;
static constexpr int kQueuesPerPool = 1 << kQueuesPerPoolBits;
static constexpr int kQueueTypeBits = 3;

// Note: lower numbers are higher priorities, zero is default priority
static int kHighPriority = 0;
static int kLowPriority = 1;

// Non-default streams
// Note: the number of MLU devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next queue
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in queue.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the CNRT has
// already been destroyed and thus invoking cudaStreamDestroy could lead to a
// crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
// the destruction.
static c10::once_flag device_flags[MLU_DEVICE_NUM_MAX];
static std::atomic<uint32_t> low_priority_counters[MLU_DEVICE_NUM_MAX];
static std::atomic<uint32_t> high_priority_counters[MLU_DEVICE_NUM_MAX];
static cnrtQueue_t low_priority_queues[MLU_DEVICE_NUM_MAX][kQueuesPerPool];
static cnrtQueue_t high_priority_queues[MLU_DEVICE_NUM_MAX][kQueuesPerPool];
static cnrtQueue_t default_queues[MLU_DEVICE_NUM_MAX];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 25 bits -- -- 2 bits --  -- 5 bits -----
// zeros         QueueIdType  stream id index
//
// Where QueueIdType:
//  00 = default cnrtQueue
//  01 = low priority cnrtQueue
//  10 = high priority cnrtQueue
//  11 = external cnrtQueue
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)
//
// We are obligated to treat the Queue ID 0 as the default Queue, per the
// invariant specified in c10::Stream.  However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
enum class QueueIdType : uint8_t {
  DEFAULT = 0x0,
  LOW = 0x1,
  HIGH = 0x2,
  EXT = 0x3,
};

std::ostream& operator<<(std::ostream& queue, QueueIdType s) {
  switch (s) {
    case QueueIdType::DEFAULT:
      queue << "DEFAULT";
      break;
    case QueueIdType::LOW:
      queue << "LOW";
      break;
    case QueueIdType::HIGH:
      queue << "HIGH";
      break;
    case QueueIdType::EXT:
      queue << "EXT";
      break;
    default:
      queue << static_cast<uint8_t>(s);
      break;
  }
  return queue;
}

static inline QueueIdType queueIdType(c10::StreamId s) {
  int mask_for_type = (1 << kQueueTypeBits) - 1;
  if (s && ((s & mask_for_type) == 0)) {
    return QueueIdType::EXT;
  }
  return static_cast<QueueIdType>(s & mask_for_type);
}

static inline size_t queueIdIndex(c10::StreamId s) {
  return static_cast<size_t>((s >> kQueueTypeBits) & ((1 << kQueuesPerPoolBits) - 1));
}

c10::StreamId makeQueueId(QueueIdType qt, size_t qi) {
  return (static_cast<c10::StreamId>(qi) << kQueueTypeBits) |
      static_cast<c10::StreamId>(qt);
}

// Thread-local current streams
static thread_local std::unique_ptr<c10::StreamId[]> current_queues = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalQueueState(DeviceIndex device_index) {
  num_mlus = device_count();
  // Check if the number of MLUs matches the expected compile-time max number
  // of MLUs.
  TORCH_CHECK(
      num_mlus <= MLU_DEVICE_NUM_MAX,
      "Number of MLU devices on the machine is larger than the compiled "
      "max number of mlus expected (",
      MLU_DEVICE_NUM_MAX,
      "). Increase that and recompile.");

  // Initializes default streams
  // CNRT doesn't support legacy default queue.
  c10::DeviceGuard device_guard{c10::Device(c10::DeviceType::MLU, device_index)};
  TORCH_CNRT_CHECK(cnrtQueueCreateWithPriority(
      &default_queues[device_index], 0, kLowPriority));
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceQueueState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  c10::DeviceGuard device_guard{c10::Device(c10::DeviceType::MLU, device_index)};

  for (const auto i : c10::irange(kQueuesPerPool)) {
    auto& lowpri_queue = low_priority_queues[device_index][i];
    auto& hipri_queue = high_priority_queues[device_index][i];
    TORCH_CNRT_CHECK(cnrtQueueCreateWithPriority(
        &lowpri_queue, 0, kLowPriority));
    TORCH_CNRT_CHECK(cnrtQueueCreateWithPriority(
        &hipri_queue, 0, kHighPriority));
  }

  low_priority_counters[device_index] = 0;
  high_priority_counters[device_index] = 0;
}

// Init front-end to ensure initialization only occurs once
static void initMLUQueuesOnce(DeviceIndex device_index) {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalQueueState, device_index);

  if (current_queues) {
    return;
  }

  // Inits current queues (thread local) to default queues
  current_queues = std::make_unique<c10::StreamId[]>(num_mlus);
  for (const auto i : c10::irange(num_mlus)) {
    current_queues[i] = makeQueueId(QueueIdType::DEFAULT, 0);
  }
}

// Helper to verify the MLU index is valid
static inline void check_mlu(c10::DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_mlus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in queue.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kQueuesPerPool;
}

Queue MLUQueueForId(DeviceIndex device_index, c10::StreamId stream_id) {
  return Queue(
      Queue::UNCHECKED,
      c10::Stream(
          c10::Stream::UNSAFE,
          c10::Device(c10::DeviceType::MLU, device_index),
          stream_id));
}

// See Note [StreamId assignment]
cnrtQueue_t Queue::queue() const {
  c10::DeviceIndex device_index = stream_.device_index();
  c10::StreamId stream_id = stream_.id();
  QueueIdType st = queueIdType(stream_id);
  size_t si = queueIdIndex(stream_id);
  switch (st) {
    case QueueIdType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized queue ",
          stream_,
          " (I think this should be the default queue, but I got a non-zero index ",
          si,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like torch_mlu::getQueueFromPool() to get a new queue.");
      return default_queues[device_index];
    case QueueIdType::LOW:
      return low_priority_queues[device_index][si];
    case QueueIdType::HIGH:
      return high_priority_queues[device_index][si];
    case QueueIdType::EXT:
      return reinterpret_cast<cnrtQueue_t>(stream_id);
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

// Returns a queue from the requested pool
// Note: when called the first time on a device, this will create the
// queue pools for that device.
Queue getQueueFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  if (device_index == -1)
    device_index = current_device();
  initMLUQueuesOnce(device_index);
  check_mlu(device_index);

  // Initializes the stream pools (once)
  c10::call_once(
      device_flags[device_index], initDeviceQueueState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(high_priority_counters[device_index]);
    return MLUQueueForId(device_index, makeQueueId(QueueIdType::HIGH, idx));
  }

  const auto idx = get_idx(low_priority_counters[device_index]);
  return MLUQueueForId(device_index, makeQueueId(QueueIdType::LOW, idx));
}

Queue getQueueFromExternal(
    cnrtQueue_t ext_queue,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return MLUQueueForId(device_index, reinterpret_cast<int64_t>(ext_queue));
}

Queue getDefaultQueue(DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  initMLUQueuesOnce(device_index);
  check_mlu(device_index);
  return MLUQueueForId(device_index, makeQueueId(QueueIdType::DEFAULT, 0));
}

Queue getCurrentQueue(DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  initMLUQueuesOnce(device_index);
  check_mlu(device_index);
  return MLUQueueForId(device_index, current_queues[device_index]);
}

cnrtQueue_t getCurQueue(DeviceIndex device_index) {
  return getCurrentQueue(device_index).queue();
}

void setCurrentQueue(Queue stream) {
  initMLUQueuesOnce(stream.device_index());
  current_queues[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const Queue& s) {
  return stream << s.unwrap();
}

}  // namespace torch_mlu
