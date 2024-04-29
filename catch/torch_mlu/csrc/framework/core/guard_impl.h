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
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software
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
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "aten/utils/exceptions.h"
#include "framework/core/device.h"
#include "framework/core/queue.h"
#include "framework/core/caching_allocator.h"
#include "utils/common.h"
#include "cnrt.h"  // NOLINT

namespace torch_mlu {
namespace mlu {

struct MLUGuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_type = at::DeviceType::MLU;
  MLUGuardImpl() {}
  explicit MLUGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == at::DeviceType::MLU);
  }
  at::DeviceType type() const override {
    return at::DeviceType::MLU;
  }

  c10::Device exchangeDevice(c10::Device device) const override {
    AT_ASSERT(device.type() == at::DeviceType::MLU);
    c10::Device old_device = getDevice();
    if (old_device.index() != device.index()) {
      setDevice(device);
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(at::DeviceType::MLU, current_device());
  }

  c10::optional<c10::Device> uncheckedGetDevice() const noexcept {
    int device;
    const auto err = TORCH_CNRT_ERROR_HANDLED(cnrtGetDevice(&device));
    TORCH_CNRT_WARN(err);
    if (err != CNRT_RET_SUCCESS) {
      return c10::nullopt;
    }
    return c10::Device(at::DeviceType::MLU, device);
  }

  void setDevice(c10::Device device) const override {
    if ( GET_MLU_DEVICE < 0 ) return;
    TORCH_INTERNAL_ASSERT(device.is_mlu());
    c10::Device current_device = getDevice();
    // TODO(shangang): Maybe aligned with pytorch gpu setDevice later.
    // Here is a little different with pytorch gpu setDevice.
    // GPU checks device index, if device index are same, it will
    // skip to call cudaSetDevice.
    // CATCH side calls cnrtSetDevice directly without check device
    // index.
    // cnrtSetDevice must be called before cnnlCreate is called.
    TORCH_CNRT_CHECK(cnrtSetDevice(device.index()));
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if ( GET_MLU_DEVICE < 0 ) return;
    // TODO(shangang): Maybe aligned with pytorch gpu setDevice later.
    // Here is a little different with pytorch gpu setDevice.
    // GPU checks device index, if device index are same, it will
    // skip to call cudaSetDevice.
    // CATCH side calls cnrtSetDevice directly without check device
    // index.
    // cnrtSetDevice must be called before cnnlCreate is called.
    TORCH_CNRT_WARN(cnrtSetDevice(device.index()));
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return getCurrentQueue(device.index()).unwrap();
  }

  c10::Stream getDefaultStream(c10::Device device) const {
    return getDefaultQueue(device.index()).unwrap();
  }

  c10::Stream getStreamFromGlobalPool(c10::Device device, bool isHighPriority = false)
      const override {
    return getQueueFromPool(isHighPriority, device.index()).unwrap();
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    Queue mlu_stream(s);
    auto old_stream = getCurrentQueue(s.device().index());
    setCurrentQueue(mlu_stream);
    return old_stream.unwrap();
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(cnrtNotifier_t* mlu_event, const c10::EventFlag flag) const {
    // Maps PyTorch's Event::Flag to MLU flag
    auto mlu_flag = CNRT_NOTIFIER_DEFAULT;
    switch (flag) {
      case c10::EventFlag::PYTORCH_DEFAULT:
      case c10::EventFlag::MLU_EVENT_DISABLE_TIMING:
        mlu_flag = CNRT_NOTIFIER_DISABLE_TIMING;
        break;
      case c10::EventFlag::BACKEND_DEFAULT:
      case c10::EventFlag::MLU_EVENT_DEFAULT:
        mlu_flag = CNRT_NOTIFIER_DEFAULT;
        break;
      default:
        TORCH_CHECK(false, "MLU event received unknown flag");
    }

    TORCH_CNRT_CHECK(cnrtNotifierCreateWithFlags(mlu_event, mlu_flag));
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto mlu_event = static_cast<cnrtNotifier_t>(event);
    int orig_device;
    TORCH_CNRT_WARN(cnrtGetDevice(&orig_device));
    TORCH_CNRT_WARN(cnrtSetDevice(device_index));
    TORCH_CNRT_WARN(cnrtNotifierDestroy(mlu_event));
    TORCH_CNRT_WARN(cnrtSetDevice(orig_device));
  }

  void record(
    void** event,
    const c10::Stream& stream,
    const c10::DeviceIndex device_index,
    const c10::EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
    "Event device index ",
    device_index,
    " does not match recording stream's device index ",
    stream.device_index(),
    ".");

    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(*event);
    Queue mlu_queue(stream);
    cnrtQueue_t mlu_stream  = mlu_queue.queue();

    // Moves to queue's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Create the Notifier
    if (!mlu_event) TORCH_CNRT_CHECK(cnrtNotifierCreate(&mlu_event));
    TORCH_CNRT_CHECK(cnrtPlaceNotifier(mlu_event, mlu_stream));
    *event = mlu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(
    void* event,
    const c10::Stream& stream) const override {
    if (!event) return;
    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(event);
    Queue mlu_stream(stream);
    const auto orig_device = getDevice();
    setDevice(stream.device());
    TORCH_CNRT_CHECK(cnrtQueueWaitNotifier(mlu_event, mlu_stream.queue(), 0));
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(event);
    const auto err = TORCH_CNRT_ERROR_HANDLED(cnrtQueryNotifier(mlu_event));
    if (err != cnrtErrorNotReady) {
      TORCH_CNRT_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)cnrtGetLastError();
    }
    return (err == cnrtSuccess);
  }

  // Stream-related functions
  bool queryStream(const c10::Stream& stream) const override {
    Queue mlu_stream{stream};
    return mlu_stream.query();
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    Queue mlu_stream{stream};
    mlu_stream.synchronize();
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const c10::Stream& stream)
      const override {
    Queue mlu_stream{stream};
    torch_mlu::recordQueue(data_ptr, mlu_stream);
  }
};

}  // namespace mlu
}  // namespace torch_mlu
