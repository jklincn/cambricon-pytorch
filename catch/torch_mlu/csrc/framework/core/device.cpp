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

#include "framework/core/device.h"
#include "framework/core/mlu_guard.h"
#include "utils/python_interface.h"

namespace torch_mlu {
namespace {

constexpr uint32_t rem_for_stack = 128 * 1024;

}  // namespace

c10::DeviceIndex device_count() {
  // PythonInterface::initDevice();
  unsigned count;
  auto err = cnrtGetDeviceCount(&count);
  if (err != cnrtSuccess) {
    cnrtRet_t last_err = cnrtGetLastError();
    (void)last_err;
    return 0;
  }
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex current_device() {
    int dev_ordinal;
    TORCH_CNRT_CHECK(cnrtGetDevice(&dev_ordinal));
    return static_cast<c10::DeviceIndex>(dev_ordinal);
}

uint32_t getDeviceAttr(cnrtDeviceAttr_t attr) {
  int dev_ordinal = 0;
  int device_attr = 1;
  TORCH_CNRT_CHECK(cnrtGetDevice(&dev_ordinal));
  TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_attr, attr, dev_ordinal));
  if (attr == cnrtAttrNramSizePerMcore) {
    device_attr -= rem_for_stack;
  }
  return device_attr;
}

at::DeviceIndex num_mlus = -1;
std::once_flag context_init_flag;
std::deque<std::once_flag> device_flags;
std::vector<DeviceProp> device_properties;

void initMLUContextVectors() {
    num_mlus = device_count();
    device_flags.resize(num_mlus);
    device_properties.resize(num_mlus);
}

void initDeviceProperty(at::DeviceIndex device_index) {
    // device attribute
    TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_properties[device_index].major,
                     cnrtAttrComputeCapabilityMajor, device_index));
    TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_properties[device_index].minor,
                     cnrtAttrComputeCapabilityMinor, device_index));
    int total_memory_in_mb = 0;
    TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&total_memory_in_mb,
                     cnrtAttrTotalMemSize, device_index));
    device_properties[device_index].total_memory = (long)total_memory_in_mb * 1024 * 1024;
;
    // device name
    cnrtDeviceProp_t info;
    TORCH_CNRT_CHECK(cnrtGetDeviceProperties(&info, device_index));
    device_properties[device_index].name = info.name;
}

DeviceProp* getDeviceProperties(int64_t device) {
    std::call_once(context_init_flag, initMLUContextVectors);
    if (device == -1) device = current_device();
    AT_ASSERT(device >= 0 && device < num_mlus);
    std::call_once(device_flags[device], initDeviceProperty, device);
    return &device_properties[device];
}

bool canDeviceAccessPeer(int64_t device, int64_t peer_device) {
  std::call_once(context_init_flag, initMLUContextVectors);
  if (device == -1) device = current_device();
  AT_ASSERT(device >= 0 && device < num_mlus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_mlus);
  unsigned int can_access = 0;
  TORCH_CNRT_CHECK(cnrtGetPeerAccessibility(&can_access, (int)device, (int)peer_device));
  return can_access != 0;
}

}  // namespace torch_mlu
