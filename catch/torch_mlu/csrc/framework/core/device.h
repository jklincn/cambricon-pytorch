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

#include <c10/core/Device.h>
#include <string>
#include "aten/utils/exceptions.h"
#include "utils/common.h"
#include "utils/version.h"
#include "cnrt.h" //NOLINT
#include "cn_api.h" //NOLINT

#define MLU_DEVICE_NUM_MAX 16

namespace torch_mlu {

c10::DeviceIndex device_count();

c10::DeviceIndex current_device();

inline void setDevice(c10::DeviceIndex device_index) {
  if (GET_MLU_DEVICE < 0 || device_index < 0) return;
  // Set both current device and PythonInterface's device_id
  TORCH_CNRT_CHECK(cnrtSetDevice(device_index));
  static std::once_flag flag;
  std::call_once(flag, checkRequirements);
}

inline void synchronize() { TORCH_CNRT_CHECK(cnrtSyncDevice()); }

uint32_t getDeviceAttr(cnrtDeviceAttr_t attr);

struct DeviceProp{
    std::string name = "";
    int major = -1;
    int minor = -1;
    long total_memory = -1;
};

DeviceProp* getDeviceProperties(int64_t device);

bool canDeviceAccessPeer(int64_t device, int64_t peer_device);

}  // namespace torch_mlu
