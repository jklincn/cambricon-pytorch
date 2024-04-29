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

#include <memory>
#include <mutex>

#include "utils/python_interface.h"

#include "cnrt.h"   // NOLINT
#include "cndev.h"  // NOLINT

#include "aten/utils/exceptions.h"

namespace torch_mlu {

#define SINGLETON(CLASS)                            \
  public:                                           \
   CLASS(const CLASS&) = delete;                    \
   CLASS& operator=(const CLASS&) = delete;         \
   static CLASS& instance() {                       \
     static CLASS instance;                         \
     return instance;                               \
   }                                                \
  private:                                          \
   CLASS();                                         \
   ~CLASS()

#define GET_RUNNING_MODE                            \
  PythonInterface::instance().getRunningMode()

#define SET_RUNNING_MODE(MODE)                      \
  PythonInterface::instance().setRunningMode(MODE)

#define GET_CORE_NUMBER                             \
  Global::instance().getCoreNumber()

#define GET_CORE_VERSION                            \
  Global::instance().getCoreVersion()

#define GET_MLU_DEVICE                              \
  Global::instance().getDevice()

#define GET_INPUT_FORMAT                            \
  Global::instance().getInputFormat()

// Humanity will defeat COVID-19 after all!
// Running mode for MLU devicie.
// enum class RunningMode { CNML_EAGER, CNML_FUSE, CNNL };

// A singleton class to hold common Catch stuff
class Global {
 SINGLETON(Global);

 public:
  // Get MLU device index
  inline int getDevice() { return PythonInterface::getDevice(); }
  cndevNameEnum_t getDeviceName() { return device_name_;}
  bool isUsingFloatingDevice() { return is_running_fp32_;}
  void setFP32RunningMode(bool run_fp32) {is_running_fp32_ = run_fp32;}

  // TF32 mode management
  bool allowCNNLTF32() const {
    return allow_tf32_cnnl_;
  }
  void setAllowCNNLTF32(bool b) {
    allow_tf32_cnnl_ = b;
  }
  bool allowMLUCustomTF32() const {
    return allow_tf32_custom_;
  }
  void setAllowMLUCustomTF32(bool b) {
    allow_tf32_custom_ = b;
  }

  // Fusion op management
  bool allowOpFusion() const {
    return enabled_fusion_;
  }
  void setAllowOpFusion(bool b) {
    enabled_fusion_ = b;
  }

 private:
  cndevNameEnum_t device_name_;
  bool is_running_fp32_;
  bool allow_tf32_cnnl_ = true;  // vs `torch.backends.cudnn.allow_tf32` currently only affect conv
  bool allow_tf32_custom_ = false;  // control wether to allow TF32 on the rest MLU ops
  bool enabled_fusion_ = true;  // control wether torch.nn.LSTM use fusion op.
};

// Hashing machinery from Pytorch for Params
// Fowler–Noll–Vo hash function
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(Params); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

}  // namespace torch_mlu
