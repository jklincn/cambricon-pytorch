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
#include <string>
#include <vector>
#include <iostream>
#include "cnrt.h"  //NOLINT
#include "aten/utils/exceptions.h"
#define ENUM_TYPE_CASE(x) case x: return(#x);

// Notice:
// PythonInterface Class is just used to transfer the parameters from _MLUC*.so to
// libaten_mlu.so and libjit_mlu.so library, it can tempararily save the parameters
// which are passed from python_variable_methods.cpp.

namespace torch_mlu {

// Running mode for MLU devicie.
enum class RunningMode { CNNL };

static inline const char *getRunningModeToString(enum RunningMode mode) {
   switch (mode) {
      ENUM_TYPE_CASE(RunningMode::CNNL)
   }
   return "Unsupported RunningMode.";
}

class PythonInterface {
 public:
  ~PythonInterface() {}

  // Returns the PythonInterface singleton
  static PythonInterface& instance();

  // Set the core number of MLU Device
  inline static void setCoreNumber(int core_number) {
    instance().core_number_ = core_number;
  }

  // Get the core number of MLU Device
  inline static int getCoreNumber() {
    return instance().core_number_;
  }
  // Set input image channels' order for first conv of MLU Device
  inline static void setInputFormat(int input_format) {
    instance().input_format_ = input_format;
  }

  // Get the input format of MLU Device
  inline static int getInputFormat() {
    return instance().input_format_;
  }

  // Set the core version of MLU Device
  inline static void setCoreVersion(const std::string& core_version) {
    instance().core_version_ = core_version;
  }

  // Get the core version of MLU Device
  inline static std::string getCoreVersion() {
    return instance().core_version_;
  }

  // Set device id for MLU device
  inline static void setDevice(int id) {
    // initDevice();
    instance().device_id_ = id;
    TORCH_CNRT_CHECK(cnrtSetDevice(id));
  }

  // get the count of MLU device
  static int deviceCount();

  // Get device id
  inline static int getDevice() { return instance().device_id_; }

  // Set save-flag of cambricon offline
  inline static void saveCambricon(std::string mname) {
    instance().cambricon_name_ = mname;
  }

  // Get save-flag of cambricon offline
  inline static std::string getCambricon() {
    return instance().cambricon_name_;
  }

  // enableCnmlMode is deprecated, only support CNNL now.
  inline static void enableCnmlMode(bool cnml_flag) {
    instance().mode_ = RunningMode::CNNL;
  }

  // Get RunningMode
  inline static RunningMode getRunningMode() {
      return instance().mode_;
  }

  // Get RunningMode
  inline static bool getRunningModeBool() {
      return instance().mode_ == RunningMode::CNNL;
  }

  inline static const char* getRunningModeString() {
      return getRunningModeToString(instance().mode_);
  }

  // Set RunningMode
  inline static void setRunningMode(RunningMode mode) {
      instance().mode_ = mode;
  }

  inline static bool getAsyncMode() {
      return instance().async_mode_;
  }

 private:
  PythonInterface()
      : core_number_(1), core_version_("MLU270"), device_id_(0),
      cambricon_name_(""), input_format_(0), mode_(RunningMode::CNNL),
      async_mode_(true) {
     auto async_mode = getenv("CATCH_ASYNC_DISABLE");
     if (async_mode != nullptr && std::strtol(async_mode, nullptr, 10) == 1) {
        async_mode_ = false;
     }
  }

  int core_number_;
  std::string core_version_;
  int device_id_;
  std::string cambricon_name_;
  int input_format_;
  RunningMode mode_;
  /*
   * async_mode_
   * Task Running mode is asynchronout by default.
   * Set system environment variables when synchronization is required:
   * export CATCH_ASYNC_DISABLE = 1
   */
  bool async_mode_;
};

}  // namespace torch_mlu
