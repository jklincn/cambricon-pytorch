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

#include <ATen/Tensor.h>
#include <ATen/core/List.h>

const int FATAL = 3;
const int ERROR = 2;
const int WARNING = 1;
const int INFO = 0;

// DEBUG=-1, INFO=0, WARNING=1, ERROR=2, FATAL=3
const std::vector<std::string> LOG_LEVEL =
    {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};

namespace torch_mlu {
int64_t MinCNLogLevelFromEnv();
int64_t MinCNLogLevel();
int64_t LogLevelStrToInt(const char* log_level_ptr);

class CNLogMessage {
 public:
  CNLogMessage(const char* file, int line, const char* func, int severity);
  ~CNLogMessage();

  std::stringstream& stream() {
    return stream_;
  }

  static int64_t MinCNLogLevel();

 private:
  std::stringstream stream_;
  int severity_;
};

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::Tensor& t) {
  os << "{ParameterType: Tensor, ";
  if (t.defined()) {
    os << "Shape: " << t.sizes() << ", Stride: " << t.strides()
       << ", Device: " << t.device() << ", Dtype: " << t.scalar_type();
  } else {
    os << "Undefined ";
  }
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::Scalar& t) {
  os << "{ParameterType: Scalar, ";
  os << "Value: " << t.to<float>();
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(std::ostream& os, const int64_t& t) {
  os << "{ParameterType: Long, ";
  os << "Value: " << t << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(std::ostream& os, const double& t) {
  os << "{ParameterType: Double, ";
  os << "Value: " << t << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(std::ostream& os, const bool& t) {
  os << "{ParameterType: Bool, ";
  t ? os << "True " : os << "False ";
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const std::string& t) {
  os << "{ParameterType: String, ";
  os << t;
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::Generator& t) {
  os << "{ParameterType: Generator, ";
  os << "Current_seed: " << t.current_seed() << ", Device: " << t.device()
     << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::ScalarType& t) {
  os << "{ParameterType: ScalarType, ";
  os << t;
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::Layout& t) {
  os << "{ParameterType: Layout, ";
  os << t;
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const c10::Storage& t) {
  os << "{ParameterType: Storage, ";
  os << "Device: " << t.device();
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const c10::MemoryFormat& t) {
  os << "{ParameterType: MemoryFormat, ";
  os << t;
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::Device& t) {
  os << "{ParameterType: Device, ";
  os << t;
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::TensorList& t) {
  os << "{ParameterType: TensorList, ";
  for (int64_t i = 0; i < t.size(); ++i) {
    GenerateBasicMessage(os, t[i]);
  }
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const c10::List<c10::optional<at::Tensor>>& t) {
  os << "{ParameterType: c10::List<c10::optional<Tensor>>, ";
  for (const c10::optional<at::Tensor>& elem : t) {
    if (elem.has_value()) {
      GenerateBasicMessage(os, elem.value());
    } else {
      os << "c10::nullopt ";
    }
  }
  os << "} ";
  return os;
}

inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::IntArrayRef& t) {
  os << "{ParameterType: IntArrayRef, ";
  os << t;
  os << "} ";
  return os;
}

template <typename T>
inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const torch::List<T>& t) {
  os << "{ParameterType: torch::List, ";
  auto vec = t.vec();
  for (int64_t i = 0; i < vec.size(); ++i) {
    GenerateBasicMessage(os, vec[i]);
  }
  os << "} ";
  return os;
}

template <typename T>
inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const at::ArrayRef<T>& t) {
  os << "{ParameterType: ArrayRef<T>, ";
  for (int64_t i = 0; i < t.size(); ++i) {
    GenerateBasicMessage(os, t[i]);
  }
  os << "} ";
  return os;
}

template <typename T>
inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const c10::optional<T>& t) {
  os << "{ParameterType: Optional<T>, ";
  if (t.has_value()) {
    GenerateBasicMessage(os, t.value());
  } else {
    os << "c10::nullopt ";
  }
  os << "} ";
  return os;
}

template <typename T, std::size_t N>
inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const std::array<T, N>& t) {
  os << "{ParameterType: std::array<T, N>, ";
  for (int64_t i = 0; i < t.size(); ++i) {
    GenerateBasicMessage(os, t[i]);
  }
  os << "} ";
  return os;
}

template <typename T, typename... Args>
inline std::ostream& GenerateBasicMessage(
    std::ostream& os,
    const T& t,
    const Args&... rest) {
  GenerateBasicMessage(os, t);
  GenerateBasicMessage(os, rest...);
  return os;
}

template <typename... Args>
inline std::string GenerateMessage(const Args&... args) {
  std::ostringstream ss;
  GenerateBasicMessage(ss, args...);
  return ss.str();
}

} // namespace torch_mlu

// extract message from args
#define MESSAGE(x) torch_mlu::GenerateMessage(x)

#define CNLOG_IS_ON(lvl) ((lvl) >= torch_mlu::CNLogMessage::MinCNLogLevel())

#define CNLOG(lvl)      \
  if (CNLOG_IS_ON(lvl)) \
  torch_mlu::CNLogMessage(__FILE__, __LINE__, __FUNCTION__, lvl).stream()
