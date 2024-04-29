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

#include "c10/core/ScalarTypeToTypeMeta.h"
#include "aten/utils/types.h"
#include "utils/cnlog.h"

namespace torch_mlu {

// Directly using scalarType to get CNNL date type will be better.
cnnlDataType_t getCnnlDataType(const caffe2::TypeMeta& data_type) {
  return getCnnlDataType(c10::typeMetaToScalarType(data_type));
}

// on-chip 64bit is not same with pytorch 64bit scalar type
// 1) From pytorch to on-chip, pytorch 64bit is using on-chip 32bit;
// 2) From on-chip to pytorch, on-chip 64bit is using pytorch 64bit;
// 3) on-chip complex double is not support.
#define CNNL_TYPE_AND_SCALAR_TYPE_WITHOUT_64BIT(_) \
  _(CNNL_DTYPE_FLOAT,         at::kFloat)          \
  _(CNNL_DTYPE_BFLOAT16,      at::kBFloat16)       \
  _(CNNL_DTYPE_HALF,          at::kHalf)           \
  _(CNNL_DTYPE_INT32,         at::kInt)            \
  _(CNNL_DTYPE_INT8,          at::kChar)           \
  _(CNNL_DTYPE_UINT8,         at::kByte)           \
  _(CNNL_DTYPE_BOOL,          at::kBool)           \
  _(CNNL_DTYPE_INT16,         at::kShort)          \
  _(CNNL_DTYPE_COMPLEX_HALF,  at::kComplexHalf)    \
  _(CNNL_DTYPE_COMPLEX_FLOAT, at::kComplexFloat)

cnnlDataType_t getCnnlDataType(const at::ScalarType& data_type) {
  switch (data_type) {
    #define DEFINE_CASE(cnnl_dtype, scalar_type) \
      case scalar_type:                          \
        return cnnl_dtype;
    CNNL_TYPE_AND_SCALAR_TYPE_WITHOUT_64BIT(DEFINE_CASE)
    #undef DEFINE_CASE
    case at::kLong:
      return CNNL_DTYPE_INT32;
    case at::kDouble:
      return CNNL_DTYPE_FLOAT;
    case at::kComplexDouble:
      return CNNL_DTYPE_COMPLEX_FLOAT;
    default:
      std::string msg("getCnnlDataType() not supported for ");
      throw std::runtime_error(msg + c10::toString(data_type));
  }
}

// Workaround to get the size of CNNL dtype by transferring
// CNNL datatype back to tensor ScalarType. And without complex double.
// TODO(zhanchendi): directly get size of CNNL data type after CNNL support
at::ScalarType cnnlType2ScalarType(cnnlDataType_t cnnl_dtype) {
  switch (cnnl_dtype) {
    #define DEFINE_CASE(cnnl_dtype, scalar_type) \
      case cnnl_dtype:                           \
        return scalar_type;
    CNNL_TYPE_AND_SCALAR_TYPE_WITHOUT_64BIT(DEFINE_CASE)
    #undef DEFINE_CASE
    case CNNL_DTYPE_INT64:
      return at::kLong;
    case CNNL_DTYPE_DOUBLE:
      return at::kDouble;
    default: {
      throw std::runtime_error("cnnlType2ScalarType() not supported.");
    }
  }
}

#define CNNL_TYPE_AND_CNRT_TYPE(_)                 \
  _(CNNL_DTYPE_FLOAT,         CNRT_FLOAT32)        \
  _(CNNL_DTYPE_HALF,          CNRT_FLOAT16)        \
  _(CNNL_DTYPE_INT32,         CNRT_INT32)          \
  _(CNNL_DTYPE_INT8,          CNRT_INT8)           \
  _(CNNL_DTYPE_UINT8,         CNRT_UINT8)          \
  _(CNNL_DTYPE_BOOL,          CNRT_BOOL)           \
  _(CNNL_DTYPE_INT16,         CNRT_INT16)          \
  _(CNNL_DTYPE_INT64,         CNRT_INT64)

cnrtDataType_t cnnlType2CnrtType(cnnlDataType_t cnnl_data_type) {
  switch (cnnl_data_type) {
    #define DEFINE_CASE(cnnl_dtype, cnrt_dtype) \
      case cnnl_dtype:                          \
        return cnrt_dtype;
    CNNL_TYPE_AND_CNRT_TYPE(DEFINE_CASE)
    #undef DEFINE_CASE
    default: {
      LOG(ERROR) << "Invalid data type from cnnl to cnrt!";
      return CNRT_INVALID;
    }
  }
}

}  // namespace torch_mlu
