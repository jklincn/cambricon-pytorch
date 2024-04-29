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

#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_div_out_internal(at::Tensor& output,
                                  const at::Tensor& input,
                                  const at::Tensor& other,
                                  const std::string& rounding_mode) {
  if (input.numel() == 0 || other.numel() == 0) {
      return output;
  }
  if (rounding_mode == "true") {
      TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()) && at::isFloatingType(other.scalar_type()),
                      "div inputs only support floating type");
  } else {
      TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()) || at::isIntegralType(input.scalar_type()),
                      "div trunc/floor inputs only support floating/integral type");
      TORCH_MLU_CHECK(at::isFloatingType(other.scalar_type()) || at::isIntegralType(other.scalar_type()),
                      "div trunc/floor inputs only support floating/integral type");
  }
  auto memory_format = output.suggest_memory_format();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_other;
  CnnlTensorDescriptor desc_output;
    // get tensor size and stride based on memory format
  auto output_size_stride = get_tensor_size_stride(output, memory_format);
  auto input_size_stride = get_tensor_size_stride(input, memory_format);
  auto other_size_stride = get_tensor_size_stride(other, memory_format);
  // get cnnl descriptor
  desc_input.set(input, std::get<0>(input_size_stride),
                std::get<1>(input_size_stride), CNNL_LAYOUT_ARRAY);
  desc_other.set(other, std::get<0>(other_size_stride),
                std::get<1>(other_size_stride), CNNL_LAYOUT_ARRAY);
  desc_output.set(output, std::get<0>(output_size_stride),
                  std::get<1>(output_size_stride), CNNL_LAYOUT_ARRAY);
  // get current handle
  auto handle = getCurrentHandle();
  auto input_impl = getMluTensorImpl(input);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto other_ptr = other_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  if (input_impl->numel() == 0)
      return output;

  // workspace
  size_t workspace_size = 0;
  if (rounding_mode == "true") {
      TORCH_CNNL_CHECK(
          cnnlGetDivWorkspaceSize(handle, desc_input.desc(), desc_other.desc(),
                                  desc_output.desc(), &workspace_size));
  } else if (rounding_mode == "trunc") {
      TORCH_CNNL_CHECK(
          cnnlGetFloorDivTruncWorkspaceSize(handle, desc_input.desc(), desc_other.desc(),
                                            desc_output.desc(), &workspace_size));
  } else if (rounding_mode == "floor") {
      TORCH_CNNL_CHECK(
          cnnlGetFloorDivWorkspaceSize(handle, desc_input.desc(), desc_other.desc(),
                                       desc_output.desc(), &workspace_size));
  }
  at::Tensor temp;
  void* temp_ptr = nullptr;
  if (workspace_size != 0) {
    temp = at::empty(
        {static_cast<long int>(workspace_size)},
        input.options().dtype(at::kByte));
    auto* temp_impl = getMluTensorImpl(temp);
    temp_ptr = temp_impl->mlu_data_ptr();
  }

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_FAST;

  // set descriptor config
  if (rounding_mode == "true") {
      TORCH_CNNL_CHECK(cnnlDiv_v2(handle, prefer, desc_input.desc(),
                                  input_ptr, desc_other.desc(), other_ptr,
                                  temp_ptr, workspace_size,
                                  desc_output.desc(), output_ptr));
  } else if (rounding_mode == "trunc") {
      TORCH_CNNL_CHECK(cnnlFloorDivTrunc(handle, CNNL_COMPUTATION_HIGH_PRECISION,
                                         desc_input.desc(), input_ptr, desc_other.desc(),
                                         other_ptr, desc_output.desc(), output_ptr,
                                         temp_ptr, workspace_size));
  } else if (rounding_mode == "floor") {
      // cnnl FloorDiv_v2 use CNNL_COMPUTATION_FAST mode will cause
      // performace go down, use CNNL_COMPUTATION_HIGH_PRECISION instead;
      prefer = CNNL_COMPUTATION_HIGH_PRECISION;
      TORCH_CNNL_CHECK(cnnlFloorDivV2(handle, prefer, desc_input.desc(), input_ptr,
                                       desc_other.desc(), other_ptr, desc_output.desc(),
                                       output_ptr, temp_ptr, workspace_size));
  }
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
