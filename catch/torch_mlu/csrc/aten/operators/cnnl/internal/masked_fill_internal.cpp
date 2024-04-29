/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {
// The cnnl_masked_fill_internal function will accept two values (both tensor
// and scalar) and will use one of them depends on mask_op
at::Tensor& cnnl_masked_fill_internal(at::Tensor& output,
                                      const at::Tensor& input,
                                      const at::Tensor& mask,
                                      const at::Tensor& value_tensor) {
  auto input_impl = getMluTensorImpl(input);
  auto mask_impl = getMluTensorImpl(mask);
  auto output_impl = getMluTensorImpl(output);
  // set descriptor config
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_mask;
  CnnlTensorDescriptor desc_output;

  // cnnlMasked_v4 only supports CNNL_LAYOUT_ARRAY layout
  auto layout = CNNL_LAYOUT_ARRAY;
  desc_input.set(input, layout);
  desc_mask.set(mask, layout);
  desc_output.set(output, layout);

  auto input_ptr = input_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // get handle
  auto handle = getCurrentHandle();

  // get workspace size
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  bool is_scalar_value = isCpuScalar(value_tensor);
  // create one more tensor desc when fill host mode,
  // but a lot of duplicate code be deleted.
  // Using new a CnnlTensorDescriptor
  // will fix this. But can we using this in internal.cpp?
  CnnlTensorDescriptor desc_value;
  if (!is_scalar_value) desc_value.set(value_tensor, layout);
  void* device_value_ptr = is_scalar_value ? nullptr :
                          getMluTensorImpl(value_tensor)->mlu_data_ptr();
  void* scalar_value_ptr = nullptr;
  auto mask_mode = is_scalar_value ? CNNL_MASKED_FILL_HOST : CNNL_MASKED_FILL;
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(handle,
                                              mask_mode,
                                              desc_input.desc(),
                                              desc_mask.desc(),
                                              is_scalar_value ? nullptr :desc_value.desc(),
                                              desc_output.desc(),
                                              &workspace_size));
  if (workspace_size != 0) {
    workspace = at::empty(workspace_size, input.options().dtype(at::ScalarType::Byte));
    workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  }
  // cnnlMasked
  // AT_DISPATCH_ALL_TYPES_AND3 support kByte, but cnnlMasked_v4 not support
  // uint8 in masked fill mode, so we need to check input type.
  TORCH_MLU_CHECK(input.scalar_type() != at::kByte,
                  "input type is not support uint8 in cnnl_masked_fill_internal");
  AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kBFloat16, at::kHalf,
    input.scalar_type(), "masked_fill_internal", [&] {
      // see Note: [Convert64BitTo32Bit] in accumulate_type.h
      // for more details
      using catch_scalar_t = torch_mlu::Convert64BitTo32Bit_t<scalar_t>;
      catch_scalar_t scalar_value;
      if (is_scalar_value) {
        scalar_value = value_tensor.item().to<catch_scalar_t>();
        scalar_value_ptr = (void*)(&scalar_value);
      }
      TORCH_CNNL_CHECK(cnnlMasked_v4(handle,
                                     mask_mode,
                                     desc_input.desc(),
                                     input_ptr,
                                     desc_mask.desc(),
                                     mask_ptr,
                                     is_scalar_value ? nullptr : desc_value.desc(),
                                     device_value_ptr,
                                     scalar_value_ptr,
                                     workspace_ptr,
                                     workspace_size,
                                     desc_output.desc(),
                                     output_ptr,
                                     nullptr));
    });
  return output;
}

}  // namespace ops
}  // namespace torch_mlu

