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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_masked_scale_internal(at::Tensor& output,
                                       const at::Tensor& input,
                                       const at::Tensor& mask,
                                       const float scale) {
  auto input_impl = getMluTensorImpl(input);
  auto mask_impl = getMluTensorImpl(mask);
  auto output_impl = getMluTensorImpl(output);
  // set descriptor config
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_mask;
  CnnlTensorDescriptor desc_output;

   // get suggest layout
  auto layout = suggest_cnnl_layout(input);

  desc_input.set(input, layout);
  desc_mask.set(mask, layout);
  desc_output.set(output, layout);

  // get handle
  auto handle = getCurrentHandle();

  // masked mode
  cnnlMaskedOp_t masked_op = CNNL_MASKED_SCALE;

  // get workspace size
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(handle,
                                              masked_op,
                                              desc_input.desc(),
                                              desc_mask.desc(),
                                              nullptr,
                                              desc_output.desc(),
                                              &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, input.options().dtype(at::ScalarType::Byte));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // cnnlMasked
  TORCH_CNNL_CHECK(cnnlMasked_v4(handle,
                                 masked_op,
                                 desc_input.desc(),
                                 input_ptr,
                                 desc_mask.desc(),
                                 mask_ptr,
                                 nullptr,
                                 nullptr,
                                 &scale,
                                 workspace_ptr,
                                 workspace_size,
                                 desc_output.desc(),
                                 output_ptr,
                                 nullptr));
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
