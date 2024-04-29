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
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
void cnnl_lerp_internal(at::Tensor& output,
                        const at::Tensor& self,
                        const at::Tensor& end,
                        const at::Tensor& weight) {
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->mlu_data_ptr();
  CnnlTensorDescriptor desc_self;
  desc_self.set(self, CNNL_LAYOUT_ARRAY);
  
  auto other_impl = getMluTensorImpl(end);
  auto end_ptr = other_impl->mlu_data_ptr();
  CnnlTensorDescriptor desc_end;
  desc_end.set(end, CNNL_LAYOUT_ARRAY);
  
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_ptr = weight_impl->mlu_data_ptr();
  CnnlTensorDescriptor desc_weight;
  desc_weight.set(weight, CNNL_LAYOUT_ARRAY);
  
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor desc_output;
  desc_output.set(output, CNNL_LAYOUT_ARRAY);

  size_t space_size = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetLerpWorkspaceSize(handle,
                                            desc_self.desc(),
                                            desc_end.desc(),
                                            desc_weight.desc(),
                                            desc_output.desc(),
                                            &space_size));
  auto  workspace = at::empty(space_size, self.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl -> mlu_data_ptr();

  TORCH_CNNL_CHECK(cnnlLerp(handle,
                            desc_self.desc(),
                            self_ptr,
                            desc_end.desc(),
                            end_ptr,
                            desc_weight.desc(),
                            weight_ptr,
                            workspace_ptr,
                            space_size,
                            desc_output.desc(),
                            output_ptr));
}
}  // namespace ops
}  // namespace torch_mlu
