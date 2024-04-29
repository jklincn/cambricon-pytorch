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
void cnnl_embedding_dense_backward_internal(const at::Tensor& grad_output,
                                                  const at::Tensor& indices,
                                                  int64_t num_weights,
                                                  int64_t padding_idx,
                                                  bool scale_grad_by_freq,
                                                  at::Tensor& output) {
  // handle scalar tensor.
  auto indices_dim = indices.dim();
  auto layout = suggest_cnnl_layout(grad_output);
  auto grad_new = grad_output;
  if (indices_dim == 0 && indices.numel() == 1){
    grad_new = at::unsqueeze(grad_output, 0);
  }

  auto grad_impl = getMluTensorImpl(grad_new);
  CnnlTensorDescriptor grad_desc;
  grad_desc.set(grad_new, layout);
  auto grad_ptr = grad_impl->mlu_data_ptr();

  auto indices_impl = getMluTensorImpl(indices);
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(indices, layout);
  auto indices_ptr = indices_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, layout);
  auto output_ptr = output_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  // get workspace size
  size_t tmp_size = 0;
  TORCH_CNNL_CHECK(cnnlGetEmbeddingBackwardWorkspaceSize(handle,
                                                         grad_desc.desc(),
                                                         output_desc.desc(),
                                                         scale_grad_by_freq,
                                                         &tmp_size));
  auto workspace = at::empty(tmp_size, output.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();
  TORCH_CNNL_CHECK(cnnlEmbeddingBackward(handle,
                                         padding_idx,
                                         scale_grad_by_freq,
                                         indices_desc.desc(),
                                         indices_ptr,
                                         grad_desc.desc(),
                                         grad_ptr,
                                         workspace_ptr,
                                         tmp_size,
                                         output_desc.desc(),
                                         output_ptr));
}

} // namespace ops
} // namespace torch_mlu
