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

#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {

cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input);

at::Tensor cnnl_contiguous(const at::Tensor& input,
                           c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

at::MemoryFormat get_channels_last_memory_format(int64_t dim);

bool is_permute(const at::Tensor& input);

std::vector<int64_t> get_permute_back_order(const at::Tensor& input);

std::vector<int64_t> get_permute_order(
        std::vector<int64_t> permute_back_order,
        c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

bool is_expand(const at::Tensor& input);

at::Tensor get_tensor_without_zero_stride(const at::Tensor& input);

at::Tensor non_overlapping_and_dense_out(at::Tensor& output, const at::Tensor& input);

at::Tensor permute_to_contiguous(const at::Tensor& input,
                                 c10::MemoryFormat memory_format);

bool is_slice(const at::Tensor& input);

std::vector<int64_t> get_slice_params(const at::Tensor& input);

at::Tensor get_contiguous_tensor_before_slice(const at::Tensor& input);

bool is_same_format_tensor(const at::TensorList& tensors);

at::Tensor Squeeze(const at::Tensor& self);

at::Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads,
                        const at::Tensor& self, bool some, bool compute_uv,
                        const at::Tensor& raw_u, const at::Tensor& sigma, const at::Tensor& raw_v);

at::Tensor unsqueeze_multiple(const at::Tensor & t, at::IntArrayRef dim, size_t n_dims);

c10::MemoryFormat switch_tensors_suggest_memory_format(const std::vector<at::Tensor>& tensor_list);

}  // namespace torch_mlu
