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
#include <functional>
#include <vector>
#include <tuple>
#include <memory>
#include "ATen/native/TensorIterator.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/cnnl_util.h"

namespace torch_mlu {

inline int getTransformDim(int dim, int length) {
  auto dim_v = dim;
  if (dim < 0) {
    dim += length;
  }
  if (dim == 1) {
    dim_v = length - 1;
  } else if (dim > 1) {
    dim_v -= 1;
  }
  return dim_v;
}

void transLayoutParameterDim(const cnnlTensorLayout_t& from,
                             const cnnlTensorLayout_t& to,
                             const int64_t& in_dim,
                             int64_t* out_dim);

int64_t modify_dim_based_on_layout(const int64_t dim,
            const c10::MemoryFormat memory_format);

std::vector<int64_t> modify_dims_based_on_layout(const std::vector<int64_t>& dim,
            const c10::MemoryFormat memory_format);

bool is_copy_necessary(const at::Tensor& output, const at::Tensor& output_contiguous);

std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_tensor_size_stride(
        const at::Tensor& self, at::MemoryFormat memory_format);

}  // namespace torch_mlu
