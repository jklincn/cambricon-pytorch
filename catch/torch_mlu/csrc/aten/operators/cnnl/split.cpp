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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// We can not reuse Pytorch impl of split_with_sizes
// and unsafe_split_with_sizes because we have to 
// record slice node in viewchain.
std::vector<at::Tensor> cnnl_split_with_sizes(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  const int64_t dim_size = self.size(dim);
  const int64_t num_splits = split_sizes.size();
  int64_t start_idx = 0;

  std::vector<at::Tensor> splits;
  splits.reserve(num_splits);
  for (const auto i : c10::irange(num_splits)) {
    auto length = split_sizes[i];
    TORCH_CHECK(
        length >= 0,
        "split_with_sizes expects split_sizes have only non-negative ",
        "entries, but got split_sizes=",
        split_sizes);
    splits.push_back(
        cnnl_slice(self, dim, start_idx, start_idx + length, 1));
    start_idx += length;
  }
  TORCH_CHECK(
      start_idx == dim_size,
      "split_with_sizes expects split_sizes to sum exactly to ",
      dim_size,
      " (input tensor's size at dimension ",
      dim,
      "), ",
      "but got split_sizes=",
      split_sizes);
  return splits;
}

std::vector<at::Tensor> cnnl_unsafe_split_with_sizes(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  auto result = cnnl_split_with_sizes(self, split_sizes, dim);
  for (auto& t : result) {
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(
          c10::VariableVersion(/*version=*/0));
    }
  }
  return result;
}

} // namespace ops
} // namespace torch_mlu
