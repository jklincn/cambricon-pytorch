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
#include "aten/viewchain/specificViewOps.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = at::maybe_wrap_dim(dim0, ndims);
  dim1 = at::maybe_wrap_dim(dim1, ndims);
  // Transpose of a tensor is a view operation.
  if (dim0 == dim1) {
    return self.alias();
  }
  std::vector<int64_t> dims(ndims);
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[dim0], dims[dim1]);
  const auto ptr = std::make_shared<PermuteOp>(dims);
  auto output = at::native::permute(self, dims);
  auto* output_impl = getMluTensorImpl(output);
  dynamic_cast<MLUTensorImpl*>(output_impl->external_.get())
      ->view_chain_.pushNodeToViewChain(self, output, ptr);
  return output;
}

at::Tensor& cnnl_transpose_(at::Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(
      !(self.layout() == at::kSparseCsr || self.layout() == at::kSparseCsc ||
        self.layout() == at::kSparseBsr || self.layout() == at::kSparseBsc),
      "torch.transpose_: in-place transposition is not supported for ",
      self.layout(),
      " layout");

  auto ndims = self.dim();
  dim0 = at::maybe_wrap_dim(dim0, ndims);
  dim1 = at::maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }
  std::vector<int64_t> dims(ndims);
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[dim0], dims[dim1]);
  auto size = self.sizes().vec();
  auto stride = self.strides().vec();
  int64_t offset = self.storage_offset();
  const auto ptr = std::make_shared<PermuteOp>(dims);
  at::native::transpose_(self, dim0, dim1);
  auto* self_impl = getMluTensorImpl(self);
  dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())
      ->view_chain_.pushNodeToViewChain(
          size,
          stride,
          offset,
          self.sizes().vec(),
          self.strides().vec(),
          self.storage_offset(),
          ptr);
  return self;
}

}  // namespace ops
}  // namespace torch_mlu
