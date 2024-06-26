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

#include <ATen/native/TensorAdvancedIndexing.h>
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/scatter_utils.h"

namespace torch_mlu {
namespace ops {

using at::native::SCATTER_GATHER_OP;
using at::native::scatter_stub;
using at::native::scatter_fill_stub;
using at::native::scatter_add_stub;
using at::native::scatter_reduce_stub;
using at::native::scatter_scalar_reduce_stub;
using at::native::scatter_reduce_two_stub;

void scatter_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  auto memory_format = at::MemoryFormat::Contiguous;
  at::Tensor index_internal = index;
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto src_contiguous = cnnl_contiguous(src, memory_format);
  // PYTORCH-8583: performance optimization
  bool stride_index_flag = canHandleStrideScatterIndex(index);
  if (!stride_index_flag) {
    index_internal = cnnl_contiguous(index, memory_format);
  }
  cnnl_scatter_internal(
      self_contiguous,
      self_contiguous,
      dim,
      index_internal,
      src_contiguous,
      CNNL_SCATTER);

  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

void scatter_fill_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  // use scatter to realize scatter_fill
  auto ndim = self.dim();
  std::vector<int64_t> shape(ndim, 1);
  auto src = at::full(shape, value, self.options().device(at::kMLU));
  scatter_mlu_kernel(self, dim, index, src);
}

void scatter_add_mlu_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  auto memory_format = at::MemoryFormat::Contiguous;
  at::Tensor index_internal = index;
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto src_contiguous = cnnl_contiguous(src, memory_format);
  // PYTORCH-8583: performance optimization
  bool stride_index_flag = canHandleStrideScatterIndex(index);
  if (!stride_index_flag) {
    index_internal = cnnl_contiguous(index, memory_format);
  }
  cnnl_scatter_internal(
      self_contiguous,
      self_contiguous,
      dim,
      index_internal,
      src_contiguous,
      CNNL_SCATTER_ADD);

  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

void scatter_reduce_mlu_kernel(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      scatter_add_mlu_kernel(self, dim, index, src);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      TORCH_CHECK(false, "MLU scatter reduce of multiply is not supported");
    default:
      break;
  }
}

void scatter_scalar_reduce_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    const SCATTER_GATHER_OP& reduce) {
  auto ndim = self.dim();
  std::vector<int64_t> shape(ndim, 1);
  auto src = at::full(shape, value, self.options().device(at::kMLU));
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      scatter_add_mlu_kernel(self, dim, index, src);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      TORCH_CHECK(false, "MLU scatter reduce of multiply is not supported");
    default:
      break;
  }
}

void scatter_reduce_two_mlu_kernel(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      scatter_add_mlu_kernel(self, dim, index, src);
      break;
    // TODO(PYTORCH-9239): support more reduce modes
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      TORCH_CHECK(false, "MLU scatter reduce of multiply is not supported");
    case SCATTER_GATHER_OP::REDUCE_MAXIMUM:
      TORCH_CHECK(false, "MLU scatter reduce of maximum is not supported");
    case SCATTER_GATHER_OP::REDUCE_MINIMUM:
      TORCH_CHECK(false, "MLU scatter reduce of minimum is not supported");
    case SCATTER_GATHER_OP::REDUCE_MEAN:
      TORCH_CHECK(false, "MLU scatter reduce of mean is not supported");
    default:
      break;
  }
}

REGISTER_MLU_DISPATCH(scatter_stub, &scatter_mlu_kernel);
REGISTER_MLU_DISPATCH(scatter_fill_stub, &scatter_fill_mlu_kernel);
REGISTER_MLU_DISPATCH(scatter_add_stub, &scatter_add_mlu_kernel);
REGISTER_MLU_DISPATCH(scatter_reduce_stub, &scatter_reduce_mlu_kernel);
REGISTER_MLU_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_mlu_kernel);
REGISTER_MLU_DISPATCH(scatter_reduce_two_stub, &scatter_reduce_two_mlu_kernel);

}  // namespace ops
}  // namespace torch_mlu
