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

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Some view op call AsStride to set size, stride and storage_offset, like permute,
// slice, expand and so on. AsStride is also a view op, so there are nested calls.
// To solve this nested calls, AsStride will not registe in view chain. To support
// AsStride op and keep tensor is always contiguous through view chain, we check tensor
// metedata with last node output tensor metadata in view chain. metadata just conclude
// size, stride, storage_offset.
at::Tensor cnnl_as_strided(const at::Tensor& self, at::IntArrayRef size,
                           at::IntArrayRef stride,
                           c10::optional<int64_t> storage_offset_) {
  auto* self_impl = getMluTensorImpl(self);
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<c10::TensorImpl>(
  c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  auto* result_impl = getMluTensorImpl(result);
  // cnnl_as_strided is a intermediate op when call slice op, permute op,
  // or other view ops. And cnnl_as_strided using storage to construct a new
  // tensor, this operator will loss tensor metadata, include view chain.
  dynamic_cast<MLUTensorImpl*>(result_impl->external_.get())->view_chain_ =
      dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())->view_chain_;
  return result;
}

}  // namespace ops
}  // namespace torch_mlu
