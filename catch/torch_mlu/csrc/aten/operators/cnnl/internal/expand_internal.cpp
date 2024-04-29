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

#include <ATen/native/Resize.h>
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_expand_out_internal(at::Tensor& output,
                                    const at::Tensor& self,
                                    at::IntArrayRef size) {
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(self.sizes().vec(), self.strides().vec(), size);
  TORCH_MLU_CHECK(size.size() >= (size_t)self.dim(),
    "expand(", self.toString(), "{", self.sizes(), "}, size=", size,
    "): the number of sizes provided (", size.size(), ") ",
    "must be greater or equal to the number of dimensions in the tensor (",
    self.dim(), ")");
  TORCH_MLU_CHECK(output.sizes().vec() == expandedSizes,
    "Output tensor sizes must match expanded sizes.");

  auto memory_format = output.suggest_memory_format();
  if (self.dim() < size.size()) {
    memory_format = c10::MemoryFormat::Contiguous;
  }
  auto input_contiguous = cnnl_contiguous(self, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto output_impl = getMluTensorImpl(output_contiguous);
  auto input_impl = getMluTensorImpl(input_contiguous);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  auto cnnl_input_size = modify_dims_based_on_layout(input_contiguous.sizes().vec(), memory_format);
  auto cnnl_output_size = modify_dims_based_on_layout(output_contiguous.sizes().vec(),
                                                      memory_format);
  input_desc.set(input_contiguous, cnnl_input_size,
                 get_contiguous_strides(cnnl_input_size),
                 CNNL_LAYOUT_ARRAY);
  output_desc.set(output_contiguous, cnnl_output_size,
                 get_contiguous_strides(cnnl_output_size),
                 CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  TORCH_CNNL_CHECK(cnnlExpand(handle, input_desc.desc(), input_ptr,
                                output_desc.desc(), output_ptr));
  if (is_copy_necessary(output, output_contiguous)) {
    output.copy_(output_contiguous);
  }
  return output;
}

at::Tensor cnnl_expand_internal(const at::Tensor& self,
                                at::IntArrayRef size,
                                bool implicit) {
  auto memory_format = self.suggest_memory_format();
  TORCH_MLU_CHECK(self.is_contiguous(memory_format),
    "Input tensor need be contiguous.");
  TORCH_MLU_CHECK(size.size() >= (size_t)self.dim(),
    "expand(", self.toString(), "{", self.sizes(), "}, size=", size,
    "): the number of sizes provided (", size.size(), ") ",
    "must be greater or equal to the number of dimensions in the tensor (",
    self.dim(), ")");

  std::vector<int64_t> input_size = self.sizes().vec();
  std::vector<int64_t> input_stride = self.strides().vec();
  if (self.dim() < size.size()) {
    memory_format = c10::MemoryFormat::Contiguous;
  }
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(input_size, input_stride, size);
  // Using shared storage when input and output tensor same.
  if (input_size == expandedSizes && input_stride == expandedStrides) {
    auto inplace_output = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(),
                     self.dtype());
    at::native::setStrided(inplace_output,
                           c10::IntArrayRef(expandedSizes),
                           c10::IntArrayRef(expandedStrides),
                           self.storage_offset());
    return inplace_output;
  }
  // output
  auto output = at::empty(expandedSizes, self.options(), memory_format);
  if (output.numel() == 0) {
    return output;
  }
  cnnl_expand_out_internal(output, self, size);
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
