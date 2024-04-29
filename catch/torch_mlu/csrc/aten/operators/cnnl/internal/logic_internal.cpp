/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

void cnnl_logic_internal(at::Tensor& output,
                         const at::Tensor& input_,
                         const at::Tensor& other_,
                         cnnlLogicOp_t logic_type,
                         const at::ScalarType& compute_dtype) {
  TORCH_CHECK(input_.dim() <= CNNL_MAX_DIM_SIZE && other_.dim() <= CNNL_MAX_DIM_SIZE,
              "all input tensors dimension should less than ", CNNL_MAX_DIM_SIZE,
              ", but now input dimension is ",
              input_.dim(), " other dimension is ", other_.dim());
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor output_desc;
  // Input and other need be some dtype, if one of them is a CPU tensor.
  // And already checked datatype convert overflow
  // in pytorch wrapped_scalar_tensor_and_check_convert function.
  auto compute_dtype_ = compute_dtype == at::ScalarType::Undefined
                        ? output.scalar_type() : compute_dtype;
  auto input = scalar_to_tensor_with_dtype(input_, compute_dtype_);
  auto other = scalar_to_tensor_with_dtype(other_, compute_dtype_);

  // get cnnl descriptor
  input_desc.set(input, CNNL_LAYOUT_ARRAY);
  other_desc.set(other, CNNL_LAYOUT_ARRAY);
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  auto input_impl = getMluTensorImpl(input);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto other_ptr = other_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // compute size of workspace
  size_t workspace_size = 0;
  void * temp_ptr = nullptr;
  at::Tensor temp;
  TORCH_CNNL_CHECK(cnnlGetLogicOpWorkspaceSize(handle,
                                               input_desc.desc(),
                                               other_desc.desc(),
                                               output_desc.desc(),
                                               &workspace_size));

  // malloc workspace
  if (workspace_size != 0) {
    temp = at::empty({static_cast<long int>(workspace_size)},
                                input.options().dtype(at::kByte));
    auto* temp_impl = getMluTensorImpl(temp);
    temp_ptr = temp_impl->mlu_data_ptr();
  }

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLogicOp(handle,
                               logic_type,
                               input_desc.desc(),
                               input_ptr,
                               other_desc.desc(),
                               other_ptr,
                               temp_ptr,
                               workspace_size,
                               output_desc.desc(),
                               output_ptr));
}

}  // namespace ops
}  // namespace torch_mlu

