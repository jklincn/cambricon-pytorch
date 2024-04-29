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

void cnnl_mse_loss_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  cnnlMSELossReduction_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_MSE_LOSS_NONE;
      break;
    case 1:
      reduction_mode = CNNL_MSE_LOSS_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_MSE_LOSS_SUM;
      break;
    default:
      TORCH_CHECK(false, "unsupported reduction mode");
      break;
  }

  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descTarget;
  descInput.set(input);
  descOutput.set(output);
  descTarget.set(target);

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto target_impl = getMluTensorImpl(target);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto target_ptr = target_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();

  // get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetMSELossWorkspaceSize(handle, descInput.desc(), &workspace_size));
  auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "MLU mse_loss", [&] {
    TORCH_CNNL_CHECK(cnnlMSELoss_v2(
        handle,
        reduction_mode,
        descInput.desc(),
        input_ptr,
        descTarget.desc(),
        target_ptr,
        workspace_ptr,
        workspace_size,
        descOutput.desc(),
        output_ptr));
  });
}

void cnnl_mse_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  cnnlMSELossReduction_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_MSE_LOSS_NONE;
      break;
    case 1:
      reduction_mode = CNNL_MSE_LOSS_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_MSE_LOSS_SUM;
      break;
    default:
      TORCH_CHECK(false, "unsupported reduction mode");
      break;
  }
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descTarget;
  CnnlTensorDescriptor descGrad;
  descInput.set(input);
  descOutput.set(grad_input);
  descTarget.set(target);
  if (reduction == 1 || reduction == 2) {
    // grad_output's shape is [1,1,1,1] after TensorIterator
    // and set [1] for CNNL size
    std::vector<int64_t> shape = {1};
    std::vector<int64_t> stride = {1};
    descGrad.set(grad_output, shape, stride);
  } else {
    descGrad.set(grad_output);
  }

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(grad_input);
  auto target_impl = getMluTensorImpl(target);
  auto grad_impl = getMluTensorImpl(grad_output);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto target_ptr = target_impl->mlu_data_ptr();
  auto grad_ptr = grad_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "MLU mse_loss_backward", [&] {
        TORCH_CNNL_CHECK(cnnlMSELossBackward(
            handle,
            reduction_mode,
            descInput.desc(),
            input_ptr,
            descTarget.desc(),
            target_ptr,
            descGrad.desc(),
            grad_ptr,
            descOutput.desc(),
            output_ptr));
      });
}

}  // namespace ops
}  // namespace torch_mlu
