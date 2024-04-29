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
#include "aten/utils/internal_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_softmax_common(const at::Tensor& input, int64_t dim_, bool half_to_float,
                         const at::Tensor& out, cnnlSoftmaxAlgorithm_t algo) {
  if (half_to_float) {
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
                "conversion is supported for Half type only");
  }
  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto out_contiguous = cnnl_contiguous(out, memory_format);

  if (input_contiguous.dim() == 0) input_contiguous = input_contiguous.view(1);
  int64_t dim = at::maybe_wrap_dim(dim_, input_contiguous.dim());
  dim = modify_dim_based_on_layout(dim, memory_format);
  TORCH_CHECK(dim >=0 && dim < input_contiguous.dim(),
              "dim must be non-negative and less than input dimensions");
  cnnl_softmax_out_internal(input_contiguous, dim, out_contiguous, algo);
  if (is_copy_necessary(out, out_contiguous)) out.copy_(out_contiguous);
}

void cnnl_softmax_backward_common(const at::Tensor& grad_output, const at::Tensor& output,
                                         int64_t dim_, at::ScalarType input_dtype,
                                         const at::Tensor& grad_input,
                                         cnnlSoftmaxAlgorithm_t algo) {
  bool half_to_float = grad_output.scalar_type() != input_dtype;
  if (half_to_float) {
    TORCH_CHECK(
      (grad_output.scalar_type() == at::ScalarType::Float &&
       input_dtype == at::ScalarType::Half),
      "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  auto memory_format = grad_input.suggest_memory_format();
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grad_input_contiguous = cnnl_contiguous(grad_input, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  if (grad_output_contiguous.dim() == 0) grad_output_contiguous = grad_output_contiguous.view(1);
  if (output_contiguous.dim() == 0) output_contiguous = output_contiguous.view(1);
  int64_t dim = at::maybe_wrap_dim(dim_, grad_output_contiguous.dim());
  dim = modify_dim_based_on_layout(dim, memory_format);
  TORCH_CHECK(dim >=0 && dim < grad_output_contiguous.dim(),
              "dim must be non-negative and less than input dimensions");
  cnnl_softmax_backward_out_internal(grad_output_contiguous, output_contiguous, dim,
                                     grad_input_contiguous, algo);
  if (is_copy_necessary(grad_input, grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

TORCH_IMPL_FUNC(_softmax_out_mlu)
(const at::Tensor& self, int64_t dim, bool half_to_float, const at::Tensor& out) {
  cnnl_softmax_common(self, dim, half_to_float, out, CNNL_SOFTMAX_ACCURATE);
}

TORCH_IMPL_FUNC(_softmax_backward_data_out_mlu)
(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType input_dtype, const at::Tensor& grad_input) {
  cnnl_softmax_backward_common(grad_output, output, dim, input_dtype,
                               grad_input, CNNL_SOFTMAX_ACCURATE);
}

TORCH_IMPL_FUNC(_log_softmax_out_mlu)
(const at::Tensor& self, int64_t dim, bool half_to_float, const at::Tensor& out) {
  cnnl_softmax_common(self, dim, half_to_float, out, CNNL_SOFTMAX_LOG);
}

TORCH_IMPL_FUNC(_log_softmax_backward_data_out_mlu)
(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType input_dtype, const at::Tensor& out) {
  cnnl_softmax_backward_common(grad_output, output, dim, input_dtype, out, CNNL_SOFTMAX_LOG);
}

}  // namespace ops
}  // namespace torch_mlu
