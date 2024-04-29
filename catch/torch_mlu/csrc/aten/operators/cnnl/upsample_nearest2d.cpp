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

TORCH_IMPL_FUNC(upsample_nearest2d_out_mlu)
(const at::Tensor& input,
 at::IntArrayRef output_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const at::Tensor& output) {
  at::TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameMLU(__func__, {input_arg, output_arg});

  if (input.numel() == 0) {
    return;
  }

  // NHWC input
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input_contiguous = cnnl_contiguous(input, memory_format);

  // NHWC output
  auto output_contiguous = cnnl_contiguous(output, memory_format);

  // cnnl interp
  bool align_corners = false;
  bool align_center = false;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_NEAREST;
  cnnl_upsample_internal(
      output_contiguous,
      input_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales_h,
      scales_w);
  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mlu)
(const at::Tensor& grad_output,
 at::IntArrayRef output_size,
 at::IntArrayRef input_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const at::Tensor& grad_input) {
  at::TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output, "grad_output", 2};
  checkAllSameMLU(
      "upsample_nearest2d_backward_out_mlu", {grad_output_arg, grad_input_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  // NHWC grad_input, grad_output
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grad_input_contiguous = cnnl_contiguous(grad_input, memory_format);

  cnnlInterpBackwardMode_t interp_mode = CNNL_INTERP_BACKWARD_NEAREST;
  bool align_center = false;
  bool align_corners = false;

  cnnl_upsample_backward_internal(
      grad_input_contiguous,
      grad_output_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales_h,
      scales_w);
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
