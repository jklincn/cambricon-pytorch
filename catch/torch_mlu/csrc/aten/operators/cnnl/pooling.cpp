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

#include <ATen/native/Pool.h>
#include "ATen/core/TensorBody.h"
#include "ATen/core/interned_strings.h"
#include "ATen/ops/empty.h"
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "build/aten/src/ATen/ops/avg_pool1d_native.h"
#include "build/aten/src/ATen/ops/max_pool1d_with_indices_native.h"
#include "c10/core/ScalarType.h"
#include "third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86operand.h"

using at::native::safe_downcast;
using at::native::pooling_output_shape;
using at::native::pool3d_shape_check;
using at::native::max_pool3d_backward_shape_check;

namespace torch_mlu {
namespace ops {

#define MAXPOOL2D_KERNEL_MAX 1535

TORCH_IMPL_FUNC(avg_pool2d_out_mlu)(
  const Tensor& input_,
  int64_t kH_,
  int64_t kW_,
  int64_t dH_,
  int64_t dW_,
  int64_t padH_,
  int64_t padW_,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& output_
) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_MLU_CHECK(!divisor_override.has_value(), "divisor_override is not supported");
  
  at::TensorArg output_arg{ output_, "output_", 1 };
  at::TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameMLU("avg_pool2d_out_mlu", {output_arg, input_arg});
  
  const int kH = safe_downcast<int, int64_t>(kH_);
  const int kW = safe_downcast<int, int64_t>(kW_);

  const int dH = safe_downcast<int, int64_t>(dH_);
  const int dW = safe_downcast<int, int64_t>(dW_);

  const int padH = safe_downcast<int, int64_t>(padH_);
  const int padW = safe_downcast<int, int64_t>(padW_);
  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 3 ? at::unsqueeze(output_, 0) : output_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto result_contiguous = cnnl_contiguous(output, memory_format);
  
  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input_.scalar_type(),
    "avg_pool2d_out_mlu",
    [&] {
      cnnl_pool2d_internal(result_contiguous, input_contiguous, kH, kW, dH, dW, padH, padW, ceil_mode,
                           count_include_pad, 0);

      if (input_.dim() == 3) { // cnnl only support batch mode.
        result_contiguous.squeeze_(0);
      }

      if (is_copy_necessary(output_, result_contiguous)) {
        output_.copy_(result_contiguous);
      }
    }
  );
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_mlu) (
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& gradInput_
) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_MLU_CHECK(!divisor_override.has_value(), "divisor_override is not supported");
  
  at::TensorArg gradInput_arg{ gradInput_, "gradInput_", 1 };
  at::TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  at::TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameMLU("avg_pool2d_backward_out_mlu",
                  {gradInput_arg, gradOutput_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  
  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }
 
  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput = input_.dim() == 3 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput = input_.dim() == 3 ? at::unsqueeze(gradInput_, 0) : gradInput_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = cnnl_contiguous(gradInput, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input.scalar_type(),
    "avg_pool2d_backward_out_mlu",
    [&] {
      cnnl_pool2d_backward_internal(gradInput_contiguous,
                                    gradOutput_contiguous,
                                    input_contiguous,
                                    {},
                                    kH, kW,
                                    dH, dW,
                                    padH, padW,
                                    ceil_mode,
                                    count_include_pad);
      if (input_.dim() == 3) { // cnnl only support batch mode.
        gradInput_contiguous.squeeze_(0);
      }

      if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
        gradInput_.copy_(gradInput_contiguous);
      }
    }
  );
}

TORCH_IMPL_FUNC(avg_pool3d_out_mlu) (
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& output_
) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_CHECK(!divisor_override.has_value(), "divisor_override is not supported");

  at::TensorArg output_arg{ output_, "output_", 1 };
  at::TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameMLU(__func__, {output_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 4 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 4 ? at::unsqueeze(output_, 0) : output_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input.scalar_type(),
    "avg_pool3d_out_mlu",
    [&] {
      cnnl_pool3d_internal(output_contiguous,
                           input_contiguous,
                           kT, kH, kW,
                           dT, dH, dW,
                           padT, padH, padW,
                           ceil_mode,
                           count_include_pad, 0);
      if (input_.dim() == 4) { // cnnl only support batch mode.
        output_contiguous.squeeze_(0);
      }
      if (is_copy_necessary(output_, output_contiguous)) {
        output_.copy_(output_contiguous);
      }
    }
  );
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_mlu) (
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& gradInput_
) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_CHECK(!divisor_override.has_value(), "divisor_override is not supported");
  
  at::TensorArg gradInput_arg{ gradInput_, "gradInput_", 1 };
  at::TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  at::TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameMLU(__func__,
                  {gradInput_arg, gradOutput_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((gradOutput_.ndimension() == 4 || gradOutput_.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");
  
  gradInput_.zero_();
  
  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }
  
  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 4 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput = input_.dim() == 4 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput = input_.dim() == 4 ? at::unsqueeze(gradInput_, 0) : gradInput_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = cnnl_contiguous(gradInput, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input.scalar_type(),
    "avg_pool3d_backward_out_mlu",
    [&] {
      cnnl_pool3d_backward_internal(gradInput_contiguous,
                                    gradOutput_contiguous,
                                    input_contiguous,
                                    {},
                                    kT, kH, kW,
                                    dT, dH, dW,
                                    padT, padH, padW,
                                    ceil_mode, count_include_pad);
      if (input_.dim() == 4) {
        gradInput_contiguous.squeeze_(0);
      }
      if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
        gradInput_.copy_(gradInput_contiguous);
      }
    }
  );
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_mlu) (
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& output_,
  const Tensor& indices_
) {
  at::NoNamesGuard guard;

  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1];
  }
  TORCH_CHECK(kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
    "max_pool2d: The kernel size should be smaller than 1535, while this kernel size is ",
    kernel_size_prod);
  constexpr char dilation_err[] =
    "max_pool2d: dilation must be either a single int, or a tuple of two ints, "
    "and cnnl pool2d only supports defalut dilation value";
  TORCH_CHECK((dilation.size() == 1 && dilation[0] == 1) || \
    (dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1), dilation_err);
  
  at::TensorArg output_arg{ output_, "output_", 1 };
  at::TensorArg indices_arg{ indices_, "indices_", 2 };
  at::TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameMLU(__func__, {output_arg, indices_arg, input_arg});
  if (output_.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 3 ? at::unsqueeze(output_, 0) : output_;
  at::Tensor indices = input_.dim() == 3 ? at::unsqueeze(indices_, 0) : indices_;
  
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input_.scalar_type(),
    "max_pool2d_with_indices_out_mlu",
    [&] {
      cnnl_max_pool2d_with_indices_internal(output_contiguous,
                                            indices_contiguous,
                                            input_contiguous,
                                            kH, kW,
                                            dH, dW,
                                            padH, padW,
                                            ceil_mode);
      if (input_.dim() == 3) { // cnnl only support batch mode.
        output_contiguous.squeeze_(0);
        indices_contiguous.squeeze_(0);
      }

      if (is_copy_necessary(output_, output_contiguous)) {
        output_.copy_(output_contiguous);
      }

      if (is_copy_necessary(indices_, indices_contiguous)) {
        indices_.copy_(indices_contiguous);
      }
    }
  );
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_mlu) (
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices_,
  const Tensor& gradInput_) {
  at::NoNamesGuard guard;

  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1];
  }
  TORCH_CHECK(kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
    "max_pool2d: The kernel size should be smaller than 1535, while this kernel size is ",
    kernel_size_prod);
  constexpr char dilation_err[] =
    "max_pool2d: dilation must be either a single int, or a tuple of two ints, "
    "and cnnl pool2d only supports defalut dilation value";
  TORCH_CHECK((dilation.size() == 1 && dilation[0] == 1) || \
    (dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1), dilation_err);
  
  at::TensorArg gradInput_arg{ gradInput_, "gradInput_", 1 };
  at::TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  at::TensorArg input_arg{ input_, "input_", 3 };
  at::TensorArg indices_arg{ indices_, "indices_", 4 };

  checkAllSameMLU(__func__,
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});
  if (gradOutput_.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  
  gradInput_.zero_();
  
  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput = input_.dim() == 3 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput = input_.dim() == 3 ? at::unsqueeze(gradInput_, 0) : gradInput_;
  at::Tensor indices = input_.dim() == 3 ? at::unsqueeze(indices_, 0) : indices_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = cnnl_contiguous(gradInput, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input_.scalar_type(),
    "max_pool2d_with_indices_out_mlu",
    [&] {
      cnnl_pool2d_backward_internal(gradInput_contiguous,
                                    gradOutput_contiguous,
                                    input_contiguous,
                                    indices_contiguous,
                                    kH, kW,
                                    dH, dW,
                                    padH, padW,
                                    ceil_mode, 0);

      if (input_.dim() == 3) { // cnnl only support batch mode.
        gradInput_contiguous.squeeze_(0);
        indices_contiguous.squeeze_(0);
      }

      if (is_copy_necessary(indices_, indices_contiguous)) {
        indices_.copy_(indices_contiguous);
      }

      if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
        gradInput_.copy_(gradInput_contiguous);
      }
    }
  );
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_max_pool3d_with_indices_out(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices) {
  constexpr char dilation_err[] =
    "max_pool3d: dilation must be either a single int, or a tuple of three ints, "
    "and cnnl pool3d only supports defalut dilation value";
  TORCH_CHECK((dilation.size() == 1 && dilation[0] == 1) ||
      (dilation.size() == 3 && dilation[0] == 1 &&
       dilation[1] == 1 && dilation[2] == 1), dilation_err);

  at::TensorArg output_arg{ output, "output", 1 };
  at::TensorArg indices_arg{ indices, "indices", 2 };
  at::TensorArg input_arg{ input, "input", 3 };

  checkAllSameMLU(__func__,
                  {output_arg, indices_arg, input_arg});

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1] * kernel_size[2];
  }
  TORCH_CHECK(kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
    "max_pool3d: The kernel size should be smaller than 1535, while this kernel size is ",
    kernel_size_prod);
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);
  
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
              "max_pool3d: dilation must be either a single int, or a tuple of "
              "three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");
  
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "cnnl_max_pool3d_with_indices_out()");

  // resize output
  bool channels_last = input.ndimension() == 5 && input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  if (input.ndimension() == 4) { // cnnl only support batch mode.
    output.resize_({nslices, otime, oheight, owidth});
    indices.resize_({nslices, otime, oheight, owidth});
  } else {
    if (channels_last) {
      output.resize_({nbatch, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
      indices.resize_({nbatch, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
    } else {
      output.resize_({nbatch, nslices, otime, oheight, owidth});
      indices.resize_({nbatch, nslices, otime, oheight, owidth});
    }
  }

  if (input.numel() == 0) {
    return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
  }

  // cnnl only support batch mode, expand to 5 dimemsion.
  at::Tensor _input = input;
  if (input.dim() == 4) {
    _input = input.unsqueeze(0);
    output = output.unsqueeze(0);
    indices = indices.unsqueeze(0);
  }

  auto memory_format = get_channels_last_memory_format(_input.dim());
  auto input_contiguous = cnnl_contiguous(_input, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input.scalar_type(),
    "cnnl_max_pool3d_with_indices_out",
    [&] {
      cnnl_max_pool3d_with_indices_internal(output_contiguous,
                                            indices_contiguous,
                                            input_contiguous,
                                            kT, kH, kW,
                                            dT, dH, dW,
                                            pT, pH, pW,
                                            ceil_mode);

      if (is_copy_necessary(output, output_contiguous)) {
        output.copy_(output_contiguous);
      }
      
      if (is_copy_necessary(indices, indices_contiguous)) {
        indices.copy_(indices_contiguous);
      }
    }
  );
  if (input.dim() == 4) {
    output.squeeze_(0);
    indices.squeeze_(0);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool3d_with_indices(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  at::NoNamesGuard guard;

  Tensor output = at::empty({0}, input.options());
  // TODO(kongweiguang): [PYTORCH-8347] will remove the following `if` code in the future.
  Tensor indices;
  if (CNNL_DTYPE_HALF == getCnnlDataType(input.dtype())) {
    indices = at::empty({0}, input.options().dtype(at::kShort));
  } else if (CNNL_DTYPE_FLOAT == getCnnlDataType(input.dtype())) {
    indices = at::empty({0}, input.options().dtype(at::kLong));
  }
  
  cnnl_max_pool3d_with_indices_out(input, kernel_size, stride,
                        padding, dilation, ceil_mode, output, indices);

  guard.reset();
  at::namedinference::propagate_names(output, input);
  at::namedinference::propagate_names(indices, input);

  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

at::Tensor& cnnl_max_pool3d_with_indices_backward_out(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& gradInput) {
  constexpr char dilation_err[] =
    "max_pool3d: dilation must be either a single int, or a tuple of three ints, "
    "and cnnl pool3d only supports defalut dilation value";
  TORCH_CHECK((dilation.size() == 1 && dilation[0] == 1) ||
      (dilation.size() == 3 && dilation[0] == 1 &&
       dilation[1] == 1 && dilation[2] == 1), dilation_err);

  at::TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  at::TensorArg gradOutput_arg{ gradOutput, "gradOutput", 2 };
  at::TensorArg input_arg{ input, "input", 3 };
  at::TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameMLU(__func__,
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
              "max_pool3d: kernel_size must either be a single int, or a tuple "
              "of three ints")
  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1] * kernel_size[2];
  }
  TORCH_CHECK(kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
    "max_pool3d: The kernel size should be smaller than 1535, while this kernel size is ",
    kernel_size_prod);
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT
                     : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
                     ? kT
                     : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
              "max_pool3d: stride must either be omitted, a single int, or a "
              "tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH =
      stride.empty() ? kH : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW =
      stride.empty() ? kW : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
              "max_pool3d: padding must be either be a single int, or a tuple "
              "of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
              "max_pool3d: dilation must be either a single int, or a tuple of "
              "three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "cnnl_max_pool3d_with_indices_backward_out(): ",
    "Expected 4D or 5D input tensor, but got ", input.sizes());

  TORCH_CHECK((gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
    "cnnl_max_pool3d_with_indices_backward_out(): ",
    "Expected 4D or 5D gradOutput tensor, but got ", gradOutput.sizes());

  // resize result tensor
  bool channels_last = input.ndimension() == 5 && input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  if (!channels_last) {
    gradInput.resize_as_(input);
  } else {
    gradInput.resize_as_(input, at::MemoryFormat::ChannelsLast3d);
  }

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  max_pool3d_backward_shape_check(input, gradOutput, indices, nslices, kT, kH,
                                  kW, dT, dH, dW, pT, pH, pW, dilationT,
                                  dilationH, dilationW, itime, iheight, iwidth,
                                  otime, oheight, owidth,
                                  "cnnl_max_pool3d_with_indices_backward_out()");

  if (input.numel() == 0) {
    gradInput.zero_();

    return gradInput;
  }

  at::Tensor work_input = input.dim() == 4 ? input.unsqueeze(0) : input;
  at::Tensor work_grad_output = input.dim() == 4 ? gradOutput.unsqueeze(0) : gradOutput;
  at::Tensor work_indices = input.dim() == 4 ? indices.unsqueeze(0) : indices;
  if (input.dim() == 4) {
    gradInput.unsqueeze_(0);
  }

  auto memory_format = get_channels_last_memory_format(work_input.dim());
  auto input_contiguous = cnnl_contiguous(work_input, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(work_grad_output, memory_format);
  auto indices_contiguous = cnnl_contiguous(work_indices, memory_format);
  auto grad_input_contiguous = cnnl_contiguous(gradInput, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,
    input.scalar_type(),
    "cnnl_max_pool3d_with_indices_backward_out",
    [&] {
      cnnl_pool3d_backward_internal(grad_input_contiguous,
                                    grad_output_contiguous,
                                    input_contiguous,
                                    indices_contiguous,
                                    kT, kH, kW,
                                    dT, dH, dW,
                                    pT, pH, pW,
                                    ceil_mode,
                                    false);

      if (is_copy_necessary(gradInput, grad_input_contiguous)) {
        gradInput.copy_(grad_input_contiguous);
      }
    }
  );
  if (input.dim() == 4) {
    gradInput.squeeze_(0);
  }
  return gradInput;
}

at::Tensor cnnl_max_pool3d_with_indices_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  auto gradInput = at::empty_like(input, input.suggest_memory_format());
  cnnl_max_pool3d_with_indices_backward_out(gradOutput, input, kernel_size,
            stride, padding, dilation, ceil_mode, indices, gradInput);
  return gradInput;
}

}  // namespace ops
}  // namespace torch_mlu
