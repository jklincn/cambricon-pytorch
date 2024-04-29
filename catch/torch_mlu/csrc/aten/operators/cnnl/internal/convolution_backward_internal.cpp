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

#include "aten/operators/cnnl/internal/convolution_internal_utils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_convolution_backward_input_internal(
    at::Tensor& input_grad, const at::Tensor& output_grad,
    const at::Tensor& weight, const int64_t* stride,
    const int64_t* padding, const int64_t* dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  auto memory_format = get_channels_last_memory_format(input_grad.dim());
  auto grad_input_impl = getMluTensorImpl(input_grad);
  auto weight_impl = getMluTensorImpl(weight);
  auto grad_impl = getMluTensorImpl(output_grad);
  CnnlTensorDescriptor grad_input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  auto layout = input_grad.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto results = process_pseudo_conv(
      input_grad, weight, output_grad, &conv_desc,
      padding, stride, dilation, groups,
      getCnnlDataType(input_grad.scalar_type()),
      allow_tf32);
  bool pseudo_conv3d = std::get<3>(results);
  if (!pseudo_conv3d) {  // conv3d or conv2d
    grad_input_desc.set(input_grad, layout);
    weight_desc.set(weight, layout);
    grad_desc.set(output_grad, layout);
    conv_desc.set(input_grad.dim(), stride, padding,
                dilation, groups, getCnnlDataType(input_grad.scalar_type()),
                allow_tf32);
  } else {  // pseudo conv3d to conv2d
    auto grad_input_size = std::get<0>(results);
    auto weight_size = std::get<1>(results);
    auto grad_size = std::get<2>(results);
    TORCH_MLU_CHECK(grad_input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    memory_format = c10::MemoryFormat::ChannelsLast;
    set_pseudo_conv_tensor_decs(input_grad, grad_input_size, layout,
                                memory_format, getCnnlDataType(input_grad.scalar_type()),
                                grad_input_desc);
    set_pseudo_conv_tensor_decs(weight, weight_size, layout,
                                memory_format, getCnnlDataType(weight.scalar_type()),
                                weight_desc);
    set_pseudo_conv_tensor_decs(output_grad, grad_size, layout,
                                memory_format, getCnnlDataType(output_grad.scalar_type()),
                                grad_desc);
  }

  // prepare conv desc
  cnnlConvolutionBwdDataPreference_t pre_t = CNNL_CONVOLUTION_BWD_DATA_FASTEST;
  cnnlConvolutionBwdDataAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
      handle, weight_desc.desc(), grad_desc.desc(), conv_desc.desc(),
      grad_input_desc.desc(), pre_t, &algo_t));
  // prepare workspace
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle, weight_desc.desc(), grad_desc.desc(), conv_desc.desc(),
      grad_input_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, weight.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  }
  // malloc mlu memory
  auto grad_input_ptr = grad_input_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto grad_ptr = grad_impl->mlu_data_ptr();
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlConvolutionBackwardData(
      /* handle         */ handle,
      /* alpha          */ alpha,
      /* weight_desc    */ weight_desc.desc(),
      /* weight         */ weight_ptr,
      /* diff_y_desc    */ grad_desc.desc(),
      /* diff_y         */ grad_ptr,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* diff_x_desc    */ grad_input_desc.desc(),
      /* diff_x         */ grad_input_ptr));
  return input_grad;
}

at::Tensor& cnnl_convolution_backward_weight_internal(
    at::Tensor& grad_weight, const at::Tensor& output_grad,
    const at::Tensor& input, const int64_t* stride,
    const int64_t* padding, const int64_t* dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  auto memory_format = get_channels_last_memory_format(grad_weight.dim());
  auto grad_weight_impl = getMluTensorImpl(grad_weight);
  auto input_impl = getMluTensorImpl(input);
  auto grad_impl = getMluTensorImpl(output_grad);
  CnnlTensorDescriptor grad_weight_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  auto layout = grad_weight.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto results = process_pseudo_conv(
      input, grad_weight, output_grad, &conv_desc,
      padding, stride, dilation, groups,
      getCnnlDataType(grad_weight.scalar_type()),
      allow_tf32);
  bool pseudo_conv3d = std::get<3>(results);
  if (!pseudo_conv3d) {  // conv3d or conv2d
    input_desc.set(input, layout);
    // auto weight_layout = (depthwise && grad_weight.dim() == 4) ? CNNL_LAYOUT_HWCN : layout;
    grad_weight_desc.set(grad_weight, layout);
    grad_desc.set(output_grad, layout);
    conv_desc.set(grad_weight.dim(), stride, padding,
        dilation, groups, getCnnlDataType(grad_weight.scalar_type()),
        allow_tf32);
  } else {  // pseudo conv3d to conv2d
    auto input_size = std::get<0>(results);
    auto grad_weight_size = std::get<1>(results);
    auto grad_size = std::get<2>(results);
    TORCH_MLU_CHECK(input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    memory_format = c10::MemoryFormat::ChannelsLast;
    set_pseudo_conv_tensor_decs(input, input_size, layout,
                                memory_format, getCnnlDataType(input.scalar_type()),
                                input_desc);
    set_pseudo_conv_tensor_decs(grad_weight, grad_weight_size, layout,
                                memory_format, getCnnlDataType(grad_weight.scalar_type()),
                                grad_weight_desc);
    set_pseudo_conv_tensor_decs(output_grad, grad_size, layout, memory_format,
                                getCnnlDataType(output_grad.scalar_type()), grad_desc);
  }

  // prepare conv desc
  cnnlConvolutionBwdFilterPreference_t pre_t =
      CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
  cnnlConvolutionBwdFilterAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), grad_desc.desc(),
      grad_weight_desc.desc(), pre_t, &algo_t));
  // prepare workspace
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc.desc(), grad_desc.desc(), grad_weight_desc.desc(),
      conv_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, grad_weight.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  }
  // malloc mlu memory

  auto grad_weight_ptr = grad_weight_impl->mlu_data_ptr();
  auto input_ptr = input_impl->mlu_data_ptr();
  auto grad_ptr = grad_impl->mlu_data_ptr();
  CnnlTransposeDescriptor trans_desc;

  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlConvolutionBackwardFilter(
      /* handle         */ handle,
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* diff_y_desc    */ grad_desc.desc(),
      /* diff_y         */ grad_ptr,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* diff_w_desc    */ grad_weight_desc.desc(),
      /* diff_w         */ grad_weight_ptr));
  return grad_weight;
}
}  // namespace ops
}  // namespace torch_mlu
