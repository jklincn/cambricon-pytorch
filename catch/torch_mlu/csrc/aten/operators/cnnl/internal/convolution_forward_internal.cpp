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

at::Tensor& cnnl_convolution_forward_internal(at::Tensor& output, const at::Tensor & input,
           const at::Tensor & weight, const at::Tensor& bias, const int64_t* padding,
           const int64_t* stride, const int64_t* dilation, int64_t groups,
           bool benchmark, bool deterministic, bool allow_tf32) {
  auto input_impl = getMluTensorImpl(input);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlTensorDescriptor output_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  cnnlTensorLayout_t layout =
      input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto results = process_pseudo_conv(
      input, weight, output, &conv_desc,
      padding, stride, dilation, groups,
      getCnnlDataType(output.scalar_type()),
      allow_tf32);
  bool pseudo_conv3d = std::get<3>(results);
  if (!pseudo_conv3d) {  // conv3d or conv2d
    input_desc.set(input, layout);
    // auto weight_layout = (depthwise && weight.dim() == 4) ? CNNL_LAYOUT_HWCN : layout;
    weight_desc.set(weight, layout);
    output_desc.set(output, layout);
    conv_desc.set(input.dim(), stride,
        padding, dilation, groups, getCnnlDataType(output.scalar_type()),
        allow_tf32);
  } else {  // pseudo conv3d to conv2d
    auto input_size = std::get<0>(results);
    auto weight_size = std::get<1>(results);
    auto output_size = std::get<2>(results);
    TORCH_MLU_CHECK(input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    set_pseudo_conv_tensor_decs(input, input_size, layout, c10::MemoryFormat::ChannelsLast,
                                getCnnlDataType(input.scalar_type()), input_desc);
    set_pseudo_conv_tensor_decs(weight, weight_size, layout, c10::MemoryFormat::ChannelsLast,
                                getCnnlDataType(weight.scalar_type()), weight_desc);
    set_pseudo_conv_tensor_decs(output, output_size, layout, c10::MemoryFormat::ChannelsLast,
                                getCnnlDataType(output.scalar_type()), output_desc);
  }

  // prepare conv desc
  cnnlConvolutionFwdPreference_t pre_t = CNNL_CONVOLUTION_FWD_FASTEST;
  cnnlConvolutionForwardAlgo_t algo_t;
  cnnlSetTensorDescriptorOnchipDataType(input_desc.desc(), getCnnlDataType(input.scalar_type()));
  cnnlSetTensorDescriptorOnchipDataType(weight_desc.desc(), getCnnlDataType(weight.scalar_type()));
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), weight_desc.desc(),
      output_desc.desc(), pre_t, &algo_t));

  // prepare bias
  void *bias_ptr = nullptr;
  int64_t bias_size = 0;
  std::vector<int64_t> bias_cnnl_size;
  if (bias.defined() && bias.dim() != 0 && bias.numel() != 0) {
    TORCH_MLU_CHECK(bias.dim() == 1, "currently only support 1-dim bias in "
      "cnnl_float_convolution_internal when bias.dim() != 0, but got ", bias.dim(), " dim.");
    bias_size = bias.sizes()[0];
    // for group parameter, bias size must be 4 or 5 dims,(1,C,1,1) or (1,C,1,1,1)
    if (input.dim() > 4 && !pseudo_conv3d) {
      layout = CNNL_LAYOUT_NDHWC;
      bias_cnnl_size = {1, 1, 1, 1, bias_size};
    } else {
      layout = CNNL_LAYOUT_NHWC;
      bias_cnnl_size = {1, 1, 1, bias_size};
    }
    bias_desc.set_size(bias, bias_cnnl_size, layout);
    auto bias_impl = getMluTensorImpl(bias);
    bias_ptr = bias_impl->mlu_data_ptr();
  }

  // prepare workspace
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
      handle, input_desc.desc(), weight_desc.desc(), output_desc.desc(),
      bias_desc.desc(), conv_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace = at::empty(workspace_size,
                          input.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  }

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  CnnlTransposeDescriptor trans_desc;

  const void * alpha = nullptr;
  const void * beta = nullptr;

  TORCH_CNNL_CHECK(cnnlConvolutionForward(
      /* handle         */ handle,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x_ptr          */ input_ptr,
      /* w_desc         */ weight_desc.desc(),
      /* w_ptr          */ weight_ptr,
      /* bias_desc      */ bias_desc.desc(),
      /* bias_ptr       */ bias_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y_ptr          */ output_ptr));
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
