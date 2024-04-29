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

#pragma once

#include "aten/cnnl/cnnlCommonDescriptors.h"

namespace torch_mlu {
class C10_API CnnlPoolingDescriptor
    : public CnnlDescriptor<cnnlPoolingStruct, &cnnlCreatePoolingDescriptor,
                            &cnnlDestroyPoolingDescriptor> {
  public:
      CnnlPoolingDescriptor() = default;

      void set(cnnlPoolingMode_t mode,
               int kernel_h, int kernel_w, int stride_h, int stride_w,
               int pad_u, int pad_d, int pad_l, int pad_r, bool ceil_mode);

      void set(cnnlPoolingMode_t mode,
               int64_t dims,
               const int kernel_size[],
               const int stride[],
               const int padding[],
               const int dilation[],
               bool ceil_mode);
};

class C10_API CnnlTransposeDescriptor
    : public CnnlDescriptor<cnnlTransposeStruct, &cnnlCreateTransposeDescriptor,
                            &cnnlDestroyTransposeDescriptor> {
  public:
      CnnlTransposeDescriptor() { }

      void set(const int p_dims, const int permute[]);
};

class C10_API CnnlReduceDescriptor
    : public CnnlDescriptor<cnnlReduceStruct, &cnnlCreateReduceDescriptor,
                            &cnnlDestroyReduceDescriptor> {
  public:
      CnnlReduceDescriptor() {}
      void set(std::vector<int64_t> axis, cnnlReduceOp_t mode,
               cnnlReduceIndices_t is_indices, cnnlIndicesType_t indices_type,
               cnnlDataType_t tensor_type);
};

class C10_API CnnlOpTensorDescriptor
    : public CnnlDescriptor<cnnlOpTensorStruct, &cnnlCreateOpTensorDescriptor,
                            &cnnlDestroyOpTensorDescriptor> {
  public:
      CnnlOpTensorDescriptor() {}

      void set(cnnlOpTensorDesc_t op_type, cnnlDataType_t op_tensor_comp_type,
               cnnlNanPropagation_t op_tensor_nan_opt);
};

class C10_API CnnlActivationDescriptor
    : public CnnlDescriptor<cnnlActivationStruct,
                            &cnnlCreateActivationDescriptor,
                            &cnnlDestroyActivationDescriptor> {
  public:
      CnnlActivationDescriptor() {}

      void set(cnnlActivationMode_t mode,
               float ceof, int64_t sliced_dim,
               float gamma, float scale,
               bool is_result, bool approximate);
};

class C10_API CnnlConvolutionDescriptor
  : public CnnlDescriptor<cnnlConvolutionStruct,
                      &cnnlCreateConvolutionDescriptor,
                      &cnnlDestroyConvolutionDescriptor> {
  public:
      CnnlConvolutionDescriptor() {}

      template<typename T,
        typename std::enable_if<std::is_same<T, int64_t>::value, bool>::type = true >
      void set(T dim, const T *stride, const T *padding, const T *dilation, T groups,
               cnnlDataType_t dtype, bool allow_tf32) {
        TORCH_CHECK(dim > 2, "Convolution input's dim must greater than 2!");
        int n = dim - 2;
        std::vector<int> padding_t(2 * n);
        std::vector<int> stride_t(n);
        std::vector<int> dilation_t(n);
        int groups_t = static_cast<int>(groups);
        for (int i = 0; i < n; ++i) {
            padding_t[2 * i] = padding[i];
            padding_t[2 * i + 1] = padding[i];
            stride_t[i] = stride[i];
            dilation_t[i] = dilation[i];
        }
        TORCH_CNNL_CHECK(cnnlSetConvolutionDescriptor(
            this->mut_desc(), dim, padding_t.data(), stride_t.data(), dilation_t.data(),
            groups_t, dtype));
        TORCH_CNNL_CHECK(cnnlSetConvolutionDescriptorAllowTF32(this->mut_desc(), allow_tf32));
    }
};

class C10_API CnnlMatmulDescriptor
  : public CnnlDescriptor<cnnlMatMulStruct,
                      &cnnlMatMulDescCreate,
                      &cnnlMatMulDescDestroy> {
  public:
      CnnlMatmulDescriptor() {}
      void set_attr(cnnlMatMulDescAttribute_t attr,
                    const void* buf, size_t size_in_bytes);
};

class C10_API CnnlUniqueDescriptor
    : public CnnlDescriptor<cnnlUniqueStruct,
                            &cnnlCreateUniqueDescriptor,
                            &cnnlDestroyUniqueDescriptor> {
  public:
      CnnlUniqueDescriptor() {}

      void set(bool sorted, int dim, bool return_inverse, bool return_counts);
};

class C10_API CnnlCTCLossDescriptor
    : public CnnlDescriptor<cnnlCTCLossStruct,
                            &cnnlCreateCTCLossDescriptor,
                            &cnnlDestroyCTCLossDescriptor> {
  public:
      CnnlCTCLossDescriptor() {}
      void set(cnnlCTCLossNormalizationMode_t norm_mode,
               cnnlCTCLossReduceMode_t reduce_mode,
               cnnlCTCLossZeroInfinityMode_t zero_infinity,
               int blank,
               int max_input_length,
               int max_label_length);
};

class C10_API CnnlNmsDescriptor
    : public CnnlDescriptor<cnnlNmsStruct,
                            &cnnlCreateNmsDescriptor,
                            &cnnlDestroyNmsDescriptor> {
  public:
      CnnlNmsDescriptor() {}
      void set(const cnnlNmsOutputMode_t mode,
               const float iou_threshold,
               const int max_output_size,
               const float confidence_threshold,
               const int input_layout);
};

class C10_API CnnlRoiAlignDescriptor
    : public CnnlDescriptor<cnnlRoiAlignStruct,
                            &cnnlCreateRoiAlignDescriptor,
                            &cnnlDestroyRoiAlignDescriptor> {
  public:
      CnnlRoiAlignDescriptor() {}
      void set(const int pooled_height,
               const int pooled_width,
               const int sampling_ratio,
               const float spatial_scale,
               const int pool_mode,
               const bool aligned);
};

class C10_API CnnlRNNDescriptor
    : public CnnlDescriptor<cnnlRNNParam,
                            &cnnlCreateRNNDescriptor,
                            &cnnlDestroyRNNDescriptor> {
 public:
  CnnlRNNDescriptor() {}
  void set(int hidden_size,
           int proj_size,
           int num_layers,
           int input_size,
           bool has_biases,
           bool bidirectional,
           cnnlRNNMode_t mode,
           cnnlDataType_t input_type);
};

// For LSTM packed or padded mode.
class C10_API CnnlSeqDataDescriptor
    : public CnnlDescriptor<cnnlSeqDataStruct,
                           &cnnlCreateSeqDataDescriptor,
                           &cnnlDestroySeqDataDescriptor> {
 public:
  CnnlSeqDataDescriptor() {}

  void set(const at::Tensor &t,
           int seq_array,
           const int* seq_array_ptr);
};

class C10_API CnnlDCNDescriptor
  : public CnnlDescriptor<cnnlDCNStruct,
                          cnnlCreateDCNDescriptor,
                          cnnlDestroyDCNDescriptor> {
public:
  CnnlDCNDescriptor() {}
  void set(int dim, int* padding, int* stride,
           int* dilation, int64_t deformable_group,
           int64_t conv_group, int64_t im2col_step,
           cnnlDataType_t dtype);
};


class C10_API CnnlTrigonDescriptor
    : public CnnlDescriptor<cnnlTrigonStruct,
                            &cnnlCreateTrigonDescriptor,
                            &cnnlDestroyTrigonDescriptor> {
  public:
      CnnlTrigonDescriptor() {}
      void set(cnnlTrigonFunctionMode_t mode,
        cnnlComputationPreference_t prefer);
};


class C10_API CnnlGridSampleDescriptor
    : public CnnlDescriptor<cnnlGridSampleStruct,
                            &cnnlCreateGridSampleDescriptor,
                            &cnnlDestroyGridSampleDescriptor> {
  public:
      CnnlGridSampleDescriptor() {}

      void set(const cnnlInterpMode_t interp_mode,
               const cnnlGridSamplePaddingMode_t padding_mode,
               const bool align_corners);
};

class C10_API CnnlInterpDescriptor
    : public CnnlDescriptor<cnnlInterpStruct,
                            &cnnlCreateInterpDescriptor,
                            &cnnlDestroyInterpDescriptor> {
  public:
      CnnlInterpDescriptor() {}
      void set(const cnnlTensorDescriptor_t input_desc,
               const cnnlInterpMode_t mode,
               const cnnlInterpCoordinateTransformationMode_t coordinate_trans_mode,
               float* scales);
};

class C10_API CnnlHistogramDescriptor
    : public CnnlDescriptor<cnnlHistogramStruct,
                            &cnnlCreateHistogramDescriptor,
                            &cnnlDestroyHistogramDescriptor> {
  public:
      CnnlHistogramDescriptor() {}
      void set(const cnnlHistogramMode_t hist_mode,
               int bins,
               float range_min,
               float range_max,
               bool ext_range,
               bool is_index);
      void set(const int64_t value_min,
               const int64_t value_max,
               const int64_t axis,
               bool binary_out);
};

class C10_API CnnlRNNTLossDescriptor
    : public CnnlDescriptor<cnnlRNNTLossStruct, &cnnlCreateRNNTLossDescriptor,
                            &cnnlDestroyRNNTLossDescriptor> {
 public:
  CnnlRNNTLossDescriptor() {}
  void set(const cnnlLossReduction_t reduce_mode, const int64_t blank,
           const double clamp, const bool fused_log_softmax,
           const int64_t max_logit_length, const int64_t max_target_length);
};

class C10_API CnnlEmbeddingBagDescriptor
    :public CnnlDescriptor<cnnlEmbeddingBagStruct,
                           &cnnlCreateEmbeddingBagDescriptor,
                           &cnnlDestroyEmbeddingBagDescriptor>{
 public:
  CnnlEmbeddingBagDescriptor() {}
  void set(const cnnlReduceMode_t mode,
           const void *max_norm,
           const void *norm_type,
           const void *padding_idx,
           const bool scale_grad_by_freq,
           const bool include_last_offset);
};

}  // end of namespace torch_mlu
