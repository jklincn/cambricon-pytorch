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

#include "aten/cnnl/cnnlTensorDescriptors.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/tensor_impl.h"
#include "aten/utils/tensor_util.h"

namespace torch_mlu {
void CnnlTensorDescriptor::set_reduce(const at::Tensor& t) {
  int t_dim = t.dim();
  std::vector<int64_t> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    auto t_vec = t.sizes().vec();
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(t_vec[i]);
    }
  }
  auto * tensor_impl = getMluTensorImpl(t);
  auto data_type = getCnnlType(tensor_impl);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptor_v2(this->mut_desc(), CNNL_LAYOUT_ARRAY,
                                              data_type, t_dim, dim_array.data()));
}

// TODO(CNNLCORE-13916) : delete after cnnl support.
void CnnlTensorDescriptor::set_reduce(const cnnlDataType_t& cnnl_dtype,
                                      std::vector<int64_t> keepdim) {
  int t_dim = keepdim.size();
  std::vector<int64_t> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(keepdim[i]);
    }
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptor_v2(this->mut_desc(), CNNL_LAYOUT_ARRAY,
                                              cnnl_dtype, t_dim, dim_array.data()));
}


void CnnlTensorDescriptor::set(const at::Tensor &t) {
  if (!t.defined()) {
    return;
  }
  auto *tensor_impl = getMluTensorImpl(t);
  cnnlDataType_t data_type = getCnnlType(tensor_impl);
  set(t, data_type);
}

void CnnlTensorDescriptor::set(const at::Tensor &t, cnnlDataType_t data_type) {
  int t_dim = t.dim();
  auto *tensor_impl = getMluTensorImpl(t);
  setCnnlType(tensor_impl, data_type);
  if (!t_dim) {
      t_dim = 1;
      std::vector<int64_t> dim_array(1, 1);
      // (sg) change CNNL_LAYOUT_NHWC to CNNL_LAYOUT_ARRAY?
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(),
                                                    CNNL_LAYOUT_ARRAY,
                                                    data_type,
                                                    t_dim,
                                                    dim_array.data(),
                                                    dim_array.data()));
      return;
  }
  std::vector<int64_t> shape_info(t.sizes().vec());
  std::vector<int64_t> stride_info(t.strides().vec());
  std::vector<int64_t> cnnl_stride_info(t_dim);
  if (t.is_contiguous()) {
    cnnl_stride_info = get_contiguous_strides(t.sizes());
  } else {
    cnnl_stride_info = get_cnnl_strides(shape_info, stride_info);
  }
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  if ((!t.is_contiguous()) && (t.suggest_memory_format() != at::MemoryFormat::Contiguous)) {
    convertShapeAndStride(shape_info, cnnl_stride_info);
    layout = suggest_cnnl_layout(t);
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(), layout,
                                               data_type, t_dim, shape_info.data(),
                                               cnnl_stride_info.data()));
}

void CnnlTensorDescriptor::set(const at::Tensor &t,
                               cnnlTensorLayout_t layout,
                               cnnlDataType_t data_type) {
  int t_dim = t.dim();
  auto *tensor_impl = getMluTensorImpl(t);
  if (data_type == CNNL_DTYPE_INVALID) {
    data_type = getCnnlType(tensor_impl);
  }
  if (!t_dim) {
      t_dim = 1;
      std::vector<int64_t> dim_array(1, 1);
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(),
                                                    CNNL_LAYOUT_ARRAY,
                                                    data_type,
                                                    t_dim,
                                                    dim_array.data(),
                                                    dim_array.data()));
      return;
  }
  std::vector<int64_t> shape_info(t.sizes().vec());
  std::vector<int64_t> stride_info(t.strides().vec());
  if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC
          || layout == CNNL_LAYOUT_NLC) {
      convertShapeAndStride(shape_info, stride_info);
  } else if (layout == CNNL_LAYOUT_HWCN) {
      auto convertDepthWiseConvShapeStride = [](const std::vector<int64_t>& vec,
                                                std::vector<int64_t>& target_vec,
                                                std::vector<int64_t>& stride_vec) {
          // NCHW --> HWCN
          target_vec[0] = static_cast<int>(vec[2]);
          target_vec[1] = static_cast<int>(vec[3]);
          target_vec[2] = static_cast<int>(vec[1]);
          target_vec[3] = static_cast<int>(vec[0]);
          // Caculate Stride just like contiguous of HWCN.
          stride_vec[3] = 1;
          stride_vec[2] = target_vec[3] * stride_vec[3];
          stride_vec[1] = target_vec[2] * stride_vec[2];
          stride_vec[0] = target_vec[1] * stride_vec[1];
      };
      convertDepthWiseConvShapeStride(t.sizes().vec(), shape_info, stride_info);
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(), layout,
                                                data_type, t_dim, shape_info.data(),
                                                stride_info.data()));
}

void CnnlTensorDescriptor::set_onchip_dtype(cnnlDataType_t onchip_dtype) {
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(this->mut_desc(), onchip_dtype));
}

void CnnlTensorDescriptor::set(const at::Tensor &t,
                               std::vector<int64_t>& tensor_cnnl_size,
                               int64_t dim) {
  cnnlDataType_t data_type = getCnnlDataType(t.dtype());
  auto *tensor_impl = getMluTensorImpl(t);
  int t_dim = tensor_cnnl_size.size();
  // cnnlSoftmaxForward/cnnlSoftmaxBackward had 3-dim input limitation
  if (!t_dim) {
    t_dim = 3;
    std::vector<int64_t> dim_array(3, 1);
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(),
                                                  CNNL_LAYOUT_ARRAY,
                                                  data_type,
                                                  t_dim,
                                                  dim_array.data(),
                                                  dim_array.data()));
    return;
  }
  const int out_dim = 3;
  int64_t inner_size = 1;
  int64_t outer_size = 1;
  std::vector<int64_t> shape_info(out_dim, 1);
  for (int64_t i = 0; i < dim; ++i) {
    outer_size *= tensor_cnnl_size[i];
  }
  const int64_t dim_size = tensor_cnnl_size[dim];
  for (int64_t i = dim + 1; i < t_dim; ++i) {
    inner_size *= tensor_cnnl_size[i];
  }
  // For best performance, keep dim in last channel as
  // same with original shape size.
  if (dim == 0 && t_dim == 1) {
    shape_info[2] = dim_size;
  } else if (dim == 0 && t_dim != 1) {
    shape_info[0] = dim_size;
    shape_info[1] = inner_size;
  } else if (dim == t_dim - 1) {
    shape_info[1] = outer_size;
    shape_info[2] = dim_size;
  } else {
    shape_info[0] = outer_size;
    shape_info[1] = dim_size;
    shape_info[2] = inner_size;
  }
  auto stride_info = get_contiguous_strides(shape_info);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(), CNNL_LAYOUT_ARRAY,
                                                data_type, out_dim, shape_info.data(),
                                                stride_info.data()));
}

void CnnlTensorDescriptor::set_dim(const at::Tensor &t) {
  const int inputDim = 1;
  cnnlDataType_t data_type = getCnnlDataType(t.dtype());
  std::vector<int64_t> cnnl_size({t.numel()});
  std::vector<int64_t> stride_size = {1};
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(),
                                                CNNL_LAYOUT_ARRAY,
                                                data_type, inputDim,
                                                cnnl_size.data(),
                                                stride_size.data()));
}

void CnnlTensorDescriptor::set_size(const at::Tensor &t,
                                    std::vector<int64_t> dim_sizes,
                                    cnnlTensorLayout_t layout) {
  int dim = dim_sizes.size();
  auto *tensor_impl = getMluTensorImpl(t);
  cnnlDataType_t data_type = getCnnlType(tensor_impl);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptor_v2(this->mut_desc(), layout,
                                              data_type, dim,
                                              dim_sizes.data()));
}

}  // end of namespace torch_mlu
