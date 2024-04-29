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
class C10_API CnnlTensorDescriptor
    : public CnnlDescriptor<cnnlTensorStruct, &cnnlCreateTensorDescriptor,
                            &cnnlDestroyTensorDescriptor> {
  public:
    // Init create Tensor descriptor
    CnnlTensorDescriptor() = default;
    // set descriptor from tensor
    void set(const at::Tensor &t);
    void set(const at::Tensor &t,
             cnnlTensorLayout_t layout,
             cnnlDataType_t data_type = CNNL_DTYPE_INVALID);
    void set(const at::Tensor &t, cnnlDataType_t dtype);
    void set_onchip_dtype(cnnlDataType_t data_type);

    // for dimension conbined, just support 0<=dim <=3;
    // dim == 0 or 1, will combine dim 1 and dim 2 to new shape dim 1;
    // dim == 2, will combine dim 2 and dim 3 to new shape dim 2;
    // dim == 3, will combine dim 0 and dim 1 to new shape dim 0;
    // Just support channel_last format type.
    void set(const at::Tensor &t,
             std::vector<int64_t>& tensor_cnnl_size,
             int64_t dim);
    void set_dim(const at::Tensor &t);

    void set_reduce(const at::Tensor &t);
    // TODO(CNNLCORE-13916) : delete after cnnl support.
    void set_reduce(const cnnlDataType_t& cnnl_dtype, std::vector<int64_t> keepdim);


    void set_size(const at::Tensor &t,
                  std::vector<int64_t> dim_sizes,
                  cnnlTensorLayout_t layout);

    template<typename T>
    void set(const at::Tensor &t,
             const std::vector<T>& shape_info,
             const std::vector<T>& stride_info,
             cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY,
             cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
      TORCH_CHECK(shape_info.size() == stride_info.size(),
          "shape size need equal to stride size.");
      int t_dim = shape_info.size();
      auto *tensor_impl = getMluTensorImpl(t);
      // data_type default value is CNNL_DTYPE_INVALID in this interface,
      // and can't transmit to cnnl. so call cnnl interface will using
      // tensor dtype value when data_type value is default.
      if (data_type == CNNL_DTYPE_INVALID) {
        data_type = getCnnlType(tensor_impl);
      }
      if (!t_dim) {
          t_dim = 1;
          std::vector<int64_t> dim_array(1, 1);
          TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(),
                                                        CNNL_LAYOUT_ARRAY,
                                                        data_type, t_dim,
                                                        dim_array.data(),
                                                        dim_array.data()));
          return;
      }
      std::vector<int64_t> real_shape_info;
      std::vector<int64_t> real_stride_info;
      if (std::is_same<typename std::decay<T>::type, int64_t>::value == true) {
        real_shape_info.assign(shape_info.begin(), shape_info.end());
        real_stride_info.assign(stride_info.begin(), stride_info.end());
      } else if (std::is_same<typename std::decay<T>::type, int>::value == true) {
        for (int i = 0; i < t_dim; i++) {
          real_shape_info.push_back(shape_info[i]);
          real_stride_info.push_back(stride_info[i]);
        }
      } else {
        TORCH_MLU_CHECK(false, "Value type of size and stride is not support.");
      }
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(this->mut_desc(), layout,
                                                    data_type, t_dim,
                                                    real_shape_info.data(),
                                                    real_stride_info.data()));
    }
};

}  // end of namespace torch_mlu
