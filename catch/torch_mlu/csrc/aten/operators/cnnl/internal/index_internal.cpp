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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_index_internal(at::Tensor& output,
                                const at::Tensor& self,
                                const std::vector<at::Tensor>& indices) {
  // To initialize indices ptr with nullptr (for dim check in cnnl).
  // TODO(CNNLCORE-13367): CNNL kernel has a weird check for this.
  std::vector<void *> indices_ptr(CNNL_MAX_DIM_SIZE);

  std::vector<CnnlTensorDescriptor> desc_pool;

  // To initialize cnnlTensorDescriptor_t with nullptr (for dim check in cnnl).
  std::vector<cnnlTensorDescriptor_t> indices_desc(CNNL_MAX_DIM_SIZE);

  bool is_include_bool_index = false;
  for (int i = 0 ; i < indices.size(); ++i) {
    if (indices[i].defined()) {
      TORCH_MLU_CHECK(indices[i].dim() > 0, "zero dimension tensor!");
      if (indices[i].scalar_type() == at::kBool ||
        indices[i].scalar_type() == at::kByte) {
        is_include_bool_index = true;
      }
      auto impl = getMluTensorImpl(indices[i]);
      desc_pool.emplace_back();
      desc_pool.back().set(indices[i]);
      indices_ptr[i] = impl->mlu_data_ptr();
      indices_desc[i] = desc_pool.back().desc();
    } else {
      indices_ptr[i] = nullptr;
      indices_desc[i] = nullptr;
    }
  }

  // Self tensor
  auto self_impl = getMluTensorImpl(self);
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  auto self_ptr = self_impl->mlu_data_ptr();

  // Get output sizes.
  auto handle = getCurrentHandle();
  int output_dim = 0;
  std::vector<int64_t> output_sizes(CNNL_MAX_DIM_SIZE);
  TORCH_CNNL_CHECK(cnnlGetAdvancedIndexOutputDim_v2(handle, self_desc.desc(),
                                                    indices_desc.data(), &output_dim,
                                                    output_sizes.data()));
  output_sizes.resize(output_dim);
  cnnl_resize_(output, output_sizes, c10::MemoryFormat::Contiguous);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // prepare cnnl workspace
  // For Bool AdavancedIndex, the workspace will be zero.
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetAdvancedIndexWorkspaceSize(handle, self_desc.desc(),
          indices_desc.data(), &workspace_size));
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                              self.options().dtype(at::ScalarType::Byte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();

  // TODO(CNNLCORE-13367): output_dim_ptr and output_dims_ptr can't be nullptr now.
  // Kernel will fix this later.
  at::Tensor output_dim_tensor = at::empty({1}, self.options().dtype(at::ScalarType::Int));
  at::Tensor output_dims_tensor = at::empty({8}, self.options().dtype(at::ScalarType::Long));
  CnnlTensorDescriptor output_dim_desc;
  output_dim_desc.set(output_dim_tensor);
  CnnlTensorDescriptor output_dims_desc;
  // set output_dims to int64 for support large tensor
  output_dims_desc.set(output_dims_tensor, CNNL_DTYPE_INT64);
  void* output_dim_ptr = getMluTensorImpl(output_dim_tensor)->mlu_data_ptr();
  void* output_dims_ptr = getMluTensorImpl(output_dims_tensor)->mlu_data_ptr();
  // call cnnl AdvancedIndex v2 interface.
  TORCH_CNNL_CHECK(cnnlAdvancedIndex_v2(handle, self_desc.desc(), self_ptr,
                              indices_desc.data(), indices_ptr.data(),
                              workspace_ptr, workspace_size, output_desc.desc(),
                              output_ptr, output_dim_desc.desc(), output_dim_ptr,
                              output_dims_desc.desc(), output_dims_ptr));

  // add synchronization point to receive output dims.
  if (is_include_bool_index) {
    auto tmp_dim = output_dim_tensor.item().to<int>();
    auto tmp_dims = output_dims_tensor.cpu();
    std::vector<int64_t> output_size(tmp_dim);
    for (int i=0; i < tmp_dim; i++) {
      output_size[i] = tmp_dims[i].item().to<int64_t>();
    }
    resize_impl_mlu_(getMluTensorImpl(output), output_size, c10::nullopt);
  }
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
