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

#include <vector>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::vector<int64_t> getRealDim(std::vector<int64_t> input_dim, int64_t t_dim) {
  // handle negative dims
  for (int64_t i = 0; i < input_dim.size(); ++i) {
    if (input_dim[i] < 0) {
      input_dim[i] = input_dim[i] + t_dim;
    }
  }
  // remove duplicate dims and sort them
  // e.g. (3,1,1) -> (1,3)
  std::vector<int64_t> dim_vec(input_dim);
  std::set<int64_t> s(dim_vec.begin(), dim_vec.end());
  dim_vec.assign(s.begin(), s.end());
  return dim_vec;
}

// Return 0,1,2,...,N-1 for all dims.
std::vector<int64_t> getAllDim(int64_t dim) {
    std::vector<int64_t> all_dims;
    if (dim == 0) {
        all_dims.push_back(0);
    }
    for (int i = 0; i < dim; i++) {
        all_dims.push_back(i);
    }
    return all_dims;
}

void cnnl_reduce_internal(const at::Tensor& input,
                          at::Tensor& output,
                          at::Tensor& index,
                          const std::vector<int64_t>& reduce_axis,
                          cnnlReduceOp_t reduce_mode,
                          const cnnlReduceIndices_t reduce_indices) {
  /*
    cnnlReduceOps does not squeeze shape, no matter if keepdim is enabled or not.
    So desc_shpae is the same length as input.dim() with only reduced axis is 1,
    and output_shape is the expect shape of output due to keepdim.
  */
  std::vector<int64_t> reduce_dim = getRealDim(reduce_axis, input.dim());
  if (reduce_axis.size() == 0) {
      reduce_dim = getAllDim(input.dim());
  }

  // input
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto input_cnnl_dtype = getCnnlType(input_impl);
  CnnlTensorDescriptor input_desc;
  input_desc.set_reduce(input);

  // index
  void* index_ptr = nullptr;
  uint32_t index_size_inbytes = 0;
  if (index.defined()) {
    auto index_impl = getMluTensorImpl(index);
    index_ptr = index_impl->mlu_data_ptr();
    index_size_inbytes = sizeof(int) * index.numel();
  }

  // output
  void* output_ptr = nullptr;
  CnnlTensorDescriptor output_desc;
  if (output.defined()) {
    auto output_impl = getMluTensorImpl(output);
    output_ptr = output_impl->mlu_data_ptr();
    output_desc.set_reduce(output);
  } else {
    // TODO(CNNLCORE-13916) : delete after cnnl support.
    output_desc.set_reduce(input_cnnl_dtype, index.sizes().vec());
  }

  // TODO:(sifengyang)
  // by defualt, the bit width is promoted in CNNL_REDUCE_ADD,
  // CNNL_REDUCE_AVG, CNNL_REDUCE_NORM1, CNNL_REDUCE_NORM2 and CNNL_REDUCE_MUL,
  // and other cnnlReduceOp_t will not.
  auto tensor_type = getCnnlDataType(input.dtype());
  CnnlReduceDescriptor reduce_desc;
  reduce_desc.set(reduce_dim, reduce_mode, reduce_indices,
                  CNNL_32BIT_INDICES, tensor_type);

  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  at::Tensor workspace;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(handle,
                                                input_desc.desc(),
                                                output_desc.desc(),
                                                reduce_desc.mut_desc(),
                                                &workspace_size));
  if (workspace_size != 0) {
    workspace = at::empty(workspace_size, input.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  }

  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlReduce(
      /* handle               */ handle,
      /* reduce_desc          */ reduce_desc.desc(),
      /* workspace            */ workspace_ptr,
      /* workspace_size       */ workspace_size,
      /* alpha                */ alpha,
      /* input_desc           */ input_desc.desc(),
      /* input                */ input_ptr,
      /* indices_size_inbytes */ index_size_inbytes,
      /* indices              */ index_ptr,
      /* beta                 */ beta,
      /* output_desc          */ output_desc.desc(),
      /* output               */ output_ptr));
}

void cnnl_var_internal(const at::Tensor& self, at::Tensor& out, at::IntArrayRef dim,
                       bool unbiased, bool keepdim) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  int64_t dim_value = 0;
  if (dim.size() == 1) {
    input_desc.set(self);
    output_desc.set(out);
    dim_value = at::maybe_wrap_dim(dim[0], self.dim());
  } else {
    input_desc.set_dim(self);
    output_desc.set_dim(out);
  }

  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = input_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(out);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlVarForward(handle, dim_value, unbiased, input_desc.desc(),
                                  input_ptr, output_desc.desc(), output_ptr));
}

void cnnl_std_internal(const at::Tensor& self, at::Tensor& out, at::IntArrayRef dim,
                       bool unbiased, bool keepdim) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  int64_t dim_value = 0;
  if (dim.size() == 1) {
    input_desc.set(self);
    output_desc.set(out);
    dim_value = at::maybe_wrap_dim(dim[0], self.dim());
  } else {
    input_desc.set_dim(self);
    output_desc.set_dim(out);
  }

  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = input_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(out);
  auto output_ptr = output_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlStdForward(handle, dim_value, unbiased, input_desc.desc(),
                                  input_ptr, output_desc.desc(), output_ptr));
}

}  // namespace ops
}  // namespace torch_mlu

