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

#include <algorithm>
#include "aten/utils/cnnl_util.h"
#include "framework/core/tensor_impl.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input) {
  auto suggest_memory_format = input.suggest_memory_format();
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  switch (input.dim()) {
    case 4:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast)
      ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
      break;
    case 5:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast3d)
      ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NCDHW;
      break;
    default:
      layout = CNNL_LAYOUT_ARRAY;
  }
  return layout;
}

at::MemoryFormat get_channels_last_memory_format(int64_t dim) {
  TORCH_MLU_CHECK((dim > 3) && (dim < 6),
    "at::MemoryFormat only support rank 4 or 5 tensor with channels_last memory format.");
  at::MemoryFormat memory_format;
  switch (dim) {
    case 4:
      memory_format = at::MemoryFormat::ChannelsLast;
      break;
    case 5:
      memory_format = at::MemoryFormat::ChannelsLast3d;
      break;
  }
  return memory_format;
}

bool pair_first_down(std::pair<int64_t, int64_t>pair1, std::pair<int64_t, int64_t>pair2) {
  return pair1.first > pair2.first;
}

// strides look like create by permute or not.
bool is_permute(const at::Tensor& input) {
  if (input.is_contiguous()) {
    return false;
  }
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  for (int64_t i = 0; i < ndim; ++i) {
    if (input_strides[i] == 0) {
      return false;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  sort(strides_sizes.begin(), strides_sizes.end(), pair_first_down);
  bool is_permute = true;
  int64_t z = 1;
  for (int64_t d = ndim - 1; d >= 0; d--) {
    auto it = strides_sizes[d];
    if (it.second != 1) {
      if (it.first == z) {
        z *= it.second;
      } else {
        is_permute = false;
        break;
      }
    }
  }
  return is_permute;
}

std::vector<int64_t> get_permute_back_order(const at::Tensor& input) {
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  sort(strides_sizes.begin(), strides_sizes.end(), pair_first_down);
  std::vector<int64_t>permute_back_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    auto pair = strides_sizes[i];
    for (int64_t j = 0; j < ndim; ++j) {
      if ((pair.first == input_strides[j]) && (pair.second == input_sizes[j])) {
        permute_back_order[i] = j;
        input_strides[j] = -1;
        input_sizes[j] = -1;
        break;
      }
    }
  }
  return permute_back_order;
}

std::vector<int64_t> get_permute_order(std::vector<int64_t> permute_back_order,
                                       c10::MemoryFormat memory_format) {
  auto ndim = permute_back_order.size();
  std::vector<int64_t>permute_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    permute_order[permute_back_order[i]] = i;
  }
  if ((memory_format != c10::MemoryFormat::Contiguous)
      && ((ndim == 4) || (ndim == 5))) {
    int64_t temp = permute_order[1];
    for (int64_t i = 1; i < ndim - 1; ++i) {
      permute_order[i] = permute_order[i + 1];
    }
    permute_order[ndim - 1] = temp;
  }
  return permute_order;
}

// make dim which has 0 stride to 1 len and 1 stride.
at::Tensor get_tensor_without_zero_stride(const at::Tensor& input) {
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  for (auto it = strides_sizes.begin(); it != strides_sizes.end(); ) {
    if ((*it).first == 0) {
      (*it).first = 1;
      (*it).second = 1;
    }
    ++it;
  }
  for (int64_t i = 0; i < ndim; ++i) {
    input_strides[i] = strides_sizes[i].first;
    input_sizes[i] = strides_sizes[i].second;
  }
  auto input_without_zero_stride = at::native::as_strided_tensorimpl(input,
                                               input_sizes, input_strides);
  return input_without_zero_stride;
}

// strides look like create by expand or not.
bool is_expand(const at::Tensor& input) {
  if (input.is_contiguous()) {
    return false;
  }
  // expand will modify stride value to zero,
  // so check stride value for skipping permute situation.
  auto stride = input.strides().vec();
  auto it = std::find(stride.begin(), stride.end(), 0);
  if (it == stride.end()) {
    return false;
  }
  auto input_without_zero_stride = get_tensor_without_zero_stride(input);
  return (input_without_zero_stride.is_contiguous() || is_permute(input_without_zero_stride));
}

// use cnnlTranspose_v2 instead of cnnlCopyWithStride
// when output is non_overlapping_and_dense in D2D copy.
at::Tensor non_overlapping_and_dense_out(at::Tensor& output, const at::Tensor& input) {
  TORCH_MLU_CHECK(output.is_non_overlapping_and_dense(),
    "output should be non_overlapping_and_dense in non_overlapping_and_dense_out.");
  TORCH_MLU_CHECK((output.sizes() == input.sizes()),
    "output sizes should be the same as input sizes in non_overlapping_and_dense_out.");
  TORCH_MLU_CHECK((output.dtype() == input.dtype()),
    "output dtype should be the same as input dtype in non_overlapping_and_dense_out.");
  auto ndim = output.dim();
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  at::Tensor input_non_overlapping_and_dense = input;
  if (!input.is_non_overlapping_and_dense()) {
    input_non_overlapping_and_dense = cnnl_contiguous(input);
  }
  auto output_permute_back_order = get_permute_back_order(output);
  at::IntArrayRef output_back_array_order(output_permute_back_order);
  auto input_permute_back_order = get_permute_back_order(input_non_overlapping_and_dense);
  at::IntArrayRef input_back_array_order(input_permute_back_order);

  // get contiguous tensor which matched storage, output_contiguous and output shared storage.
  auto output_contiguous = torch_mlu::ops::cnnl_permute(output, output_back_array_order);
  auto input_contiguous = torch_mlu::ops::cnnl_permute(input_non_overlapping_and_dense,
                                                       input_back_array_order);
  auto input_permute_order = get_permute_order(input_permute_back_order, memory_format);

  // output_contiguous equal to
  // input_contiguous.permute(input_permute_order).permute(output_permute_back_order)
  std::vector<int64_t>input_to_output_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    input_to_output_order[i] = input_permute_order[output_permute_back_order[i]];
  }
  at::IntArrayRef input_to_output_array(input_to_output_order);
  torch_mlu::ops::cnnl_permute_out_internal(output_contiguous,
                                            input_contiguous,
                                            input_to_output_array);
  return output;
}

bool can_get_contiguous_tensor_by_strdies(const at::Tensor& input) {
  auto input_squeeze = Squeeze(input);
  auto strides = input_squeeze.strides().vec();
  if (strides[input_squeeze.dim() - 1] != 1) {
    return false;
  }
  for (int64_t i = 0; i < (input_squeeze.dim() - 1); ++i) {
    if ((strides[i] % strides[i + 1]) != 0) {
      return false;
    }
  }
  return true;
}

int64_t get_slice_dim(const at::Tensor& input) {
  auto sizes = input.sizes();
  auto strides = input.strides();
  auto contiguous_strides = get_contiguous_strides(sizes);
  for (int64_t i = input.dim() - 1; i >= 0 ; --i) {
    if (strides[i] != contiguous_strides[i]) {
      // step = 1
      if (can_get_contiguous_tensor_by_strdies(input)) {
        return i + 1;
      } else {  // step > 1
        return i;
      }
    }
  }
  return -1;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_sizes_and_strides_before_slice(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  auto strides = input.strides().vec();
  auto slice_dim = get_slice_dim(input);
  int64_t step;
  int64_t slice_dim_size;
  int64_t slice_dim_stride;
  // slice_dim is lowest dim
  if (slice_dim == (input.dim() - 1)) {
    slice_dim_stride = 1;
    step = strides[slice_dim];
    if (input.dim() == 1) {
      slice_dim_size = (sizes[slice_dim] - 1) * step + 1;
    } else {
      slice_dim_size = strides[slice_dim - 1];
    }
  } else if (slice_dim == 0) {  // slice_dim is highest dim
    slice_dim_stride = sizes[slice_dim + 1] * strides[slice_dim + 1];
    step = strides[slice_dim] / slice_dim_stride;
    slice_dim_size = (sizes[slice_dim] - 1) * step + 1;
  } else {  // slice_dim is middle dim
    slice_dim_size = strides[slice_dim - 1] / (sizes[slice_dim + 1] * strides[slice_dim + 1]);
    slice_dim_stride = sizes[slice_dim + 1] * strides[slice_dim + 1];
  }
  sizes[slice_dim] = slice_dim_size;
  strides[slice_dim] = slice_dim_stride;
  return std::tuple<std::vector<int64_t>, std::vector<int64_t>> {sizes, strides};
}

// slice forward:
//
//    new_offset = old_offset + start * old_strides[dim]
//    new_sizes[dim] = (end - start + step - 1) / step (round-up)
//    new_strides[dim] = old_strides[dim] * step
//    old_strides[dim] = new_sizes[dim + 1] * new_strides[dim + 1] (dim < input.dim() - 1)
//                       or 1 (dim = input.dim() - 1)
//
// we can get new_offset, new_sizes, new_strides by input
// firstly, get dim which may be sliced
// secondly, infer sizes and strides before slicing
// finally, is_slice is true when strides are same as sizes' contiguous strides
// and input's storage has enough bytes
bool is_slice(const at::Tensor& input) {
  if (input.is_contiguous(input.suggest_memory_format())) {
    return false;
  }
  int64_t storage_numel = input.storage().nbytes() / input.itemsize();
  auto sizes = input.sizes().vec();
  auto strides = input.strides().vec();
  for (int64_t i = 0; i < input.dim(); ++i) {
    if (strides[i] == 0) {
      return false;
    }
  }
  auto slice_dim = get_slice_dim(input);
  if (slice_dim == -1) {
    return false;
  }
  if ((input.dim() > 1) && (slice_dim < (input.dim() - 1))) {
    if (((strides[slice_dim] % (sizes[slice_dim + 1] * strides[slice_dim + 1])) != 0)) {
      return false;
    }
  }
  if ((input.dim() > 2) && (0 < slice_dim) && (slice_dim < (input.dim() - 1))) {
    if (((strides[slice_dim - 1] % (sizes[slice_dim + 1] * strides[slice_dim + 1])) != 0)) {
      return false;
    }
  }
  auto old_sizes_and_strides = get_sizes_and_strides_before_slice(input);
  auto old_sizes = std::get<0>(old_sizes_and_strides);
  auto old_strides = std::get<1>(old_sizes_and_strides);
  int64_t step = strides[slice_dim] / old_strides[slice_dim];
  if (((old_sizes[slice_dim] + step - 1) / step) < sizes[slice_dim]) {
    return false;
  }
  auto contiguous_strides = get_contiguous_strides(old_sizes);
  int64_t contiguous_numel = 1;
  for (int64_t i = 0; i < input.dim(); ++i) {
    contiguous_numel *= old_sizes[i];
    if ((old_sizes[i] != 1) && (contiguous_strides[i] != old_strides[i])) {
      return false;
    }
  }
  if ((contiguous_numel + (input.storage_offset() % old_strides[slice_dim])) > storage_numel) {
    return false;
  }
  return true;
}

std::vector<int64_t> get_slice_params(const at::Tensor& input) {
  TORCH_MLU_CHECK(is_slice(input), "is_slice is not true.");
  auto sizes = input.sizes().vec();
  auto strides = input.strides().vec();
  // dim, start, end, step;
  std::vector<int64_t> params(4, -1);

  // get dim
  params[0] = get_slice_dim(input);

  // get step
  auto old_sizes_and_strides = get_sizes_and_strides_before_slice(input);
  auto old_sizes = std::get<0>(old_sizes_and_strides);
  auto old_strides = std::get<1>(old_sizes_and_strides);
  params[3] = strides[params[0]] / old_strides[params[0]];

  // get start and end
  int64_t storage_numel = input.storage().nbytes() / input.itemsize();
  // sizes[dim] = (len + step - 1) / step
  int64_t len = (sizes[params[0]] - 1) * params[3] + 1;
  int64_t contiguous_tensor_numel = 1;
  for (int64_t i = 0; i < input.dim(); ++i) {
    contiguous_tensor_numel *= old_sizes[i];
  }
  // get start
  if ((contiguous_tensor_numel + input.storage_offset()) <= storage_numel) {
    params[1] = 0;
  } else {
    params[1] = (contiguous_tensor_numel + input.storage_offset()
                 - storage_numel + old_strides[params[0]] - 1)
                 / old_strides[params[0]];
  }
  // end = len + start
  params[2] = len + params[1];
  return params;
}

at::Tensor get_contiguous_tensor_before_slice(const at::Tensor& input) {
  TORCH_MLU_CHECK(is_slice(input), "is_slice is not true.");
  auto params = get_slice_params(input);
  int64_t contiguous_offset = 0;
  auto old_sizes_and_strides = get_sizes_and_strides_before_slice(input);
  auto old_sizes = std::get<0>(old_sizes_and_strides);
  auto old_strides = std::get<1>(old_sizes_and_strides);
  contiguous_offset = input.storage_offset() - (params[1] * old_strides[params[0]]);
  return torch_mlu::ops::cnnl_as_strided(input, old_sizes, old_strides, contiguous_offset);
}

// return true if tensors with same format
bool is_same_format_tensor(const at::TensorList& tensors) {
  TORCH_MLU_CHECK(tensors.size() > 0, "Input tensor num need be greater than 0.");
  const auto& size = tensors[0].sizes();
  const auto& stride = tensors[0].strides();
  for (int i = 1; i < tensors.size(); i++) {
    TORCH_MLU_CHECK(tensors[i].defined(), "Input tensor is not defined.");
  }
  for (const auto& t : tensors) {
    if (t.is_non_overlapping_and_dense() == false) {
      return false;
    }
    if (t.sizes() != size || t.strides() != stride) {
      return false;
    }
  }
  return true;
}

std::tuple<at::DimVector, at::DimVector>
inferSqueezeGeometry(const at::Tensor &tensor) {
  at::DimVector sizes;
  at::DimVector strides;

  for (const auto d : c10::irange(tensor.dim())) {
    if (tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(std::move(sizes), std::move(strides));
}

at::Tensor Squeeze(const at::Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  at::Tensor result;
  result = at::native::as_strided_tensorimpl(self, std::get<0>(g), std::get<1>(g));
  return result;
}

at::Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads,
                        const at::Tensor& self, bool some, bool compute_uv,
                        const at::Tensor& raw_u, const at::Tensor& sigma, const at::Tensor& raw_v) {
  TORCH_CHECK(compute_uv,
    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = sigma.size(-1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(-1, 0, k);
    v = raw_v.narrow(-1, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(-1, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(-1, 0, k);
    }
  }
  auto vh = v.conj().transpose(-2, -1);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    gsigma = gsigma.to(self.dtype());
    // computes u @ diag(gsigma) @ vh
    sigma_term = at::matmul(u * gsigma.unsqueeze(-2), vh);
  } else {
    sigma_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto uh = u.conj().transpose(-2, -1);
  auto sigma_inv = sigma.pow(-1);
  auto sigma_sq = sigma.pow(2);
  auto F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    auto guh = gu.conj().transpose(-2, -1);
    u_term = at::matmul(u, F.mul(at::matmul(uh, gu) - at::matmul(guh, u)) * sigma.unsqueeze(-2));
    if (m > k) {
      // projection operator onto subspace orthogonal to span(U) defined as I - UU^H
      auto proj_on_ortho_u = -at::matmul(u, uh);
      proj_on_ortho_u.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
      u_term = u_term + proj_on_ortho_u.matmul(gu * sigma_inv.unsqueeze(-2));
    }
    u_term = at::matmul(u_term, vh);
  } else {
    u_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (gv.defined()) {
    auto gvh = gv.conj().transpose(-2, -1);
    v_term = sigma.unsqueeze(-1) * at::matmul(F.mul(at::matmul(vh, gv) - at::matmul(gvh, v)), vh);
    if (n > k) {
      // projection operator onto subspace orthogonal to span(V) defined as I - VV^H
      auto proj_on_v_ortho = -at::matmul(v, vh);
      proj_on_v_ortho.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
      v_term = v_term + sigma_inv.unsqueeze(-1) * at::matmul(gvh, proj_on_v_ortho);
    }
    v_term = at::matmul(u, v_term);
  } else {
    v_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (self.is_complex() && gu.defined()) {
    at::Tensor L = at::matmul(uh, gu).diagonal(0, -2, -1);
    at::real(L).zero_();
    at::imag(L).mul_(sigma_inv);
    at::Tensor imag_term = at::matmul(u * L.unsqueeze(-2), vh);
    return u_term + sigma_term + v_term + imag_term;
  }

  return u_term + sigma_term + v_term;
}

at::Tensor unsqueeze_multiple(const at::Tensor & t, at::IntArrayRef dim, size_t n_dims) {
    auto dims_to_unsqueeze = at::dim_list_to_bitset(dim, n_dims);
    at::Tensor res = t;
    for (size_t i = 0; i < n_dims; i++) {
      if (dims_to_unsqueeze[i]) {
        res = res.unsqueeze(i);
      }
    }
    return res;
}

c10::MemoryFormat switch_tensors_suggest_memory_format(const std::vector<at::Tensor>& tensor_list) {
    if (tensor_list.size() == 0) {
        return c10::MemoryFormat::Contiguous;
    }
    std::vector<c10::MemoryFormat> tensors_memory_format;
    auto ndim = tensor_list[0].dim();
    bool all_ndim_same = true;
    for(auto tensor:tensor_list) {
        tensors_memory_format.push_back(tensor.suggest_memory_format());
        all_ndim_same &= (tensor.dim() == ndim);
    }
    /* 1. If the ndim of all input tensors is same
    *       If tensors_memory_format contains ChannelsLast, return ChannelsLast,
    *       otherwise return Contiguous
    *  2. If the ndim of all input tensors is not same, return Contiguous
    */
    if (all_ndim_same) {
        auto channel_last_3d_it = std::find(tensors_memory_format.begin(),
                                            tensors_memory_format.end(),
                                            c10::MemoryFormat::ChannelsLast3d);
        if (channel_last_3d_it != tensors_memory_format.end() && ndim == 5) {
            return c10::MemoryFormat::ChannelsLast3d;
        } 
        auto channel_last_2d_it = std::find(tensors_memory_format.begin(),
                                            tensors_memory_format.end(),
                                            c10::MemoryFormat::ChannelsLast);
        if (channel_last_2d_it != tensors_memory_format.end() && ndim == 4) {
            return c10::MemoryFormat::ChannelsLast;
        }
        return c10::MemoryFormat::Contiguous;
    }
    return c10::MemoryFormat::Contiguous;
}
}  // namespace torch_mlu
