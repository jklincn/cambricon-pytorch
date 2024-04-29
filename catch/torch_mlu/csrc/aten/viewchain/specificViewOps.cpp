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

#include "aten/viewchain/specificViewOps.h"
#include "aten/utils/cnnl_util.h"

/**
 * Note [viewOps]
 * ~~~~~~~~~~~~~~~~
 * viewOps is an information record of a series of view operators.
 */

namespace torch_mlu {

/**
 * Note [PermuteOp]
 * ~~~~~~~~~~~~~~~~
 * PermuteOp class is for store premute op info, and specify
 * the corresponding cnnl IO kernel.
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
bool PermuteOp::hasCnnlSpecificFunction() {
  TORCH_MLU_CHECK(!this->v_dims.empty(), "Permute op parameters is null.");
  return true;
}

// Infer tensor's `sizes` and `strides` through geometric method.
// Permute only change tensor size and stride.
// PermuteOp only support contiguous output tensor.
bool PermuteOp::inferShape() {
  auto nDims = v_dims.size();
  const TensorInfo& input_info = this->getInputTensorInfo();
  std::vector<int64_t> outputSizes(input_info.v_sizes.begin(), input_info.v_sizes.end());
  for (decltype(nDims) i = 0; i < nDims; ++i) {
    auto dim = v_dims[i];
    outputSizes[i] = input_info.v_sizes[dim];
  }
  this->updateOutputTensorInfo(TensorInfo(std::move(outputSizes),
                               torch_mlu::get_contiguous_strides(outputSizes),
                               input_info.i_storageOffset));
  return true;
}

// Call Cnnl Specific IO kernel.
at::Tensor PermuteOp::runCnnlSpecificFunction(const at::Tensor& input,
                                              c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(this->e_type == ViewsType::kPermute, "ViewType need be kPermute.");
  auto inter_tensor = input;
  // Tensor memory format must be contiguous when flow through view chain.
  // (TODO)shangang: permute_to_contiguous and cnnl_permute_internal maybe fused to a
  // permute call.
  if (!inter_tensor.is_contiguous(c10::MemoryFormat::Contiguous)) {
    inter_tensor = torch_mlu::permute_to_contiguous(inter_tensor,
                        c10::MemoryFormat::Contiguous);
  }
  return torch_mlu::ops::cnnl_permute_internal(inter_tensor, v_dims);
}

std::string PermuteOp::parameterToString() const {
  std::string parameter_info(torch_mlu::viewTypeToString(this->e_type));
  int i = 0;
  parameter_info += ", dims: [";
  for (auto e : this->v_dims) {
    if (i++ > 0)
      parameter_info += ", ";
    parameter_info += std::to_string(e);
  }
  parameter_info += "].";
  return parameter_info;
}

/**
 * Note [SliceOp]
 * ~~~~~~~~~~~~~~~~
 * SliceOp class is for store slice op info, and specify
 * the corresponding cnnl IO kernel.
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
bool SliceOp::hasCnnlSpecificFunction() {
  return true;
}

// Infer tensor's `sizes` and `strides` through geometric method.
// Parameter value already modified in cnnl_slice, so don't need to check value here.
// SliceOp support CL and CF contiguous tensor.
bool SliceOp::inferShape() {
  const TensorInfo& input_info = this->getInputTensorInfo();
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  // Caculate input tensor memory format based on geometric calculation.
  {
    bool is_cl_contiguous = torch_mlu::geometry_is_cl_contiguous(input_info.v_sizes,
                                                                 input_info.v_strides);
    if (is_cl_contiguous) {
      memory_format = input_info.v_sizes.size() == 4 ?
                       c10::MemoryFormat::ChannelsLast
                       : c10::MemoryFormat::ChannelsLast3d;
    }
  }
  // Update output tensor info.
  std::vector<int64_t> outputSizes(input_info.v_sizes.begin(), input_info.v_sizes.end());
  int64_t storageOffset = input_info.i_storageOffset;
  for (int i = 0; i < v_dims.size(); ++i) {
    outputSizes[v_dims[i]] = (v_ends[i] - v_starts[i] + v_steps[i] - 1) / v_steps[i];
    storageOffset += v_starts[i] * input_info.v_strides[v_dims[i]];
  }
  this->updateOutputTensorInfo(TensorInfo(std::move(outputSizes),
                               torch_mlu::get_contiguous_strides(outputSizes, memory_format),
                               storageOffset));
  return true;
}

// Call Cnnl Specific IO kernel.
at::Tensor SliceOp::runCnnlSpecificFunction(const at::Tensor& input,
                                            c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(this->e_type == ViewsType::kSlice, "ViewType need to be kSlice.");
  auto output = torch_mlu::ops::cnnl_multi_dims_slice_internal(input,
                                    this->v_dims, this->v_starts,
                                    this->v_ends, this->v_steps);
  return output;
}

std::string SliceOp::parameterToString() const {
  std::stringstream parameter_info;
  parameter_info << torch_mlu::viewTypeToString(this->e_type);
  parameter_info << ", dims: " << this->v_dims;
  parameter_info << ", start: " << this->v_starts;
  parameter_info << ", end: " << this->v_ends;
  parameter_info << ", step: " << this->v_steps;
  return parameter_info.str();
}

/**
 * Note [ExpandOp]
 * ~~~~~~~~~~~~~~~~
 * ExpandOp class is for store expand op info, and specify
 * the corresponding cnnl IO kernel.
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
bool ExpandOp::hasCnnlSpecificFunction() {
  TORCH_MLU_CHECK(!this->v_dims.empty(), "Expand op parameters is null.");
  return true;
}

// Infer tensor's `sizes` and `strides` through geometric method.
// Expand op support cl and cf contiguous tensor.
bool ExpandOp::inferShape() {
  const TensorInfo& input_info = this->getInputTensorInfo();
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  // Caculate input tensor memory format based on geometric calculation.
  {
    bool is_cl_contiguous = geometry_is_cl_contiguous(input_info.v_sizes,
                                                      input_info.v_strides);
    const int input_size_dim = input_info.v_sizes.size();
    if (is_cl_contiguous && input_size_dim == v_dims.size()) {
      memory_format = input_size_dim == 4 ? c10::MemoryFormat::ChannelsLast
                                      : c10::MemoryFormat::ChannelsLast3d;
    }
  }
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(input_info.v_sizes,
                                             input_info.v_strides, v_dims);
  this->updateOutputTensorInfo(TensorInfo(std::move(expandedSizes),
                      torch_mlu::get_contiguous_strides(expandedSizes, memory_format),
                      input_info.i_storageOffset));
  return true;
}

// Call Cnnl Specific IO kernel.
at::Tensor ExpandOp::runCnnlSpecificFunction(const at::Tensor& input,
                                             c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(this->e_type == ViewsType::kExpand, "ViewType need be kExpand.");
  if (input.sizes() == this->v_dims) {
    return input;
  }
  const int dim_size = this->v_dims.size();
  const int input_size = input.dim();
  // Whether all expand dims in paramter are euqal to value 1.
  // This situation no need to call expand internal, just set Tensor size.
  std::vector<int64_t> temp_dims(input.sizes().vec());
  temp_dims.insert(temp_dims.begin(), dim_size - input_size, 1);
  at::Tensor inter_tensor = input;
  if (input_size < dim_size &&
      !inter_tensor.is_contiguous(c10::MemoryFormat::Contiguous)) {
    inter_tensor = torch_mlu::permute_to_contiguous(inter_tensor,
                      c10::MemoryFormat::Contiguous);
  }
  if (temp_dims == this->v_dims) {
    auto inter_tensor_impl = getMluTensorImpl(inter_tensor);
    inter_tensor_impl->set_sizes_contiguous(this->v_dims);
    return inter_tensor;
  }
  auto output = torch_mlu::ops::cnnl_expand_internal(inter_tensor,
                                                      this->v_dims,
                                                      this->b_implicit);
  return output;
}

std::string ExpandOp::parameterToString() const {
  std::string parameter_info(torch_mlu::viewTypeToString(this->e_type));
  int i = 0;
  parameter_info += ", dims: [";
  for (auto e : this->v_dims) {
    if (i++ > 0)
      parameter_info += ", ";
    parameter_info += std::to_string(e);
  }
  parameter_info += "], ";
  parameter_info += this->b_implicit ? "implicit: true." : "implicit: false.";
  return parameter_info;
}

/**
 * Note [ReshapeOp]
 * ~~~~~~~~~~~~~~~~
 * ReshapeOp class is for store reshape op info, and specify
 * the corresponding cnnl IO kernel.
 * 
 * Note: Reshape op not support Channel_last and Channel_last_3d, so
 * we just consider contiguous input tensor.
 * 
 * Why don't support CL contiguous, you can see more details in below:
 * cnnl_view op in torch_mlu/csrc/aten/operators/cnnl/reshape.cpp
 * computeStride function in aten/src/ATen/TensorUtils.cpp
 * 
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
bool ReshapeOp::hasCnnlSpecificFunction() {
  TORCH_MLU_CHECK(!this->v_shape.empty(), "Permute op parameters is null.");
  return true;
}

// Infer tensor's `sizes` and `strides` through geometric method.
// ReshapeOp only support CF contiguous tensor.
bool ReshapeOp::inferShape() {
  const TensorInfo& input_info = this->getInputTensorInfo();
  this->updateOutputTensorInfo(TensorInfo(v_shape,
                               torch_mlu::get_contiguous_strides(v_shape),
                               input_info.i_storageOffset));
  return true;
}

// Call Cnnl Specific IO kernel.
at::Tensor ReshapeOp::runCnnlSpecificFunction(const at::Tensor& input,
                                              c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(this->e_type == ViewsType::kReshape, "ViewType need be kReshape.");
  TORCH_MLU_CHECK(input.is_contiguous(input.suggest_memory_format()),
    "Reshape op input tensor memory format is not contiguous.");
  TORCH_MLU_CHECK(!output_opt.has_value(),
    "Reshape no need take real output tensor to run specific function.");

  // convert input tensor to channels first tensor.
  at::Tensor internal_tensor = input;
  if (internal_tensor.suggest_memory_format() != c10::MemoryFormat::Contiguous) {
    internal_tensor = torch_mlu::permute_to_contiguous(input,
                        c10::MemoryFormat::Contiguous);
  }
  auto internal_tensor_impl = getMluTensorImpl(internal_tensor);
  internal_tensor_impl->set_sizes_contiguous(this->v_shape);
  return internal_tensor;
}

std::string ReshapeOp::parameterToString() const {
  std::string parameter_info(torch_mlu::viewTypeToString(this->e_type));
  int i = 0;
  parameter_info += ", shape: [";
  for (auto e : this->v_shape) {
    if (i++ > 0)
      parameter_info += ", ";
    parameter_info += std::to_string(e);
  }
  parameter_info += "]. ";
  return parameter_info;
}

/**
 * Note [UnfoldOp]
 * ~~~~~~~~~~~~~~~~
 * UnfoldOp class is for store unfold op info, and specify
 * the corresponding cnnl IO kernel.
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
bool UnfoldOp::hasCnnlSpecificFunction() {
  return true;
}

// Infer tensor's `sizes` and `strides` through geometric method.
// UnfoldOp only support CF contiguous tensor.
bool UnfoldOp::inferShape() {
  const TensorInfo& input_info = this->getInputTensorInfo();
  const int self_dim_num = input_info.v_sizes.size();
  std::vector<int64_t> new_size(self_dim_num + 1);
  new_size[self_dim_num] = i_size;
  for (int d = 0; d < self_dim_num; ++d) {
    const auto& self_size = input_info.v_sizes[d];
    const auto& self_stride = input_info.v_strides[d];
    if (d == i_dimension) {
      new_size[d] = (self_size - i_size) / i_step + 1;
    } else {
      new_size[d] = self_size;
    }
  }
  this->updateOutputTensorInfo(TensorInfo(new_size,
                                          torch_mlu::get_contiguous_strides(new_size),
                                          input_info.i_storageOffset));
  return true;
}

// Call Cnnl Specific IO kernel.
// Based on this op logic, output can't using shared storage of input.
at::Tensor UnfoldOp::runCnnlSpecificFunction(const at::Tensor& input,
                                             c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(this->e_type == ViewsType::kUnfold, "ViewType need be kUnfold.");
  auto inter_tensor = input;
  if (!inter_tensor.is_contiguous(c10::MemoryFormat::Contiguous)) {
    inter_tensor = torch_mlu::permute_to_contiguous(inter_tensor,
                      c10::MemoryFormat::Contiguous);
  }
  this->inferShape();
  at::Tensor output = at::empty(this->getOutputTensorInfo().v_sizes,
                                inter_tensor.options(),
                                c10::MemoryFormat::Contiguous);
  torch_mlu::ops::cnnl_unfold_internal(output,
                                       inter_tensor,
                                       this->i_dimension,
                                       this->i_size,
                                       this->i_step);
  return output;
}

std::string UnfoldOp::parameterToString() const {
  std::string parameter_info(torch_mlu::viewTypeToString(this->e_type));
  parameter_info += ", dimension: ";
  parameter_info += std::to_string(this->i_dimension) + ", size: ";
  parameter_info += std::to_string(this->i_size) + ", step: ";
  parameter_info += std::to_string(this->i_step) + ".";
  return parameter_info;
}

/**
 * Note [DiagonalOp]
 * ~~~~~~~~~~~~~~~~
 * DiagonalOp class is for store diagonal op info, and specify
 * the corresponding cnnl IO kernel.
 */

// Determine whether it can be optimized,
// Reserved interfaces prevent low performance of special case.
/* bool DiagonalOp::hasCnnlSpecificFunction() {
  return false;
}

// Infer tensor's `sizes` and `strides` through geometric method.
bool DiagonalOp::inferShape() {
  return false;
}

// Call Cnnl Specific IO kernel.
at::Tensor DiagonalOp::runCnnlSpecificFunction(const at::Tensor& input,
                                               c10::optional<at::Tensor> output_opt) {
  TORCH_MLU_CHECK(false, "ViewType kDiagonal is not support now.");
  TORCH_MLU_CHECK(this->e_type == ViewsType::kDiagonal, "ViewType need be kDiagonal.");
}

std::string DiagonalOp::parameterToString() const {
  std::string parameter_info(torch_mlu::viewTypeToString(this->e_type));
  parameter_info += ", offset: ";
  parameter_info += std::to_string(this->i_offset) + ", dim1: ";
  parameter_info += std::to_string(this->i_dim1) + ", dim2: ";
  parameter_info += std::to_string(this->i_dim2) + ".";
  return parameter_info;
} */

}  // end of namespace torch_mlu

