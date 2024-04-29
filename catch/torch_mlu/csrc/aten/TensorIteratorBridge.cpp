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

#include "ATen/native/Resize.h"
#include "aten/utils/cnnl_util.h"
#include "aten/operators/cnnl/cnnlOpParams.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

// This function is a little different with pytorch original function.
// In pytorch side, the function is defined in TensorIterator.h, and the function
// is using op.target_dtype to create a TensorOptions.
// In Catch side, we need to create internal tensor with compute dtype and create
// a output tensor with oprand target dtype.
at::TensorOptions
TensorIteratorBridge::original_options(const at::OperandInfo& op,
                                        const at::ScalarType& scalar_dtype) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor_base().options().dtype(scalar_dtype).device(op.device);
  } else {
    return at::TensorOptions(scalar_dtype).device(op.device);
  }
}

// Modify op tensor info and keep origin tensor info in op.original_tensor_base.
// Always keep pytorch original tensor in op.original_tensor, and keep
// internal tensor in op.tensor.
// 1) op.original_tensor is undefined, exchange tensor and original_tensor,
// and update operand original tensor and operand tensor.
// 2) Otherwise just cast tensor to target dtype and update operand tensor.
void TensorIteratorBridge::update_operand_tensor_info(at::OperandInfo& op,
                                                      at::Tensor&& tensor) {
  if (!op.original_tensor_base().defined()) {
    op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(
      std::move(tensor)));
  } else {
    op.tensor(c10::MaybeOwned<at::TensorBase>::owned(
      std::move(tensor)));
  }
  op.current_dtype = op.tensor_base().scalar_type();
}

// Check cpu scalar tensor.
bool TensorIteratorBridge::is_cpu_scalar(const at::OperandInfo& op) {
  return op.tensor().numel() == 1 && op.tensor().device().type() == at::DeviceType::CPU;
}

// Compare current dtype and target dtype, if not equal, CATCH need cast tensor to
// target dtype for cnnl kernel compute.
// But a very special situation in CATCH side is MLU tensor don't support 64bit dtype,
// if tensor dtype is 64bit, the real memory dtype is already 32bit, so no need to call
// cast op. And this cast is not easy to find, cause the cast will be copy in cast internal.
bool TensorIteratorBridge::is_same_current_and_target_dtype(const at::OperandInfo& op) {
  if (op.current_dtype == op.target_dtype ||
      get_mlu_scalar_type(op.current_dtype) == op.target_dtype ||
      op.current_dtype == get_mlu_scalar_type(op.target_dtype)) {
    return true;
  }
  return false;
}

/**
 * Note [CATCH mixed types]
 * ~~~~~~~~~~~~~~~~
 * CATCH mixed types is decided by CNNL op kernel, and the mixed types list is
 * stored in opParamsMap.
 *
 * Mixed type data format is a vector of vector, and the first elements are
 * input types, the last element is output type. For example: 'half + float = half'
 * and 'half + float = float', the mixed types list is {{half, float, half},
 * {half, float, float}}.
 *
 * Do we need to cast tensor dtype when input tensor is mixed types?
 * example:
 *   Tensor A dtype: short, Tensor B dtype: float, and cnnl op support mix types is
 *    'half + float'.
 *   In this case, we may have two way to do:
 *   1) cast A to half, and then call cnnl op, this need handle the A tensor dtype in
 *      mixed types.
 *   2) cast A to float, and then call cnnl op. this need handle the A tensor dtype in
 *      commont types.
 *  Currently, we choose the second way, but maybe call cast op one more time.
 */
std::vector<at::ScalarType>
TensorIteratorBridge::getOutputDtypeWithInputMixedTypes(at::TensorIteratorBase& iter,
                                                         const CnnlOpParams& params) {
  const int ninputs = iter.ninputs();
  std::vector<at::ScalarType> output_scalar_types;
  if (ninputs < 2) return output_scalar_types;
  const int ntensors = iter.ntensors();
  const int noutputs = iter.noutputs();
  TORCH_MLU_CHECK(params.input_mixed_types_list_.size() > 0 &&
    params.input_mixed_types_list_[0].size() == ntensors && noutputs == 1,
    "mixed types list is less than 0 or mix type num is not equal with iter ntensors.");
  std::vector<at::ScalarType> input_scalar_types;
  for (int i = noutputs; i < ntensors; ++i) {
    input_scalar_types.push_back(iter.dtype(i));
  }
  for (const auto& item : params.input_mixed_types_list_) {
    // Get supported input mix types.
    std::vector<at::ScalarType> tmp;
    for (int i = 0; i < ninputs; ++i) {
      tmp.push_back(item[i]);
    }
    // Only support one output dtype right now.
    if (input_scalar_types == tmp) {
      output_scalar_types.push_back(item[ninputs]);
    }
  }
  return output_scalar_types;
}

// when operator has not inputs tensor, we only need to guarantee two conditions:
// 1. output tensor's memory is dense and no overlapping.
// 2. output tensor's dtype can be supported by cnnl kernel.
bool TensorIteratorBridge::nullary_input(at::TensorIteratorBase& iter,
                                         const CnnlOpParams& params,
                                         const std::string& op_name) {
  auto noutputs = iter.noutputs();
  if (iter.ntensors() != noutputs) return false;
  TORCH_CHECK(noutputs == 1,
    "Currently TensorIteratorBridge only support single output if nullary op.");
  // nullary op common_dtype() is undefined, and get from output tensor dtype.
  auto common_dtype = iter.dtype();
  // Check whether fix output dtype.
  this->compute_dtype_ = this->fix_output_dtype_;
  if (this->fix_output_dtype_ == at::ScalarType::Undefined) {
    this->compute_dtype_  = get_catch_promote_types(common_dtype,
                                                   params.support_types_,
                                                   op_name,
                                                   params.allow_implicit_type_convert_);
  }
  // Only one output tensor, so we can use the first operand to represent the output.
  auto& op = iter.operand(0);
  op.target_dtype = this->compute_dtype_;
  const bool is_same_dtype = is_same_current_and_target_dtype(op);
  // Support strided memory.
  if (params.allow_strided_memory_) {
    if (!is_same_dtype) {
      // CNNL Cast op is not support stride, so we will get a contiguous tensor.
      op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(
                           op.tensor().to(op.target_dtype)));
      op.current_dtype = op.target_dtype;
    }
    return true;
  }
  // Don't need to convert common_dtype to 32-bit dtype, common dtype is return dtype,
  // which is decided in TensorIteratorBase.
  if (!op.tensor_base().is_non_overlapping_and_dense() || !is_same_dtype) {
    // Nullary op will_resize is always false, so don't need to check will_resize.
    // target_dtype tensor is using for internal cnnl compute.
    op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(
                         at::empty(op.tensor_base().sizes(),
                         original_options(op, op.target_dtype))));
    op.current_dtype = op.target_dtype;
  }
  return true;
}

/**
 * Note [CATCH dtype promote]
 * Using operand current dtype and target dtype to represent tensor status.
 * 1) current dtype is mean the pytorch output tensor dtype;
 * 2) target dtype is mean the cnnl compute dtype.
 *
 * The basic policy of dtype assign is:
 * 1) If mixed input types is supported by op, then check input tensor types with mix type list.
 * 2) If mixed input types matched and output dtype is matched exactly, just using output dtype;
 * 3) If mixed input types matched and common dtype is matched exactly, using common dtype
 *    and set output operand target dtype to common dtype;
 * 4) If mixed input types matched and output/common dtype are not matched, then fallback to
 *    compute dtype.
 * 5) Using common dtype to get cnnl compute dtype, and set target dtype of each operand to
 *    compute dtype.
 *
 * The basic policy of output tensor dtype is fellow the pytorch rule, and the output dytpe must
 * same with gpu side.
 *
 */
void TensorIteratorBridge::compute_types(at::TensorIteratorBase& iter,
                                          const CnnlOpParams& params,
                                          const std::string& op_name) {
  // Do nothing when op don't need to check types.
  if (params.allow_different_input_types_ == true) return;
  auto common_dtype = iter.common_dtype();
  const int noutputs = iter.noutputs();
  const int ninputs = iter.ninputs();
  if (params.isSupportMixedInputTypes() && noutputs == 1 && ninputs >= 2) {
    // Check input tensor dtypes.
    auto first_input_dtype = iter.dtype(noutputs);
    bool is_different_input_dtypes = false;
    for (int i = (noutputs + 1); i < (noutputs + ninputs); ++i) {
      if (first_input_dtype != iter.dtype(i)) {
        is_different_input_dtypes = true;
        break;
      }
    }
    // Input tensor dtypes is different, so to match mixed types.
    if (is_different_input_dtypes == true) {
      std::vector<at::ScalarType> v_output_types =
        std::move(getOutputDtypeWithInputMixedTypes(iter, params));
      if (v_output_types.size() != 0) {
        // Magic number, cause only one output tensor.
        auto& op = iter.operand(0);
        this->compute_dtype_ = common_dtype;
        // If fix dtype is setted, using fix dtype first.
        if (this->fix_output_dtype_ != at::ScalarType::Undefined) {
          op.target_dtype = this->fix_output_dtype_;
          return;
        }
        // Check output dtype.
        auto it = std::find(v_output_types.begin(), v_output_types.end(),
                            op.current_dtype);
        // output dtype is matched exactly.
        if (it != v_output_types.end()) return;
        // output dtype is not matched exactly, fallback to common dtype.
        it = std::find(v_output_types.begin(), v_output_types.end(), common_dtype);
        // common dtype is matched exactly.
        if (it != v_output_types.end()) {
          op.target_dtype = common_dtype;
          return;
        }
      }
    }
  }
  // Failed to match mix types, fallback to compute dtype promote.
  this->compute_dtype_ = get_catch_promote_types(common_dtype,
                                               params.support_types_,
                                               op_name,
             /* convert_dtype */               params.allow_implicit_type_convert_);
  // Set operand target dtype by using cnnl comput dtype.
  for (int i = 0; i < iter.ntensors(); ++i) {
    auto& op = iter.operand(i);
    // Check fixed output dtype.
    if (this->fix_output_dtype_ != at::ScalarType::Undefined && op.is_output == true) {
      op.target_dtype = this->fix_output_dtype_;
      continue;
    }
    op.target_dtype = this->compute_dtype_;
  }
}

// Almost pytorch binary ops support cpu scalar tensor. And When the first operand is cpu
// scalar tensor, the device guard can't switch to the correct device. So we need to
// switch to the correct device manually. In Pytorch side, this operation is done in
// gpu_kernel_with_scalars function.
// https://github.com/pytorch/pytorch/blob/release/1.6/aten/src/ATen/native/cuda/Loops.cuh#L79
// But in catch side, we can't find a common function before each internal op call.
// So we add this in TensorIteratorBridge.
void TensorIteratorBridge::switch_to_correct_device(at::TensorIteratorBase& iter) {
  // First operand is cpu scalar tensor and second tensor is defined.
  // Don't need to check second tensor' device type, because TensorIterator already
  // do this check in 'compute_types'.
  if (iter.ntensors() == 3 && (iter.ntensors() - iter.noutputs() == 2) &&
    is_cpu_scalar(iter.operand(1)) && iter.operand(2).tensor_base().defined()) {
    device_guard_.reset_device(iter.operand(2).tensor_base().device());
  }
}

// Broadcast input tensor to common shape. And this broadcast is not real expand,
// just expand value 1 in begin of tensor size and call expand op to change tensors
// size and stride.
// For mlu pytorch training, broadcast has two ways.
// 1. Catch will broadcast each tensor dims to common shape dims;
// 2. CNNL will broadcast each tensor dim value to to common shape dim value.
//   example:        a (2,3,4)   b (2,2,1,4) common shape (2,2,3,4)
//   Catch handle:   a (1,2,3,4) b (2,2,1,4)
//   CNNL handle:    a (2,2,3,4) b (2,2,3,4)
void TensorIteratorBridge::input_tensor_broadcast(at::TensorIteratorBase& iter,
                                                  const CnnlOpParams& params) {
  if (params.allow_different_input_sizes_ == true) return;
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    TORCH_CHECK(op.tensor_base().defined(), "Input tensor is not defined.");
    TORCH_CHECK(!op.original_tensor_base().defined(), "Input original_tensor is defined.");
    // Now using cnnl op broadcast function, just expand value 1 in begin of tensor size.
    // Or maybe expand tensor to common shape directly, in this situation device memory
    // usage will be arising, but performance will be arising too.
    // So this requires a trade-off.
    // Now we just expand value 1 in begin of tensor size.
    const int tensor_ndim = op.tensor_base().dim();
    TORCH_CHECK(this->ndim_ >= tensor_ndim, "Output dim is less than input dim.");
    if (!is_cpu_scalar(op) && this->ndim_ > tensor_ndim) {
      std::vector<int64_t> shape = op.tensor_base().sizes().vec();
      shape.insert(shape.begin(), this->ndim_ - tensor_ndim, 1);
      // Reduce at::expand on mlu profile timechart for align ops stack same with
      // original pytorch. This operation has no impact on performance.
      // TensorIteratorBridge.cpp file is compiled before cnnl_expand file,
      // so cnnl_expand is not available.
      // Write a mirror code to instead cnnl_expand and cnnl_as_stride.
      std::vector<int64_t> expandedSizes;
      std::vector<int64_t> expandedStrides;
      std::tie(expandedSizes, expandedStrides) =
        at::inferExpandGeometry(op.tensor_base().sizes(),
                                op.tensor_base().strides(),
                                shape);
      auto* self_impl = getMluTensorImpl(op.tensor());
      // generate a broadcast tensor. More details in cnnl_as_stride.cpp.
      auto broadcast_tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(op.tensor_base().storage()),
                     op.tensor_base().key_set(),
                     op.tensor_base().dtype());
      // Set size and stride.
      auto* broadcast_impl = getMluTensorImpl(broadcast_tensor);
      broadcast_impl->set_sizes_and_strides(expandedSizes, expandedStrides,
                                            op.tensor_base().storage_offset());
      // copy view chain from tensor to broadcast_tensor.
      dynamic_cast<MLUTensorImpl*>(broadcast_impl->external_.get())->view_chain_ =
          dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())->view_chain_;
      // push a expand node to view chain. More details in cnnl_expand.cpp.
      const auto ptr = std::make_shared<ExpandOp>(shape, false);
      dynamic_cast<MLUTensorImpl*>(broadcast_impl->external_.get())
        ->view_chain_.pushNodeToViewChain(op.tensor(), broadcast_tensor, ptr);
      // Move broadcast_tensor to op.tensor.
      update_operand_tensor_info(op, std::move(broadcast_tensor));
    }
  }
}

// Default memory format in mlu side will always be treated as
// Channels_last or Channels_last_3d if tensor dims is equal to 4 or 5.
// MluFastSetupType::NON_OVERLAPPING_DENSE can't convert to compatible memory format,
// so you need to handle it out of this function.
c10::MemoryFormat
TensorIteratorBridge::get_tensor_iterator_memory_format(int dim_num) {
  TORCH_MLU_CHECK(this->setup_type_ != MluFastSetupType::NON_OVERLAPPING_DENSE,
    "NON_OVERLAPPING_DENSE can't convert to compatible memory format.");
  auto memory_format = switch_mlu_setup_type_to_memory_format(this->setup_type_);
  if (memory_format == c10::MemoryFormat::Preserve) {
    memory_format = c10::MemoryFormat::Contiguous;
    if (dim_num == 4 || dim_num == 5) {
      memory_format = torch_mlu::get_channels_last_memory_format(dim_num);
    }
  }
  return memory_format;
}

/**
 * Note [CATCH FAST_SETUP_TYPE]
 * CATCH fast setup type is only based on is_non_overlapping_and_dense input tensors.
 * And the fast setup type priority is:
 * channels_last > channels_last_3d > contiguous > is_non_overlapping_and_dense.
 * 1) Collect tensor list from input tensors without is_non_overlapping_and_dense
 *    tensor and cpu scalar tensor;
 * 2) Call compute_tensors_setup_type to get mlu setup type for tensor list;
 * 3) If is_all_non_overlapping_and_dense == true and setup_type is NON_OVERLAPPING_DENSE,
 *    using setup type NON_OVERLAPPING_DENSE. Otherwise fall back to channels_last_3d or
 *    channels_last or contiguous.
 * 4) Modify setup type based on common shape dim.
 * After this, setup_type_ will be set. And using setup_type_ to get common stride.
 *
 */
void TensorIteratorBridge::compute_mlu_setup_type(const at::TensorIteratorBase& iter,
                                                  const CnnlOpParams& params) {
  if (params.support_memory_format_ != c10::MemoryFormat::Preserve) {
    if (this->shape_.size() == 4 || this->shape_.size() == 5) {
      this->setup_type_ =
        switch_memory_format_to_mlu_setup_type(params.support_memory_format_);
      this->strides_ = get_contiguous_strides(this->shape_,
                          params.support_memory_format_);
    } else {
      this->setup_type_ = MluFastSetupType::CONTIGUOUS;
      this->strides_ = get_contiguous_strides(this->shape_);
    }
    return;
  }
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  bool is_all_non_overlapping_and_dense = true;
  std::vector<at::TensorBase> tensor_vec;
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    TORCH_CHECK(op.tensor_base().defined(), "Input tensor is not defined.");
    // Skip scalar tensor in mlu setup type decision.
    if (is_cpu_scalar(op)) continue;
    // Overlapping or not dense tensor don't need to compute tensors
    // mlu setup type.
    if (!op.tensor_base().is_non_overlapping_and_dense()) {
      is_all_non_overlapping_and_dense = false;
      continue;
    }
    tensor_vec.push_back(op.tensor_base());
  }
  this->setup_type_ = compute_tensors_setup_type(tensor_vec);
  // Setup type is decided by is_non_overlapping_and_dense tensors, so need
  // to check with common shape dim.
  // Example: 6-dim tensor is not contiguous, and 5-dim tensor is contiguos.
  // so mlu setup type is CHANNELS_LAST_3D, but common dim is 6, so need to
  // fall back to contiguous.
  if ((this->ndim_ > 5 || this->ndim_ < 4) &&
      (this->setup_type_ == MluFastSetupType::CHANNELS_LAST ||
       this->setup_type_ == MluFastSetupType::CHANNELS_LAST_3D)) {
    this->setup_type_ = MluFastSetupType::CONTIGUOUS;
  }
  // Example: 5-dim tensor is not contiguous, and 4-dim tensor is contiguos.
  // so mlu setup type is CHANNELS_LAST, but common dim is 5, so need to
  // fall back to CHANNELS_LAST_3D.
  if (this->setup_type_ == MluFastSetupType::CHANNELS_LAST && this->ndim_ == 5) {
    this->setup_type_ = MluFastSetupType::CHANNELS_LAST_3D;
  }
  if (is_all_non_overlapping_and_dense == false &&
    this->setup_type_ == MluFastSetupType::NON_OVERLAPPING_DENSE) {
    // cnnl kernel not support overlapping or not dense tensor, so finally return
    // NONE to call cnnl_contiguous to get channels_first contiguous tensors.
    // (TODO)shangang: pytorch is not support this, and requires a lot of work.
    // More effective way is to move tensor MluFastSetupType to NON_OVERLAPPING_DENSE
    // if some tensors MluFastSetupType are NON_OVERLAPPING_DENSE.
    this->setup_type_ = MluFastSetupType::NONE;
  }
  // Record common stride.
  // 1. using first no broadcast tensor stride size when
  // setup_type is NON_OVERLAPPING_DENSE
  // 2. Calculation strides based on memory format.
  if (this->setup_type_ == MluFastSetupType::NON_OVERLAPPING_DENSE) {
    this->strides_ = tensor_vec[0].strides().vec();
  } else {
    auto memory_format = get_tensor_iterator_memory_format(this->ndim_);
    this->strides_ = get_contiguous_strides(this->shape_, memory_format);
  }
}

// Based on common setup type, modify input tensor size and stride for cnnl
// kernel compute.
// If common setup type is NON_OVERLAPPING_DENSE, this mean all input tensor is
// NON_OVERLAPPING_DENSE, so nothing need to do.
// Otherwise call cnnl_contiguous to get a new tensor, this is not mean malloc
// a new device memory for this one. If tensor is contiguous, this tensor will
// shared storage with original tensor, Otherwise need malloc a new device memory
// for this new tensor.
void TensorIteratorBridge::resize_or_cast_input_tensor(at::TensorIteratorBase& iter,
                                                       const CnnlOpParams& params) {
  const bool is_non_overlapping_dense =
    this->setup_type_ == MluFastSetupType::NON_OVERLAPPING_DENSE;
  auto memory_format = is_non_overlapping_dense == true ? c10::MemoryFormat::Preserve :
    get_tensor_iterator_memory_format(this->ndim_);
  if (params.support_memory_format_ != c10::MemoryFormat::Preserve) {
    if (this->shape_.size() == 4 || this->shape_.size() == 5) {
      memory_format = params.support_memory_format_;
    } else {
      memory_format = c10::MemoryFormat::Contiguous;
    }
  }
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    // Skip cpu scalar tensor in contiguous.
    if (is_cpu_scalar(op)) continue;
    const bool is_same_dtype = is_same_current_and_target_dtype(op);
    const bool is_same_memory_format = is_non_overlapping_dense ||
      op.tensor_base().is_contiguous(memory_format);
    if (is_same_dtype && is_same_memory_format) continue;
    if (is_same_dtype && !is_same_memory_format) {
      at::Tensor internal_tensor =
        torch_mlu::cnnl_contiguous(op.tensor(), memory_format);
      update_operand_tensor_info(op, std::move(internal_tensor));
    } else if (!is_same_dtype && is_same_memory_format) {
      at::Tensor internal_tensor = op.tensor().to(op.target_dtype);
      update_operand_tensor_info(op, std::move(internal_tensor));
    } else {
      // If tensor is not contiguous and dtype is not same with target dtype,
      // get contiguous tensor first, and then cast to target dtype.
      auto temp_tensor = torch_mlu::cnnl_contiguous(op.tensor(), memory_format);
      update_operand_tensor_info(op, temp_tensor.to(op.target_dtype));
    }
  }
}

// Based on output operand info, create a new tensor or resize tensor.
// prerequisite:
//   1. output tensor is defined;
//   2. current dtype is pytorch decided; and target dtype is needed by
//      cnnl op kernel.
// Output tensor is define, op.original_tensor need be set, and there
// will be have two different situation:
//   2.1 will_resize == True, this mean output tensor can be modified,
//       so call set_output to modify tensor info stored in op struct;
//   2.2 will_resize == False, this mean output tensor can't be modified.
//       So check the output tensor whether satisfied common setup type,
//       if satisfied, just using output tensor, otherwise need to create
//       a new tensor with common setup type.
void TensorIteratorBridge::malloc_or_resize_output_tensor(at::TensorIteratorBase& iter) {
  auto noutputs = iter.noutputs();
  for (int i = 0; i < noutputs; ++i) {
    auto& op = iter.operand(i);
    const auto& tensor_size = op.tensor_base().sizes();
    // current_dtype and target_dtype are setted when operand initialized.
    if (op.will_resize == true) {
      // Call set_output interface to resize output tensor.
      iter.set_output_raw_strided(i, tensor_size, this->strides_,
                                  original_options(op, op.current_dtype),
                                  iter.get_dim_names());
      // Check current and compute dtype, if not same, create a new tensor for
      // kernel compute.
      if (!is_same_current_and_target_dtype(op)) {
        at::Tensor internal_tensor = at::empty(tensor_size, original_options(op, op.target_dtype));
        internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(tensor_size, this->strides_);
        update_operand_tensor_info(op, std::move(internal_tensor));
      }
    } else {
      // If output tensor is inplace of first input tensor, so no need to create a new one.
      // So just point to the first input tensor.
      // Pytorch just support one output tensor inplace to first input tensor. If support
      // multi-output tensors inplace input tensors, need to modify here.
      if (op.is_read_write == true) {
        // Find inplace input tensor.
        // 1) If any input tensor is same with this output tensor, this mean inplace tensor
        //    is not changed, just using output tensor.
        const auto& ntensors = iter.ntensors();
        bool is_inplaced_tensor_changed = true;
        for (int j = noutputs; j < ntensors; ++j) {
          if (iter.operand(j).tensor().is_same(op.tensor())) {
            is_inplaced_tensor_changed = false;
            break;
          }
        }
        if (is_inplaced_tensor_changed == false) continue;
        // 2) If inplaced input tensor is changed, we need to find this inplaced input tensor,
        //    and reuse this input operand info for output.
        int inplace_index = 0;
        for (int j = noutputs; j < ntensors; ++j) {
          const auto& input_op = iter.operand(j);
          if (input_op.original_tensor_base().defined() &&
              input_op.original_tensor().is_same(op.tensor())) {
            inplace_index = j;
            break;
          }
        }
        TORCH_MLU_CHECK(inplace_index != 0,
          "Can't find a inplace tensor when output operand is_read_write flag is true,");
        auto& inplace_op = iter.operand(inplace_index);
        // op.original_tensor is undefined, and already checked in to_build.
        // Using input inplace op info to reduce cast op or IO kernel op of output tensor.
        op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::borrowed(inplace_op.tensor()));
        op.current_dtype = inplace_op.current_dtype;
        op.target_dtype = inplace_op.target_dtype;
      } else {
        // TODO(shangang): output tensor will resize flag is False when size is same
        // with common shape in pytorch. This will cause output original tensor will
        // be a strided tensor, and using copy with stride to copy data from output
        // tensor to output original tensor.
        // If output tensor size and stride are same with common size and stride, and
        // current dtype is same with target dtype, this tensor can be used in kernel
        // launch.
        // Otherwise need to malloc a new tensor for output.
        // Normally pytorch TensorIterator has already checked size.
        // Reduce op has already malloc output tensor in make_reduction, so don't need
        // to malloc a new output tensor. And this function will be add when reduct op
        // intergrated with TensorIteratorBridge.
        if (!is_same_current_and_target_dtype(op) ||
          op.tensor_base().strides() != this->strides_) {
          at::Tensor internal_tensor = at::empty(tensor_size,
                                                 original_options(op, op.target_dtype));
          internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(tensor_size,
                                                                       this->strides_);
          update_operand_tensor_info(op, std::move(internal_tensor));
        }
      }
    }
  }
}

// Op support stride overlap, so don't need to check tensor stride, just check the
// tensor dtype.
// Add stride function here is to avoid unnecessary code logic excution.
void TensorIteratorBridge::cast_input_output_tensors_with_stride(at::TensorIteratorBase& iter) {
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  // Cast input tensors.
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    const bool is_same_dtype = is_same_current_and_target_dtype(op);
    // Skip cpu scalar tensor in contiguous.
    if (is_cpu_scalar(op) || is_same_dtype) continue;
    // CNNL Cast op is not support stride, so we will get a contiguous tensor.
    update_operand_tensor_info(op, op.tensor().to(op.target_dtype));
  }
  // Cast output tensors.
  for (int i = 0; i < noutputs; ++i) {
    auto& op = iter.operand(i);
    if (is_cpu_scalar(op)) continue;
    // If output tensor is inplace of first input tensor, so no need to create a new one.
    // So just point to the first input tensor.
    // Pytorch just support one output tensor inplace to first input tensor. If support
    // multi-output tensors inplace input tensors, need to modify here.
    if (op.is_read_write == true) {
      // Find inplace input tensor.
      // 1) If any input tensor is same with this output tensor, this mean inplace tensor
      //    is not changed, just using output tensor.
      const auto& ntensors = iter.ntensors();
      bool is_inplaced_tensor_changed = true;
      for (int j = noutputs; j < ntensors; ++j) {
        if (iter.operand(j).tensor().is_same(op.tensor())) {
          is_inplaced_tensor_changed = false;
          break;
        }
      }
      if (is_inplaced_tensor_changed == false) continue;
      // 2) If inplaced input tensor is changed, we need to find this inplaced input tensor,
      //    and reuse this input operand info for output.
      int inplace_index = 0;
      for (int j = noutputs; j < ntensors; ++j) {
        const auto& input_op = iter.operand(j);
        if (input_op.original_tensor_base().defined() &&
            input_op.original_tensor().is_same(op.tensor())) {
          inplace_index = j;
          break;
        }
      }
      TORCH_MLU_CHECK(inplace_index != 0,
        "Can't find a inplace tensor when output operand is_read_write flag is true,");
      auto& inplace_op = iter.operand(inplace_index);
      // op.original_tensor is undefined, and already checked in to_build.
      // Using input inplace op info to reduce cast op or IO kernel op of output tensor.
      op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::borrowed(inplace_op.tensor()));
      op.current_dtype = inplace_op.current_dtype;
      op.target_dtype = inplace_op.target_dtype;
    } else if (!is_same_current_and_target_dtype(op)) {
      // CNNL Cast op is not support stride, so we will get a contiguous tensor.
      update_operand_tensor_info(op, op.tensor().to(op.target_dtype));
    }
  }
}

void TensorIteratorBridge::to_build(at::TensorIteratorBase& iter,
                                    const std::string& op_name) {
  // All Tensor in TensorIterator need to be defined after mlu add a patch in
  // TensorIteratorBase::build function. That patch is to set a operand will_resize
  // flag is true when malloc a new output tensor (which is not defined in user side.)
  // with common dtype and common shape.
  // After this patch, all output tensor will be defined.
  for (int i = 0; i < iter.noutputs(); ++i) {
    auto& op = iter.operand(i);
    TORCH_MLU_CHECK(!op.original_tensor_base().defined(), "Output original_tensor is defined.");
    TORCH_MLU_CHECK(op.tensor_base().defined(), "Output tensor is not defined.");
  }
  // Updata ndim of TensorIteratorBridge.
  // common shape and ndim has been coalesced in pytorch side. Output tensor always
  // created by common shape, so using first output tensor size to broadcast input tensor.
  this->ndim_ = iter.operand(0).tensor_base().dim();
  this->shape_ = iter.operand(0).tensor_base().sizes();
  // Get fix output dtype and align with gpu side for better performance.
  // Like logic op using fixed bool as output dtype.
  // TensorIteratorBase is build by build_borrowing_unary_force_boolean_op
  // or build_comparison_op or build_borrowing_comparison_op.
  // Also this configuration is a little werid, cause just a few ops
  // support mixed inputs, almost cnnl op kernel need same dtype in support
  // types.
  this->fix_output_dtype_ = iter.get_static_dtype().has_value() ?
    iter.get_static_dtype().value() : at::ScalarType::Undefined;
  // Get cnnl op params from op name.
  const auto& params = getCnnlOpParams(op_name);
  // nullary op can be handled simply. Avoid error call to common_dtype().
  if (nullary_input(iter, params, op_name)) return;
  // switch to mlu correct device
  switch_to_correct_device(iter);
  // compute the result dtype that be support in mlu.
  compute_types(iter, params, op_name);
  // Support strided memory.
  if (params.allow_strided_memory_ == true) {
    // broadcast if necessary
    input_tensor_broadcast(iter, params);
    // Only cast tensor to target dtype and keep tensor stride.
    // This need cnnl cast kernel support stride.
    cast_input_output_tensors_with_stride(iter);
    return;
  }
  // compute mlu setup type based on input tensors
  compute_mlu_setup_type(iter, params);
  // broadcast if necessary
  input_tensor_broadcast(iter, params);
  // cast or contiguous mlu input tensors.
  resize_or_cast_input_tensor(iter, params);
  // malloc or resize output tensor
  malloc_or_resize_output_tensor(iter);
}

}  // namespace torch_mlu
