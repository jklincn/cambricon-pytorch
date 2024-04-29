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
#include <stdint.h>  // NOLINT
#include "ATen/native/Resize.h"  // NOLINT
#include "ATen/NativeFunctions.h"  // NOLINT
#include "aten/operators/cnnl/cnnl_kernel.h"  // NOLINT
#include "aten/operators/cnnl/internal/cnnl_internal.h"  // NOLINT
#include "aten/utils/cnnl_util.h" // NOLINT
#include "aten/utils/internal_util.h"
#include "aten/utils/tensor_util.h"

/**
 * Note [cnnl_contiguous]
 * ~~~~~~~~~~~~~~~~
 * cnnl_contiguous is almost same with pytorch contigous in functionally.
 * cnnl_contiguous is a series of different ways of IO processing. 
 * The basic is to use permute to complete the target memory format change
 * when tensor is contiguous;
 * The second is to use special IO function to complete memory contiguous;
 * The third is to use view chain to complete memory contiguous;
 * The last one is to use stride copy to complete memory contiguous.
 */

namespace torch_mlu {

static bool DISABLE_VIEW_SPECIFIC_IO_KERNEL = std::getenv(
                                    "DISABLE_VIEW_SPECIFIC_IO") != nullptr &&
                                    (strcmp(std::getenv("DISABLE_VIEW_SPECIFIC_IO"), "ON") == 0 ||
                                    strcmp(std::getenv("DISABLE_VIEW_SPECIFIC_IO"), "on") == 0 ||
                                    strcmp(std::getenv("DISABLE_VIEW_SPECIFIC_IO"), "1") == 0)
                                    ? true    // Disable specific IO kernel and viewchain.
                                    : false;  // Enable specific IO kernel. Base on this, disable or enable viewchain by DISABLE_VIEWCHAIN_FUNC environ. // NOLINT

at::Tensor permute_to_contiguous(const at::Tensor& input,
                                 c10::MemoryFormat memory_format) {
  // If input tensor is contigous, return input.
  if (input.is_contiguous(memory_format)) {
    return input;
  }
  auto permute_back_order = get_permute_back_order(input);
  at::IntArrayRef back_array_order(permute_back_order);
  auto input_before_permute = torch_mlu::ops::cnnl_permute(input, back_array_order);
  auto permute_order = get_permute_order(permute_back_order, memory_format);
  at::IntArrayRef array_order(permute_order);
  auto input_contiguous = torch_mlu::ops::cnnl_permute_internal(input_before_permute, array_order);
  if (memory_format != c10::MemoryFormat::Contiguous) {
    auto strides = get_contiguous_strides(input.sizes(), memory_format);
    getMluTensorImpl(input_contiguous)->set_sizes_and_strides(input.sizes(), strides);
  }
  TORCH_MLU_CHECK(input.sizes() == input_contiguous.sizes(),
    "input sizes must equal to output sizes.");
  return input_contiguous;
}

// only support original tensor memory format, memory_format just affect output.
// size in shared_storage tensor is always relayable.
at::Tensor cnnl_contiguous(const at::Tensor& input,
                           c10::MemoryFormat memory_format) {
  // Check tensor device type, and call native contiguous() if cpu tensor.
  if (input.is_mlu() == false) {
    return input.contiguous(memory_format);
  }
  if (!input.defined()) return input;
  TORCH_MLU_CHECK(memory_format != c10::MemoryFormat::Preserve,
    "Preserve memory format is unsupported by the contiguous operator.");
  // Channels last or channels last3d only support 4 or 5 dimensions.
  TORCH_MLU_CHECK(
    !(memory_format == at::MemoryFormat::ChannelsLast && input.dim() != 4),
    "required rank 4 tensor to use channels_last format");
  TORCH_MLU_CHECK(
    !(memory_format == at::MemoryFormat::ChannelsLast3d && input.dim() != 5),
    "required rank 5 tensor to use ChannelsLast3d format");
  if (input.is_contiguous(memory_format)) {
    return input;
  }
  if (DISABLE_VIEW_SPECIFIC_IO_KERNEL) {
    TORCH_WARN_ONCE("You export DISABLE_VIEW_SPECIFIC_IO environment to "
                    "implement cnnl_contiguous function by CnnlCopy.");
    auto output = at::empty(input.sizes(), input.options(), memory_format);
    torch_mlu::ops::cnnl_copy_without_contiguous_internal(output, input);
    return output;
  }
  // Get tensor with reverse memory_format.
  if (input.is_contiguous(input.suggest_memory_format())) {
    return permute_to_contiguous(input, memory_format);
  }
  if (dynamic_cast<MLUTensorImpl*>(getMluTensorImpl(input)->external_.get())
          ->view_chain_.getViewChainNodeSize() != 1) {
    if (is_permute(input)) {
      return permute_to_contiguous(input, memory_format);
    }
    if (is_expand(input)) {
      auto input_without_zero_stride = get_tensor_without_zero_stride(input);
      input_without_zero_stride = permute_to_contiguous(input_without_zero_stride, memory_format);
      TORCH_MLU_CHECK(input_without_zero_stride.is_contiguous(memory_format),
                      "input_without_zero_stride should be contiguous with ", memory_format);
      auto contiguous_strides = get_contiguous_strides(input_without_zero_stride.sizes(),
                                                        memory_format);
      auto input_without_zero_stride_impl = getMluTensorImpl(input_without_zero_stride);
      input_without_zero_stride_impl->set_sizes_and_strides(input_without_zero_stride.sizes(),
                                                            contiguous_strides);
      auto output = at::empty(input.sizes(), input.options(), memory_format);
      torch_mlu::ops::cnnl_expand_out_internal(output, input_without_zero_stride,
                                               input.sizes());
      return output;
    }
    if ((memory_format == c10::MemoryFormat::Contiguous) && is_slice(Squeeze(input))) {
      auto input_sizes = input.sizes();
      auto input_squeeze = Squeeze(input);
      auto params = get_slice_params(input_squeeze);
      auto contiguous_tensor = get_contiguous_tensor_before_slice(input_squeeze);
      auto output_squeeze = torch_mlu::ops::cnnl_slice_internal(contiguous_tensor,
                                                                params[0], params[1],
                                                                params[2], params[3]);
      return at::native::as_strided_tensorimpl(output_squeeze, input_sizes,
                                               get_contiguous_strides(input_sizes),
                                               output_squeeze.storage_offset());
    }
  }
  // call view chain func to get contiguous tensor.
  const auto input_size = input.sizes();
  auto* input_impl = getMluTensorImpl(input);
  // Using copy kernel when can't using view chain to run specific IO kernel.
  if (dynamic_cast<MLUTensorImpl*>(input_impl->external_.get())
          ->view_chain_.canRunViewChain(input)) {
    // Optimization view chain and then call each specific IO kernel in view chain.
    // (TODO) shangang: Support output tensor may reduce some transpose in some
    // special cases.
    at::Tensor view_chain_output =
        dynamic_cast<MLUTensorImpl*>(input_impl->external_.get())
            ->view_chain_.runViewChain(input);

    // modify view chain output based on memory format.
    if (view_chain_output.suggest_memory_format() != memory_format) {
      return permute_to_contiguous(view_chain_output, memory_format);
    }

    return view_chain_output;
  }

  // Finally call copy kernnel.
  at::Tensor output = at::empty(input_size, input.options(), memory_format);
  return torch_mlu::ops::cnnl_copy_without_contiguous_internal(output, input);
}

}  // namespace torch_mlu
