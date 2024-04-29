/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(replication_pad2d_out_mlu) (
    const at::Tensor& input,
    at::IntArrayRef padding,
    const at::Tensor& output) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "replication_pad2d_out", [&] {
    
    if (input.numel() == 0) {
      return;
    }
    auto input_ = input.dim() == 3 ? at::unsqueeze(input, 0) : input;
    auto output_ = input.dim() == 3 ? at::unsqueeze(output, 0) : output;
    auto input_contiguous = cnnl_contiguous(input_, input_.suggest_memory_format());
    auto output_contiguous = cnnl_contiguous(output_, input_.suggest_memory_format());
    cnnl_replication_pad2d_internal(output_contiguous, input_contiguous, padding);
    if (input.dim() == 3) { // cnnl only support batch mode.
        output_contiguous.squeeze_(0);
    }
    if (is_copy_necessary(output, output_contiguous)) {
      output.copy_(output_contiguous);
    }
  });
}

// TODO: currently, replication pad 1D uses cnnlReplicationPad2d, and this func takes extra squeeze,
// unsqueeze and copy calls. Re-adapt this operator after cnnlReplicationPad1d is ready.
TORCH_IMPL_FUNC(replication_pad1d_out_mlu) (
    const at::Tensor& input,
    at::IntArrayRef padding,
    const at::Tensor& output) {
  auto padding_vec = padding.vec();
  std::vector<int64_t> pad(4);
  for (int i = 0; i < padding_vec.size(); i++) {
    pad[i] = static_cast<int>(padding_vec[i]);
  }
  pad[2] = 0;
  pad[3] = 0;
  at::IntArrayRef padding_(pad);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "replication_pad1d_out", [&] {
    
    if (input.numel() == 0) {
      return;
    }
    auto input_ = input.dim() == 2 ? at::unsqueeze(input, 0) : input;
    input_ = at::unsqueeze(input_, 2);
    auto output_ = output.dim() == 2 ? at::unsqueeze(output, 0) : output;
    output_ = at::unsqueeze(output_, 2);
    auto input_contiguous = cnnl_contiguous(input_, input_.suggest_memory_format());
    auto output_contiguous = cnnl_contiguous(output_, input_.suggest_memory_format());
    cnnl_replication_pad2d_internal(output_contiguous, input_contiguous, padding_);
    output_contiguous.squeeze_(2);
    if (input.dim() == 2) { // cnnl only support batch mode.
        output_contiguous.squeeze_(0);
    }
    if (is_copy_necessary(output, output_contiguous)) {
      output.copy_(output_contiguous);
    }
  });
}

} // namespace ops
} // namespace torch_mlu
