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

#include <torch/autograd.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> cnnl_ctc_loss_forward(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  
  auto probs_contiguous = probs.device()==at::Device(at::kMLU)
                          ? cnnl_contiguous(probs)
                          : cnnl_contiguous(probs.to(at::Device(at::kMLU)));
  auto targets_contiguous = targets.device()==at::Device(at::kMLU)
                            ? cnnl_contiguous(targets)
                            : cnnl_contiguous(targets.to(at::Device(at::kMLU)));
  auto input_lengths_contiguous = input_lengths.device()==at::Device(at::kMLU)
                                  ? cnnl_contiguous(input_lengths)
                                  : cnnl_contiguous(input_lengths.to(at::Device(at::kMLU)));
  auto target_lengths_contiguous = target_lengths.device()==at::Device(at::kMLU)
                                   ? cnnl_contiguous(target_lengths)
                                   : cnnl_contiguous(target_lengths.to(at::Device(at::kMLU)));
  return cnnl_ctc_loss_internal(
      probs_contiguous,
      targets_contiguous,
      input_lengths_contiguous,
      target_lengths_contiguous,
      il,
      tl,
      blank,
      reduction,
      zero_infinity,
      normalization);
}

class CTCLossFunction : public torch::autograd::Function<CTCLossFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& probs,
      const at::Tensor& targets,
      const at::Tensor& input_lengths,
      const at::Tensor& target_lengths,
      at::IntArrayRef il,
      at::IntArrayRef tl,
      int64_t blank,
      int64_t reduction,
      bool zero_infinity,
      int64_t normalization) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = cnnl_ctc_loss_forward(
        probs,
        targets,
        input_lengths,
        target_lengths,
        il,
        tl,
        blank,
        reduction,
        zero_infinity,
        normalization);

    auto result0 = std::get<0>(result);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({result1});

    return {result0, result1};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto raw_grad = saved[0];
    auto result =
        cnnl_ctc_loss_backward(grad_outputs[0], raw_grad);
    return {
      result,
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable(),
      torch::autograd::Variable()};
  }
};

at::Tensor cnnl_warp_ctc_loss_autograd(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  TORCH_CHECK(reduction == 1, "warp_ctc_loss only support sum mode.");
  TORCH_CHECK(
      normalization == 0,
      "warp_ctc_loss's input doesn't go through log_softmax.");
  TORCH_CHECK(
      (input_lengths.scalar_type() == at::ScalarType::Long ||
       input_lengths.scalar_type() == at::ScalarType::Int),
      "input_lengths must be long or int");
  TORCH_CHECK(
      (input_lengths.scalar_type() == at::ScalarType::Long ||
       target_lengths.scalar_type() == at::ScalarType::Int),
      "target_lengths must be long or int");
  // get scalar value of input_lengths and target_lengths
  at::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
  at::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  auto result = CTCLossFunction::apply(
      probs,
      targets,
      input_lengths,
      target_lengths,
      il,
      tl,
      blank,
      2, // reduction=sum
      zero_infinity,
      normalization)[0]; // normalization=none
  // loss is 1-dim for warp_ctc_loss and 0-dim for nn.CTCLoss
  return result.unsqueeze(0);
}

at::Tensor cnnl_ctc_loss(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  TORCH_CHECK(
      (input_lengths.scalar_type() == at::ScalarType::Long ||
       input_lengths.scalar_type() == at::ScalarType::Int),
      "input_lengths must be long or int");
  TORCH_CHECK(
      (input_lengths.scalar_type() == at::ScalarType::Long ||
       target_lengths.scalar_type() == at::ScalarType::Int),
      "target_lengths must be long or int");
  // get scalar value of input_lengths and target_lengths
  at::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
  at::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return CTCLossFunction::apply(
      probs,
      targets,
      input_lengths,
      target_lengths,
      il,
      tl,
      blank,
      reduction,
      zero_infinity,
      2)[0]; // normalization=log_softmax
}

at::Tensor cnnl_ctc_loss(
    const at::Tensor& probs,
    const at::Tensor& targets,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  // get scalar value of input_lengths and target_lengths
  // TODO(PYTORCH-8642) 2 extra H2D copies are used compare with cuda ctc_loss.
  at::Tensor input_lengths = at::tensor(il).to(at::Device(at::kMLU));
  at::Tensor target_lengths = at::tensor(tl).to(at::Device(at::kMLU));

  return CTCLossFunction::apply(
      probs,
      targets,
      input_lengths,
      target_lengths,
      il,
      tl,
      blank,
      reduction,
      zero_infinity,
      2)[0]; // normalization=log_softmax
}

at::Tensor cnnl_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& raw_grad) {
  auto grad_out_contiguous = cnnl_contiguous(grad_out);
  auto raw_grad_contiguous = cnnl_contiguous(raw_grad);
  if (grad_out.sizes().empty()) {
    return raw_grad_contiguous * grad_out_contiguous;
  } else {
    return raw_grad_contiguous * grad_out_contiguous.unsqueeze(0).unsqueeze(2);
  }
}

}  // namespace ops
}  // namespace torch_mlu
