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


#include <ATen/native/UnaryOps.h>
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

using at::native::multinomial_with_replacement_stub;

at::Tensor& cnnl_multinomial_out(
    const at::Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<at::Generator> gen,
    at::Tensor& result) {
  return at::native::multinomial_out(self, n_sample, with_replacement, gen, result);
}

at::Tensor cnnl_multinomial(
    const at::Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<at::Generator> gen) {
  return at::native::multinomial(self, n_sample, with_replacement, gen);
}

void multinomial_with_replacement_mlu_kernel(
    at::Tensor& result,
    const at::Tensor& self,
    const int64_t n_sample,
    c10::optional<at::Generator> gen) {
  auto self_contiguous = cnnl_contiguous(self);
  auto result_contiguous = cnnl_contiguous(result);
  cnnl_multinomial_internal(result_contiguous, self_contiguous, n_sample, true, gen);
  if (!result.is_same(result_contiguous)) {
    result.copy_(result_contiguous);
  }
}

REGISTER_MLU_DISPATCH(multinomial_with_replacement_stub, &multinomial_with_replacement_mlu_kernel);

}  // namespace ops
}  // namespace torch_mlu

