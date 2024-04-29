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
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "framework/generator/generator_impl.h"

namespace torch_mlu {
namespace ops {

void cnnl_multinomial_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  // get tensor impl
  auto* input_impl = getMluTensorImpl(self);
  auto* output_impl = getMluTensorImpl(output);

  // create the descriptor
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  desc_input.set(self, CNNL_LAYOUT_ARRAY);
  desc_output.set(output, CNNL_LAYOUT_ARRAY);

  // get current handle
  auto handle = getCurrentHandle();

  // workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetRandGenerateMultinomialWorkspaceSize(
      handle, desc_input.desc(), &workspace_size));
  auto workspace = at::empty(workspace_size, self.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);

  // get mlu ptr
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto workspace_ptr = workspace_impl->mlu_data_ptr();

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      gen, getDefaultMLUGenerator());
  const int64_t nelem = output.numel();
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(cnnlRandGetSimulateThreadNum(handle, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }
  cnnlRandGenerator_t g_mlu = nullptr;
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  size_t seed = (size_t)rng_engine_inputs.seed_;
  size_t offset = (size_t)rng_engine_inputs.offset_.val;
  TORCH_CNNL_CHECK(cnnlRandCreateGenerator_v2(&g_mlu,
                                              rng_type,
                                              0, /*invalid param*/
                                              seed,
                                              0, /*invalid param*/
                                              offset));

  // The rows of input do not need to sum to one (in which case we use the
  // values as weights), but must be non-negative, finite and have a non-zero
  // sum. So set is_logits false.
  bool is_logits = false;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "MLU multinomial", [&] {
    TORCH_CNNL_CHECK(cnnlRandGenerateMultinomial_v2(
        handle,
        g_mlu,
        desc_input.desc(),
        input_ptr,
        replacement,
        is_logits,
        nullptr, /*invaid param in philox algorithm.*/
        workspace_ptr,
        workspace_size,
        desc_output.desc(),
        output_ptr));
  });

  TORCH_CNNL_CHECK(cnnlRandDestroyGenerator(g_mlu));
}

} // namespace ops
} // namespace torch_mlu
