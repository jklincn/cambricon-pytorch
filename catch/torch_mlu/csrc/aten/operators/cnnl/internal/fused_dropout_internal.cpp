#include "framework/generator/generator_impl.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/philox_utils.h"

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

void fused_dropout_internal(at::Tensor& output, at::Tensor& mask,
                            const at::Tensor& self, double p,
                            c10::optional<at::Generator> gen) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descMask;
  descInput.set(self);
  descOutput.set(output);
  descMask.set(mask);
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto mask_impl = getMluTensorImpl(mask);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(gen, getDefaultMLUGenerator());
  const int64_t nelem = self.numel();
  cnnlRandGenerator_t g_mlu = nullptr;
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(cnnlRandGetSimulateThreadNum(handle, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }
  size_t seed = (size_t)rng_engine_inputs.seed_;
  size_t offset = (size_t)rng_engine_inputs.offset_.val;
  TORCH_CNNL_CHECK(cnnlRandCreateGenerator_v2(&g_mlu,
                                              rng_type,
                                              0, /*invalid param*/
                                              seed,
                                              0, /*init subsequence*/
                                              offset));

  // fused dropout
  TORCH_CNNL_CHECK(cnnlFusedDropout_v2(handle,
                                       g_mlu,
                                       descInput.desc(),
                                       input_ptr,
                                       1-p,
                                       nullptr, /*it's invalid in philox algorithm.*/
                                       descMask.desc(),
                                       mask_ptr,
                                       descOutput.desc(),
                                       output_ptr));

  TORCH_CNNL_CHECK(cnnlRandDestroyGenerator(g_mlu));
  return;
}

}  // namespace ops
}  // namespace torch_mlu
