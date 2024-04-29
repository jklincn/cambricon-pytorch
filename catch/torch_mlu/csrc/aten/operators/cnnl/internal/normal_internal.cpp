#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "framework/generator/generator_impl.h"

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_normal_internal(at::Tensor& output, double mean,
                                 double std, c10::optional<at::Generator> gen) {
  size_t output_num = static_cast<size_t>(output.numel());
  if (output_num == 0) {
    return output;
  }

  // prepare output tensor
  auto* output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();

  // get current handle
  auto handle = getCurrentHandle();
  auto output_type = getCnnlDataType(output.dtype());

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(gen, getDefaultMLUGenerator());
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

  // generate random uniform data
  TORCH_CNNL_CHECK(cnnlRandGenerateNormal(handle,
                                          g_mlu,
                                          output_type,
                                          nullptr, /*It's invalid param in philox algorithm.*/
                                          output_num,
                                          mean,
                                          std,
                                          output_ptr));
  TORCH_CNNL_CHECK(cnnlRandDestroyGenerator(g_mlu));

  return output;
}

}  // namespace ops
}  // namespace torch_mlu
