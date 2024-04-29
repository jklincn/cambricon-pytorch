#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "framework/generator/generator_impl.h"

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

// Uniform is using self tensor to generate a random data, no need to malloc output tensor
// and return output. Also check contiguous in internal.
at::Tensor& cnnl_uniform_internal(at::Tensor& self, c10::optional<at::Generator> gen,
                                  float min, float max) {
  size_t self_num = static_cast<size_t>(self.numel());
  if (self_num == 0) {
    return self;
  }

  // prepare output tensor
  auto* self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();

  // get current handle
  auto handle = getCurrentHandle();
  auto self_type = getCnnlDataType(self.dtype());

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(gen, getDefaultMLUGenerator());
  const int64_t nelem = self.numel();
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
  TORCH_CNNL_CHECK(cnnlRandGenerateUniform(handle,
                                           g_mlu,
                                           self_type,
                                           nullptr, /*It's invalid in philox algorithm.*/
                                           self_num,
                                           min,
                                           max,
                                           self_ptr));
  TORCH_CNNL_CHECK(cnnlRandDestroyGenerator(g_mlu));
  return self;
}

}  // namespace ops
}  // namespace torch_mlu

