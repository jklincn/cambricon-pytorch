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

#include <framework/generator/generator_impl.h>
#include <framework/core/tensor_impl.h>
#include <aten/utils/tensor_util.h>
#include <ATen/Utils.h>
#include "framework/core/device.h"
#include "framework/core/guard_impl.h"
#include "framework/core/mlu_guard.h"

namespace torch_mlu {

// Ensures we only call mluGetDeviceCount only once
static std::once_flag num_mlu_init_flag;

// Total number of mlus in the system.
static int64_t num_mlus;

// Ensures default_gens_mlu is initialized once.
static std::deque<std::once_flag> mlu_gens_init_flag;

// Default, global MLU generators, one per MLU.
static std::vector<at::Generator> default_gens_mlu;

/*
* Populates the global variables related to MLU generators
* Warning: this function must only be called once!
*/
static void initMLUGenVector() {
  num_mlus = device_count();
  mlu_gens_init_flag.resize(num_mlus);
  default_gens_mlu.resize(num_mlus);
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultMLUGenerator gets the default generator for a particular
 * mlu device.
 */
const at::Generator& getDefaultMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_mlus);
  }
  std::call_once(mlu_gens_init_flag[idx], [&] {
    default_gens_mlu[idx] = at::make_generator<MLUGeneratorImpl>(idx);
    default_gens_mlu[idx].seed();
  });
  return default_gens_mlu[idx];
}

/**
 * Utility to create a MLUGeneratorImpl. Returns a shared_ptr
 */
at::Generator createMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_mlus, "The device_index is invalid.");
  auto generator = at::make_generator<MLUGeneratorImpl>(idx);
  auto gen_impl = at::check_generator<MLUGeneratorImpl>(generator);
  gen_impl->set_current_seed(c10::default_rng_seed_val);
  gen_impl->set_philox_offset_per_thread(0);
  return generator;
}

/**
 * MLUGeneratorImpl class implementation
 */
MLUGeneratorImpl::MLUGeneratorImpl(at::DeviceIndex device_index)
  : c10::GeneratorImpl{at::Device(at::DeviceType::MLU, device_index),
              at::DispatchKeySet(c10::DispatchKey::MLU)} {
}

/**
 * Sets the seed to be used by cnnlRandGenerator_t
 * Resets the philox_offset_per_thread_ to 0
 */
void MLUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::current_seed() const {
  return seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 */
uint64_t MLUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the current internal state of MLUGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> MLUGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  // The following line is just here for BC reason. sizeof curandStateMtgp32 is 4120.
  // It used to be static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  // MAX_NUM_BLOCKS was 200 and sizeof(curandStateMtgp32) is 4120. Hardcoding these numbers here
  // because this is just host side code and we don't want to worry about linking with cuda
  static const size_t states_size = 200 * sizeof(4120);
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, at::ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  // since curandStateMTGP is not used anymore, fill gen_states of THCGenerator with deterministic garbage value of -1
  // gen_states in THCGenerator struct was an array of curandStateMtgp32s.
  memset(rng_state, -1, states_size);
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(this->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as std::atomic<int64_t>
  memcpy(rng_state + states_size, &current_seed, seed_size);
  memcpy(rng_state + states_size + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of MLUGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of MLUGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void MLUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t states_size = 200 * sizeof(4120); // this line is just here for BC reason
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed;
  auto new_rng_state = new_state.data<uint8_t>();
  memcpy(&input_seed, new_rng_state + states_size, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + states_size + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the philox_offset_per_thread_ to be used by cnnlRandGenerator_t
 */
void MLUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::philox_offset_per_thread() const {
  return philox_offset_per_thread_;
}

void MLUGeneratorImpl::capture_prologue(int64_t* seed_extragraph, int64_t* offset_extragraph) {
  TORCH_CHECK(false, "MLUGeneratorImpl::capture_prologue() not implement");
}

uint64_t MLUGeneratorImpl::capture_epilogue() {
  TORCH_CHECK(false, "MLUGeneratorImpl::capture_epilogue() not implement");
}

PhiloxMLUState MLUGeneratorImpl::philox_mlu_state(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return PhiloxMLUState(this->seed_, offset);
}

std::pair<uint64_t, uint64_t> MLUGeneratorImpl::philox_engine_inputs(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of MLUGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType MLUGeneratorImpl::device_type() {
  return at::DeviceType::MLU;
}

/**
 * Public clone method implementation
 */
std::shared_ptr<MLUGeneratorImpl> MLUGeneratorImpl::clone() const {
  return std::shared_ptr<MLUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 */
MLUGeneratorImpl* MLUGeneratorImpl::clone_impl() const {
  auto gen = new MLUGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

/**
 * set rng state
 **/
void set_rng_state(at::DeviceIndex idx, at::Tensor state) {
  auto gen = getDefaultMLUGenerator(idx);
  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(gen, getDefaultMLUGenerator());
  std::lock_guard<std::mutex> lock(gen_impl->mutex_);
  gen_impl->set_state(*(state.getIntrusivePtr().get()));
}

/**
 * set manual seed
 **/
void manual_seed(at::DeviceIndex idx, uint64_t seed) {
  auto defaultGenerator = getDefaultMLUGenerator(idx);
  defaultGenerator.set_current_seed(seed);
}

/**
 * seed
 **/
void seed(at::DeviceIndex idx) {
  auto defaultGenerator = getDefaultMLUGenerator(idx);
  defaultGenerator.seed();
}

/**
 * initial seed
 **/
uint64_t initial_seed(at::DeviceIndex idx) {
  auto defaultGenerator = getDefaultMLUGenerator(idx);
  auto seed = defaultGenerator.current_seed();
  return seed;
}

}  // namespace torch_mlu
