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

#include "torch/csrc/python_headers.h"  // the python headers should be first included
#include "python/python_variable_methods.h"
#include "python/memory.h"
#include "python/Stream.h"
#include "python/Event.h"
#include "python/ProcessGroupCNCL.h"

#include "c10/core/Device.h"
#include "c10/util/Optional.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/Generator.h"

#include "framework/core/memory_allocator.h"
#include "framework/core/caching_allocator.h"
#include "framework/generator/generator_impl.h"
#include "framework/core/device.h"
#include "utils/python_interface.h"
#include "utils/version.h"
#include "utils/cndumper.h"
#include "python/THMP.h"
#include "python/Generator.h"

#if USE_PROFILE
#include "profile_mlu.h"  // NOLINT
#endif

namespace torch_mlu {
namespace {

void PythonVariableMethods(py::module& m) {
  THMPStorage_init(m.ptr());
  THCGenerator_init(m.ptr());
  THMPStream_init(m.ptr());
  THMPEvent_init(m.ptr());
  THMPProcessGroupCNCL_init(m.ptr());
  THMPModule_methods(m.ptr());
  registerMLUDeviceProperties(m.ptr());

  // Device Management.
  m.def("_get_device", []() { return PythonInterface::getDevice(); });
  m.def("_synchronize", []() {return torch_mlu::synchronize(); });
  m.def("_get_device_properties", [](int device) -> torch_mlu::DeviceProp* {
          return torch_mlu::getDeviceProperties(device);
          }, pybind11::return_value_policy::reference);

  // Memory Management.
  m.def("_mluHostAllocator", []() -> uint64_t {return reinterpret_cast<uint64_t>(torch_mlu::getMLUCachingHostAllocator()); });
  m.def("_mlu_resetAccumulatedMemoryStats", [](int param) { torch_mlu::resetAccumulatedStats(param); });
  m.def("_mlu_resetPeakMemoryStats", [](int param) { torch_mlu::resetPeakStats(param); });
  m.def("_mlu_memoryStats", [](int param) -> py::dict {return mlu_memoryStats(param);});
  m.def("_mlu_memorySnapshot", []() -> py::list { return mlu_memorySnapshot(); });
  m.def("_mlu_recordMemoryHistory", [](bool enabled) { return mlu_recordMemoryHistory(enabled); });

  m.def("_memory_allocated", [](int param) -> uint64_t {return
        torch_mlu::currentMemoryAllocated(param); });
  m.def("_memory_cached", [](int param) -> uint64_t {return
        torch_mlu::currentMemoryCached(param); });
  m.def("_max_memory_allocated", [](int param) -> uint64_t {return
        torch_mlu::maxMemoryAllocated(param); });
  m.def("_max_memory_cached", [](int param) -> uint64_t {return
        torch_mlu::maxMemoryCached(param); });
  m.def("_empty_cache", [](){ torch_mlu::emptyCache(); });
  m.def("_current_device", []() -> int { return static_cast<int>(torch_mlu::current_device()); });
  m.def("_mem_get_info", [](int device) -> std::pair<size_t, size_t> {return torch_mlu::MemGetInfo(device); });
  m.def("_set_memory_strategy",
        [](bool native_memory_strategy) {
        torch_mlu::set_memory_strategy(native_memory_strategy); });
  m.def("_get_memory_strategy",
        []() -> bool {
        return torch_mlu::is_native_memory_strategy(); });
  m.def("_set_memory_fraction",
        [](double fraction, int64_t device) {
        torch_mlu::setMemoryFraction(fraction, device); });
  m.def("_mlu_mluCachingAllocator_raw_alloc",
        [](size_t size, int64_t queue) -> void* { return mluCachingAllocator_raw_alloc(size, static_cast<cnrtQueue_t>((void*)queue)); });
  m.def("_mlu_mluCachingAllocator_raw_delete",
        [](void* mem_ptr) { mluCachingAllocator_raw_delete(mem_ptr); });
  m.def("_mlu_mluCachingAllocator_set_allocator_settings",
        [](std::string& env) {
        torch_mlu::setAllocatorSettings(env); });
  //m.def("_memory_debug", [](at::Tensor data){
  //    torch_mlu::memoryDebug(&data.storage().data_ptr()); });
  //m.def("_memory_debug", [](){
  //    torch_mlu::memoryDebug(); });

  // TF32 mode management
  m.def("_get_cnnl_allow_tf32", []() -> bool {
        return torch_mlu::Global::instance().allowCNNLTF32(); });
  m.def("_set_cnnl_allow_tf32", [](bool b) {
        torch_mlu::Global::instance().setAllowCNNLTF32(b); });
  m.def("_get_mlu_custom_allow_tf32", []() -> bool {
        return torch_mlu::Global::instance().allowMLUCustomTF32(); });
  m.def("_set_mlu_custom_allow_tf32", [](bool b) {
        torch_mlu::Global::instance().setAllowMLUCustomTF32(b); });

  // Fusion Ops Management, currently only torch.nn.LSTM.
  m.def("_get_mlufusion_enabled", []() -> bool {
        return torch_mlu::Global::instance().allowOpFusion(); });
  m.def("_set_mlufusion_enabled", [](bool b) {
        torch_mlu::Global::instance().setAllowOpFusion(b); });

  // Dumptools API
  m.def("_dump_cnnl_gencase", [&](int mode) {torch_mlu::_dump_cnnl_gencase(mode); });
  m.def("_get_version", []() { return torch_mlu::getVersion(); });

  // profiler API
  #if USE_PROFILE
  m.def("_enable_mlu_profiler", &torch_mlu::profiler::enableMluProfiler);
  #endif
}
}  // namespace

}  // namespace torch_mlu

// Init methods of py::module
void initMLUModule(PyObject* m) {
  auto t = py::handle(m).cast<py::module>();
  torch_mlu::PythonVariableMethods(t);
}
