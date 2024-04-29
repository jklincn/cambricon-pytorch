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

#include <pybind11/pybind11.h>
#include <array>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <ATen/ATen.h>
#include "framework/generator/generator_impl.h"  // MLU generator
#include "utils/python_interface.h"
#include <python/THMP.h>
#include "python/Generator.h"
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/mlu_lazy_init.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cuda/python_comm.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include "framework/core/queue.h"
#include "framework/core/device.h"

static bool in_bad_fork = false;  // True for children forked after mlu init

#ifndef WIN32
// Called in the forked child if mlu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_mlu_init(true);
}
#endif

// Should be called before the first mlu call.
// Note: This is distinct from initExtension because a stub mlu implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static std::once_flag flag;
  std::call_once(flag, []{ pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

static at::Tensor dispatch_to(const at::Tensor & self, Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THMPModule_initExtension(PyObject *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork);  // Handled at python level
  poison_fork();
  at::globalContext().lazyInitMLU();
  auto m = THPObjectPtr(PyImport_ImportModule("torch.mlu"));
  if (!m) throw python_error();
  THMPStorage_postInit(m);

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_mlus = torch_mlu::device_count();
  auto default_mlu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_mlus));
  for (int i = 0; i < num_mlus; i++) {
    auto gen = torch_mlu::getDefaultMLUGenerator(i);
    auto cast_gen = (THPGenerator*)THCGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_mlu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_mlu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Callback for python part. Used for additional initialization of python classes.
// Note: Unlike THMPModule_initExtension above, what THPModule_initExtension_mlu
// does is device-independent(so far, only THMPStorage_postInit is called), and
// they have the same relationship to each other as THPModule_initExtension and
// THCPModule_initExtension.
static PyObject* THPModule_initExtension_mlu(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  auto module = THPObjectPtr(PyImport_ImportModule("torch.mlu"));
  if (!module)
    throw python_error();

  THMPStorage_postInit(module);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THMPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    torch_mlu::getCurrentQueue(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THMPModule_getDefaultStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    torch_mlu::getDefaultQueue(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THMPModule_setStream_wrap(PyObject *self, PyObject *obj) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
    throw python_error();
  }
  auto stream = torch_mlu::Queue::unpack(bits);
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(torch_mlu::current_device());
  if (device != stream.device_index()) {
    THMPModule_setDevice(stream.device_index());
  }
  torch_mlu::setCurrentQueue(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  torch::utils::mlu_lazy_init();
  auto can_access = torch_mlu::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::mlu_lazy_init();
  auto device = static_cast<int>(torch_mlu::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(torch_mlu::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_mlu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "mlu(Tensor self, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "mlu(Tensor self, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto device = r.isNone(1) ? at::Device(at::DeviceType::MLU) : r.device(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK(device.is_mlu(), "Invalid device, must be mlu device");
  torch::utils::mlu_lazy_init();
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THMPModule_methods[] = {
  {"_initExtension", THPModule_initExtension_mlu, METH_NOARGS, nullptr},
  {"_mlu_init", THMPModule_initExtension, METH_NOARGS, nullptr},
  {"_mlu_getCurrentStream", THMPModule_getCurrentStream_wrap, METH_O, nullptr},
  {"_mlu_getDefaultStream", THMPModule_getDefaultStream_wrap, METH_O, nullptr},
  {"_mlu_setStream", THMPModule_setStream_wrap, METH_O, nullptr},
  {"_mlu_setDevice", THMPModule_setDevice_wrap, METH_O, nullptr},
  {"_mlu_canDeviceAccessPeer", THMPModule_canDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
  {"_mlu_getDevice", THMPModule_getDevice_wrap, METH_NOARGS, nullptr},
  {"_mlu_getDeviceCount", THMPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_mlu_isInBadFork", THMPModule_isInBadFork, METH_NOARGS, nullptr},
  {"mlu", castPyCFunctionWithKeywords(THPVariable_mlu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

void THMPModule_methods(PyObject *module) {
  if (PyModule_AddFunctions(module, _THMPModule_methods) < 0) {
    throw python_error();
  }
}

void THMPModule_initModule() {
  THMPModule_initExtension(nullptr, nullptr);
}

void THMPModule_setDevice(int device) {
  torch_mlu::PythonInterface::setDevice(device);
}

PyObject* THMPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  torch::utils::mlu_lazy_init();
  THMPModule_setDevice(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

void registerMLUDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  pybind11::class_<torch_mlu::DeviceProp>(m, "_MLUDeviceProperties")
    .def_readonly("name", &torch_mlu::DeviceProp::name)
    .def_readonly("major", &torch_mlu::DeviceProp::major)
    .def_readonly("minor", &torch_mlu::DeviceProp::minor)
    .def_readonly("total_memory", &torch_mlu::DeviceProp::total_memory)
    .def("__repr__", [](const torch_mlu::DeviceProp & prop) {
        std::ostringstream stream;
        stream << "_MLUDeviceProperties(name='" << prop.name << "', major=" << prop.major
            << ", minor=" << prop.minor
            << ", total_memory=" << prop.total_memory / (1024 * 1024)
            << "MB)";
        return stream.str();
    });
}
