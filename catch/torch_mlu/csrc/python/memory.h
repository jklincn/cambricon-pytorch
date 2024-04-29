#pragma once
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_compat.h>
#include "framework/core/caching_allocator.h"

struct Frame {
  PyCodeObject* code;
  int lasti;
};

struct StackContext : public torch_mlu::CachingAllocatorContext {
  std::vector<Frame> frames;
  ~StackContext() {
    py::gil_scoped_acquire acquire;
    for (auto& f : frames) {
      Py_XDECREF((PyObject*)f.code);
    }
  }
  static std::unique_ptr<torch_mlu::CachingAllocatorContext> gather() {
    py::gil_scoped_acquire acquire;
    auto r = std::make_unique<StackContext>();
    PyFrameObject* f = PyEval_GetFrame();
    Py_XINCREF(f);
    while (f) {
      r->frames.emplace_back(Frame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
    }
    return r;
  }
};

py::dict mlu_memoryStats(int device);

py::list mlu_memorySnapshot();

void mlu_recordMemoryHistory(bool enabled);

void* mluCachingAllocator_raw_alloc(size_t size, cnrtQueue_t queue);

void mluCachingAllocator_raw_delete(void* mem_ptr);
