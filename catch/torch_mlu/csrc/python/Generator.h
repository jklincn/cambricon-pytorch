#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/utils/object_ptr.h>

// Creates a new Python object wrapping the default at::Generator. The reference
// is borrowed. The caller should ensure that the at::Generator object lifetime
// last at least as long as the Python wrapper.
TORCH_PYTHON_API PyObject* THCGenerator_initDefaultGenerator(
    at::Generator cdata);

bool THCGenerator_init(PyObject* module);
