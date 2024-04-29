#include <pybind11/pybind11.h>

extern void initMLUModule(PyObject* m);

PYBIND11_MODULE(_MLUC, m) { initMLUModule(m.ptr()); }
