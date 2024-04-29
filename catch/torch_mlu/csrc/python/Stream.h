#ifndef THMP_STREAM_INC
#define THMP_STREAM_INC

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
// #include <THC/THC.h>
#include "framework/core/queue.h"

struct THMPStream : THPStream{
  torch_mlu::Queue mlu_stream;
};
extern PyObject *THMPStreamClass;

void THMPStream_init(PyObject *module);

inline bool THMPStream_Check(PyObject* obj) {
  return THMPStreamClass && PyObject_IsInstance(obj, THMPStreamClass);
}

#endif  // THMP_STREAM_INC
