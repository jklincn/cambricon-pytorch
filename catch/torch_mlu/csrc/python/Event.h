#ifndef THMP_EVENT_INC
#define THMP_EVENT_INC

#include <torch/csrc/python_headers.h>
#include "framework/core/notifier.h"
#include "python/Stream.h"

struct THMPEvent {
  PyObject_HEAD
  torch_mlu::Notifier mlu_event;
};
extern PyObject *THMPEventClass;

void THMPEvent_init(PyObject *module);

inline bool THMPEvent_Check(PyObject* obj) {
  return THMPEventClass && PyObject_IsInstance(obj, THMPEventClass);
}

#endif  // THMP_EVENT_INC

