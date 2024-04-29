#include <torch/csrc/python_headers.h>
#include <structmember.h>
#include <c10/core/CPUAllocator.h>
#include <libshm.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include "python/THMP.h"
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <c10/util/intrusive_ptr.h>
#include "framework/storage/StorageMethods.h"
#include "framework/storage/StorageSharing.h"
#include "framework/core/caching_allocator.h"

PyObject* THMPStorageClass = nullptr;

PyObject* THMPStorage_New(c10::intrusive_ptr<c10::StorageImpl> ptr) {
  AT_ASSERT(ptr);
  PyTypeObject* type = (PyTypeObject*)THMPStorageClass;
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THMPStorage*)obj)->cdata = ptr.release();
  }
  return obj;
}

static void THMPStorage_dealloc(THMPStorage* self) {
  if (self->cdata) {
    c10::raw::intrusive_ptr::decref(self->cdata);
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THMPStorage_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({
      THMPStorageStr "(*, int64_t allocator=None, Device device=None)",
      THMPStorageStr
      "(int64_t size, *, int64_t allocator=None, Device device=None)",
      THMPStorageStr
      "(PyObject* sequence, *, int64_t allocator=None, Device device=None)",
  });
  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  int64_t allocator_arg_idx = 0;
  int64_t device_arg_idx = 1;

  if (r.idx > 0) {
    allocator_arg_idx = 1;
    device_arg_idx = 2;
  }

  c10::optional<int64_t> allocator_opt = r.toInt64Optional(allocator_arg_idx);
  c10::optional<at::Device> device_opt = r.deviceOptional(device_arg_idx);

  TORCH_CHECK(
      !allocator_opt.has_value() || !device_opt.has_value(),
      THMPStorageStr,
      "(): only one or neither of 'allocator' or 'device' can ",
      "be given, but not both");

  THMPStoragePtr self((THMPStorage*)type->tp_alloc(type, 0));
  THPUtils_assert(self, "failed to allocate a " THMPStorageStr " object");
  c10::Allocator* allocator = nullptr;
  at::OptionalDeviceGuard device_guard;
  if (allocator_opt.has_value()) {
    allocator = reinterpret_cast<c10::Allocator*>(allocator_opt.value());
  } else if (device_opt.has_value()) {
    at::Device device = device_opt.value();
    if (device.type() == at::kCPU) {
      allocator = c10::GetDefaultCPUAllocator();
    } else if (device.type() == at::kMLU) {
      at::globalContext().lazyInitMLU();
      allocator = torch_mlu::getMLUCachingAllocator();
    } else {
      TORCH_CHECK(
          false,
          THMPStorageStr,
          "(): This Storage is only suggested applicable to MLU devices, but gives ",
          device.type());
    }
    device_guard.reset_device(device);
  } else {
    allocator = c10::GetDefaultCPUAllocator();
  }

  // torch.Storage(*, ...)
  if (r.idx == 0) {
    self->cdata = c10::make_intrusive<at::StorageImpl>(
                      c10::StorageImpl::use_byte_size_t(),
                      0,
                      allocator,
                      /*resizable=*/true)
                      .release();
    return (PyObject*)self.release();

    // torch.Storage(size, *, ...)
  } else if (r.idx == 1) {
    int64_t size = r.toInt64(0);
    self->cdata = c10::make_intrusive<at::StorageImpl>(
                      c10::StorageImpl::use_byte_size_t(),
                      size,
                      allocator,
                      /*resizable=*/true)
                      .release();
    return (PyObject*)self.release();

    // torch.Storage(sequence, *, ...)
  } else if (r.idx == 2) {
    PyObject* sequence = r.pyobject(0);
    Py_ssize_t length = PySequence_Length(sequence);
    TORCH_CHECK(
        PySequence_Check(sequence),
        THMPStorageStr,
        "(): Expected a sequence type, but got ",
        THPUtils_typename(sequence));
    TORCH_CHECK(
        length >= 0,
        THMPStorageStr,
        "(): Could not obtain the length of sequence of type ",
        THPUtils_typename(sequence));
    self->cdata = c10::make_intrusive<at::StorageImpl>(
                      c10::StorageImpl::use_byte_size_t(),
                      length,
                      allocator,
                      /*resizable=*/true)
                      .release();
    THPObjectPtr item;

    try {
      for (Py_ssize_t i = 0; i < length; i++) {
        item = PySequence_GetItem(sequence, i);
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        uint8_t value = THPByteUtils_unpackReal(item.get());
        if (allocator == c10::GetDefaultCPUAllocator()) {
          self->cdata->unsafe_data<uint8_t>()[i] = value;
        } else {
          // TODO: this might be slow - consider batched updates?
          storage_set(
              at::unsafeStorageFromTH(self->cdata, /*retain=*/true), i, value);
        }
      }
    } catch (const std::exception& e) {
      THPUtils_setError(
          THMPStorageStr
          "(): tried to construct a storage from a sequence (%s), "
          "but one of the items was of type %s instead of int",
          THPUtils_typename(sequence),
          THPUtils_typename(item.get()));
      return nullptr;
    }
    return (PyObject*)self.release();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THMPStorage_length(THMPStorage* self) {
  HANDLE_TH_ERRORS
  return self->cdata->nbytes() / sizeof(uint8_t);
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THMPStorage_get(THMPStorage* self, PyObject* index) {
  HANDLE_TH_ERRORS
  /* Integer index */
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    if (nindex < 0)
      nindex += (self->cdata->nbytes() / sizeof(uint8_t));
    if (nindex < 0 ||
        nindex >=
            static_cast<int64_t>(self->cdata->nbytes() / sizeof(uint8_t))) {
      std::string msg = "index " + std::to_string(nindex) +
          " out of range for storage of size " + std::to_string(self->cdata->nbytes() / sizeof(uint8_t));
      PyErr_SetString(PyExc_IndexError, msg.c_str());

      return nullptr;
    }
    uint8_t value = storage_get(
        at::unsafeStorageFromTH(self->cdata, /*retain=*/true), nindex);
    return THPByteUtils_newReal(value);
    /* Slice index */
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = self->cdata->nbytes() / sizeof(uint8_t);
    if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
      return nullptr;
    if (step != 1) {
      THPUtils_setError(
          "Trying to slice with a step of %lld, but only a step of "
          "1 is supported",
          (long long)step);
      return nullptr;
    }

    uint8_t* data = self->cdata->data<uint8_t>();

    at::StorageImpl* old_storage = self->cdata;
    c10::raw::intrusive_ptr::incref(old_storage);
    auto new_storage = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
        slicelength * sizeof(quantized_t),
#else
        slicelength * sizeof(uint8_t),
#endif
        at::DataPtr(
            static_cast<void*>(data + start),
            old_storage,
            [](void* s) {
              c10::raw::intrusive_ptr::decref(static_cast<at::StorageImpl*>(s));
            },
            old_storage->device()),
        old_storage->allocator(),
        /* resizable */ false);

    PyObject* _ret = THMPStorage_New(std::move(new_storage));
    return _ret;
  }
  PyErr_Format(
      PyExc_TypeError,
      "can't index a " THMPStorageStr " with %s",
      THPUtils_typename(index));
  return nullptr;
  END_HANDLE_TH_ERRORS
}

static int THMPStorage_set(THMPStorage* self, PyObject* index, PyObject* value) {
  HANDLE_TH_ERRORS
  if (!THPByteUtils_checkReal(value)) {
    THPUtils_setError(
        "can only set storage content with a int types, but got "
        "%s instead",
        THPUtils_typename(value));
    return -1;
  }
  uint8_t rvalue = THPByteUtils_unpackReal(value);
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    storage_set(
        at::unsafeStorageFromTH(self->cdata, /*retain=*/true), nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = self->cdata->nbytes() / sizeof(uint8_t);
    if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
      return -1;
    if (step != 1) {
      THPUtils_setError(
          "Trying to slice with a step of %lld, but only a step of "
          "1 is supported",
          (long long)step);
      return 0;
    }
    for (; start < stop; start++)
      storage_set(
          at::unsafeStorageFromTH(self->cdata, /*retain=*/true), start, rvalue);
    return 0;
  }
  THPUtils_setError(
      "can't index a " THMPStorageStr " with %s", THPUtils_typename(index));
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THMPStorage_mappingmethods = {
    (lenfunc)THMPStorage_length,
    (binaryfunc)THMPStorage_get,
    (objobjargproc)THMPStorage_set};

// TODO: implement equality
PyTypeObject THMPStorageType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch_mlu._MLUC." THMPStorageBaseStr, /* tp_name */
    sizeof(THMPStorage), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THMPStorage_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THMPStorage_mappingmethods, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr,
    /* will be assigned in init */ /* tp_methods */
    nullptr,
    /* will be assigned in init */ /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THMPStorage_pynew, /* tp_new */
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMemberDef THMPStorage_members[] = {
    {(char*)"_cdata",
     T_ULONGLONG,
     offsetof(THMPStorage, cdata),
     READONLY,
     nullptr},
    {nullptr}};

static PyObject* THMPStorage_device(THMPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata->device());
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THMPStorage_properties[] = {
    {"device", (getter)THMPStorage_device, nullptr, nullptr, nullptr},
    {nullptr}};

bool THMPStorage_init(PyObject* module) {
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THMPStorage_getMethods());
  THPUtils_addPyMethodDefs(methods, THMPStorage_getSharingMethods());

  THMPStorageType.tp_methods = methods.data();
  THMPStorageType.tp_members = THMPStorage_members;
  THMPStorageType.tp_getset = THMPStorage_properties;
  if (PyType_Ready(&THMPStorageType) < 0)
    return false;
  Py_INCREF(&THMPStorageType);
  PyModule_AddObject(module, THMPStorageBaseStr, (PyObject*)&THMPStorageType);
  return true;
}

void THMPStorage_postInit(PyObject* module) {
  THMPStorageClass = PyObject_GetAttrString(module, "UntypedStorage");
  if (!THMPStorageClass)
    throw python_error();
}
