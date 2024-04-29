#include <atomic>
#include <string>
#include <torch/csrc/python_headers.h>
#include <structmember.h>
#include <c10/core/CPUAllocator.h>
#include <libshm.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include "python/THMP.h"
#include <torch/csrc/copy_utils.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/MapAllocator.h>
#include <torch/csrc/utils/python_numbers.h>
#include "framework/storage/StorageSharing.h"
#include "framework/storage/Storage.h"

static PyObject* THMPStorage_sharedDecref(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THMPStorage*)_self;
  c10::DeviceType device_type = self->cdata->device_type();
  if (device_type == at::kCPU) {
    c10::StorageImpl* storage = self->cdata;
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage->data_ptr());
    if (ctx) {
      ctx->decref();
    }
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_sharedIncref(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THMPStorage*)_self;
  c10::DeviceType device_type = self->cdata->device_type();
  if (device_type == at::kCPU) {
    c10::StorageImpl* storage = self->cdata;
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage->data_ptr());
    if (ctx) {
      ctx->incref();
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_pyNewFilenameStorage(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }

  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
  std::string handle = at::NewProcessWideShmHandle();
  // For cpu-only methods, return a THPStorage type object.
  return THPStorage_New(c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      THManagedMapAllocator::makeDataPtr("", handle.c_str(), flags, size),
      /*allocator=*/nullptr,
      /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_shareFilename(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      reinterpret_cast<THMPStorage*>(_self)->cdata->device_type() == at::kCPU,
      "_share_filename_: only available on CPU");
  auto self = (THMPStorage*)_self;
  c10::StorageImpl* storage = self->cdata;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  THManagedMapAllocator* ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
    std::string handle = at::NewProcessWideShmHandle();
    at::Storage new_storage(c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        storage->nbytes(),
        THManagedMapAllocator::makeDataPtr(
            "", handle.c_str(), flags, storage->nbytes()),
        /*allocator=*/nullptr,
        /*resizable=*/false));

    at::Storage _self_aten = torch::createStorage(_self);
    {
      // Copying into shared memory can be slow, so release the GIL
      pybind11::gil_scoped_release no_gil;
      storage_copy(new_storage, _self_aten);
    }

    std::swap(*storage, *new_storage.unsafeGetStorageImpl());
    ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle()));
  if (!manager_handle)
    return nullptr;
  THPObjectPtr storage_handle(PyBytes_FromString(ctx->filename()));
  if (!storage_handle)
    return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage->nbytes() / sizeof(uint8_t)));
  if (!size)
    return nullptr;

  THPObjectPtr tuple(PyTuple_New(3));
  if (!tuple)
    return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_newSharedFilename(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  PyObject* _manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject* _object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size = PyTuple_GET_ITEM(args, 2);
  if (!PyBytes_Check(_manager_handle) || !PyBytes_Check(_object_handle) ||
      !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file system mode",
        1,
        "a handle (string/bytes) and storage size (int)");
    return nullptr;
  }
  const char* manager_handle = PyBytes_AS_STRING(_manager_handle);
  const char* object_handle = PyBytes_AS_STRING(_object_handle);
  int64_t size = THPUtils_unpackLong(_size);
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  // For cpu-only methods, return a THPStorage type object.
  return THPStorage_New(c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      THManagedMapAllocator::makeDataPtr(
          manager_handle, object_handle, flags, size),
      /*allocator=*/nullptr,
      /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static c10::intrusive_ptr<c10::StorageImpl> THMPStorage_newFdStorage(
    ptrdiff_t size) {
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE |
      at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_UNLINK;
  std::string handle = at::NewProcessWideShmHandle();
  auto sptr = at::MapAllocator::makeDataPtr(
      handle.c_str(), flags, size * sizeof(uint8_t), nullptr);
  return c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      std::move(sptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
}

static PyObject* THMPStorage_pyNewFdStorage(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  // For cpu-only methods, return a THPStorage type object.
  return THPStorage_New(THMPStorage_newFdStorage(size));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_shareFd(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      reinterpret_cast<THMPStorage*>(_self)->cdata->device_type() == at::kCPU,
      "_share_fd_: only available on CPU");
  auto self = (THMPStorage*)_self;
  c10::StorageImpl* storage = self->cdata;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  at::MapAllocator* ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = at::MapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    at::Storage new_storage(THMPStorage_newFdStorage(storage->nbytes()));
    at::Storage _self_aten = torch::createStorage(_self);
    {
      // Copying into shared memory can be slow, so release the GIL
      pybind11::gil_scoped_release no_gil;
      storage_copy(new_storage, _self_aten);
    }

    std::swap(*storage, *new_storage.unsafeGetStorageImpl());
    ctx = at::MapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr storage_handle(THPUtils_packInt32(ctx->fd()));
  if (!storage_handle)
    return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage->nbytes() / sizeof(uint8_t)));
  if (!size)
    return nullptr;

  THPObjectPtr tuple(PyTuple_New(2));
  if (!tuple)
    return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_newSharedFd(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* _tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject* _size = PyTuple_GET_ITEM(args, 1);
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file descriptor mode",
        1,
        "a file descriptor (int) and storage size (int)");
    return nullptr;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int fd;
  int tmp_fd = (int)THPUtils_unpackLong(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  if ((fd = dup(tmp_fd)) == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return nullptr;
  }

  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE |
      at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_FROMFD;
  // For cpu-only methods, return a THPStorage type object.
  return THPStorage_New(c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      at::MapAllocator::makeDataPtr(at::WITH_FD, "", fd, flags, size, nullptr),
      /*allocator=*/nullptr,
      /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_weakRef(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto self = (THMPStorage*)_self;
  c10::StorageImpl* storage = self->cdata;
  return PyLong_FromVoidPtr(c10::raw::intrusive_ptr::make_weak(storage));
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_newWithWeakPtr(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "_new_with_weak_ptr(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  if (auto* storage = c10::raw::weak_intrusive_ptr::lock(weak_storage)) {
    return THMPStorage_New(
        c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_freeWeakRef(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (arg == Py_None) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg), "_free_weak_ref(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  c10::raw::weak_intrusive_ptr::decref(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_expired(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "_expired(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  return PyBool_FromLong(
      c10::raw::weak_intrusive_ptr::use_count(weak_storage) == 0);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_sharedFd(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THMPStorage*)_self;
  at::MapAllocator* ctx = nullptr;
  if (self->cdata->device_type() == at::kCPU) {
    c10::StorageImpl* storage = self->cdata;
    ctx = at::MapAllocator::fromDataPtr(storage->data_ptr());
  }

  THPUtils_assert(ctx, "couldn't retrieve a shared file descriptor");
  return THPUtils_packInt32(ctx->fd());
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_isShared(PyObject* _self, PyObject* noargs) {
  auto self = (THMPStorage*)_self;
  if (self->cdata->device_type() == at::kMLU) {
    Py_RETURN_TRUE;
  }
  if (at::MapAllocator::fromDataPtr(self->cdata->data_ptr()) ||
      THManagedMapAllocator::fromDataPtr(self->cdata->data_ptr())) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THMPStorage_sharingMethods[] = {
    {"_new_with_weak_ptr",
     THMPStorage_newWithWeakPtr,
     METH_O | METH_CLASS,
     nullptr},
     //TODO(mengpenghui):To be adapted on MLU.
     /*
    {"_share_cuda_", THPStorage_shareCuda, METH_NOARGS, nullptr},
    {"_new_shared_cuda",THPStorage_newSharedCuda,METH_VARARGS | METH_STATIC, nullptr},
    {"_release_ipc_counter_cuda", THPStorage_releaseIPCCounter, METH_VARARGS | METH_STATIC, nullptr},
    */
    {"_share_fd_cpu_", THMPStorage_shareFd, METH_NOARGS, nullptr},
    {"_new_shared_fd_cpu",
     THMPStorage_newSharedFd,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_fd_cpu",
     THMPStorage_pyNewFdStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_share_filename_cpu_", THMPStorage_shareFilename, METH_NOARGS, nullptr},
    {"_new_shared_filename_cpu",
     THMPStorage_newSharedFilename,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_filename_cpu",
     THMPStorage_pyNewFilenameStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_weak_ref", THMPStorage_weakRef, METH_NOARGS, nullptr},
    {"_free_weak_ref", THMPStorage_freeWeakRef, METH_O | METH_STATIC, nullptr},
    {"_expired", THMPStorage_expired, METH_O | METH_STATIC, nullptr},
    {"_shared_decref", THMPStorage_sharedDecref, METH_NOARGS, nullptr},
    {"_shared_incref", THMPStorage_sharedIncref, METH_NOARGS, nullptr},
    {"_get_shared_fd", THMPStorage_sharedFd, METH_NOARGS, nullptr},
    {"is_shared", THMPStorage_isShared, METH_NOARGS, nullptr},
    {nullptr}};

PyMethodDef* THMPStorage_getSharingMethods() {
  return THMPStorage_sharingMethods;
}
