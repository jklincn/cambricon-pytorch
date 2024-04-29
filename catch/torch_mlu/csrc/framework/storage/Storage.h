#ifndef THMP_STORAGE_INC
#define THMP_STORAGE_INC

#include <torch/csrc/Types.h>
#define THMPStorageStr "torch.mlu.UntypedStorage"
#define THMPStorageBaseStr "StorageBase"
#define THMPStoragePtr THPStoragePtr
#define THMPStorage THPStorage

TORCH_PYTHON_API PyObject* THMPStorage_New(
    c10::intrusive_ptr<c10::StorageImpl> ptr);
extern PyObject* THMPStorageClass;

bool THMPStorage_init(PyObject* module);
void THMPStorage_postInit(PyObject* module);

extern PyTypeObject THMPStorageType;

#endif
