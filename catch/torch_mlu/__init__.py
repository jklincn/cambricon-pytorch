import os

import torch
from torch._utils import classproperty
from torch_mlu import _MLUC
from torch_mlu.core import mlu_model
from torch_mlu.utils._utils import _get_device_index
from torch_mlu.core import lazy_init
from torch_mlu.core import memory
from torch_mlu.core.backends.cnnl import CnnlModule
from torch_mlu.core.backends.mlufusion import MlufusionModule
from torch_mlu.utils import random
from torch_mlu.utils.counter import _check_gencase

_check_gencase()

def get_version():
    return _MLUC._get_version()

__version__=get_version()

from torch_mlu.core import mlu_model
import torch_mlu.optimizers
import torch_mlu.amp
import torch_mlu.profiler


################################################################################
# Define Storage
################################################################################

from .storage import UntypedStorage
from torch.storage import _LegacyStorage

class _MluLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        raise RuntimeError('from_buffer: Not available for MLU storage')

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError('_new_with_weak_ptr: Not available for MLU storage')

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError('_new_shared_filename: Not available for MLU storage')

class ByteStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.uint8

class DoubleStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.double

class FloatStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.float

class HalfStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.half

class LongStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.long

class IntStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int

class ShortStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.short

class CharStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int8

class BoolStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bool

class BFloat16Storage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bfloat16

class ComplexDoubleStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.cdouble

class ComplexFloatStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.cfloat

del _LegacyStorage
del _MluLegacyStorage

_mlu_storage_classes = [UntypedStorage, DoubleStorage, FloatStorage, LongStorage, IntStorage, ShortStorage, CharStorage, ByteStorage,
                        HalfStorage, BoolStorage, BFloat16Storage, ComplexDoubleStorage, ComplexFloatStorage]
for r in _mlu_storage_classes:
    torch._storage_classes.add(r)
    torch.mlu.__setattr__(r.__name__, r)

### add torch.mlu.random module
torch.mlu.__setattr__("random", torch_mlu.utils.random)

### torch.backends
torch._C._set_cublas_allow_tf32 = torch_mlu.core.backends.mlu.fake_set_cublas_allow_tf32
torch._C._set_cublas_allow_fp16_reduced_precision_reduction = torch_mlu.core.backends.mlu.fake_set_cublas_allow_fp16_reduced_precision_reduction
torch.backends.__setattr__("mlu", torch_mlu.core.backends.mlu)
torch.backends.__setattr__("cnnl", CnnlModule(torch_mlu.core.backends.cnnl, "torch.backends.cnnl"))
torch.backends.__setattr__("mlufusion", MlufusionModule(torch_mlu.core.backends.mlufusion, "torch.backends.mlufusion"))

_MLUC._initExtension()

torch.version.mlu = None
