from __future__ import print_function
from typing import Union, Tuple, Optional, TypeVar, Any

import ctypes
import warnings
import pickle
import threading
import traceback
import torch
import torch_mlu
from typing import Union, Dict, Any
from torch_mlu.core.device import Device  # pylint: disable=W0611
from torch_mlu.core.device import Device as _device  # pylint: disable=W0611, W0404
from torch_mlu.utils._utils import _get_device_index

default_generators: Tuple[torch._C.Generator] = ()

#### Memory management
def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other MLU application and visible in
    `cnmon info`.
    """
    torch_mlu._MLUC._empty_cache()

def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns a dictionary of MLU memory allocator statistics for a
    given device.
    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_memoryStats(device)

def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the current MLU memory occupied by tensors in bytes for a given
    device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device)["allocated_bytes.all.current"]

def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum MLU memory occupied by tensors in bytes for a given
    device.
    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.mlu.reset_peak_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device)["allocated_bytes.all.peak"]

def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the current MLU memory managed by the caching allocator in bytes
    for a given device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device)["reserved_bytes.all.current"]

def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum MLU memory managed by the caching allocator in bytes
    for a given device.
    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.mlu.reset_peak_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device)["reserved_bytes.all.peak"]

def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.memory_reserved`."""
    warnings.warn(
        "torch.mlu.memory_cached has been renamed to torch.mlu.memory_reserved",
        DeprecationWarning)
    return memory_reserved(device=device)

def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.max_memory_reserved`."""
    warnings.warn(
        "torch.mlu.max_memory_cached has been renamed to torch.mlu.max_memory_reserved",
        DeprecationWarning)
    return max_memory_reserved(device=device)

### Torch Module

T = TypeVar('T', bound='Module')


def module_mlu(self: T, device: Optional[Union[int, _device]] = None) -> T:
    if device is None:
        device = torch.mlu.current_device()
    device = torch.device(device)
    device = torch.device('mlu', device.index)
    return self._apply(lambda t: t.to(device))

torch.nn.Module.mlu = module_mlu

### Torch Tensor Dtype

tensor_dtype_dict = {
    "DoubleTensor": torch.float64,
    "FloatTensor": torch.float32,
    "HalfTensor": torch.float16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.int16,
    "IntTensor": torch.int32,
    "LongTensor": torch.int64,
    "BoolTensor": torch.bool,
    "BFloat16Tensor": torch.bfloat16
}

def dtype_tensor_wrap(func):
    def wrap(*args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = torch.device('mlu')
        else:
            kwargs['device'] = torch.device(kwargs['device'])
        if kwargs['device'].type != 'mlu':
            raise RuntimeError("legacy constructor expects device type: mlu")
        if len(args) == 0:
            torch.tensor([], device='mlu')
        if isinstance(args[0], list):
            kwargs['dtype'] = tensor_dtype_dict[func.__name__]
            return torch.tensor(*args, **kwargs)
        kwargs['dtype'] = tensor_dtype_dict[func.__name__]
        return torch.empty(args, **kwargs)

    return wrap

### Random sampling


def manual_seed(seed):
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    torch.mlu.manual_seed(seed)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers for the current MLU.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-MLU model, this function is insufficient
        to get determinism.  To seed all MLUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)
    torch.mlu.manual_seed_all(seed)


### Torch MLU


def is_mlu_tensor(tensor):
    return tensor.device.type == 'mlu'


def mlu_device(n=None, devkind=None):  # pylint:disable=W0613
    return torch.device('mlu')

def get_device():
    return torch_mlu._MLUC._get_device()

def empty_cached_memory():
    r"""
        cnrtFree all cached memory
    """
    torch_mlu._MLUC._empty_cached_memory()


def to(optimizer, device):
    for state_perparam in optimizer.state.values():
        for k, v in state_perparam.items():
            if isinstance(v, torch.Tensor):
                state_perparam[k] = v.to(device)
    return optimizer


def prepare_save(model, optimizer=None, device=torch.device('mlu')):
    cpu = torch.device('cpu')
    mlu = torch.device('mlu')
    model.to(cpu)
    if optimizer is not None:
        to(optimizer, cpu)

def set_memory_strategy(native_memory_strategy:bool):
    torch_mlu._MLUC._set_memory_strategy(native_memory_strategy)

def memory_debug(tensor=None):
    r"""
        start memory debugging
    """
    if tensor is None:
        return torch_mlu._MLUC._memory_debug()
    else:
        return torch_mlu._MLUC._memory_debug(tensor)

torch.mlu.__setattr__("default_generators", default_generators)

### torch.backends
#torch.backends.__setattr__("mlu", torch_mlu.core.backends.mlu)
#torch.backends.__setattr__("cnnl", CnnlModule(torch_mlu.core.backends.cnnl, "torch.backends.cnnl"))
