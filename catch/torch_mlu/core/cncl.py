import sys
from typing import Union
import torch
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple
)
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group, get_backend, _GLOO_AVAILABLE
from torch._C._distributed_c10d import _ProcessGroupWrapper
import torch_mlu

thismodule = sys.modules[__name__]

_CNCL_AVAILABLE = True

def version():
    major, minor, patch= torch_mlu._MLUC._cncl_version()
    return (major, minor, patch)


def is_cncl_available():
    return _CNCL_AVAILABLE


_mlu_streams = None
def _get_mlu_stream(device: int):
    global _mlu_streams
    if device == -1:
        return None
    if _mlu_streams is None:
        _mlu_streams = [None] * torch.mlu.device_count()
    if _mlu_streams[device] is None:
        _mlu_streams[device] = torch.mlu.Stream(device)
    return _mlu_streams[device]


def _recursive_to(inputs, target_mlu, use_side_stream_for_tensor_copies):
    r"""
    Recursively moves input to the target_mlu.
    """

    def to_map(obj):
        if isinstance(obj, torch.Tensor):
            if obj.device == torch.device("mlu", target_mlu):
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_mlu),)
            else:
                # Perform CPU -> MLU copies in a background stream.
                stream = _get_mlu_stream(target_mlu)
                with torch.mlu.stream(stream):
                    output = obj.to(target_mlu)
                with torch.mlu.device(target_mlu):
                    current_stream = torch.mlu.current_stream()
                    current_stream.wait_stream(stream)
                    output.record_stream(current_stream)
                return (output,)                

        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]
    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res

def _check_for_cncl_backend(group):
    pg = group or _get_default_group()
    # Gate PG wrapper check on Gloo availability.
    if _GLOO_AVAILABLE:
        # It is not expected for PG to be wrapped many times, but support it just
        # in case
        while isinstance(pg, _ProcessGroupWrapper):
            pg = pg.wrapped_pg
    return get_backend(pg) == "cncl"


def _get_pg_device(group: ProcessGroup):
    """
    Returns the device to use with ``group``.
    This is mlu for CNCL and CPU for everything else
    """
    if _check_for_cncl_backend(group):
        return torch.device('mlu', torch.mlu.current_device())
    return torch.device("cpu")

torch.distributed.distributed_c10d.__setattr__("_get_pg_device", _get_pg_device)
torch.distributed.utils.__setattr__("_recursive_to", _recursive_to)

torch.mlu.__setattr__("cncl", thismodule)
torch.distributed.__setattr__("is_cncl_available", is_cncl_available)

if hasattr(torch_mlu._MLUC, 'ProcessGroupCNCL'):
    from torch_mlu._MLUC import ProcessGroupCNCL
    torch.distributed.__setattr__("ProcessGroupCNCL", ProcessGroupCNCL)
    ProcessGroupCNCL.__module__ = "torch.distributed.distributed_c10d"

