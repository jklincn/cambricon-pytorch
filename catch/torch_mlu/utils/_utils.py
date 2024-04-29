import torch
import types
from typing import Optional
from typing import Any

def _get_device_index(device: Any, optional: bool = False,
                      allow_cpu: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MLU device. Note that for a MLU device without a specified index,
    i.e., ``torch.device('mlu')``, this will return the current default MLU
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MLU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["mlu", "cpu"]:
                raise ValueError('Expected a mlu or cpu device, but got: {}'.format(device))
        elif device.type != 'mlu':
            raise ValueError('Expected a mlu device, but got: {}'.format(device))
    if not torch.jit.is_scripting():
        if isinstance(device, torch.mlu.device):
            return device.idx
    return torch._utils._get_device_index(device, optional, allow_cpu)

def get_current_device_index() -> int:
    r"""Checks if there are MLU devices available and
    returns the device index of the current default MLU device.
    Returns -1 in case there are no MLU devices available.
    Arguments: ``None``
    """
    if torch.mlu.device_count() > 0:
        return torch.mlu.current_device()
    return -1

def _get_available_device_type():
    if torch.mlu.is_available():
        return "mlu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    return None

def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == "mlu":
        return get_member(torch.mlu)
    if device_type and device_type.lower() == "xpu":
        return get_member(torch.xpu)  # type: ignore[attr-defined]
    return None

# monkey patch three functions called by torch._utils._get_device_index
torch._utils.__setattr__("get_current_device_index", get_current_device_index)
torch._utils.__setattr__("_get_device_attr", _get_device_attr)
torch._utils.__setattr__("_get_available_device_type", _get_available_device_type)
# same with torch.cuda._utils._get_device_index
torch.mlu.__setattr__("_utils", types.ModuleType("_utils"))
torch.mlu._utils.__setattr__("_get_device_index", _get_device_index)