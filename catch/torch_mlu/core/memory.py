from __future__ import print_function
from typing import Union, Dict, Any, Tuple
import collections
import warnings

import torch
import torch._six
from torch.cuda._memory_viz import segments as _segments, memory as _memory
import torch_mlu
from torch.types import Device
from torch_mlu.utils._utils import _get_device_index

__all__ = ['caching_allocator_alloc', 'caching_allocator_delete', 'set_per_process_memory_fraction',
           'empty_cache', 'memory_stats', 'memory_stats_as_nested_dict', 'reset_accumulated_memory_stats',
           'reset_peak_memory_stats', 'reset_max_memory_allocated', 'reset_max_memory_cached',
           'memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved',
           'memory_cached', 'max_memory_cached', 'memory_snapshot', 'memory_summary', 'mem_get_info',
           'set_memory_strategy', 'get_memory_strategy']

def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Performs a memory allocation using the MLU memory allocator.
    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.mlu.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
        stream (torch.mlu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.
    """
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.mlu.current_stream(device)
    if isinstance(stream, torch.mlu.Stream):
        stream = stream.mlu_stream
    if not isinstance(stream, int):
        raise TypeError('Invalid type for stream argument, must be '
                        '`torch.mlu.Stream` or `int` representing a pointer '
                        'to a exisiting stream')
    with torch.mlu.device(device):
        return torch_mlu._MLUC._mlu_mluCachingAllocator_raw_alloc(size, stream)

def caching_allocator_delete(mem_ptr):
    r"""Deletes memory allocated using the MLU memory allocator.

    Memory allocated with :func:`~torch.mlu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.
    """
    torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete(mem_ptr)

def set_per_process_memory_fraction(fraction, device: Union[Device, int] = None) -> None:
    r"""Set memory fraction for a process.
    The fraction is used to limit an caching allocator to allocated memory on a MLU device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    torch.mlu._lazy_init()
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError('Invalid type for fraction argument, must be `float`')
    if fraction < 0 or fraction > 1:
        raise ValueError('Invalid fraction value: {}. '
                         'Allowed range: 0~1'.format(fraction))

    torch_mlu._MLUC._set_memory_fraction(fraction, device)

def empty_cache() -> None:
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

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cnrtMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cnrtMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the documentation).
    This helps avoid memory framentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cnrtMalloc()``.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)

def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_memoryStats(device)

def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "accumulated" (historical) stats tracked by the MLU memory allocator.
    See :func:`~torch.mlu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetAccumulatedMemoryStats(device)

def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "peak" stats tracked by the MLU memory allocator.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetPeakMemoryStats(device)

def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum MLU memory occupied by
    tensors for a given device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_allocated now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        DeprecationWarning)
    return reset_peak_memory_stats(device=device)

def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum MLU memory managed by the
    caching allocator for a given device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    .. warning::
        This function now calls :func:`~torch.mlu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_cached now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        DeprecationWarning)
    return reset_peak_memory_stats(device=device)

def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the current MLU memory occupied by tensors in bytes for a given
    device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


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
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)

def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the current MLU memory managed by the caching allocator in bytes
    for a given device.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)

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
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)

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

def memory_snapshot():
    r"""Returns a snapshot of the mlu memory allocator state across all devices.
    Interpreting the output of this function requires familiarity with the
    memory allocator internals.
    """
    return torch_mlu._MLUC._mlu_memorySnapshot()

def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Returns the global free and total MLU memory occupied for a given
    device using cnrtMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    return torch_mlu._MLUC._mem_get_info(device)

def set_memory_strategy(native_memory_strategy:bool):
    r"""Deprecated; Use `PYTORCH_MLU_MEMORY_STRATEGY` instead.`
    """
    warnings.warn("torch.mlu.set_memory_strategy is deprecated, "
            "use environment variable `PYTORCH_MLU_MEMORY_STRATEGY` "
            "to set memory strategy instead. For more info, see the "
            "Cambricon PyTorch User Guide.")
    torch_mlu._MLUC._set_memory_strategy(native_memory_strategy)

def get_memory_strategy():
    r"""Returns the memory strategy used by Allocator, `True` represents PyTorch native memory strategy, `False` represents CATCH memory strategy.
    """
    return torch_mlu._MLUC._get_memory_strategy()

# def memory_debug(tensor=None):
#     r"""
#         start memory debugging
#     """
#     if tensor is None:
#         return torch_mlu._MLUC._memory_debug()
#     else:
#         return torch_mlu._MLUC._memory_debug(tensor)


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.
    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return "{:7d} {}".format(sz, prefix)

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return "{:7d} {} ".format(cnt, prefix)

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("reserved_bytes", "MLU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "MLU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch MLU memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append("  {_:9} MLU OOMs: {num_ooms:<12d} | {_:6} cnrtMalloc retries: {num_alloc_retries:<8d}  ")
    lines.append("=" * 75)
    lines.append("        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  ")

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = None, None, None, None

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(" {:<21} | {} | {} | {} | {} ".format(
                submetric_name,
                formatter(current, current_prefval),
                formatter(peak, peak_prefval),
                formatter(allocated, allocated_prefval),
                formatter(freed, freed_prefval)),
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize MLU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(" {:<21} | {} | {} | {} | {} ".format(
            metric_name,
            formatter(current, current),
            formatter(peak, peak),
            formatter(allocated, allocated),
            formatter(freed, freed)),
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"

def _record_memory_history(enabled: bool, device: Union[Device, int] = None):
    with torch.mlu.device(device):
        torch_mlu._MLUC._mlu_recordMemoryHistory(enabled)

def _snapshot(device: Union[Device, int] = None):
    with torch.mlu.device(device):
        return torch_mlu._MLUC._mlu_memorySnapshot()

def _save_segment_usage(filename='output.svg', snapshot=None):
    if snapshot is None:
        snapshot = memory_snapshot()
    with open(filename, 'w') as f:
        f.write(_segments(snapshot))

def _save_memory_usage(filename='output.svg', snapshot=None):
    if snapshot is None:
        snapshot = memory_snapshot()
    with open(filename, 'w') as f:
        f.write(_memory(snapshot))

def _set_allocator_settings(env: str):
    return torch_mlu._MLUC._mlu_mluCachingAllocator_set_allocator_settings(env)

memory_interface = [caching_allocator_alloc, caching_allocator_delete, set_per_process_memory_fraction,
           empty_cache, memory_stats, memory_stats_as_nested_dict, reset_accumulated_memory_stats,
           reset_peak_memory_stats, reset_max_memory_allocated, reset_max_memory_cached,
           memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved,
           memory_cached, max_memory_cached, memory_snapshot, memory_summary, mem_get_info,
           set_memory_strategy, get_memory_strategy, _record_memory_history,
           _snapshot, _save_segment_usage, _save_memory_usage, _set_allocator_settings]

for m in memory_interface:
    torch.mlu.__setattr__(m.__name__, m)
