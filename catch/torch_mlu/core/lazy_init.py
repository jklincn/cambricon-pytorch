from __future__ import print_function

import ctypes
import warnings
import pickle
import threading
import traceback
import torch
import torch_mlu

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch_mlu._MLUC, "_mlu_isInBadFork", lambda: False)

### torch.mlu
def init():
    r"""Initialize PyTorch's MLU state.
    """
    _lazy_init()

def is_initialized():
    r"""Returns whether PyTorch's MLU state has been initialized."""
    return _initialized and not _is_in_bad_fork()

def _lazy_call(callable):
    if is_initialized():
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))

def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                    "Cannot re-initialize MLU in forked subprocess. To use MLU with multiprocessing, you must use the "
                    "'spawn' start method")
        torch_mlu._MLUC._mlu_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = ("MLU call failed lazily at initialization with error: {}\n\n"
                    "MLU call was originally invoked at:\n\n{}").format(str(e), orig_traceback)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True

flist = [init, is_initialized, _lazy_init, _lazy_call]
for f in flist:
    torch.mlu.__setattr__(f.__name__, f)
torch.mlu.__setattr__("_is_in_bad_fork", _is_in_bad_fork)
torch.mlu.__setattr__("_initialized", _initialized)
