import os
from typing import Optional, List, Tuple, Union, Any, cast, Sequence
import contextlib
import functools
import unittest
import inspect
import torch
import torch_mlu
from functools import wraps
from torch.testing._internal.common_device_type import (onlyOn, skipIf, OpDTypes)
from torch.testing._internal.common_device_type import dtypes as org_dtypes
from torch.testing._internal.common_device_type import ops as org_ops

RUN_MLU = torch.is_mlu_available()
RUN_MLU_HALF = RUN_MLU
RUN_MLU_MULTI_MLU = RUN_MLU and torch.mlu.device_count() > 1
TEST_MLU = torch.is_mlu_available()
TEST_MULTIMLU = TEST_MLU and torch.mlu.device_count() >= 2
MLU_DEVICE = torch.device("mlu:0") if TEST_MLU else None

def onlyMLU(fn):
    return onlyOn('mlu')(fn)

def onlyOnCPUAndMLU(fn):
    @wraps(fn)
    def only_fn(self, device, *args, **kwargs):
        if self.device_type != 'cpu' and self.device_type != 'mlu':
            reason = "onlyOnCPUAndMLU: doesn't run on {0}".format(self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, device, *args, **kwargs)

    return only_fn

# Skips a test on MLU.
class skipMLU(skipIf):

    def __init__(self, reason):
        super().__init__(True, reason, device_type='mlu')

# Skips a test on MLU if the condition is true.
class skipMLUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='mlu')

class MLUDtypeBlackList(object):
    def __init__(self):
        self.dtypes_map = {'uint8': torch.uint8,
                           'int8': torch.int8,
                           'int16': torch.int16,
                           'int32': torch.int32,
                           'int64': torch.int64,
                           'float16': torch.float16,
                           'float32': torch.float32,
                           'float64': torch.float64,
                           'bfloat16': torch.bfloat16,
                           'complex64': torch.complex64,
                           'complex128': torch.complex128,
                           'bool': torch.bool}
        # Default
        self.dtype_black_list = [torch.float64, torch.int64, torch.complex128]
        env_black_list = os.getenv('MLU_DTYPE_BLACK_LIST', 'null')
        if env_black_list != 'null':
            def rewrite_black_list():
                new_black_list = []
                for type_str in env_black_list.split(','):
                    assert type_str in self.dtypes_map.keys(), \
                        "MLU_DTYPE_BLACK_LIST only support keys: {}. But got {}.".format(
                                                                self.dtypes_map.keys(), type_str)
                    new_black_list.append(self.dtypes_map[type_str])
                return new_black_list

            self.dtype_black_list = rewrite_black_list() if env_black_list else []

    def all(self):
        return self.dtype_black_list

    def has(self, dtype):
        if dtype in self.dtype_black_list:
            return True
        return False

    def opposite(self):
        white_list = []
        for dtype in self.dtypes_map.values():
            if dtype not in self.dtype_black_list:
                white_list.append(dtype)
        return white_list

global_mlu_black_list = MLUDtypeBlackList()

MLU_TEST_LONG_AND_DOUBLE = False if (global_mlu_black_list.has(torch.int64) \
                                    or global_mlu_black_list.has(torch.float64)) else True

class ops(org_ops):
    def __init__(self, op_list, *, dtypes: Union[OpDTypes, Sequence[torch.dtype]] = OpDTypes.supported,
                 allowed_dtypes: Optional[Sequence[torch.dtype]] = None):
        mlu_allowed_dtypes = set(global_mlu_black_list.opposite())
        final_allowed_dtypes = None if dtypes == OpDTypes.none else \
                               mlu_allowed_dtypes if allowed_dtypes is None else \
                               mlu_allowed_dtypes.intersection(set(allowed_dtypes))
        super().__init__(op_list, dtypes=dtypes, allowed_dtypes=final_allowed_dtypes)


class dtypes(org_dtypes):

    def __init__(self, *args, device_type="all"):

        def filter_dtypes():
            dtype_black_list = global_mlu_black_list.all()
            dtypes = []
            for dtype in args:
                if type(dtype) is not torch.dtype:
                    # ((dtype1, dtype2), (dtype1, dtype3), ...)
                    inter_set = [dt for dt in dtype if dt in dtype_black_list]
                    if len(inter_set) == 0:
                        dtypes.append(dtype)
                elif dtype not in dtype_black_list:
                    # (dtype1, dtype2, dtype3, ...)
                    dtypes.append(dtype)
            return dtypes

        super().__init__(*filter_dtypes(), device_type=device_type)

# Overrides specified dtypes on MLU.
class dtypesIfMLU(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='mlu')



# Test whether hardware TF32 math mode enabled. It is enabled only on:
# >= MLU500
def tf32_is_not_fp32():
    if not torch.mlu.is_available():
        return False
    if torch.mlu.get_device_properties(torch.mlu.current_device()).major < 5:
        return False
    return True


@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = torch.backends.mlu.matmul.allow_tf32
    try:
        torch.backends.mlu.matmul.allow_tf32 = False
        with torch.backends.cnnl.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.mlu.matmul.allow_tf32 = old_allow_tf32_matmul


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_allow_tf32_matmul = torch.backends.mlu.matmul.allow_tf32
    old_precison = self.precision
    try:
        torch.backends.mlu.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cnnl.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            yield
    finally:
        torch.backends.mlu.matmul.allow_tf32 = old_allow_tf32_matmul
        self.precision = old_precison


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as pecified by the
# argument. For example:
#    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
#    @tf32_on_and_off(0.005)
#    def test_matmul(self, device, dtype):
#        a = ...; b = ...;
#        c = torch.matmul(a, b)
#        self.assertEqual(c, expected)
# In the above example, when testing torch.float32 and torch.complex64 on MLU
# on MLU590, the matmul will be running at
# TF32 mode and TF32 mode off, and on TF32 mode, the assertEqual will use reduced
# precision to check values.
#
# This decorator can be used for function with or without device/dtype, such as
# @tf32_on_and_off(0.005)
# def test_my_op(self)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device, dtype)
# @tf32_on_and_off(0.005)
# def test_my_op(self, dtype)
# if device is specified, it will check if device is mlu
# if dtype is specified, it will check if dtype is float32 or complex64
# tf32 and fp32 are different only when all the three checks pass
def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        # with tf32_on(self, tf32_precision):
        #     function_call()
        org_precison = self.precision
        self.precision = tf32_precision
        function_call()
        self.precision = org_precison

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            cond = tf32_is_not_fp32()
            if 'device' in kwargs:
                cond = cond and (torch.device(kwargs['device']).type == 'mlu')
            if 'dtype' in kwargs:
                cond = cond and (kwargs['dtype'] in {torch.float32, torch.complex64})
            if cond:
                # TODO(PYTORCH-8982): Need to implement cudnn.flag()
                # Only test tf32 on mlu590
                # with_tf32_disabled(kwargs['self'], lambda: f(**kwargs))
                with_tf32_enabled(kwargs['self'], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped
    return wrapper


# This is a wrapper that wraps a test to run it with TF32 turned off.
# This wrapper is designed to be used when a test uses matmul or convolutions
# but the purpose of that test is not testing matmul or convolutions.
# Disabling TF32 will enforce torch.float tensors to be always computed
# at full precision.
def with_tf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped


# override from torch/testing/_legacy.py
def get_all_device_types() -> List[str]:
    return ['cpu'] if not torch.is_mlu_available() else ['cpu', 'mlu']

