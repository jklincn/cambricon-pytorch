
import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import TEST_NUMBA, IS_WINDOWS, TEST_WITH_ROCM
import inspect
import contextlib
from distutils.version import LooseVersion

TEST_MLU = torch.mlu.is_available()
TEST_MULTIMLU = TEST_MLU and torch.mlu.device_count() >= 2