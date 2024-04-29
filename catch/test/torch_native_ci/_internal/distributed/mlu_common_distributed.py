
import torch
import torch_mlu
import sys
import os
import torch.distributed as c10d
from functools import (
    wraps
)
from typing import NamedTuple, Optional, Union


from torch.testing._internal.common_utils import (
    TestCase,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    FILE_SCHEMA,
    find_free_port,
    retry_on_connect_failures,
    IS_SANDCASTLE,
    sandcastle_skip_if,
    sandcastle_skip,
)

class TestSkip(NamedTuple):
    exit_code: int
    message: str

TEST_SKIPS = {
    "backend_unavailable": TestSkip(
        72, "Skipped because distributed backend is not available."
    ),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "odd_worldsize": TestSkip(87, "Skipped due to odd world size."),
    "no_mlu": TestSkip(74, "MLU is not available."),
    "multi-mlu-1": TestSkip(75, "Need at least 1 MLU device"),
    "multi-mlu-2": TestSkip(77, "Need at least 2 MLU devices"),
    "multi-mlu-3": TestSkip(80, "Need at least 3 MLU devices"),
    "multi-mlu-4": TestSkip(81, "Need at least 4 MLU devices"),
    "multi-mlu-5": TestSkip(82, "Need at least 5 MLU devices"),
    "multi-mlu-6": TestSkip(83, "Need at least 6 MLU devices"),
    "multi-mlu-7": TestSkip(84, "Need at least 7 MLU devices"),
    "multi-mlu-8": TestSkip(85, "Need at least 8 MLU devices"),
    "cncl": TestSkip(76, "c10d not compiled with CNCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no MLU peer access"),
    "generic": TestSkip(
        86, "Test skipped at subprocess level, look at subprocess log for skip reason"
    ),
}

# HELPER FOR MULTIMLU TESTS
def init_multimlu_helper(world_size: int, backend: str):
    """Multimlu tests are designed to simulate the multi nodes with multi
    MLUs on each node. Nccl backend requires equal #MLUs in each process.
    On a single node, all visible MLUs are evenly
    divided to subsets, each process only uses a subset.
    """
    nMLUs = torch.mlu.device_count()
    visible_devices = range(nMLUs)

    # if backend == "cncl":
    #     # This is a hack for a known CNCL issue using multiprocess
    #     # in conjunction with multiple threads to manage different MLUs which
    #     # may cause cnclCommInitRank to fail.
    #     # http://docs.nvidia.com/deeplearning/sdk/cncl-release-notes/rel_2.1.4.html#rel_2.1.4
    #     # It slows down the performance of collective operations.
    #     # Without this setting CNCL might throw unhandled error.
    #     os.environ["CNCL_MAX_NRINGS"] = "1"

    # If rank is less than or equal to number of available MLU's
    # then each rank can be mapped to corresponding MLU.
    nMLUs_per_process = 1
    if world_size > nMLUs:
        nMLUs_per_process = nMLUs // world_size
    rank_to_MLU = {
        i: list(
            visible_devices[i * nMLUs_per_process : (i + 1) * nMLUs_per_process]
        )
        for i in range(world_size)
    }
    return rank_to_MLU

def requires_cncl():
    return sandcastle_skip_if(
        not c10d.is_cncl_available(),
        "c10d was not compiled with the CNCL backend",
    )


def requires_cncl_version(version, msg):
    if not c10d.is_cncl_available():
        return sandcastle_skip(
            "c10d was not compiled with the CNCL backend",
        )
    else:
        return sandcastle_skip_if(
            torch.mlu.cncl.version() < version,
            "Requires CNCL version greater than or equal to: {}, found: {}, reason: {}".format(
                version, torch.mlu.cncl.version(), msg
            ),
        )

def skip_if_lt_x_mlu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.mlu.is_available() and torch.mlu.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-mlu-{x}"].exit_code)

        return wrapper

    return decorator


# This decorator helps avoiding initializing mlu while testing other backends
def cncl_skip_if_lt_x_mlu(backend, x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend != "cncl":
                return func(*args, **kwargs)
            if torch.mlu.is_available() and torch.mlu.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-mlu-{x}"].exit_code)

        return wrapper

    return decorator


def with_cncl_blocking_wait(func):
    """
    Convenience decorator to set/unset CNCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of CNCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both CNCL_BLOCKING_WAIT and
    CNCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save and unset CNCL_ASYNC_ERROR_HANDLING
        try:
            cached_cncl_async_error_handling: Union[str, None] = os.environ[
                "CNCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["CNCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # CNCL_ASYNC_ERROR_HANDLING was unset
            cached_cncl_async_error_handling = None

        # Save val of CNCL_BLOCKING_WAIT and set it.
        try:
            cached_cncl_blocking_wait: Union[str, None] = os.environ[
                "CNCL_BLOCKING_WAIT"
            ]
        except KeyError:
            cached_cncl_blocking_wait = None
        finally:
            os.environ["CNCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            # restore old values.
            if cached_cncl_async_error_handling is not None:
                os.environ[
                    "CNCL_ASYNC_ERROR_HANDLING"
                ] = cached_cncl_async_error_handling

            if cached_cncl_blocking_wait is not None:
                os.environ["CNCL_BLOCKING_WAIT"] = cached_cncl_blocking_wait

    return wrapper

def mlus_for_rank(world_size):
    """Multimlu tests are designed to simulate the multi nodes with multi
    MLUs on each node. Nccl backend requires equal #MLUs in each process.
    On a single node, all visible MLUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.mlu.device_count()))
    mlus_per_process = torch.mlu.device_count() // world_size
    mlus_for_rank = []
    for rank in range(world_size):
        mlus_for_rank.append(
            visible_devices[rank * mlus_per_process : (rank + 1) * mlus_per_process]
        )
    return mlus_for_rank

def require_n_mlus_for_cncl_backend(n, backend):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend == "CNCL" and torch.mlu.device_count() < n:
                sys.exit(TEST_SKIPS[f"multi-mlu-{n}"].exit_code)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator