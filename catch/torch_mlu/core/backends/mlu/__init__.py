import torch    # pylint: disable=W0611
import warnings
from torch.backends.cuda import cuFFTPlanCacheAttrContextProp, cuFFTPlanCache
import torch_mlu

__all__ = ["matmul", "custom", "CnnlMatmulTF32Controller", "MLUCustomTF32Controller",
  "fake_set_cublas_allow_tf32", "fake_set_cublas_allow_fp16_reduced_precision_reduction"]

class cnFFTPlanCache(cuFFTPlanCache):
    r"""
    Represents a specific plan cache for a specific `device_index`. The
    attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cnFFT plan cache.
    """
    size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_size,
        '.size is a read-only property showing the number of plans currently in the '
        'cache. To change the cache capacity, set cnfft_plan_cache.max_size.')


class cnFFTPlanCacheManager(object):
    r"""
    Represents all cnFFT plan caches. When indexed with a device object/index,
    this object returns the `cnFFTPlanCache` corresponding to that device.

    Finally, this object, when used directly as a `cnFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cnFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.mlu._utils._get_device_index(device)
        dev_cnt = torch.mlu.device_count() if hasattr(torch.mlu, "device_count") else 0
        if index < 0 or index >= dev_cnt:
            raise RuntimeError(
                ("cnfft_plan_cache: expected 0 <= device index < {}, but got "
                 "device with index {}").format(dev_cnt, index))
        if len(self.caches) == 0:
            self.caches.extend(cnFFTPlanCache(index) for index in range(dev_cnt))
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.mlu.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.mlu.current_device()], name, value)
        else:
            return super(cnFFTPlanCacheManager, self).__setattr__(name, value)

class CnnlMatmulTF32Controller:
    r"""
    Control wether to allow TF32 on matmul, same function as `torch.backends.cuda.matmul.allow_tf32`.
    """
    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch._C._get_cnmatmul_allow_tf32()
        raise AssertionError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch._C._set_cnmatmul_allow_tf32(value)
        raise AssertionError("Unknown attribute " + name)

class MLUCustomTF32Controller:
    r"""
    Control wether to allow TF32 on the rest MLU ops, not controlled by
    `CnnlMatmulTF32Controller` and `CnnlTF32Controller`.
    """
    def __getattr__(self, name):
        assert name == "allow_tf32", "Unknown attribute " + name
        return torch_mlu._MLUC._get_mlu_custom_allow_tf32()

    def __setattr__(self, name, value):
        assert name == "allow_tf32", "Unknown attribute " + name
        if not isinstance(value, bool):
            raise  RuntimeError("set_mlu_custom_allow_tf32 expects a bool, "
                "but got {}".format(type(value)))
        return torch_mlu._MLUC._set_mlu_custom_allow_tf32(value)

# Because the float32_matmul_precision flag is device independent,
# we need to prevent torch._C._set_cublas_allow_tf32 from modifying the flg.
def fake_set_cublas_allow_tf32(value):
    if not isinstance(value, bool):
        raise  RuntimeError("set_allow_tf32_cublas expects a bool, "
            "but got {}".format(type(value)))
    warnings.warn("When using MLU device, the cuda API does not take effect. "
      "Please use torch.backends.mlu.matmul.allow_tf32.")

# MLU side is not support matmul using compute type float16, so we
# need to prevent torch._C.allow_fp16_reduced_precision_reduction to modifying the flg.
def fake_set_cublas_allow_fp16_reduced_precision_reduction(value):
    if not isinstance(value, bool):
        raise  RuntimeError("set_allow_tf32_cublas expects a bool, "
            "but got {}".format(type(value)))
    warnings.warn("When using MLU device, the cublas_allow_fp16_reduced_precision_reduction"
      " API does not take effect. And MLU only support compute type float32.")

cnfft_plan_cache = cnFFTPlanCacheManager()
matmul = CnnlMatmulTF32Controller()
custom = MLUCustomTF32Controller()
