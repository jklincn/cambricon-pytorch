# 使用 MLUExtension 机制编译自定义算子步骤
## 概述
通常因为用户需求紧急或者是其它原因需要手动实现算子，CATCH 的 MLUExtension 提供了对编译BangC程序功能的封装。用户可以自行实现BangC程序然后通过MLUExtension编译然后使用PyBind11绑定Python接口从而实现对算子的调用。下面的例子将从0开始实现一个sigmoid函数。
为了更方便的描述整个流程，我们将在一个目录中描述整个实现：
```
# mlu_extension
├── README.md：描述算子功能的说明文档
├── mlu_custom_ext：生成的module模块用于在python层导入。
│   ├── __init__.py：python包固有文件
│   ├── mlu：mlu代码文件，根据实际情况自己创建，在setup.py中修改即可（建议使用目录管理BangC代码）。
│   │   ├── include：头文件目录（头文件和实现分离，属于代码习惯，建议采用此布局）
│   │   │   ├── bang_sigmoid_sample.h：实现对mlu函数的封装。
│   │   │   ├── kernel.h：BangC代码中的宏，良好的组织代码的需要。
│   │   │   └── custom_ops.h：算子对外头文件。
│   │   └── src
│   │       ├── bang_sigmoid.cpp：对PyTorch层面Tensor的封装，和自定义算子中xxx_internal里面的实现类似。
│   │       ├── bang_sigmoid_sample.mlu：核心BangC实现。
│   └── mlu_functions：合理的组织自己的代码方便后续调用。
│       ├── __init__.py：包必备文件。
│       └── mlu_functions.py：对C++代码的封装。
├── setup.py：构建包的脚本。
└── tests
    └── test_sigmoid.py：对绑定代码的python侧测试。
  
```
先看看算子层实现`bang_sigmoid.cpp`
```


#include "bang_sigmoid_sample.h"
#include "customed_ops.h"

#include "ATen/Tensor.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

using namespace torch_mlu;
torch::Tensor active_sigmoid_mlu(torch::Tensor x) {
  auto x_contiguous = torch_mlu::cnnl_contiguous(x);
  auto x_impl = getMluTensorImpl(x_contiguous);
  auto x_ptr = x_impl->mlu_data_ptr();

  auto y = at::empty_like(x_contiguous);
  auto y_contiguous = torch_mlu::cnnl_contiguous(y);
  auto y_impl = getMluTensorImpl(y_contiguous);
  auto y_ptr = y_impl->mlu_data_ptr();

  int32_t size = x_contiguous.numel();

  cnrtQueue_t queue = getCurQueue();
  bang_sigmoid_kernel_entry(
      queue,
      reinterpret_cast<float*>(y_ptr),
      reinterpret_cast<float*>(x_ptr),
      size);

  return y;
}

PYBIND11_MODULE(libmlu_custom_ext, m) {
  m.def("active_sigmoid_mlu", &active_sigmoid_mlu);
}
  
```
需要重点介绍的是Pybind这一部分，`libmlu_custom_ext`描述了最后生成的库用于导入，最后生成的库格式大概是`libmlu_custom_ext.cpython-37m-x86_64-linux-gnu`。具体查看[这里](https://pybind11.readthedocs.io/en/stable/basics.html)，其中`m.def("active_sigmoid_mlu", &active_sigmoid_mlu);`表示将`"active_sigmoid_mlu"`绑定一个执行函数`active_sigmoid_mlu`。正如上面所说，我们需要实现`bang_sigmoid_kernel_entry`。这里简单说明一下BangC程序的执行分别对应于BangC程序的三部分：

1. host部分：主要使用cnrt接口完成device内存分配，将host内存拷贝到对应的device位置，创建执行队列之类的任务。
2. host对device的调用：通常用于描述任务如何执行，比如是block任务还是union任务，设置对应任务启动的参数。
3. device部分：这部分通常使用`__mlu_global__`前缀，过于复杂的逻辑通常使用`__mlu_func__`封装成一个函数。

```
#include <bang_sigmoid_sample.h>
#include <kernel.h>
__nram__ char NRAM_BUFFER[MAX_NRAM_SIZE];
template<typename T>
__mlu_global__ void bang_sigmoid_kernel(T *d_dst, T *d_src, int N) {
  const int NRAM_LIMIT_SIZE = FLOOR_ALIGN(MAX_NRAM_SIZE / 2, 64);
  int nram_limit = NRAM_LIMIT_SIZE / sizeof(T);
  // 对列数据切分
  int32_t num_per_core = N / taskDim;
  int32_t repeat = num_per_core / nram_limit;
  int32_t rem = num_per_core % nram_limit;

  T *d_input_per_task = d_src + taskId * nram_limit;
  T *d_output_per_task = d_dst + taskId * nram_limit;
  T *nram_out = (T *)NRAM_BUFFER;
  T *nram_in = (T *)(NRAM_BUFFER + NRAM_LIMIT_SIZE);

  const int align_rem = CEIL_ALIGN(rem, 64);

  int i = 0;
  for (; i < repeat; i++) {
    //
    __memcpy_async(nram_in, d_input_per_task + i * nram_limit, NRAM_LIMIT_SIZE,
                   GDRAM2NRAM);
    __sync_io();
    __bang_active_sigmoid(nram_out, nram_in, nram_limit);
    __sync_compute();

    __memcpy_async(d_output_per_task + i * nram_limit, nram_out,
                   NRAM_LIMIT_SIZE, NRAM2GDRAM);

    __sync_io();
  }
  if (rem > 0) {
    __memcpy_async(nram_in, d_input_per_task + i * nram_limit,
                   rem * sizeof(T), GDRAM2NRAM);
    __sync_io();
    __bang_active_sigmoid(nram_out, nram_in, align_rem);
    __sync_compute();
    __memcpy_async(d_output_per_task + i * nram_limit, nram_out,
                   rem * sizeof(T), NRAM2GDRAM);
    __sync_io();
  }
}
template<typename T>
void bang_sigmoid_kernel_entry(cnrtQueue *queue, T *d_dst, T *d_src,
                           int elem_count) {
  cnrtDim3_t dim = {1, 1, 1};
  int taskDims = dim.x * dim.y * dim.z;
  cnrtFunctionType_t c = CNRT_FUNC_TYPE_BLOCK;
  if (elem_count < taskDims) {
    dim.x = 1;
    dim.y = 1;
  }
  bang_sigmoid_kernel<<<dim, c, queue>>>(d_dst, d_src, elem_count);
  cnrtQueueSync(queue);
}
template<typename T>
void bang_sigmoid_sample(T *h_dst, T *h_src, const int elem_count) {

  T *d_src, *d_dst;
  cnrtQueue_t queue;
  cnrtQueueCreate(&queue);
  cnrtRet_t ret;
  ret =
      cnrtMalloc(reinterpret_cast<void **>(&d_src), elem_count * sizeof(T));
  ret =
      cnrtMalloc(reinterpret_cast<void **>(&d_dst), elem_count * sizeof(T));

  ret = cnrtMemcpy(d_src, h_src, elem_count * sizeof(T),
                   CNRT_MEM_TRANS_DIR_HOST2DEV);

  bang_sigmoid_kernel_entry(queue, d_dst, d_src, elem_count);
  cnrtQueueSync(queue);
  ret = cnrtMemcpy(h_dst, d_dst, elem_count * sizeof(T),
                   CNRT_MEM_TRANS_DIR_DEV2HOST);

  ret = cnrtQueueDestroy(queue);
}
template void bang_sigmoid_sample(float*, float*, int);
template void bang_sigmoid_kernel_entry(cnrtQueue *, float *, float *, int);

```
- `bang_sigmoid_kernel`：调用BangC指令完成计算。
- `bang_sigmoid_kernel_entry`：host侧对device的入口。
- `bang_sigmoid_sample`：host接口，用来测试算子实现，非必须。
```
template void bang_sigmoid_sample(float*, float*, int);
template void bang_sigmoid_kernel_entry(cnrtQueue *, float *, float *, int);
```
这两行用来显示特例化模板函数，主要用在头文件和模板分离的情况避免未定义的引用。第一个特例化非必须，主要用于方便在host端测试整个函数逻辑；第二个函数是真正特例化的部分。
```
// bang_sigmoid_sample.h
#pragma once
#include <cnrt.h>
template <typename T>
void bang_sigmoid_kernel_entry(
    cnrtQueue* queue,
    T* d_dst,
    T* d_src,
    int elem_count);

```
上述头文件主要包含的头文件为cnrt.h(cnrtQueue需要)暴露给PyTorch算子层，以通过其完成最终的计算。完成上述三段代码之后PyTorch C++层面的功能就已经就绪。接下来通过setup借用python编译代码为动态库：
```
import os
from setuptools import setup, find_packages

from torch.utils import cpp_extension
from torch_mlu.utils.cpp_extension import MLUExtension, BuildExtension
import glob

mlu_custom_src = "mlu_custom_ext" # c++、mlu源码目录
cpath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                     os.path.join(mlu_custom_src, "mlu"))


def source(src):
    cpp_src = glob.glob("{}/*.cpp".format(src))
    mlu_src = glob.glob("{}/*.mlu".format(src))
    cpp_src.extend(mlu_src)
    return cpp_src


mlu_extension = MLUExtension(name="libmlu_custom_ext",#生成库的名称，导入python 模块的时候需要
                             sources=source(os.path.join(cpath, 'src')),
                             include_dirs=[os.path.join(cpath, "include")],
                             verbose=True,
                             extra_link_args=["-Wl,--as-needed"],# 额外的链接选项，在gcc最后加入，本例下这个参数不生效
                             extra_compile_args={ # 额外的编译选项
                                 "cxx": [
                                     "-O3",
                                     "-std=c++14",
                                 ],
                                 "cncc": [
                                     "-O3", "--bang-mlu-arch=mtp_372",#cncc编译的架构支持
                                     "-I{}".format(
                                         os.path.join(cpath, "include"))#包含的自定义模块的头文件
                                 ]
                             })

setup(name="mlu_custom_ext",# python包的名称和模块名称可以不同
      version="0.1",# 当前扩展的版本
      packages=find_packages(),
      ext_modules=[mlu_extension],
      cmdclass={
          "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
      })

```
到这里`python setup.py install`之后就可以通过：`from libmlu_custom_ext import active_sigmoid_mlu`导入自己实现的函数了，这里我们做一些额外的非必须的工作（可选）。
通过扩展自定义算子实现其反向，这个操作主要来自于torch本身的能力，详情查看[autograd](https://pytorch.org/docs/stable/torch.html?highlight=autograd#module-torch.autograd)。
```
# mlu_custom_ext/mlu_functions
├── __init__.py       # Python包必备文件
└── mlu_functions.py  # 实现核心功能
```
mlu_functions.py代码如下：
```
from turtle import forward
import torch
import torch.nn as nn
import torch.jit as jit

from typing import Any

from libmlu_custom_ext import *

# 通过autograd计算前向和反向
class sigmoid_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = active_sigmoid_mlu(x)
        ctx.save_for_backward(*[x, y])
        return y

    @staticmethod
    def backward(ctx: Any, d_r: Any) -> Any:
        d_r = d_r.contiguous()
        x, y = ctx.saved_variables
        dx = y * (1 - y) * d_r
        return dx

@jit.ignore
def sigmoid(x: torch.Tensor) -> torch.Tensor: # 包装模块
    return sigmoid_function.apply(x)

```
经过此操作后你可以导入sigmoid_function或者直接使用sigmoid，他们都在module里面的mlu_functions里面。这样便于组织你的代码，对一些算法复杂，但是核心C++逻辑不多的代码可以通过此方法将核心逻辑封装，然后在外部用python进一步封装。

