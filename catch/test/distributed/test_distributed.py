from __future__ import absolute_import, division, print_function, unicode_literals
#pylint: disable=C0413,C0411,C0302
import copy
import multiprocessing
import os
import sys
from sys import path
from os.path import dirname
from itertools import product
import time
import unittest
import math
from functools import reduce, wraps
import distutils.dir_util
from argparse import ArgumentParser
from multiprocessing.managers import BaseManager
from queue import Queue
import random as rd

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import socket
from contextlib import closing

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

path.append(dirname(path[0]))
from common_utils import TestCase, TEST_BFLOAT16

INIT_METHOD = os.getenv("INIT_METHOD", "env://")
DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_distributedDataParallel":200, "test_pressure":200, "test_barrier":300}
SKIP_IF_BACKEND_UNAVAILABLE = 78
cwd = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(cwd, "tmp")

def find_free_port():
    """
    Finds an available port and returns that port number.

    NOTE: If this function is being used to allocate a port to Store (or
    indirectly via init_process_group or init_rpc), it should be used
    in conjuction with the `retry_on_connect_failures` decorator as there is a potential
    race condition where the allocated port may become unavailable before it can be used
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', 0))
        _, port = sock.getsockname()
        return port

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Catch distributed training unittest")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes participate in testing, "
                             "this is set ot 1 by default.")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node testing, "
                             "this is set to 0 by default.")
    parser.add_argument("--nproc_per_node", type=int, default= 4,
                        help="The number of processes to launch on each node, "
                             "for multi-node testing, this is set to 4 by default.")
    parser.add_argument("--connects", default=rd.randint(-1, 4), type=int, choices=range(-1, 4),
                        help="We support testing for different technologies of "
                             "connection. Different techs have different priority "
                             "levels. In this script, MLU‑Link > P2P > SHM > SOCKET. "
                             "when input is -1, no cncl environment will be set; "
                             "input is 0, all techs can be used; input is 1, only "
                             "P2P, SHM and SOCKET can be used, MLU‑Link is prohibited; "
                             "2: SHM, SOCKET; 3: SOCKET. By default, every techs "
                             "have chances to be tested. Note: When here are multi "
                             "node, connects would be set to -1 forcibly, please "
                             "make sure different node use the same cncl environments "
                             "according to cncl user guide doc.")
    parser.add_argument("--master_addr", default="127.0.0.10", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1, now this is "
                             "set to 127.0.0.10 by default.")
    parser.add_argument("--master_port", default=find_free_port(), type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training, now this is set to 20000 by default. "
                             "In addition, we also use (master_port + 10) for sync.")
    parser.add_argument("--delay_time", default=0, type=int,
                        help="The communication time may be different between "
                             "different environment. So we provide a choice for "
                             "user to add additional delay time for all test case.")

    parser.add_argument('unittest_args', nargs='*')

    return parser.parse_args()

def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).mlu(device_id)

def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT

class QueueManager(BaseManager):
    pass

# flg: flg==True means force run OP on CPU, to avoid MLU caculations.
class Linear_mlu(nn.Linear):   # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            if self.bias is not None:
                bias_cpu = self.bias.cpu()
            else:
                bias_cpu = None
            return F.linear(input_.cpu(), self.weight.cpu(), bias_cpu)
        else:
            return F.linear(input_, self.weight, self.bias)

class Conv2d_mlu(nn.Conv2d):  # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            return self._conv_forward(input_.cpu(), self.weight.cpu(), None)
        else:
            return self._conv_forward(input_, self.weight, None)

class _FC2(nn.Module):   # pylint: disable=W0223
    def __init__(self):
        super(_FC2, self).__init__()   # pylint: disable=R1725
        self.fc = Linear_mlu(10, 12, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x, flg):
        x = self.fc(x, flg)
        return x

class Net(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(Net, self).__init__()   # pylint: disable=R1725
        self.fc1 = Linear_mlu(2, 10, bias=False)
        self.fc2 = _FC2()
        self.conv = Conv2d_mlu(2, 6, 2, bias=False)
        self.fc3 = Linear_mlu(12, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.Tensor([2, 2]).long(), requires_grad=False)

    def forward(self, x, flg):
        x = self.relu(self.fc1(x, flg))
        x = self.relu(self.fc2(x, flg))
        x = self.conv(x.view(-1, 2, 3, 2), flg).view(-1, 12)
        x = self.fc3(x, flg)
        return F.softmax(x, dim=1)

class BatchNormNet(nn.Module):  # pylint: disable=W0223
    def __init__(self, affine=True):
        super(BatchNormNet, self).__init__()   # pylint: disable=R1725
        self.fc1 = Linear_mlu(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4, affine=affine)
        self.fc2 = Linear_mlu(40, 4, bias=False)

    def forward(self, x, flg):
        x = torch.reshape(self.fc1(x, flg), (-1, 4, 10))
        x = self.bn(x.to("mlu")).cpu()
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x, flg)
        return F.softmax(x, dim=1)

class OnlyBatchNormNet(nn.Module):  # pylint: disable=W0223
    def __init__(self, module):
        super(OnlyBatchNormNet, self).__init__()   # pylint: disable=R1725
        self.bn = module

    def forward(self, x, flg):   #pylint: disable=W0613
        x = self.bn(x.to("mlu")).cpu()
        return x

class Foo:
    def __init__(self, x):
        # Can be tensor or int
        self.x = x

    def __eq__(self, other):
        def eq(value, other):
            if isinstance(value, torch.Tensor):
                return torch.equal(value, other)
            return value == other

        for attr, value in self.__dict__.items():
            other_value = other.__dict__[attr]
            if not eq(value, other_value):
                return False
        return True

class _DistTestBase(object):  # pylint: disable=R0205, R0904
    args = None
    rank = 0
    world_size = 0

    @classmethod
    def _init_global_test(cls):
        group = [i for i in range(0, dist.get_world_size())]  # pylint: disable=R1721
        rank = dist.get_rank()
        return (group, rank)

    def _test_broadcast_helper(self, group, rank):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        type_info = [
            ("torch.FloatTensor", -1e-10, True),
            ("torch.HalfTensor", -0.1, True),
            ("torch.CharTensor", -2, True),
            ("torch.ByteTensor", 129, True),
            ("torch.IntTensor", -1e5, True),
            ("torch.LongTensor", 1e5, True),
            ("torch.DoubleTensor", -1e-10, True),
        ]
        if TEST_BFLOAT16:
            type_info.append(("torch.BFloat16Tensor", -1e-10, True))
        for ttype, value, is_test in type_info:
            if not is_test:
                continue
            for src in group:
                #expected_tensor = self.to_device(torch.tensor([]))
                expected_tensor = self.to_device(_build_tensor(src + 1, value).type(ttype))
                if rank == src:
                    dist.broadcast(expected_tensor, src)
                else:
                    tensor = self.to_device(_build_tensor(src + 1, -1).type(ttype))
                    dist.broadcast(tensor, src)
                    self.assertEqual(tensor.size(), expected_tensor.size())
                    self.assertTrue(tensor.type(torch.float).cpu().eq(
                        expected_tensor.type(torch.float).cpu()).min().item())

    #@unittest.skip("not test")
    def test_broadcast(self):
        group, rank = self._init_global_test()
        self._test_broadcast_helper(group, rank)

    # @unittest.skip("not test")
    def test_broadcast_object_list(self):
        # Case where rank != MLU device.
        next_rank = (self.rank + 1) % torch.mlu.device_count()
        torch.mlu.set_device(next_rank)

        torch.manual_seed(0)
        f = Foo(10)
        f.bar = 1
        foo_cpu_tensor = Foo(torch.randn(3, 3))
        foo_mlu_tensor = Foo(torch.randn(3,3).mlu(0))
        COLLECTIVES_OBJECT_TEST_LIST = [
            {"key1": 3, "key2": 4, "key3": {"nested": True}},
            f,
            foo_cpu_tensor,
            foo_mlu_tensor,
            "foo",
            [1, 2, True, "string", [4, 5, "nested"]],
        ]

        src_rank = 0

        objects = (
            COLLECTIVES_OBJECT_TEST_LIST
            if self.rank == src_rank
            else [None for _ in COLLECTIVES_OBJECT_TEST_LIST]
        )

        # Single object test
        single_obj_list = [objects[0]]
        if self.rank != src_rank:
            self.assertNotEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])
        dist.broadcast_object_list(single_obj_list, src=0)
        self.assertEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])

        # Multiple input objects test
        if self.rank != src_rank:
            self.assertNotEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)

        dist.broadcast_object_list(objects, src=0)
        # Test mlu tensor broadcast successfully
        self.assertTrue(objects[3].x.device.type == "mlu")
        self.assertEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)

    def _test_async_helper(self, group, rank, op, master_value,
                           worker_value, expected_value):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertTensorsEqual(
                    tensor.cpu(), _build_tensor(src + 1, expected_value), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertTensorsEqual(
                    tensor.cpu(), _build_tensor(src + 1, expected_value), 3e-3)
            self.assertTrue(work.is_completed())
            self.assertTrue(work.is_success())

    #@unittest.skip("not test")
    def test_async(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_async_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    def _test_reduce_helper(self, group, rank, op, master_value,
                            worker_value, expected_value):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                dist.reduce(tensor, src, op)
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    'mlu')).abs().cpu().max().item(), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                dist.reduce(tensor, src, op)

    #@unittest.skip("not test")
    def test_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    #@unittest.skip("not test")
    def test_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    #@unittest.skip("not test")
    def test_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    #@unittest.skip("not test")
    def test_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    def _test_all_reduce_helper(self, group, rank, op, master_value,
                                worker_value, expected_value):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                dist.all_reduce(tensor, op)
                #print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    'mlu')).abs().cpu().max().item(), 3e-3)
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                dist.all_reduce(tensor, op)
                #print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess((tensor.float() - _build_tensor(src + 1, expected_value).to(
                    'mlu')).abs().cpu().max().item(), 3e-3)

    #@unittest.skip("not test")
    def test_all_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    #@unittest.skip("not test")
    def test_all_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    #@unittest.skip("not test")
    def test_all_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    #@unittest.skip("not test")
    def test_all_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    # @unittest.skip("not test")
    def test_empty_tensors(self):
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        pg = _get_default_group()

        ys = [self.to_device(torch.FloatTensor([]))]
        xs = [[self.to_device(torch.FloatTensor([])) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    def _test_reduce_scatter_helper(self, rank, op, expected_value):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        type_info = [
            ("torch.FloatTensor", True),
            ("torch.HalfTensor", True),
            ("torch.CharTensor", True),
            ("torch.ByteTensor", True),
            ("torch.IntTensor", True),
            ("torch.LongTensor", True),
        ]
        if TEST_BFLOAT16:
            type_info.append(("torch.BFloat16Tensor", True))
        for ttype, is_test in type_info:
            if not is_test:
                continue
            else:
                output = self.to_device(torch.tensor([0]).type(ttype))
                if op == dist.ReduceOp.PRODUCT:
                    tensor_list = [
                        self.to_device(torch.tensor(
                            [(rank + i) % self.world_size + 1]).type(ttype))
                        for i in range(0, self.world_size)
                    ]
                else:
                    tensor_list = [
                        self.to_device(torch.tensor([rank + i]).type(ttype))
                        for i in range(0, self.world_size)
                    ]
                dist.reduce_scatter(output, tensor_list, op)
                # mlu bfloat16 is not support item() now.
                self.assertEqual(expected_value.cpu() if isinstance(expected_value, torch.Tensor) else expected_value, \
                                 output.cpu() if isinstance(output, torch.Tensor) else output)

    # @unittest.skip("not test")
    def test_reduce_scatter_sum(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank,
            dist.ReduceOp.SUM,
            float(self.world_size * (self.world_size - 1) / 2) + rank * self.world_size,
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_min(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank,
            dist.ReduceOp.MIN,
            float(rank)
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_max(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank,
            dist.ReduceOp.MAX,
            float(rank + self.world_size - 1)
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_product(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank,
            dist.ReduceOp.PRODUCT,
            float(math.factorial(self.world_size))
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_tensor(self):
        _, rank = self._init_global_test()
        size = 2
        tensor_out = torch.zeros(size, dtype=torch.int64).mlu(rank % torch.mlu.device_count())

        # Concatenated input
        tensor_in = torch.arange(self.world_size * size).mlu(rank % torch.mlu.device_count())
        dist.reduce_scatter_tensor(tensor_out, tensor_in)
        # Check result
        expected_tensor = torch.arange(rank * size, (rank + 1) * size) * self.world_size
        self.assertEqual(tensor_out.cpu(), expected_tensor)

        # Stacked input
        tensor_out = torch.zeros(size, dtype=torch.int64).mlu(rank % torch.mlu.device_count())
        tensor_in = torch.reshape(
          tensor_in, (self.world_size, size)).mlu(rank % torch.mlu.device_count())
        dist.reduce_scatter_tensor(tensor_out, tensor_in)
        # Check result
        # Should be the same as the result in concatenated case
        self.assertEqual(tensor_out.cpu(), expected_tensor)

    def _test_all_gather_helper(self, group, rank, times=1):
        def _build_tensor(size, value):
            return torch.arange(value, size*size*size + value).view(size, size, size).float()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        loop_list = list(range(times))
        dtype_list = [("torch.FloatTensor", True),
                      ("torch.HalfTensor", True),
                      ("torch.CharTensor", True),
                      ("torch.ByteTensor", True),
                      ("torch.IntTensor", True),
                      ("torch.LongTensor", True),
                      ("torch.DoubleTensor", True),
                      ("torch.BoolTensor", True)]
        if TEST_BFLOAT16:
            dtype_list.append(("torch.BFloat16Tensor", True))
        list_list = [loop_list, dtype_list, group]
        for _, dtype_tuple, src in product(*list_list):
            if not dtype_tuple[1]:
                continue
            ttype = dtype_tuple[0]
            tensor = self.to_device(_build_tensor(src + 1, rank).type(ttype))
            tensors = [self.to_device(_build_tensor(src + 1, -1).type(ttype))
                        for i in group]
            dist.all_gather(tensors, tensor)
            expected_tensors = [self.to_device(_build_tensor(src + 1, i).type(ttype))
                                for i in group]
            for t1, t2 in zip(tensors, expected_tensors):
                self.assertTrue(t1.cpu().eq(t2.cpu()).min().item())

    #@unittest.skip("not test")
    def test_all_gather(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank)

    #@unittest.skip("not test")
    def test_all_gather_object(self):
        _, rank = self._init_global_test()
        next_rank = (rank + 1) % torch.mlu.device_count()
        torch.mlu.set_device(next_rank)

        f = Foo(10)
        f.bar = 1    #pylint: disable=C0104,W0201
        torch.manual_seed(0)
        foo_cpu_tensor = Foo(torch.randn(3, 3))
        foo_mlu_tensor = Foo(torch.randn(3,3).mlu(0))

        gather_objects = [
            {"key1": 3, "key2": 4, "key3": {"nested": True}},
            f,
            foo_cpu_tensor,
            foo_mlu_tensor,
            "foo",
            [1, 2, True, "string", [4, 5, "nested"]],
        ]

        output_gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            output_gathered, gather_objects[rank % len(gather_objects)]
        )

        for i, val in enumerate(output_gathered):
            expected = gather_objects[i % len(gather_objects)]
            self.assertEqual(val, expected)

            output_gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                output_gathered, gather_objects[rank % len(gather_objects)]
            )

    # @unittest.skip("not test")
    def test_all_gather_into_tensor_ops(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = torch.tensor([self.rank]).mlu()
        output_t = torch.zeros((self.world_size), dtype=tensor.dtype).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

    # @unittest.skip("not test")
    def test_all_gather_into_cat_tensor(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        size = 2
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = (torch.ones([size, size]) * self.rank).mlu()
        output_t = (torch.ones([self.world_size * size, size], dtype=tensor.dtype)*(-1)).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Check result
        # Concatenate all blocks into a bigger tensor
        expected_tensor = torch.cat([
            torch.ones([size, size]) * i for i in range(self.world_size)
        ])
        # Verification
        self.assertEqual(output_t, expected_tensor)

    # @unittest.skip("not test")
    def test_all_gather_into_stack_tensor(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        size = 2
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = (torch.ones([size, size]) * self.rank).mlu()
        output_t = (torch.ones([self.world_size, size, size], dtype=tensor.dtype) * (-1)).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Check result
        # Concatenate all blocks into a bigger tensor
        expected_tensor = torch.stack([
            torch.ones([size, size]) * i for i in range(self.world_size)
        ])
        # Verification
        self.assertEqual(output_t, expected_tensor)

    # @unittest.skip("not test")
    def test_all_gather_into_tensor_basics(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        # anticpate an error
        with self.assertRaisesRegex(RuntimeError, "output tensor size must be equal to \
world_size times input tensor size"):
            tensor = torch.tensor([self.rank]).mlu()
            output_t = torch.zeros((self.world_size + 1), dtype=tensor.dtype).mlu()
            # fails the check because output_t is not correctly sized
            dist.all_gather_into_tensor(output_t, tensor)

        # anticpate an error
        with self.assertRaisesRegex(RuntimeError, "output tensor must have the same type \
as input tensor"):
            tensor = torch.tensor([self.rank], dtype=torch.float).mlu()
            output_t = torch.zeros((self.world_size + 1), dtype=torch.long).mlu()
            # fails the check because the dtype is different
            dist.all_gather_into_tensor(output_t, tensor)

    #@unittest.skip("not test")
    def test_pressure(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank, times=20)

    def _test_p2pop_helper(self, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        dtype_list = ["torch.FloatTensor", "torch.HalfTensor", "torch.CharTensor",
            "torch.ByteTensor", "torch.IntTensor", "torch.LongTensor",
            "torch.DoubleTensor", "torch.BoolTensor"]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        dist.barrier()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for ttype in dtype_list:
            send_tensor = torch.tensor(range(10)).type(ttype).to("mlu")
            recv_tensor = torch.zeros(10).type(ttype).to("mlu")
            p2p_op_list = []
            if rank == 0:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, 1))
            elif rank == 1:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, 0))
            if rank in [0, 1]:
                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()
            dist.barrier()
            if rank == 1:
                self.assertTensorsEqual(recv_tensor.float().cpu(), send_tensor.float().cpu(), 0)

    #@unittest.skip("not test")
    def test_p2pop(self):
        _, rank = self._init_global_test()
        self._test_p2pop_helper(rank)

    #@unittest.skip("not test")
    def test_batch_isend_irecv(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        world_size = dist.get_world_size()
        recv_tensors = [None for _ in range(world_size)]
        expected_tensors = [None for _ in range(world_size)]

        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        for val in ["1", "0"]:
            p2p_op_list = []
            os.environ["CNCL_BLOCKING_WAIT"] = val
            for src in range(0, dist.get_world_size()):
                send_tensor = _build_tensor(rank + 1, device_id=device_id).fill_(src)
                recv_tensors[src] = _build_tensor(src + 1, value=-1, device_id=device_id)
                expected_tensors[src] = _build_tensor(src + 1, value=-1, device_id=device_id).fill_(rank)
                recv_op = dist.P2POp(dist.irecv, recv_tensors[src], src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src)
                p2p_op_list.append(send_op)
            
            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()
            
            for src in range(0, world_size):
                self.assertEqual(recv_tensors[src], expected_tensors[src])

        dist.barrier()
        
    # @unittest.skip("not test")
    def test_batch_isend_irecv_cncl_self(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        p2p_op_list = []

        if rank == 0:
            send_tensor = _build_tensor(rank + 1, device_id=device_id)
            recv_tensor = _build_tensor(rank + 1, value=-1, device_id=device_id)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
            p2p_op_list.append(recv_op)
            send_op = dist.P2POp(dist.isend, send_tensor, 0)
            p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

        dist.barrier()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_tensor_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        if rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "Tensors must be MLU and dense"
            ):
                send_tensor = _build_tensor(rank + 1)
                send_op = dist.P2POp(dist.isend, send_tensor, 1)
                req = dist.batch_isend_irecv([send_op])
                req.wait()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_op_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        if rank == 0:
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            with self.assertRaisesRegex(
                RuntimeError, "^Invalid ``op``"
            ):
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                send_op = dist.P2POp(dist.broadcast, send_tensor, 1)
                req = dist.batch_isend_irecv([send_op])
                req.wait()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_op_list_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        if rank == 0:
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            with self.assertRaisesRegex(
                RuntimeError, "^Invalid ``p2p_op_list``"
            ):
                send_tensor = _build_tensor(rank + 1)
                req = dist.batch_isend_irecv([1, 2])
                req.wait()

        dist.barrier()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_mixed_backend_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        group_gloo = dist.new_group(ranks=[0, 1], backend="gloo")
        group_cncl = dist.new_group(ranks=[0, 1], backend="cncl")
        if rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "All ops need to use the same group"
            ):
                send_tensor = _build_tensor(rank + 1)
                send_op_gloo = dist.P2POp(dist.isend, send_tensor, 1, group_gloo)
                send_op_cncl = dist.P2POp(dist.isend, send_tensor, 1, group_cncl)
                req = dist.batch_isend_irecv([send_op_gloo, send_op_cncl])
                req.wait()
        dist.destroy_process_group(group_cncl)

    def test_batch_isend_irecv_ring_exchange_cncl(self):
            _, rank = self._init_global_test()
            dist.barrier()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)

            send_tensor = _build_tensor(world_size, device_id=device_id)
            recv_tensor = _build_tensor(world_size, value=-1, device_id=device_id)
            send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

            dist.barrier()

    def test_batch_isend_irecv_no_rank_zero_cncl(self):
        _, rank = self._init_global_test()
        world_size = dist.get_world_size()
        # Ensure the process group has been fully initialized (needed by
        # the first sub-group batch_isend_irecv call)
        dist.barrier()
        if world_size > 2 :
            rank = dist.get_rank()
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            p2p_op_list = []

            if rank == 1:
                peer = 2
            elif rank == 2:
                peer = 1

            if rank in [1, 2]:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(peer + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, peer)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, peer)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            dist.barrier()         

    def _test_barrier_helper(self, group, rank):
        dist.barrier()   # test barrier before set device
        torch.mlu.set_device(rank % torch.mlu.device_count())
        WAIT_TIME = 10  # seconds

        # Because MLU does not support Double currently, the precision of the float cast result
        # of time.time() is not enough, so we remainder the value by 100000
        for src in group:
            expected_time = self.to_device(torch.FloatTensor(1).fill_(0.0))
            if src == rank:
                expected_time.fill_(time.time() % 100000 + WAIT_TIME)
                dist.broadcast(expected_time, src)
                time.sleep(WAIT_TIME + 0.1)
                dist.barrier()
            else:
                dist.broadcast(expected_time, src)
                dist.barrier()
                finish_time = time.time() % 100000
                self.assertGreaterEqual(float(finish_time), float(expected_time.item()),
                  "destination rank: %d, my rank: %d" % (src, rank))

    #@unittest.skip("not test")
    def test_barrier(self):
        group, rank = self._init_global_test()
        self._test_barrier_helper(group, rank)

    def _test_all_to_all_single_equal_split_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        in_tensor = (torch.ones([size, size]) * rank).mlu()
        expected_tensor = torch.cat([torch.ones([1, size]) * i for i in group])
        out_tensor = (torch.ones([size, size]) * -1).mlu()
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertTensorsEqual(out_tensor.cpu(), expected_tensor, 0)

    def _test_all_to_all_single_unequal_split_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        in_splits = [i + 1 for i in group]
        out_splits = [rank + 1 for _ in group]
        in_tensor = (torch.ones([sum(in_splits), size]) * rank).mlu()
        out_tensor = (torch.zeros([(rank + 1) * size, size])).mlu()
        expected_tensor = torch.cat([torch.ones([rank + 1, size]) * i for i in group])
        dist.all_to_all_single(out_tensor, in_tensor, out_splits, in_splits)
        self.assertTensorsEqual(out_tensor.cpu(), expected_tensor, 0)

    def _test_all_to_all_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        in_splits = [i + 1 for i in group]
        dtype_list = ["torch.FloatTensor", "torch.HalfTensor", "torch.CharTensor",
            "torch.ByteTensor", "torch.IntTensor", "torch.LongTensor",
            "torch.DoubleTensor", "torch.BoolTensor"]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        for ttype in dtype_list:
            in_tensors = [
                (torch.ones([in_splits[i], size]) * rank).type(ttype).mlu() for i in group
            ]
            out_tensors = [torch.zeros([(rank + 1), size]).type(ttype).mlu() for _ in group]
            expected_tensors = [(torch.ones([rank + 1, size]) * i).type(ttype) for i in group]
            dist.all_to_all(out_tensors, in_tensors)
            for out, expt in zip(out_tensors, expected_tensors):
                self.assertTensorsEqual(out.cpu(), expt, 0)

    #@unittest.skip("not test")
    def test_all_to_all_single_equal_split(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_single_equal_split_helper(group, rank)

    #@unittest.skip("not test")
    def test_all_to_all_single_unequal_split(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_single_unequal_split_helper(group, rank)

    #@unittest.skip("not test")
    def test_all_to_all(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_helper(group, rank)

    def test_ddp_grad_div_uneven_inputs(self):
        # Test gradient division during training with join() API. If
        # divide_by_initial_world_size=False, we scale by the effective world
        # size when allreducing grads.
        dim = 5
        batch = 1
        grad_scale = 50
        model = nn.Linear(dim, dim, bias=False)
        inp = torch.ones(batch, dim, device=self.rank) * grad_scale
        net = torch.nn.parallel.DistributedDataParallel(
            model.mlu(self.rank), device_ids=[self.rank], bucket_cap_mb=1
        )
        n_iters = 3
        if self.rank > 0:
            n_iters += 2

        with net.join(divide_by_initial_world_size=False):
            for _ in range(n_iters):
                loss = net(inp).sum()
                loss.backward()
                # The grad is always expected_grad, since we divide by the number
                # of currently active processes and inactive processes contribute
                # zero gradient. If we kept dividing by static initial world
                # size as processes leave, the grad would be smaller.
                expected_grad = torch.ones(dim, dim, device=self.rank) * grad_scale
                param = list(net.parameters())[0]
                self.assertEqual(expected_grad, param.grad)
                # Avoid accumulating grads so that it's the same every iteration
                net.zero_grad()
                torch.mlu.synchronize(device=self.rank)

        # If divide_by_initial_world_size=True (default), we always scale grads
        # by the initial world_size.
        with net.join(divide_by_initial_world_size=True):
            for i in range(n_iters):
                loss = net(inp).sum()
                loss.backward()
                effective_ws = dist.get_world_size()
                if i >= 3:
                    effective_ws -= 1
                expected_grad = (
                    torch.ones(dim, dim, device=self.rank)
                    * grad_scale
                    * effective_ws
                ) / dist.get_world_size()
                param = list(net.parameters())[0]
                self.assertEqual(expected_grad, param.grad)
                # Avoid accumulating grad so that it's the same every iteration.
                net.zero_grad()
                torch.mlu.synchronize(device=self.rank)

    @classmethod
    def _model_step(cls, model):
        for param in model.parameters():
            if param.grad is not None:
                param.data = param.data + param.grad
                param.grad.detach_()
                param.grad.zero_()

    # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
    @classmethod
    def _test_DDP_helper(cls, model, input_var, target, loss, flg,  scale_factor=1.0):
        model.train()
        output = model(input_var, flg)
        l = loss(output, target) *  scale_factor
        l.backward()

    def _assert_equal_param(self, param, param_DDP):
        self.assertEqual(len(param), len(param_DDP))
        for p, p_DDP in zip(param, param_DDP):
            self.assertEqual(p, p_DDP.cpu())

    def _test_multi_nodes_helper(self, param_DDP, rank):
        ps = []
        file_name = "params_" + str(self.world_size) + "cards.pt"
        single_node_params_file = os.path.join(TEMP_DIR, file_name)
        if  self.args.nnodes == 1:
            if rank == 0:
                for p in param_DDP:
                    ps.append(p.cpu())
                torch.save(ps, single_node_params_file)
        else:
            if os.path.exists(single_node_params_file):
                ps = torch.load(single_node_params_file, map_location = torch.device('cpu'))
                for p_sing, p_mult in zip(ps, param_DDP):
                    self.assertTensorsEqual(p_sing, p_mult.cpu(), 0)
            else:
                print("WARNING: " + single_node_params_file + " not found, if you want to "
                      "compare with single mlu card parameters of Net, please run single "
                      "node version of test_distributed.py firstly!")

    def _test_DDP_5iter(self, model_base, model_DDP, input_data, target, loss, local_bs,
                        rank, batch_size, base_is_mlu=False, offset=None):
        for _ in range(5):
            # single cpu training
            self._test_DDP_helper(model_base, input_data, target, loss, base_is_mlu)

            # DDP training, DDP scatters subsets of input_cpu to nodes/MLUs
            if offset is None:
                offset = rank * local_bs
            self._test_DDP_helper(model_DDP, input_data[offset: offset + local_bs],
                target[offset: offset + local_bs], loss, True,
                dist.get_world_size() * local_bs / batch_size)

            # Update weights and run a second iteration to shake out errors
            self._model_step(model_base)
            self._model_step(model_DDP)
            self._assert_equal_param(list(model_base.parameters()),
                                     list(model_DDP.module.parameters()))

            # Shuffle the input so that DDP input is different
            input_data = input_data[torch.randperm(batch_size)]
        self._test_multi_nodes_helper(list(model_DDP.module.parameters()), rank)

    def _test_DistributedDataParallel(self, rank):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())

        # cpu training setup
        model = Net()
        #model.fc1.weight.register_hook(hook)

        # DDP training setup
        model_DDP = copy.deepcopy(model)
        model_DDP.to('mlu')
        # can use find_unused_parameters=True
        model_DDP = nn.parallel.DistributedDataParallel(model_DDP,
            device_ids=[rank % torch.mlu.device_count()])
        def hook(grad):     # pylint: disable=W0612
            print("hook no_grad_param: ", model_DDP.module.no_grad_param.size(),
              model_DDP.module.no_grad_param.cpu())
            return grad
        #model_DDP.module.fc1.weight.register_hook(hook)

        # dummy data initialization
        local_bs = 1
        global_bs = self.args.nproc_per_node * self.args.nnodes * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model,
            model_DDP,
            input_cpu,
            target,
            loss,
            local_bs,
            rank,
            global_bs,
        )

    #@unittest.skip("not test")
    def test_distributedDataParallel(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)


    #@unittest.skip("not test")
    def test_distributedDataParallel_side_stream(self):
        torch.manual_seed(1)
        os.environ["PYTORCH_DDP_USE_SIDE_STREAM"]="0"
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)
        os.environ["PYTORCH_DDP_USE_SIDE_STREAM"]="1"
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)
        try:
            os.environ.pop("PYTORCH_DDP_USE_SIDE_STREAM")
        except Exception as e:
            # Ignore Errors of os.environ.
            pass


    #@unittest.skip("not test")
    def test_distributedDataParallel_non_default_stream(self):
        _, rank = self._init_global_test()
        dev_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(dev_id)
        queue = torch.mlu.Stream()
        with torch.mlu.stream(queue):
            net = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Linear(1, 1, bias=False).mlu(dev_id), device_ids=[rank]
            )
            for i in range(1000):
                # Clear gradients manually
                grad = net.module.weight.grad
                if grad is not None:
                    grad.detach_()
                    grad.zero_()
                # Forward + BW
                batch = torch.tensor([rank]).float().mlu(dev_id)
                loss = net(batch).sum()
                loss.backward()
                # For each worker, the gradient on the weight should be worker_rank.
                grad = net.module.weight.grad
                avg = grad.clone()
                # All-reducing the gradient averages should give us the gradient
                # average. If not, then one of the workers has not correctly
                # written back the averaged gradient before this all-reduce call.
                dist.all_reduce(avg)
                world_size = self.world_size
                avg.div_(world_size)
                expected_grad = sum(i for i in range(world_size)) / world_size
                self.assertEqual(
                    avg[0, 0],
                    expected_grad,
                    msg=f"Expected gradient of {expected_grad} but got {avg} on rank {rank}",
                )

    def _test_DistributedDataParallel_SyncBatchNorm(self, rank, model, size_i, size_t,
        diff_input_bs=False):
        # mlu training setup
        model_mlu = copy.deepcopy(model)
        model_mlu.to('mlu')

        # DDP training setup
        model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
        model_DDP.to('mlu')
        model_DDP = nn.parallel.DistributedDataParallel(model_DDP,
            device_ids=[rank % torch.mlu.device_count()])

        # dummy data initialization
        local_bs = rank + 2 if diff_input_bs else 2
        bs_offset = int((rank + 3) * rank / 2) if diff_input_bs else None
        global_bs =  int((self.world_size + 3) * self.world_size / 2) if diff_input_bs \
            else self.world_size * local_bs
        input_cpu = torch.randn(global_bs, *size_i)
        target = torch.randn(global_bs, *size_t)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model_mlu,
            model_DDP,
            input_cpu,
            target,
            loss,
            local_bs,
            rank,
            global_bs,
            True,
            offset=bs_offset
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet()
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [4])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_No_Affine(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet(False)
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [4])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_2D_Input(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = OnlyBatchNormNet(nn.BatchNorm1d(2))
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [2])

    #@unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_5D_Input(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = OnlyBatchNormNet(nn.BatchNorm3d(99))
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model,
          [99, 10, 215, 7], [99, 10, 215, 7])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Gradient(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet()
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [4], diff_input_bs=True)

    #@unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        model = nn.SyncBatchNorm(2, momentum=0.99).to('mlu')
        model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        input_var = []
        for i in range(self.world_size):
            input_var_rank = torch.cat([
                torch.ones(2, 1, 2 ** (i + 1)) * (0.1 ** (i - 1)),
                torch.ones(2, 1, 2 ** (i + 1)) * (0.3 ** (i - 1))
            ], dim=1)
            input_var.append(input_var_rank)

        all_input_var = torch.cat(
            [x.permute(1, 0, 2).contiguous().view(model.num_features, -1) for x in input_var],
            dim=1)

        for i in range(100):
            y = model_ddp(input_var[rank].to("mlu"))
            y.mean().backward()

        running_mean, running_var = model_ddp.module.running_mean, model_ddp.module.running_var
        torch.testing.assert_allclose(running_mean.cpu(), all_input_var.cpu().mean(1))
        torch.testing.assert_allclose(running_var.cpu(), all_input_var.cpu().var(1))

    #@unittest.skip("not test")
    def test_abnormal_and_api(self):
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        tensors = [self.to_device(torch.randn(2))]
        pg = _get_default_group()

        # test basic api
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)
        self.assertTrue(dist.is_initialized())

        # test unsupported communicate op
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.allgather_coalesced([tensors], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.allreduce_coalesced(tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.gather([tensors], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.scatter(tensors, [tensors])
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.recv_anysource(tensors, 0)
        # use abnormal input tensors to test
        with self.assertRaisesRegex(RuntimeError, "Tensor list input to scatter/gather " +
                                                  "must match number of collective participants"):
            pg.allgather([[tensors[0] for _ in range(self.world_size + 1)]], tensors)
        with self.assertRaisesRegex(RuntimeError, "Expecting all tensors on the same device"):
            pg.allgather([[tensors[0].cpu() for _ in range(self.world_size)]], tensors)
        with self.assertRaisesRegex(RuntimeError, "All tensor operands to scatter/gather " +
                                                  "must have the same number of elements"):
            pg.allgather([[tensors[0] for _ in range(self.world_size)]],
                [self.to_device(torch.randn(3))])
        with self.assertRaisesRegex(RuntimeError, "MLU Tensors must be on a single MLU " +
                                                  "device per process"):
            pg.allgather([tensors], [tensors[0], tensors[0]])
        with self.assertRaisesRegex(RuntimeError, "MLU Tensors must be on a single MLU " +
                                                  "device per process"):
            pg.allreduce([tensors[0], tensors[0]])
        with self.assertRaisesRegex(RuntimeError, "Cannot use ReduceOp.BAND with CNCL"):
            pg.allreduce(tensors[0], dist.ReduceOp.BAND)
        with self.assertRaisesRegex(RuntimeError, "Tensor list must be nonempty"):
            pg.broadcast([])
        with self.assertRaisesRegex(
            RuntimeError, "Tensor list mustn't be larger than the number of available MLUs"):
            exceed_tensor_list = [tensors[0]
                                  for _ in range(torch.mlu.device_count() + 1)]
            pg.broadcast(exceed_tensor_list)
        with self.assertRaisesRegex(RuntimeError, "Tensors must be MLU and dense"):
            pg.broadcast([tensors[0].cpu()])
        with self.assertRaisesRegex(
            RuntimeError, "Size of input tensor list not equal group size"):
            pg.alltoall(tensors, tensors, dist.AllToAllOptions())
        with self.assertRaisesRegex(RuntimeError, "Tensors must be contiguous"):
            pg.alltoall([tensors[0] for _ in range(self.world_size)],
                [self.to_non_dense(tensors[0]) for _ in range(self.world_size)],
                dist.AllToAllOptions())
        with self.assertRaisesRegex(RuntimeError,
            "input tensor must be the same type as the output tensor"):
            pg._reduce_scatter_base(tensors[0], tensors[0].half())
        with self.assertRaisesRegex(RuntimeError,
            "input tensor must be the same size as output size times world size"):
            pg._reduce_scatter_base(tensors[0], tensors[0])

        dist.destroy_process_group()
        self.assertFalse(dist.is_initialized())
        pg = None

class TestDistBackend(TestCase, _DistTestBase):
    MANAGER_PROCESS_RANK = -1
    sync_manager = None

    @staticmethod
    def manager_join(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MANAGER_PROCESS_RANK:
                self._join_and_reduce()  # pylint: disable=W0212
            else:
                fn(self)

        return wrapper

    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = cls.args.master_addr
        os.environ["MASTER_PORT"] = str(cls.args.master_port)
        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                if not getattr(fn, "__unittest_skip__", False):
                    setattr(cls, attr, cls.manager_join(fn))
        if cls.args.node_rank == 0:
            queue = Queue()
            QueueManager.register(str("get_queue"), lambda:queue)
            cls.sync_manager = QueueManager(address=("", cls.args.master_port + 10),
                                            authkey=b'abc')
            cls.sync_manager.start()
        else:
            QueueManager.register(str("get_queue"))
            cls.sync_manager = QueueManager(address=(cls.args.master_addr,
                                                     cls.args.master_port + 10), authkey=b'abc')

    @classmethod
    def tearDownClass(cls):
        if cls.args.node_rank == 0:
            queue = cls.sync_manager.get_queue()
            while queue.empty() is False:
                time.sleep(0.1)
            cls.sync_manager.shutdown()

    def setUp(self):
        super(TestDistBackend, self).setUp()   # pylint: disable=R1725
        self.processes = []
        self.rank = self.MANAGER_PROCESS_RANK
        for rank in range(self.args.node_rank * self.args.nproc_per_node,
                self.args.node_rank * self.args.nproc_per_node + self.args.nproc_per_node):
            self.processes.append(self._spawn_process(rank))

    def tearDown(self):
        super(TestDistBackend, self).tearDown()   # pylint: disable=R1725

        for p in self.processes:
            p.terminate()

    def _spawn_process(self, rank):
        os.environ["RANK"] = str(rank)
        name = "process " + str(rank)
        process = multiprocessing.Process(
            target=self._run, name=name, args=(rank, ))
        process.start()
        return process

    def _barrier(self, rank):
        self.sync_manager.connect()
        q = self.sync_manager.get_queue()
        if rank == 0:
            for _ in range(self.world_size - 1):
                q.put(str(os.getpid()))
        else:
            print("received finish signal from Process", q.get())

    def _run(self, rank):
        if torch.mlu.device_count() < args.nproc_per_node:
            print("Lack MLU Device !!!!!!")
            sys.exit(0)
        
        self.rank = rank
        self.world_size = self.args.nproc_per_node * self.args.nnodes
        try:
            print("begin init Process", os.getpid())
            dist.init_process_group(backend='cncl', init_method=INIT_METHOD,
                world_size=self.world_size, rank=self.rank)
            print("end init Process", os.getpid())
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(SKIP_IF_BACKEND_UNAVAILABLE)
            raise

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()

        # must close the current listenning socket before doing barrier,
        # otherwise the connecting request of the pg of the next test
        # case might be listened
        if dist.is_initialized():
            dist.destroy_process_group()
        self._barrier(rank)

        sys.exit(0)

    def _join_and_reduce(self):
        join_timeout = get_timeout(self.id()) + self.args.delay_time
        for rank, process in enumerate(self.processes):
            process.join(join_timeout)
            self.assertFalse(process.is_alive(),
                             "Timeout waiting for rank %d to terminate" % rank)

        first_process = self.processes[0]
        for p in self.processes:
            self.assertEqual(p.exitcode, first_process.exitcode)

        if first_process.exitcode == SKIP_IF_BACKEND_UNAVAILABLE:
            raise unittest.SkipTest("Compiled without the cncl backend")

        self.assertEqual(first_process.exitcode, 0)

if __name__ == '__main__':
    args = parse_args()

    if args.nnodes > 1:
        args.connects = -1

    if args.connects > 0:
        os.environ["CNCL_MLULINK_DISABLE"] = "1"
    if args.connects > 1:
        os.environ["CNCL_P2P_LEVEL"] = "0"
    if args.connects > 2:
        os.environ["CNCL_SHM_DISABLE"] = "1"

    distutils.dir_util.mkpath(TEMP_DIR)

    _DistTestBase.args = args
    unittest.main(argv=[sys.argv[0]] + args.unittest_args, verbosity=2)
