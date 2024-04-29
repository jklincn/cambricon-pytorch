from __future__ import print_function

import sys
import logging
import copy
import os
import unittest
from itertools import product
import torch
from torch.nn import GroupNorm

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

NC = [(0, 1), (1, 2), (2, 9), (4, 16), (3, 64), (15, 8), (7, 27)]
G = [1, 1, 3, 16, 4, 8, 27]

HxW = [(), (1,), (2, 0), (1,1), (2, 7), (3, 5, 4, 2), (1, 1, 1, 1, 1, 1), (5, 8, 4, 1, 7, 3)]

affines = [True, False]
dtypes = [torch.float, torch.half, torch.double]

def shape_list():
    shapes = []
    groups = []
    nd_shapes = []
    for i, sf in enumerate(NC):
        group = G[i]
        for ef in HxW:
            shape = sf + ef
            shapes.append(shape)
            groups.append(group)
            nd_shape = shape + (2,)
            nd_shapes.append(nd_shape)
    return shapes, groups, nd_shapes

shapes, groups, nd_shapes = shape_list()

class TestGroupNormOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_contiguous(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group in zip(shapes, groups):
                layer = GroupNorm(group, shape[1], affine=affine)
                x = torch.randn(shape).to(dtype)
                x_mlu = x.to('mlu')
                out_cpu = layer(x.to(torch.float))
                layer = layer.to('mlu').to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_channel_last(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group in zip(shapes, groups):
                layer = GroupNorm(group, shape[1], affine=affine)
                x = self.convert_to_channel_last(torch.randn(shape).to(dtype))
                x_mlu = x.to('mlu')
                out_cpu = layer(x.to(torch.float))
                layer = layer.to('mlu').to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_not_dense(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group, nd_shape in zip(shapes, groups, nd_shapes):
                layer = GroupNorm(group, shape[1], affine=affine)
                x_nd = torch.randn(nd_shape).to(dtype)
                x = x_nd[..., 1]
                x_mlu = x.to('mlu')
                out_cpu = layer(x.to(torch.float))
                layer = layer.to('mlu').to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_backward(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group, nd_shape in zip(shapes, groups, nd_shapes):
                if dtype == torch.half:
                    dtype = torch.float
                # (FIXME): group_norm_backward is insufficient accuracy in some dtype and some shape
                er = 0.05
                layer = GroupNorm(group, shape[1], affine=affine)
                x_nd = torch.randn(nd_shape, dtype=dtype, requires_grad=True)
                x = x_nd[..., 1]
                x_mlu = x.to('mlu')
                out_cpu = layer(x.to(torch.float))
                grad = torch.randn(nd_shape, dtype=dtype)
                grad_cpu = grad.to(out_cpu.dtype)[..., 1].view(out_cpu.shape)
                out_cpu.backward(grad_cpu)
                x_grad_cpu = copy.deepcopy(x_nd.grad)
                x_nd.grad.zero_()
                if affine:
                    gamma_grad_cpu = copy.deepcopy(layer.weight.grad)
                    beta_grad_cpu = copy.deepcopy(layer.bias.grad)
                    layer.weight.grad.zero_()
                    layer.bias.grad.zero_()
                layer = layer.to('mlu').to(dtype)
                out_mlu = layer(x_mlu)
                grad_mlu = grad.to('mlu').to(out_mlu.dtype)[..., 1].view(out_mlu.shape)
                out_mlu.backward(grad_mlu)
                x_grad_mlu = copy.deepcopy(x_nd.grad)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), er, use_MSE=True)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x_grad_mlu.cpu().float(), er, use_MSE=True)
                if affine:
                    gamma_grad_mlu = copy.deepcopy(layer.weight.grad)
                    beta_grad_mlu = copy.deepcopy(layer.bias.grad)
                    self.assertTensorsEqual(
                        gamma_grad_cpu, gamma_grad_mlu.cpu().float(), er, use_MSE=True)
                    self.assertTensorsEqual(
                        beta_grad_cpu, beta_grad_mlu.cpu().float(), er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_exceptions(self):
        shape = (3, 9, 3, 3)
        x = torch.randn(shape)
        x_mlu = x.to('mlu')
        layer = GroupNorm(3, 6)
        layer = layer.to('mlu')
        msg = "Expected weight to be a vector of size equal to the number of channels in " + \
              "input, but got weight of shape [6] and input of shape [3, 9, 3, 3]"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)

        layer = GroupNorm(3, 9)
        layer = layer.to('mlu')
        x_mlu = x_mlu.to(torch.int)
        msg = "GroupNorm only support float, half and double type inputs, but got dtype: Int"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)

        layer = GroupNorm(3, 9)
        layer = layer.to('mlu')
        x_mlu = x_mlu.to(torch.half)
        msg = "GroupNorm only support same dtypes of input, weight and bias, but got " + \
              "input dtype: Half weight dtype: Float bias dtype: Float"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)


if __name__ == '__main__':
    unittest.main()
