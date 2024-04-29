from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import torch
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestMaskedScatter(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_tensor(self):
        types = [torch.half, torch.float, torch.double]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100, ), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=t)
            mask = torch.randn(shape) > 0
            source = torch.rand(shape, dtype=t)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            source_mlu = self.to_device(source)
            ori_ptr = x_mlu.data_ptr()
            if t == torch.half:
                x, source = x.float(), source.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_scatter_(x, mask, source)
            out_mlu = torch.Tensor.masked_scatter_(x_mlu, mask_mlu, source_mlu)
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(
                x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_channels_last_and_not_dense(self):
        shape = (100, 512, 2, 5)
        # channels last
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.randn(shape) > 0
        source = torch.rand(shape, dtype=torch.float)
        x = x.to(memory_format = torch.channels_last)
        mask = mask.to(memory_format = torch.channels_last)
        source = source.to(memory_format = torch.channels_last)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        source_mlu = self.to_device(source)
        x_mlu = x.to(memory_format = torch.channels_last)
        mask_mlu = mask.to(memory_format = torch.channels_last)
        source_mlu = source.to(memory_format = torch.channels_last)
        out_cpu = torch.Tensor.masked_scatter_(x, mask, source)
        out_mlu = torch.Tensor.masked_scatter_(x_mlu, mask_mlu, source_mlu)
        self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                use_MSE = True)
        # not dense
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.ones(shape, dtype=torch.bool)
        source = torch.rand(shape, dtype=torch.float)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        source_mlu = self.to_device(source)
        out_cpu = torch.Tensor.masked_scatter_(x[...,2], mask[...,2], source[...,2])
        out_mlu = torch.Tensor.masked_scatter_(x_mlu[...,2], mask_mlu[...,2], source_mlu[...,2])
        self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003,
                                use_MSE = True)
    #@unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_backward(self):
        x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],
                         dtype = torch.float, device = 'cpu', requires_grad = True)
        x_mlu = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],
                             dtype = torch.float, device = 'mlu', requires_grad = True)
        mask = torch.tensor([[0,1,1],[1,0,0],[0,0,1]]).bool()
        mask_mlu = mask.mlu()
        source = torch.zeros(3,3).to(torch.float)
        source_mlu = torch.zeros((3,3), dtype = torch.float, device = 'mlu')
        out_cpu = torch.masked_scatter(x, mask, source)
        out_mlu = torch.masked_scatter(x_mlu, mask_mlu, source_mlu)
        grad = torch.randn(3,3)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        out_grad_cpu = x.grad
        out_grad_mlu = x_mlu.grad
        self.assertTensorsEqual(
            out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(
            out_grad_cpu, out_grad_mlu.cpu(), 0.0, use_MSE=True)
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_exception(self):
        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float, device='mlu')
        mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device='mlu')
        src = torch.zeros(2, dtype=torch.float, device='mlu')
        ref_msg = r"Number of elements of source < number of ones in mask"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            dest.masked_scatter_(mask, src)
        ref_msg = r"masked_scatter: expected self and source to have same dtypes but got"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            dest.masked_scatter_(mask, src.double())
        ref_msg = r"masked_scatter: expected BoolTensor or ByteTensor for mask"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            dest.masked_scatter_(mask.int(), src)
        src_legal = torch.zeros(10, device = 'mlu', dtype = torch.float)
        ref_msg = r"masked_scatter_ received a mask with dtype torch.uint8, "
        with self.assertWarnsRegex(UserWarning, ref_msg):
            dest.masked_scatter_(mask.to(torch.uint8), src_legal)
    
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_with_mix_memory_format(self):
        x = self.convert_to_channel_last(torch.rand(2,2,2,2, dtype=torch.float))
        x_mlu = copy.deepcopy(x).mlu()
        mask = self.convert_to_channel_last(
               torch.tensor([[[[1., 1.], [0., 1.]],[[0., 0.],[0., 1.]]],
                            [[[0., 0.],[0., 0.]],[[0., 0.],[0., 1.]]]]).bool())
        source = torch.rand(5, dtype=torch.float)
        torch.ops.aten.masked_scatter_(x, mask, source)
        torch.ops.aten.masked_scatter_(x_mlu, self.to_device(mask), self.to_device(source))
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

if __name__ == '__main__':
    unittest.main()

