from __future__ import print_function
import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")

from common_utils import (
    testinfo, run_tests, TestCase, TEST_LARGETENSOR, largeTensorTest, TEST_BFLOAT16)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestClampOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_clamp(self):
        shape_list = [(1, 2, 3, 4), (2, 144, 7, 15), (23, 4), (256)]
        dtype_list = [torch.float,
                      torch.half,
                      torch.double,
                      torch.long,
                      torch.int,
                      torch.int16,
                      torch.int8]  # half is not support for cpu
        min_list = (1, 2, None)
        max_list = (10, 100, None)
        product_list = product(shape_list,
                               dtype_list,
                               min_list,
                               max_list)
        for shape, dtype, min_, max_ in product_list:
            if max_ is None and min_ is None:
                continue
            if dtype == torch.half:
                x = torch.randn(shape, dtype=torch.float)
                y = torch.randn(shape, dtype=torch.float)
            else:
                x = torch.randn(shape).to(dtype)
                y = torch.randn(shape).to(dtype)
            out_cpu = torch.clamp(x, min_, max_)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype), min_, max_)
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test clamp.out
            out_cpu = torch.clamp(x, min_, max_, out=y)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype),
                                  min_, max_,
                                  out=self.to_mlu_dtype(y, dtype))
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test inplace operation
            x_cpu = copy.deepcopy(x)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x_cpu.clamp_(min_, max_)
            x_mlu.clamp_(min_, max_)
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)


    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_clamp_bfloat16(self):
        shape_list = [(1, 2, 3, 4), (2, 144, 7, 15), (23, 4), (256)]
        dtype_list = [torch.bfloat16]
        min_list = (1, 2, None)
        max_list = (10, 100, None)
        #TODO(guwei):Need to check the err threshold
        err = 3e-2
        product_list = product(shape_list,
                               dtype_list,
                               min_list,
                               max_list)
        for shape, dtype, min_, max_ in product_list:
            if max_ is None and min_ is None:
                continue
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.clamp(x, min_, max_)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype), min_, max_)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # test clamp.out
            out_cpu = torch.clamp(x, min_, max_, out=y)
            out_mlu = torch.clamp(self.to_mlu_dtype(x, dtype),
                                  min_, max_,
                                  out=self.to_mlu_dtype(y, dtype))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # test inplace operation
            x_cpu = copy.deepcopy(x)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x_cpu.clamp_(min_, max_)
            x_mlu.clamp_(min_, max_)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_backward(self):
        for shape in [(2,3), (8, 224, 224), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)

            out_cpu = torch.clamp(x, 0.1, 10)
            out_mlu = torch.clamp(self.to_device(x), 0.1, 10)

            grad = torch.randn(out_cpu.shape, dtype=torch.float)

            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)

            x_cpu = copy.deepcopy(x)
            x.grad.zero_()

            outmlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(outmlu_ptr, out_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)
            self.assertTensorsEqual(x, x_cpu, 0)

            # test inplace operation
            x.grad.zero_()

            x_mlu = self.to_device(x)
            x_mlu.clamp_(0.1, 10)
            xmlu_ptr = x_mlu.data_ptr()
            x_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(xmlu_ptr, x_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_not_dense(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = x[:,:,:,3:6]
            out_cpu = torch.clamp(input_cpu, 0.1, 10)
            input_mlu = self.to_device(x)
            input_mlu = input_mlu[:,:,:,3:6]
            out_mlu = torch.clamp(input_mlu, 0.1, 10)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_channel_last(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14, 25)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = self.convert_to_channel_last(x)
            out_cpu = torch.clamp(input_cpu, 0.1, 10)
            input_mlu = self.to_device(x)
            input_mlu = self.convert_to_channel_last(input_mlu)
            out_mlu = torch.clamp(input_mlu, 0.1, 10)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_exception(self):
        a = torch.randn(3).to('mlu')
        ref_msg = "torch.clamp: At least one of 'min' or 'max' must not be None"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.clamp()
        with self.assertRaises(RuntimeError) as info:
            a=torch.randn(1,2,3,4).to('mlu')
            b=torch.randn(4).to('mlu')
            a.clamp(min=b, max=b)
        ref_msg = "torch_mlu unsupport clamp.Tensor."
        self.assertEqual(info.exception.args[0],ref_msg)

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`")
    @largeTensorTest('46GB')
    def test_clamp_large(self):
        for shape in [(5, 1024, 1024, 1024)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = torch.clamp(x, 0.1, 10)
            input_mlu = self.to_device(x)
            out_mlu = torch.clamp(input_mlu, 0.1, 10)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)



if __name__ == '__main__':
    run_tests()
