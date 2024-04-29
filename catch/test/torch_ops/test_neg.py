from __future__ import print_function
import os

import sys
import logging
import unittest
import torch
import copy
from itertools import product
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import (
    testinfo, run_tests, TestCase, TEST_LARGETENSOR, largeTensorTest, TEST_BFLOAT16)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

class TestNegOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_neg(self):
        shape_list = [(5, 10, 2, 2), (2, 3, 4),
                      (2, 2, 2, 1, 1, 3)]
        dtype_list = [torch.float, torch.half, \
                      torch.double, torch.long, torch.int]
        func_list = [lambda x:x, self.to_non_dense, self.convert_to_channel_last]
        torch_func_list = [torch.neg, torch.negative]
        err = 0.0
        for shape, dtype, func, torch_func in \
          product(shape_list, dtype_list, func_list, torch_func_list):
            x = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
            x_mlu = copy.deepcopy(x).mlu()
            out_cpu = torch_func(func(x))
            out_mlu = torch_func(func(x_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_neg_bfloat16(self):
        shape_list = [(5, 10, 2, 2), (2, 3, 4),
                      (2, 2, 2, 1, 1, 3)]
        dtype_list = [torch.bfloat16]
        func_list = [lambda x:x, self.to_non_dense, self.convert_to_channel_last]
        torch_func_list = [torch.neg, torch.negative]
        err = 0.0
        for shape, dtype, func, torch_func in \
          product(shape_list, dtype_list, func_list, torch_func_list):
            x = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
            x_mlu = copy.deepcopy(x).mlu()
            out_cpu = torch_func(func(x))
            out_mlu = torch_func(func(x_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_inplace(self):
        shape_list = [(5, 10, 2, 2), (2, 3, 4),
                      (2, 2, 2, 1, 1, 3)]
        dtype_list = [torch.float, torch.half, \
                      torch.double, torch.long, torch.int]
        func_list = [lambda x:x, self.to_non_dense, self.convert_to_channel_last]
        torch_func_list = [torch.neg_, torch.negative_]
        for shape, dtype, func, torch_func in \
          product(shape_list, dtype_list, func_list, torch_func_list):
            x = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
            x_cpu = func(x)
            x_mlu = func(copy.deepcopy(x).mlu())
            ori_ptr = x_mlu.data_ptr()
            torch_func(x_cpu)
            torch_func(x_mlu)
            self.assertEqual(x_cpu.dtype, x_mlu.dtype)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_out(self):
        shape_list = [(5, 2, 2, 2, 4), (10, 3, 3, 3), (2, 3, 4),
                      (2, 2, 2, 1, 1, 3), (10,), ()]
        dtype_list = [torch.float, torch.half, \
                      torch.double, torch.long, torch.int]
        func_list = [lambda x:x, self.to_non_dense, self.convert_to_channel_last]
        torch_func_list = [torch.neg, torch.negative]
        for shape, dtype, func, torch_func in \
          product(shape_list, dtype_list, func_list, torch_func_list):
            x = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
            x_cpu = func(x)
            x_mlu = func(copy.deepcopy(x).mlu())
            out_cpu = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
            out_mlu = copy.deepcopy(out_cpu).mlu()
            ori_ptr = out_mlu.data_ptr()
            out_cpu = torch_func(x_cpu, out=out_cpu)
            out_mlu = torch_func(x_mlu, out=out_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertEqual(ori_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)
            # resize output
            out_cpu = torch.empty((0,), dtype=dtype)
            out_mlu = copy.deepcopy(out_cpu).mlu()
            ori_ptr = out_mlu.data_ptr()
            out_cpu = torch_func(x_cpu, out=out_cpu)
            out_mlu = torch_func(x_mlu, out=out_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTrue(ori_ptr != out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True)

    # Test is_non_overlapping_and_dense
    # @unittest.skip("not test")
    @testinfo()
    def test_neg_permute(self):
        shape_list = [(5, 10, 2, 2, 4), (10, 3, 3, 3), (2, 3, 4),
                (2, 2, 2, 1, 1, 3)]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range (4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(permute_shape[i])
            # test output
            torch.neg(x, out = out)
            torch.neg(x_mlu, out = out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)
            # test functional
            out = torch.neg(x)
            out_mlu = torch.neg(x_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)
            # test inplace
            torch.neg_(x)
            torch.neg_(x_mlu)
            self.assertTrue(x.stride() == x_mlu.stride())
            self.assertTrue(x.storage_offset() == x_mlu.storage_offset())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_neg_inf_nan_and_zero_element(self):
        # test inf and nan
        input_cpu = torch.randn(3, 2, 2, dtype=torch.float)
        input_cpu[0][0][0] = float('inf')
        input_cpu[0][0][1] = float('-inf')
        input_cpu[0][1][0] = float('nan')
        input_mlu = copy.deepcopy(input_cpu).mlu()
        out_cpu = torch.neg(input_cpu)
        out_mlu = torch.neg(input_mlu)
        self.assertEqual(out_cpu, out_mlu.cpu())
        # test zero element
        input_cpu = torch.randn(0, 2, 2, dtype=torch.float)
        input_mlu = copy.deepcopy(input_cpu).mlu()
        out_cpu = torch.neg(input_cpu)
        out_mlu = torch.neg(input_mlu)
        self.assertEqual(out_cpu, out_mlu.cpu())

    # @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`")
    @largeTensorTest('44GB')
    def test_neg_large(self):
        shape = (5, 1024, 1024, 1024)
        dtype, err = torch.half, 3e-3
        x = torch.testing.make_tensor(shape, dtype=dtype, device='cpu')
        x_mlu = copy.deepcopy(x).mlu()
        out_cpu = torch.neg(x)
        out_mlu = torch.neg(x_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True)

if __name__ == '__main__':
    run_tests()
