from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import copy

import torch # pylint: disable=C0413

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo, run_tests, TestCase, TEST_LARGETENSOR, largeTensorTest)

logging.basicConfig(level=logging.DEBUG)

def to_mlu(tensor_cpu):
    return tensor_cpu.to('mlu')


class TestOps(TestCase):
    #@unittest.skip("not test")
    @testinfo()
    def test_std(self):
        keepdim_v = [True, False]
        unbiased_v = [True, False]
        def func_not_dense(x):
            return x[..., :-1]
        func_list = [lambda x:x, self.convert_to_channel_last, func_not_dense]
        dims = [-3, -2, -1, 0, 1, 2]
        for keepdim in keepdim_v:
            for unbiased in unbiased_v:
                for t in [torch.float, torch.double]:
                    for func in func_list:
                        for dim in dims:
                            x = torch.randn(10, 64, 32, dtype=t)
                            output_cpu = torch.std(
                                func(x), dim, keepdim=keepdim, unbiased=unbiased)
                            output_cpu_all = torch.std(
                                func(x), [], keepdim=keepdim, unbiased=unbiased)
                            output_mlu = torch.std(
                                func(to_mlu(x)), dim, keepdim=keepdim, unbiased=unbiased)
                            output_mlu_all = torch.std(
                                func(to_mlu(x)), [], keepdim=keepdim, unbiased=unbiased)
                            self.assertTensorsEqual(
                                output_cpu.float(), output_mlu.cpu(), 3e-3, use_MSE=True)
                            self.assertTensorsEqual(
                                output_cpu_all.float(), output_mlu_all.cpu(), 3e-3, use_MSE=True)

        unbiased_v = [True, False]
        for unbiased in unbiased_v:
            x = torch.randn(2, 3, 2, 3, dtype=torch.float)
            output_cpu = torch.std(x.float(), unbiased=unbiased)
            output_mlu = torch.std(to_mlu(x), unbiased=unbiased)
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_empty_tensor(self):
        keepdim_v = [True, False]
        unbiased_v = [True, False]
        def func_not_dense(x):
            return x[..., :-1]
        func_list = [lambda x:x, self.convert_to_channel_last, func_not_dense]
        for keepdim in keepdim_v:
            for unbiased in unbiased_v:
                for t in [torch.float, torch.double]:
                    for func in func_list:
                        x = torch.randn(10, 0, 32, dtype=t)
                        output_cpu = torch.std(
                            func(x), 2, keepdim=keepdim, unbiased=unbiased)
                        output_cpu_all = torch.std(
                            func(x), [], keepdim=keepdim, unbiased=unbiased)
                        output_mlu = torch.std(
                            func(to_mlu(x)), 2, keepdim=keepdim, unbiased=unbiased)
                        output_mlu_all = torch.std(
                            func(to_mlu(x)), [], keepdim=keepdim, unbiased=unbiased)
                        self.assertEqual(
                            output_cpu.shape, output_mlu.shape)
                        self.assertEqual(
                            output_cpu_all.shape, output_mlu_all.shape)

    #@unittest.skip("not test")
    @testinfo()
    def test_std_exceptions(self):
        x_shapes = [(2,0,4), (2,1,4)]
        dim = -5
        for shape in x_shapes:
            x = torch.randn(shape, dtype = torch.float)
            with self.assertRaises(IndexError) as info:
                torch.std(to_mlu(x), dim)
            ref_msg = "Dimension out of range (expected to be in range of [" +\
            str(-len(shape)) + ", " + str(len(shape)-1) + "], but got " + str(dim) + ")"
            self.assertEqual(info.exception.args[0], ref_msg)

    #@unittest.skip("not test")
    @testinfo()
    def test_std_out(self):
        keepdim_v = [True, False]
        unbiased_v = [True, False]
        dims = [-3, -2, -1, 0, 1, 2]
        for keepdim in keepdim_v:
            for unbiased in unbiased_v:
                for dim in dims:
                    x = torch.randn(2, 3, 4)
                    output_cpu = torch.randn_like(x)
                    output_mlu = output_cpu.mlu()
                    torch.std(x, dim, keepdim=keepdim, unbiased=unbiased, out=output_cpu)
                    torch.std(to_mlu(x), dim, keepdim=keepdim, unbiased=unbiased, out=output_mlu)
                    self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu(),
                                            3e-3, use_MSE=True)
                    # test with empty dim
                    output_cpu_all = torch.randn_like(x)
                    output_mlu_all = output_cpu_all.mlu()
                    torch.std(x, [], keepdim=keepdim, unbiased=unbiased, out=output_cpu_all)
                    torch.std(to_mlu(x), [], keepdim=keepdim, unbiased=unbiased, out=output_mlu_all)
                    self.assertTensorsEqual(output_cpu_all.float(), output_mlu_all.cpu(),
                                            3e-3, use_MSE=True)

        unbiased_v = [False, True]
        for unbiased in unbiased_v:
            x = torch.randn(2, 3, 2, 3, dtype=torch.float)
            output_cpu = torch.randn_like(x)
            output_mlu = output_cpu.mlu()
            torch.std(x.float(), [], unbiased=unbiased, out=output_cpu)
            torch.std(to_mlu(x), [], unbiased=unbiased, out=output_mlu)
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu(), 3e-3, use_MSE=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_std_out_nan(self):
        keepdim_v = [True, False]
        unbiased_v = [True, False]
        for keepdim in keepdim_v:
            for unbiased in unbiased_v:
                x = torch.randn(2, 0, 4, 3)
                output_cpu = torch.randn_like(x)
                output_mlu = output_cpu.mlu()
                torch.std(x, 1, keepdim=keepdim, unbiased=unbiased, out=output_cpu)
                torch.std(to_mlu(x), 1, keepdim=keepdim, unbiased=unbiased, out=output_mlu)
                self.assertEqual(output_cpu, output_mlu.cpu())
                # test with empty dim
                output_cpu_all = torch.randn_like(x)
                output_mlu_all = output_cpu_all.mlu()
                torch.std(x, [], keepdim=keepdim, unbiased=unbiased, out=output_cpu_all)
                torch.std(to_mlu(x), [], keepdim=keepdim, unbiased=unbiased, out=output_mlu_all)
                self.assertEqual(output_cpu, output_mlu.cpu())

    # TODO(hyl): dependcy div.out op
    #@unittest.skip("not test")
    @testinfo()
    def test_std_backward(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6),
                      (2, 512, 8), (1, 100),(24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.std(x, item[1], keepdim=item[0])
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to('mlu')
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu=torch.std(x_mlu, item[1], keepdim=item[0])
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`")
    @largeTensorTest('24GB')
    def test_std_large_exceptions(self):
        # [cnnlStdForward] Check failed: input_element_number < 0x80000000.
        # The tensor element number should < 2G, but the input tensor has 2698444800 elements.
        ref_msg = r"CNNL error: CNNL_STATUS_BAD_PARAM"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            shape_list = [(48, 4096, 13725),(1, 4096*48*13725)]
            for shape in shape_list:
                x = torch.randn(shape, dtype=torch.float32)
                out_cpu = torch.std(x,dim=0)
                out_mlu = torch.std(self.to_mlu(x),dim=0)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

if __name__ == "__main__":
    run_tests()
