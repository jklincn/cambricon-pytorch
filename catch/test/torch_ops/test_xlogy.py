from __future__ import print_function
import logging
import unittest
import sys
import os
import copy
import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import (
    testinfo, run_tests, TestCase, TEST_LARGETENSOR, largeTensorTest)
logging.basicConfig(level=logging.DEBUG)

class TestXlogyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_xlogy_contiguous(self):
        shape_list = [(16,384,3072), (16, 0, 88)]
        data_types = [torch.float, torch.half, torch.double]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                y = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.xlogy(x, y)
                out_mlu = torch.xlogy(self.to_mlu_dtype(x, data_type), self.to_mlu_dtype(y, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True, allow_inf=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_xlogy_out(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x_mlu = torch.randn(shape, dtype=data_type, device='mlu')
                y_mlu = torch.randn(shape, dtype=data_type, device='mlu')
                x = x_mlu.cpu()
                y = y_mlu.cpu()
                out_cpu = torch.randn(shape, dtype=data_type, device='mlu').cpu()
                out_mlu = torch.randn(shape, dtype=data_type, device='mlu')
                torch.xlogy(x, y, out=out_cpu)
                torch.xlogy(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True, allow_inf=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_xlogy_type(self):
        shape_list = [(1, 3, 16, 16)]
        type_list = [torch.double, torch.float, torch.half, torch.long,
                     torch.int, torch.short, torch.bool]
        for shape in shape_list:
            for type in type_list:
                x_cpu = torch.randn(shape).to(type)
                x_mlu = self.to_mlu(x_cpu)
                y_cpu = torch.randn(shape).to(type)
                y_mlu = self.to_mlu(y_cpu)
                if type == torch.half:
                    x_cpu = x_cpu.float()
                    y_cpu = y_cpu.float()
                out_cpu = torch.xlogy(x_cpu, y_cpu)
                out_mlu = torch.xlogy(x_mlu, y_mlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True, allow_inf=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_xlogy_nan_or_zero(self):
        x = torch.zeros(6,).to('mlu')
        y = torch.tensor([-1, 0, 1, float('inf'), 0, float('nan')]).to('mlu')
        print(y)
        output = torch.xlogy(x, y)
        output_baseline = torch.tensor([0., 0., 0., 0., 0., float('nan')])
        print(output)
        self.assertTensorsEqual(output.cpu(), output_baseline, 0.003, use_MSE=True, allow_inf=True)

    #TODO(PYTORCH-10129): cnnlLogic not implement large tensor.
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`")
    @largeTensorTest('26GB')
    def test_xlogy_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        data_types = [torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                y = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.xlogy(x, y)
                out_mlu = torch.xlogy(self.to_mlu_dtype(x, data_type), self.to_mlu_dtype(y, data_type))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE = True, allow_inf=True)


if __name__ == '__main__':
    unittest.main()
