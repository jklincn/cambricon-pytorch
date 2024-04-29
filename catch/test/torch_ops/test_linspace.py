from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(2)

class TestLinspaceOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_linspace(self):
        start_list = [1, 3, 3.5, 3.5, 4.1, 8.9, 11]
        end_list = [2, 5, 2.5, 10.5, 11.3, 99.1, 121]
        steps_list = [0, 3, 1, 11, 6, 100, 121]
        dtype_list = [(torch.float, 0), (torch.half, 1e-3)]
        for dtype, err in dtype_list:
            for start, end, steps in product(start_list, end_list, steps_list):
                x = torch.linspace(start, end, steps=steps, device="cpu", dtype=torch.float)
                x_mlu = torch.linspace(start, end, steps=steps, device="mlu", dtype=dtype)
                self.assertTensorsEqual(x, x_mlu.cpu(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_linspace_out(self):
        out_shape_list = [(1,), (2, 4), (2, 1, 3), (16, 8, 64)]
        start_list = [1, 3, 3.5, 3.5, 4.1, 8.9, 11]
        end_list = [2, 5, 2.5, 10.5, 11.3, 99.1, 121]
        steps_list = [0, 3, 1, 1, 11, 6]#121
        # start=1, end=5, steps=121 cause different results for cpu and cuda,
        # while mlu and cuda are consistent
        dtype_list = [(torch.float, 0), (torch.half, 1e-3), (torch.int32, 0)]
        for dtype, err in dtype_list:
            for out_shape, start, end, steps in product(out_shape_list, start_list, end_list, steps_list):
                out = torch.randn(out_shape)
                if dtype == torch.int32:
                    out = out.to(torch.int32)
                out_mlu = self.to_mlu_dtype(out, dtype)
                x = torch.linspace(start, end, steps=steps,
                                        device="cpu", out=out)
                x_mlu = torch.linspace(start, end, steps=steps,
                                        device="mlu", out=out_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
                self.assertTensorsEqual(out, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_linspace_permute(self):
        res = torch.randn((3, 3, 1000))
        res = res.permute(2, 0, 1)
        res_mlu = res.to('mlu')
        torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device='cpu', out=res)
        torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, out=res_mlu)
        self.assertEqual(
            res_mlu.flatten(), torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device='mlu', dtype=torch.float32))
        self.assertEqual(
            res.flatten(), torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device='cpu', dtype=torch.float32))
        self.assertEqual(res_mlu.cpu(), res)

    # @unittest.skip("not test")
    @testinfo()
    def test_linspace_exception(self):
        ref_msg = "number of steps must be non-negative"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.linspace(1, 10, -1, device='mlu')
        
        ref_msg = "element number of output should be less than or equal to 2\^25 in cnnl_linspace."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.linspace(1, 10, 33554433, device='mlu')
if __name__ == '__main__':
    unittest.main()
