from __future__ import print_function

import sys
import os
import unittest
import logging
import torch
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase


class TestRollOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_roll(self):
        test_case_list = [((128, 15, 30), (-5, 20, -14), (0, 2, 1)),
                          ((10, 128, 15, 30), (-5), ()),
                          ((10, 128, 15, 30), (11, 5), (0, 1)),
                          ((10, 128, 15, 30), (8, -10, 6), (1, 0, 2)),
                          ((10, 128, 15, 30), (-5, 20, -14, 20), (0, 2, 0, 3)),
                          ((2, 10, 128, 15, 30), (-5, 20, -14, 20), (0, 2, 0, 3))]
        func_list = [self.convert_to_channel_last,
                     self.to_non_dense, lambda x:x]
        for (shape, shifts, dims), func in product(test_case_list, func_list):
            input_self = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.roll(func(input_self), shifts, dims)
            out_mlu = torch.roll(func(self.to_mlu(input_self)), shifts, dims)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_roll2(self):
        test_case_list = [((10, 10, 64, 30), (11, 5), (1, -15), (0, 1))]
        for shape, shifts, shifts_1, dims in test_case_list:
            input_self = torch.randn(shape, dtype=torch.float)
            out_cpu_1 = torch.roll(input_self, shifts, dims)
            out_cpu_2 = torch.roll(input_self, shifts_1, dims)
            self.assertTensorsEqual(out_cpu_1, out_cpu_2, 0)
            out_mlu_1 = torch.roll(self.to_mlu(input_self), shifts, dims)
            out_mlu_2 = torch.roll(self.to_mlu(input_self), shifts_1, dims)
            self.assertTensorsEqual(out_mlu_1.cpu(),
                                    out_mlu_2.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_roll_dtype(self):
        dtype_list = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
                      torch.half, torch.float, torch.double, torch.complex32, torch.complex64, torch.complex128]
        for dtype in dtype_list:
            x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype).view(4, 2)
            out_cpu = torch.roll(x, 1)
            out_mlu = torch.roll(x.mlu(), 1)
            self.assertEqual(out_mlu.cpu(),out_cpu, 0)


if __name__ == '__main__':
    unittest.main()
