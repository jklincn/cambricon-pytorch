from __future__ import print_function

import sys
import logging
import os
import itertools
import copy
import unittest
import torch
from torch.nn.functional import grid_sample

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411
logging.basicConfig(level=logging.DEBUG)

class TestGridSamplerOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_grid_sampler_2d(self):
        # Accuracy: MLU's result is consistent with GPU but not CPU for
        # some non-deterministic behavior on device. Therefore, some
        # input data can not reach accuracy threshold compared with CPU.
        shape_list = [((1, 1, 6, 6), (1, 8, 8, 2)),
                      ((3, 1, 4, 4), (3, 9, 9, 2)),
                      ((6, 5, 3, 2), (6, 0, 3, 2)),
                      ((6, 0, 3, 2), (6, 4, 4, 2))]
        interp_mode=["bilinear", "nearest"]
        padding_mode=["zeros", "reflection"]
        align_corners = [False, True]
        type_list = [torch.float32]
        func_x_list = [lambda x:x, self.convert_to_channel_last, lambda x:x[...,::2]]
        func_g_list = [lambda x:x, self.convert_to_channel_last]
        param_list = [shape_list, type_list, interp_mode, padding_mode, \
                      align_corners, func_x_list, func_g_list]
        for shape, type, im, pm, ac, func_x, func_g in itertools.product(*param_list):
            if im == "nearest" and pm == "reflection":
                continue
            input = torch.randn(shape[0], dtype=type)
            input.requires_grad=True
            input_mlu = copy.deepcopy(input)
            grid = torch.randn(shape[1], dtype=type)
            out_cpu = grid_sample(func_x(input.float()), func_g(grid.float()), im, pm, ac)
            out_mlu = grid_sample(func_x(input_mlu.mlu()), func_g(grid.mlu()), im, pm, ac)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            grad = torch.ones_like(out_cpu)
            func_x(out_cpu).backward(func_x(grad))
            func_x(out_mlu).backward(func_x(grad.mlu()))
            self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_grid_sampler_2d_exception(self):
        input = torch.randn(1, 1, 28, 28).mlu()
        grid = torch.randn(1, 230, 352, 2).mlu()
        ref_msg = "interpolation_mode only support bilinear or nearest."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            grid_sample(input, grid, 'bicubic', 'zeros', False)
        ref_msg = "bilinear only support zeros or reflection padding."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            grid_sample(input, grid, 'bilinear', 'border', False)
        ref_msg = "nearest only support zeros padding."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            grid_sample(input, grid, 'nearest', 'reflection', False)

if __name__ == '__main__':
    unittest.main()
