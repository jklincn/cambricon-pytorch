from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo, run_tests, TestCase, TEST_LARGETENSOR, largeTensorTest)
logging.basicConfig(level=logging.DEBUG)


class TestAddcdivOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_inplace_contiguous(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 0, 7), (128, 64, 0, 7), (128, 64, 0, 1)),
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                a.addcdiv_(b, c, value = 0.35)

                ori_ptr = a_mlu.data_ptr()
                a_mlu.addcdiv_(self.to_mlu_dtype(b, data_type),\
                               self.to_mlu_dtype(c, data_type), value = 0.35)
                self.assertEqual(a_mlu.data_ptr(), ori_ptr)
                self.assertTensorsEqual(a.float(),
                                        a_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_inplace_channel_last(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                a.addcdiv_(b, c, value = 0.35)

                ori_ptr = a_mlu.data_ptr()
                a_mlu.addcdiv_(self.to_mlu_dtype(b, data_type),\
                               self.to_mlu_dtype(c, data_type), value = 0.35)
                self.assertEqual(a_mlu.data_ptr(), ori_ptr)
                self.assertTensorsEqual(a.float(),
                                        a_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_inplace_not_dense(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)[:, :, :, :int(shape_a[-1] / 2)]
                b = torch.rand(shape_b, dtype=torch.float)[:, :, :, :int(shape_b[-1] / 2)]
                c = torch.randint(low = 1, high = 10,\
                                  size = shape_c, dtype=torch.float)[:, :, :, :int(shape_c[-1] / 2)]

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                a.addcdiv_(b, c, value = 0.35)

                ori_ptr = a_mlu.data_ptr()
                a_mlu.addcdiv_(self.to_mlu_dtype(b, data_type),\
                               self.to_mlu_dtype(c, data_type), value = 0.35)
                self.assertEqual(a_mlu.data_ptr(), ori_ptr)
                self.assertTensorsEqual(a.float(),
                                        a_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_contiguous(self):
        shape_list = [((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
                      ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
                      ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out = torch.addcdiv(a, b, c, value = 0.35)
                out_mlu = torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_channel_last(self):
        shape_list = [((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
                      ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
                      ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out = torch.addcdiv(a, b, c, value = 0.35)
                out_mlu = torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_channel_last_expand(self):
        shape_list = [((128, 64, 7, 7), (1,1), (7,1)),
                      ((1024, 512, 3), (512,3), (512,3)),
                      ((512, 256, 3, 3, 4), [1], [1]),
                      ((512, 256, 3, 3, 4), (), ())]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out = torch.addcdiv(a, b, c, value = 0.35)
                out_mlu = torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)
    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_channel_last_expand(self):
        shape_list = [((128, 64, 7, 7), (1,1), (7,1)),
                      ((1024, 512, 3), (512,3), (512,3)),
                      ((1024, 512, 8, 6), (1024, 1, 8,6), (8, 6)),
                      ((8, 6), (1024, 10, 8,6), (8, 6)),
                      ((6), (1024, 10, 8,6), (8, 6)),
                      ((512, 256, 3, 3, 4), [1], [1]),
                      ((512, 256, 3, 3, 4), (), ())]
        data_types = [torch.float, torch.half]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)
                out = torch.randn(shape_a, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, value = 0.35, out=out)
                torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35,
                                        out=out_mlu)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_not_dense(self):
        shape_list = [((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
                      ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
                      ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)[:, :, :, :int(shape_a[-1] / 2)]
                b = torch.rand(shape_b, dtype=torch.float)[:, :, :, :int(shape_b[-1] / 2)]
                c = torch.randint(low = 1, high = 10,\
                                  size = shape_c, dtype=torch.float)[:, :, :, :int(shape_c[-1] / 2)]

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out = torch.addcdiv(a, b, c, value = 0.35)
                out_mlu = torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_contiguous(self):
        shape_list = [((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
                      ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
                      ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)
                out = torch.randn((1024, 512, 3, 3), dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),
                              self.to_mlu_dtype(c, data_type), out=out_mlu)
                self.assertEqual(ori_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_channel_last(self):
        shape_list = [((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
                      ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
                      ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                out = torch.randn((1024, 512, 3, 3), dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),
                              self.to_mlu_dtype(c, data_type), out=out_mlu)
                self.assertEqual(ori_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_not_dense(self):
        shape_list = [((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
                      ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
                      ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2))]
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)[:, :, :, :int(shape_a[-1] / 2)]
                b = torch.rand(shape_b, dtype=torch.float)[:, :, :, :int(shape_b[-1] / 2)]
                c = torch.randint(low = 1, high = 10,\
                                  size = shape_c, dtype=torch.float)[:, :, :, :int(shape_c[-1] / 2)]

                out = torch.randn((1024, 512, 3, 3), dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),
                              self.to_mlu_dtype(c, data_type), out=out_mlu)
                self.assertEqual(ori_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_shape_contiguous(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)
                out = torch.randn(1, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),\
                              self.to_mlu_dtype(c, data_type), out=out_mlu)
                assert ori_ptr != out_mlu.data_ptr()
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_shape_channel_last(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                out = torch.randn(1, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),\
                                     self.to_mlu_dtype(c, data_type), out=out_mlu)
                assert ori_ptr != out_mlu.data_ptr()
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_out_shape_not_dense(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)[:, :, :, :int(shape_a[-1] / 2)]
                b = torch.rand(shape_b, dtype=torch.float)[:, :, :, :int(shape_b[-1] / 2)]
                c = torch.randint(low = 1, high = 10,\
                                  size = shape_c, dtype=torch.float)[:, :, :, :int(shape_c[-1] / 2)]
                out = torch.randn(1, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(copy.deepcopy(out), data_type)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                torch.addcdiv(a, b, c, out=out)
                ori_ptr = out_mlu.data_ptr()
                torch.addcdiv(a_mlu, self.to_mlu_dtype(b, data_type),\
                                     self.to_mlu_dtype(c, data_type), out=out_mlu)
                assert ori_ptr != out_mlu.data_ptr()
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu(),
                                        0.003,
                                        use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addcdiv_exception(self):
        # Test broadcast rules of shape
        a = torch.randn(1, device="mlu")
        b = torch.randn([3, 2, 3], device="mlu")
        c = torch.randn([3, 2, 3], device="mlu")
        reg_msg = "output with shape \[.*\] doesn't match the broadcast shape \[.*\]"  # pylint: disable=W1401
        with self.assertRaisesRegex(RuntimeError, reg_msg):
            a.addcdiv_(b, c, value=1.0)

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`")
    @largeTensorTest('46GB')
    def test_addcdiv_large(self):
        shape_list = [((5, 1024, 1024, 1024), (5, 1024, 1, 1024), (5, 1024, 1024, 1))]
        data_types = [torch.half]
        for shape_a, shape_b, shape_c in shape_list:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.randint(low = 1, high = 10, size = shape_c, dtype=torch.float)
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out = torch.addcdiv(a, b, c, value = 0.35)
                out_mlu = torch.addcdiv(a_mlu,
                                        self.to_mlu_dtype(b, data_type),
                                        self.to_mlu_dtype(c, data_type),
                                        value = 0.35)
                self.assertTensorsEqual(out.float(),
                                        out_mlu.cpu().float(),
                                        0.003,
                                        use_MSE=True)


if __name__ == '__main__':
    run_tests()
