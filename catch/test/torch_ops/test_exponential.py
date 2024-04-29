from __future__ import print_function
import logging
import unittest
import sys
import os
import torch
import torch_mlu
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import TestCase
logging.basicConfig(level=logging.DEBUG)

class TestExponentialOp(TestCase):
    def _test_exponential(self, shape, lambd, dtype):
        x = torch.randn(shape, dtype=dtype, device='mlu')
        x.exponential_(lambd=lambd)
        self.assertGreater(x.min(), 0)

        y = torch.randn(shape, dtype=dtype, device='mlu')
        torch.manual_seed(123)
        x.exponential_(lambd=lambd)
        torch.manual_seed(123)
        y.exponential_(lambd=lambd)
        self.assertEqual(x, y)

    # @unittest.skip("not test")
    def test_exponential(self):
        for dtype in [torch.float, torch.half, torch.double]:
            for shape in [[], [1], [2,3], [2,3,4], [2,3,4,5], [2,3,4,5,6]]:
                for lambd in [0.1, 1.0, 1.2, 8.8]:
                    self._test_exponential(shape, lambd, dtype)

    # @unittest.skip("not test")
    def test_exponential_lambda(self):
        x = torch.randn((1), dtype=torch.float, device='mlu')
        x.exponential_(lambd=0.0)
        self.assertEqual(x.cpu(), torch.tensor([float('inf')]))

        x.exponential_(lambd=float('inf'))
        self.assertEqual(x.cpu(), torch.tensor([0.0]))

    # @unittest.skip("not test")
    def test_exponential_lambda_exception(self):
        ref_msg = f'expects lambda >= 0.0, but found lambda'
        for lambd in [-1.2]:
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                self._test_exponential([2,3,4], lambd, torch.float)

if __name__ == '__main__':
    unittest.main()


