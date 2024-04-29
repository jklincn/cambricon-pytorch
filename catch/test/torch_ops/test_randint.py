from __future__ import print_function
import sys
import os
import unittest
import logging
import torch
from scipy import stats

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir+"/../")
from common_utils import TestCase, testinfo  # pylint: disable=C0413
import torch_mlu  # pylint: disable=W0611
logging.basicConfig(level=logging.DEBUG)

class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_randint(self):
        shape_list = [(200,),(20,80),(6,70,16),(8,20,80,14),(7,6,9,20,50),(2,8,10,14,27)]
        lows = [-100,-50,-10,-1,0,1,100]
        highs = [-50,50,0,10,10,50,1000]
        device = 'mlu'
        failure_count = 0
        total_count = 0
        for shape in shape_list:
            for i in range(len(lows)):
                mlu_tensor = torch.randint(lows[i],highs[i],shape,device=device)
                res = stats.ks_2samp(mlu_tensor.cpu().numpy().reshape(-1), \
                                     torch.randint(lows[i],highs[i],shape).numpy().reshape(-1))
                total_count += 1
                if res.pvalue < 0.01:
                    failure_count += 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_randint_high(self):
        shape_list = [(200,),(20,80),(6,70,16),(8,20,80,14),(7,6,9,20,50),(2,8,10,14,27)]
        highs = [10,20,50,100,500,1000]
        device = 'mlu'
        failure_count = 0
        total_count = 0
        for shape in shape_list:
            for i in range(len(highs)):
                mlu_tensor = torch.randint(highs[i],shape,device=device)
                res = stats.ks_2samp(mlu_tensor.cpu().numpy().reshape(-1), \
                                     torch.randint(highs[i],shape).numpy().reshape(-1))
                total_count += 1
                if res.pvalue < 0.01:
                    failure_count += 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_randint_out(self):
        shape_list = [(200,),(20,80),(6,70,16),(8,20,80,14),(7,6,9,20,50),(2,8,10,14,27)]
        lows = [-100,-50,-10,-1,0,1,100]
        highs = [-50,50,0,10,10,50,1000]
        device = 'mlu'
        failure_count = 0
        total_count = 0
        for shape in shape_list:
            for i in range(len(lows)):
                mlu_tensor = torch.tensor([],dtype=torch.int64,device=device)
                torch.randint(lows[i], highs[i],shape,device=device, out=mlu_tensor)
                res = stats.ks_2samp(mlu_tensor.cpu().numpy().reshape(-1), \
                                     torch.randint(lows[i],highs[i],shape).numpy().reshape(-1))
                total_count += 1
                if res.pvalue < 0.01:
                    failure_count += 1
        self.assertTrue(failure_count < total_count * 0.1)

if __name__ == "__main__":
    unittest.main()
