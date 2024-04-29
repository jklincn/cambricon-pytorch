from __future__ import print_function
import os
import sys
import collections
import unittest
import torch
from torch._six import inf
import gc
import tempfile
from random import randint
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging
logging.basicConfig(level=logging.DEBUG)

TEST_MLU = torch.mlu.is_available()
TEST_MULTIMLU = TEST_MLU and torch.mlu.device_count() >= 2

class TestCachingAllocator(TestCase):

    def _check_memory_stat_consistency(self):
        snapshot = torch.mlu.memory_snapshot()
        expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

        for segment in snapshot:
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            expected["segment.all.current"] += 1
            expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            is_split = len(segment["blocks"]) > 1
            for chunk in segment["blocks"]:
                if chunk["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if chunk["state"].startswith("active_"):
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if chunk["state"] == "inactive" and is_split:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += chunk["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += chunk["size"]
        for device, expected in expected_each_device.items():
            stats = torch.mlu.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch.mlu.current_device()

        m0 = torch.mlu.memory_allocated(device)
        last_m_arr = [torch.mlu.memory_allocated(device)]
        max_m_arr = [torch.mlu.max_memory_allocated(device)]
        last_r_arr = [torch.mlu.memory_reserved(device)]
        max_r_arr = [torch.mlu.max_memory_reserved(device)]

        def alloc(*size):
            with torch.mlu.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch.mlu.FloatTensor(*size)

        def assert_change(comp=1, empty_cache=False, reset_peak=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch.mlu.memory_allocated(device)
            new_max_m = torch.mlu.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_r = torch.mlu.memory_reserved(device)
            new_max_r = torch.mlu.max_memory_reserved(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_r, new_max_r)
            self.assertGreaterEqual(new_max_r, max_r_arr[0])
            last_r_arr[0] = new_r
            max_r_arr[0] = new_max_r

            if empty_cache:
                torch.mlu.empty_cache()
                new_r = torch.mlu.memory_reserved(device)
                new_max_r = torch.mlu.max_memory_reserved(device)
                self.assertLessEqual(new_r, last_r_arr[0])
                self.assertLessEqual(new_r, new_max_r)
                self.assertEqual(new_max_r, max_r_arr[0])
                last_r_arr[0] = new_r

            if reset_peak:
                torch.mlu.reset_peak_memory_stats(device)
                self.assertEqual(torch.mlu.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.mlu.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch.mlu.memory_reserved(device), last_r_arr[0])
                self.assertEqual(torch.mlu.max_memory_reserved(device), last_r_arr[0])
                max_r_arr[0] = last_r_arr[0]

        assert_change(0)
        assert_change(0, reset_peak=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch.mlu.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        # small chunks with allocation smaller than 1MB
        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        # large chunks with allocation larger than 1MB
        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_peak=(i % 2 == 0))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        # now the memory of tensor2 is used by permute
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        # now the memory of permute is used by tensor2
        assert_change(0, reset_peak=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_peak=True)
        assert_change(0)
        self.assertEqual(torch.mlu.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_peak=True)
        self.assertEqual(torch.mlu.memory_allocated(device), m0)

        # if int(os.environ.get("ENABLE_CATCH_MEMORY_DEBUG")) :
        #     t3 = alloc(100)
        #     torch.mlu.memory_debug(t3)
        #     torch.mlu.memory_debug()
        #     assert_change(1)

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)

    #@unittest.skip("not test")
    @testinfo()
    def test_memory_stats(self):
        # os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '0'
        gc.collect()
        torch.mlu.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()
        # torch.mlu.empty_cache()
        # for _ in self._test_memory_stats_generator(self):
        #     pass
        # os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '1'
        # torch.mlu.empty_cache()
        # for _ in self._test_memory_stats_generator(self):
        #     self._check_memory_stat_consistency()
        # torch.mlu.empty_cache()
        # for _ in self._test_memory_stats_generator(self):
        #     pass
        # os.environ["ENABLE_CATCH_MEMORY_DEBUG"] = '0'

    #@unittest.skip("not test")
    @testinfo()
    def test_memory_allocation(self):
        gc.collect()
        torch.mlu.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.mlu.memory_allocated()
            mem = torch.mlu.caching_allocator_alloc(size)
            self.assertGreater(torch.mlu.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.mlu.caching_allocator_delete(mem)
                self.assertEqual(torch.mlu.memory_allocated(), prev)

    #@unittest.skip("not test")
    @testinfo()
    def test_memory_strategy(self):
        # test catch strategy env.
        os.environ["PYTORCH_MLU_MEMORY_STRATEGY"] = '1'
        # make the env work.
        torch.mlu._set_allocator_settings("")
        assert(not torch.mlu.get_memory_strategy())
        torch.mlu.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

        torch.mlu.set_memory_strategy(True)
        assert(torch.mlu.get_memory_strategy())
        torch.mlu.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()
        # unset env for other case.
        os.environ["PYTORCH_MLU_MEMORY_STRATEGY"] = '0'

    @unittest.skipIf(not TEST_MULTIMLU, "only one MLU detected")
    @testinfo()
    def test_memory_stats_multimlu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch.mlu.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device='mlu:0', N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('mlu:1'), N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.mlu.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('mlu:1'), N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    #@unittest.skip("not test")
    @testinfo()
    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device='mlu')

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 500.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 500, dtype=torch.int8, device='mlu')

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate more than 1EB memory"):
            torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device='mlu')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    #@unittest.skip("not test")
    @testinfo()
    def test_set_per_process_memory_fraction(self):
        # test invalid fraction value.
        with self.assertRaisesRegex(TypeError, "Invalid type"):
            torch.mlu.set_per_process_memory_fraction(int(1))
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.mlu.set_per_process_memory_fraction(-0.1)
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.mlu.set_per_process_memory_fraction(2.0)

        tensor = torch.empty(1024, device='mlu')
        torch.mlu.empty_cache()
        total_memory = torch.mlu.get_device_properties(0).total_memory
        torch.mlu.set_per_process_memory_fraction(0.5, 0)

        # test 0.4 allocation is ok.
        application = int(total_memory * 0.4) - torch.mlu.max_memory_reserved()
        tmp_tensor = torch.empty(application, dtype=torch.int8, device='mlu')
        del tmp_tensor
        torch.mlu.empty_cache()

        application = int(total_memory * 0.5)
        # it will get OOM when try to allocate more than half memory.
        with self.assertRaisesRegex(RuntimeError, "MLU out of memory."):
            torch.empty(application, dtype=torch.int8, device='mlu')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    #@unittest.skip("not test")
    @testinfo()
    def test_memory_snapshot(self):
        try:
            torch.mlu.empty_cache()
            torch.mlu._record_memory_history(True)
            x = torch.rand(311, 411, device='mlu')

            # create a bunch of tensors that all will tile into the
            # same segment to  exercise the history merging code
            # 512B is the minimum block size,
            # so we allocate all the tensors to this size to make sure
            # they tile evenly
            tensors = [torch.rand(128, device='mlu') for _ in range(1000)]
            while tensors:
                del tensors[randint(0, len(tensors) - 1)]

            # exercise the history trimming code
            torch.rand(128 * 5, device='mlu')

            ss = torch.mlu._snapshot()
            found_it = False
            for seg in ss:
                for b in seg['blocks']:
                    if 'history' in b:
                        for h in b['history']:
                            if h['real_size'] == 311 * 411 * 4:
                                self.assertTrue('test_caching_allocator' in h['frames'][0]['filename'])
                                found_it = True
            self.assertTrue(found_it)
            with tempfile.NamedTemporaryFile() as f:
                torch.mlu._save_segment_usage(f.name)
                with open(f.name, 'r') as f2:
                    self.assertTrue('test_caching_allocator.py' in f2.read())

        finally:
            torch.mlu._record_memory_history(False)

    #@unittest.skip("not test")
    @testinfo()
    def test_allocator_settings(self):
        def power2_div(size, div_factor):
            pow2 = 1
            while pow2 < size:
                pow2 = pow2 * 2
            if pow2 == size:
                return pow2
            step = pow2 / 2 / div_factor
            ret = pow2 / 2
            while ret < size:
                ret = ret + step
            return ret

        torch.mlu.empty_cache()
        key = 'active_bytes.all.allocated'

        nelems = 21 * 1024 * 1024
        nbytes = 4 * nelems  # floats are 4 bytes

        start_mem = torch.mlu.memory_stats()[key]
        torch.mlu._set_allocator_settings("")
        # NOTE: Do not use torch.rand which may include extra memory cost.
        x = torch.mlu.FloatTensor(nelems)

        reg_mem = torch.mlu.memory_stats()[key]
        torch.mlu._set_allocator_settings("roundup_power2_divisions:4")
        y = torch.mlu.FloatTensor(nelems)

        pow2_div4_mem = torch.mlu.memory_stats()[key]

        self.assertTrue(reg_mem - start_mem == nbytes)
        self.assertTrue(pow2_div4_mem - reg_mem == power2_div(nbytes, 4))

        torch.mlu._set_allocator_settings("garbage_collection_threshold:0.5")
        torch.mlu._set_allocator_settings("garbage_collection_threshold:0.5,max_split_size_mb:40")

        # should have reset the power2 divisions now
        torch.mlu.empty_cache()
        start_mem = torch.mlu.memory_stats()[key]
        z = torch.mlu.FloatTensor(nelems)
        reg_mem = torch.mlu.memory_stats()[key]
        self.assertTrue(reg_mem - start_mem == nbytes)


        with self.assertRaises(RuntimeError):
            torch.mlu._set_allocator_settings("foo:1,bar:2")

        with self.assertRaises(RuntimeError):
            torch.mlu._set_allocator_settings("garbage_collection_threshold:1.2")

        with self.assertRaises(RuntimeError):
            torch.mlu._set_allocator_settings("max_split_size_mb:2")

    #@unittest.skip("not test")
    @testinfo()
    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cnrt_malloc_retry."""
        stream = torch.mlu.Stream()

        with torch.mlu.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device='mlu')

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device='mlu')
            with torch.mlu.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.mlu.empty_cache()

if __name__ == '__main__':
    unittest.main()
