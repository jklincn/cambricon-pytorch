# Owner(s): ["oncall: distributed"]

import sys
import test_c10d_spawn
import torch
import torch.distributed as c10d
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions
from _internal.distributed.common_mlu import TEST_MULTIMLU
from _internal.distributed.mlu_common_distributed import (
    requires_cncl,
    skip_if_lt_x_mlu,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    sandcastle_skip_if,
    TEST_WITH_DEV_DBG_ASAN,
)

NO_CNCL = not hasattr(c10d, "ProcessGroupCNCL")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619
if sys.version_info < (3, 9):

    class ProcessGroupShareTensorTest(
        test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase
    ):
        @classmethod
        def _init_pg_cncl(cls, rank, filename, world_size):
            store = c10d.FileStore(filename, world_size)
            return c10d.ProcessGroupCNCL(store, rank, world_size)

        @sandcastle_skip_if(not TEST_MULTIMLU, "At least 2 MLUs needed")
        @sandcastle_skip_if(NO_CNCL, "CNCL needed")
        def test_shared_broadcast_cncl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_broadcast_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_cncl,
                1,
            )

        @sandcastle_skip_if(not TEST_MULTIMLU, "At least 2 MLUs needed")
        @sandcastle_skip_if(NO_CNCL, "CNCL needed")
        def test_shared_allreduce_cncl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allreduce_process,
                [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_cncl,
                1,
            )

        @classmethod
        def _test_reduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
        ):
            pg = init_pg(rank, filename, world_size)
            x = shared_tensors[rank]
            pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
            if rank == 0:
                c2p.put((rank, torch.ones(2, 2) * 2, x.to("cpu")))
            else:
                c2p.put((rank, torch.ones(2, 2), x.to("cpu")))
            p2c.get()

        @sandcastle_skip_if(not TEST_MULTIMLU, "At least 2 MLUs needed")
        @sandcastle_skip_if(NO_CNCL, "CNCL needed")
        def test_shared_reduce_cncl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_reduce_process,
                [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_cncl,
                1,
            )

        @sandcastle_skip_if(not TEST_MULTIMLU, "At least 2 MLUs needed")
        @sandcastle_skip_if(NO_CNCL, "CNCL needed")
        def test_shared_allgather_cncl(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allgather_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_cncl,
                self.world_size,
            )


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsCncl(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("cncl")

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_reduce(self):
            self._test_reduce("cncl")

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_allreduce(self):
            self._test_allreduce("cncl")

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_gather(self):
            self._test_all_gather("cncl")

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_to_all(self):
            self._test_all_to_all("cncl")

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_to_all_single(self):
            self._test_all_to_all_single("cncl")

        # Test Ops only supported in CNCL.
        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_reduce_scatter(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='cncl')
            device = torch.device(f"mlu:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + self.rank
            x1 = torch.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.empty_like(x0)
            expected = (1 + self.world_size) * self.world_size / 2 + self.world_size * self.rank
            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            self.assertEqual(y, torch.ones(5, 5, device=device) * expected)
            z = y.sin().sum()
            z.backward()
            expected_0 = (1 + self.world_size) * self.world_size / 2
            expected_1 = expected_0 + self.world_size
            x_s_0 = (expected_0 * torch.ones(5, 5, device=device)).cos()
            x_s_1 = (expected_1 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s_0)
            self.assertEqual(x1.grad, x_s_1)

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_reduce_scatter_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='cncl')
            device = torch.device(f"mlu:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):

                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x0 = torch.rand(5, 5, device=device, requires_grad=True)
            x1 = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.empty(5, 5, device=device)

            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_cncl()
        @skip_if_lt_x_mlu(2)
        @sandcastle_skip_if(not _torch_dist_nn_available, "torch.distributed.nn is not available")
        def test_all_gather_base(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='cncl')

            device = torch.device(f"mlu:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True

            output = torch.empty(5 * self.world_size, 5, device=device)
            output = torch.distributed.nn.functional._all_gather_base(output, x)
            self.assertEqual(output.size(), torch.Size((5 * self.world_size, 5)))

            for idx in range(self.world_size):
                self.assertEqual(output[5 * idx : 5 * (idx + 1)], torch.ones(5, 5, device=device) + idx)

            y = torch.sum(output.view(self.world_size, 5, 5), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()
