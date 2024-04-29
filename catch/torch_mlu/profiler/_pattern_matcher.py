import json
import math
import os
import re
from typing import Dict, List, Optional, Set

import torch
from torch.profiler import profile
import torch.utils.benchmark as benchmark
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
from torch._C._profiler import (_ProfilerEvent, _ExtraFields_TorchOp,
                                _ExtraFields_PyCCall, _ExtraFields_PyCall,
                                _EventType)
from torch.profiler._pattern_matcher import (
    Pattern,
    ForLoopIndexingPattern,
    OptimizerSingleTensorPattern,
    SynchronizedDataLoaderPattern,
    GradNotSetToNonePattern,
    Conv2dBiasFollowedByBatchNorm2dPattern,
    source_code_location,
    input_shapes,
    input_dtypes
)

# Patterns

class ExtraMLUCopyPattern(Pattern):
    '''
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to MLU.
    example: torch.zeros((100, 100)).to("mlu")

    Pattern:
    build-in method                 |build-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    '''

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Extra MLU Copy Pattern"
        self.description = "Filled a CPU tensor and immediately moved it to MLU. Please initalize it on MLU."
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device"
        self.init_ops = {
            "aten::fill_", "aten::zero_", "aten::normal_", "aten::uniform_"
        }

    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes

    def match(self, event):
        # TODO: We should also check tensor identities
        if event.name != "aten::to":
            return False
        to_event = event
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != "aten::_to_copy":
            return False
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != "aten::copy_":
            return False
        # aten::copy_ should have the first 2 args dtype the same
        dtypes = input_dtypes(event)
        if len(dtypes) < 2:
            return False
        if dtypes[0] != dtypes[1]:
            return False
        event = to_event
        # Up one level
        event = event.parent
        if event is None:
            return False
        # .mlu() call stack is:
        #  | torch_mlu/core/device.py(141): _mlu
        #  | <built-in function mlu>
        #  | aten::to
        # One level deeper than the GPU call stack
        if '): _mlu' in event.parent.name:
            event = event.parent
        # Check if we have a aten::fill_ in previous leaf
        event = self.prev_of(event)
        if event is None:
            return False
        while event.children:
            event = event.children[-1]
            # aten::zero_ is a special optimzation case where fill_ is not called
            if event.name in self.init_ops:
                return True
        return event.name in self.init_ops
        # TODO: Check if tensor is reused

    def benchmark(self, events: List[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            size = shape[0]
            to_timer = benchmark.Timer(stmt='torch.ones(size).to("mlu")',
                                       globals={'size': size})
            de_timer = benchmark.Timer(stmt='torch.ones(size, device="mlu")',
                                       globals={'size': size})
            to_time = to_timer.timeit(10).mean
            de_time = de_timer.timeit(10).mean
            shapes_factor_map[shape] = de_time / to_time
        return shapes_factor_map


class FP32MatMulPattern(Pattern):

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "FP32 MatMul Pattern"
        self.description = (
            "You are currently using MLU that supports TF32. "
            "Please enable TF32 by setting 'torch.backends.mlu.matmul.allow_tf32 = True'"
        )
        self.url = "https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"

    @property
    def skip(self):
        has_tf32 = torch.mlu.get_device_properties(torch.mlu.current_device()).major >= 5
        return has_tf32 is False or super().skip or not self.prof.record_shapes

    def match(self, event: _ProfilerEvent):
        # If we saw this pattern once, we don't need to match it again
        if event.tag != _EventType.TorchOp:
            return False
        assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
        if event.name == "aten::mm":
            if event.extra_fields.allow_tf32_cnmatmul is False:
                return True
        return False

    def report(self, event: _ProfilerEvent):
        return self.description

    def benchmark(self, events: List[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            matrixA = torch.randn(shape[0], device="mlu", dtype=torch.float32)
            matrixB = torch.randn(shape[1], device="mlu", dtype=torch.float32)
            fp32_timer = benchmark.Timer(stmt='torch.mm(matrixA, matrixB)',
                                         globals={
                                             "matrixA": matrixA,
                                             "matrixB": matrixB
                                         })
            tf32_timer = benchmark.Timer(
                stmt='torch.mm(matrixA, matrixB)',
                setup='torch.backends.mlu.matmul.allow_tf32 = True',
                globals={
                    "matrixA": matrixA,
                    "matrixB": matrixB
                })
            torch.backends.mlu.matmul.allow_tf32 = False
            fp32_time = fp32_timer.timeit(10).mean
            tf32_time = tf32_timer.timeit(10).mean
            shapes_factor_map[shape] = tf32_time / fp32_time
        return shapes_factor_map


class MatMulDimInFP16Pattern(Pattern):

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Matrix Multiplication Dimension Not Aligned Pattern"
        self.description = "Detected matmul with dimension not aligned. Please use matmul with aligned dimension."
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-mixed-precision-and-amp"

    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes

    def match(self, event: _ProfilerEvent):

        def mutiple_of(shapes, multiple):
            return all(dim % multiple == 0 for shape in shapes
                       for dim in shape[-2:])

        if event.name not in ("aten::mm", "aten::bmm", "aten::addmm"):
            return False
        if not input_dtypes(event):
            return False
        arg_dtype = input_dtypes(event)[0]
        # TODO: Have a better way to check dtype_size
        if (arg_dtype.endswith("c10::BFloat16")
                or arg_dtype.endswith("c10::Half")) and not mutiple_of(
                    input_shapes(event), 8):
            return True
        return False

    def benchmark(self, events: List[_ProfilerEvent]):

        def closest_multiple(shapes, multiple):
            return [multiple * math.ceil(shape / multiple) for shape in shapes]

        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            matrixA = torch.randn(shape[0], device="mlu", dtype=torch.float16)
            matrixB = torch.randn(shape[1], device="mlu", dtype=torch.float16)
            not_aligned_dim_timer = benchmark.Timer(
                stmt='torch.mm(matrixA, matrixB)',
                globals={
                    "matrixA": matrixA,
                    "matrixB": matrixB
                })
            matrixA = torch.randn(closest_multiple(shape[0], 8),
                                  device="mlu",
                                  dtype=torch.float16)
            matrixB = torch.randn(closest_multiple(shape[1], 8),
                                  device="mlu",
                                  dtype=torch.float16)
            aligned_dim_timer = benchmark.Timer(
                stmt='torch.mm(matrixA, matrixB)',
                globals={
                    "matrixA": matrixA,
                    "matrixB": matrixB
                })
            not_aligned_dim_time = not_aligned_dim_timer.timeit(10).mean
            aligned_dim_time = aligned_dim_timer.timeit(10).mean
            shapes_factor_map[shape] = aligned_dim_time / not_aligned_dim_time
        return shapes_factor_map


def report_all_anti_patterns(prof,
                             should_benchmark: bool = False,
                             print_enable: bool = True,
                             json_report_dir: str = None):
    report_dict: Dict = {}
    anti_patterns = [
        ExtraMLUCopyPattern(prof, should_benchmark),
        # ForLoopIndexingPattern(prof, should_benchmark),
        FP32MatMulPattern(prof, should_benchmark),
        OptimizerSingleTensorPattern(prof, should_benchmark),
        SynchronizedDataLoaderPattern(prof, should_benchmark),
        GradNotSetToNonePattern(prof, should_benchmark),
        Conv2dBiasFollowedByBatchNorm2dPattern(prof, should_benchmark),
        MatMulDimInFP16Pattern(prof, should_benchmark)
    ]
    reported = set()
    summaries = []
    message_list = [f"{'-'*40}TorchTidy Report{'-'*40}"]
    message_list.append("Matched Events:")

    for anti_pattern in anti_patterns:
        matched_events = anti_pattern.matched_events()
        if not matched_events:
            continue
        summaries.append(anti_pattern.summary(matched_events))
        for event in matched_events:
            report_msg = anti_pattern.report(event)
            if report_msg not in reported:
                message_list.append(report_msg)
                reported.add(report_msg)
                src_location, line_no = source_code_location(event).split(":")
                report_dict.setdefault(src_location, []).append({
                    "line_number": int(line_no),
                    "name": anti_pattern.name,
                    "url": anti_pattern.url,
                    "message": anti_pattern.description,
                })

    if json_report_dir is not None:
        json_report_path = os.path.join(json_report_dir,
                                        "torchtidy_report.json")
        if os.path.exists(json_report_path):
            with open(json_report_path, "r") as f:
                exisiting_report = json.load(f)
                exisiting_report.update(report_dict)
                report_dict = exisiting_report
        with open(json_report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

    message_list.append("Summary:")
    message_list += summaries
    message_list.append(f"{'-'*40}TorchTidy Report{'-'*40}")
    if print_enable:
        print("\n".join(message_list))
