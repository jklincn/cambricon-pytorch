import torch
from .profiler_util import (
    EventList,
    FunctionEvent,
    supported_activities
)
from ._pattern_matcher import (
    ExtraMLUCopyPattern,
    FP32MatMulPattern,
    MatMulDimInFP16Pattern,
    report_all_anti_patterns
)

torch.autograd.profiler.__setattr__("EventList", EventList)
torch.autograd.profiler.__setattr__("FunctionEvent", FunctionEvent)
torch.profiler.profiler.__setattr__("supported_activities", supported_activities)
torch.profiler.__setattr__("supported_activities", supported_activities)

torch.profiler._pattern_matcher.__setattr__("ExtraMLUCopyPattern", ExtraMLUCopyPattern)
torch.profiler._pattern_matcher.__setattr__("FP32MatMulPattern", FP32MatMulPattern)
torch.profiler._pattern_matcher.__setattr__("MatMulDimInFP16Pattern", MatMulDimInFP16Pattern)
torch.profiler._pattern_matcher.__setattr__("report_all_anti_patterns", report_all_anti_patterns)
