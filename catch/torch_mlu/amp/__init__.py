import torch
import torch.mlu.amp as amp

from .autocast_mode import autocast, custom_fwd, custom_bwd # noqa: F401
from .grad_scaler import GradScaler # noqa: F401

amp.__setattr__("autocast", autocast)
amp.__setattr__("custom_fwd", custom_fwd)
amp.__setattr__("custom_bwd", custom_bwd)
amp.__setattr__("GradScaler", GradScaler)
