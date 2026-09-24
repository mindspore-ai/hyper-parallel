# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Mixed-precision and optimizer configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from torch.optim import Optimizer  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.swap_optimizer_base import validate_state_keys
from hyper_parallel.trainer.config.target import Target, _serialize_config_value


@dataclass
class MixedPrecisionConfig:
    """Mixed-precision parameters exposed by the initial YAML schema."""

    enabled: bool = False


@dataclass
class OptimizerSwapConfig:
    """Optimizer-state swap options for the Trainer-built optimizer.

    Swapping moves Adam/AdamW state tensors to host memory between updates, so
    the peak device footprint drops by the optimizer state size while the
    copies overlap the surrounding step. It is independent of
    ``fsdp_config.enable_offload``, which offloads weights and gradients.

    Args:
        enabled: Whether the Trainer wraps the built optimizer with state swap.
        swap_times: Number of pipeline partitions the swapped state is split
            into; each partition is prefetched one batch ahead of its update.
        min_numel: State tensors smaller than this element count stay on device.
        state_keys: Logical Adam/AdamW state keys to swap. ``None`` uses the
            adapter defaults (``exp_avg``, ``exp_avg_sq``, ``max_exp_avg_sq``).
        include_master_params: Whether optimizer-owned fp32 master parameters
            are swapped too. Only meaningful together with
            ``optimizer.fp32_main_params``, which is the wrapper owning them.
        packed_swap: Whether the backend packs swapped state into two reusable
            staging buffers. ``None`` keeps the backend default (enabled on
            PyTorch, and for MindFormers AdamW on MindSpore).
    """

    enabled: bool = False
    swap_times: int = 16
    min_numel: int = 1024
    state_keys: Optional[List[str]] = None
    include_master_params: bool = False
    packed_swap: Optional[bool] = None

    def __post_init__(self) -> None:
        """Reject a pipeline width or state key set the swap runtime cannot use."""
        if self.swap_times < 1:
            raise ValueError(
                f"optimizer.swap.swap_times must be at least 1, got {self.swap_times}"
            )
        if self.min_numel < 0:
            raise ValueError(
                f"optimizer.swap.min_numel must be non-negative, got {self.min_numel}"
            )
        if self.state_keys is not None:
            validate_state_keys(self.state_keys)


@dataclass
class OptimizerConfig:
    """Optimizer target plus Trainer-owned parameter precision policy."""

    target: Target[Optimizer]
    fp32_main_params: bool = False
    swap: OptimizerSwapConfig = field(default_factory=OptimizerSwapConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize optimizer options in their compact target YAML shape."""
        config = self.target.to_dict()
        config["fp32_main_params"] = self.fp32_main_params
        config["swap"] = _serialize_config_value(self.swap)
        return config
