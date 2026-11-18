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
"""Parameter-free mHC post computation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from hyper_parallel.components.functional.mhc_post import mhc_post
from ._utils import _num_stream, _validate_settings


class MhcPostModule(nn.Module):
    """Combine transformed and residual streams after an MHC-wrapped block."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
    ) -> None:
        """Build the high-performance MHC post module from an existing module.

        Args:
            module: Source MHC post module.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.
        """
        super().__init__()
        del module_fqn, context
        if (module is None) == (config is None):
            raise ValueError("provide exactly one of config or module")
        if config is not None:
            _validate_settings(config.num_stream)
            self.config = config
            self.num_stream = config.num_stream
            return
        self.config = getattr(module, "config", None)
        self.num_stream = _num_stream(module)
        self.train(module.training)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the wrapped block output into the residual streams."""
        return mhc_post(
            x,
            residual,
            h_post,
            h_res,
            self.num_stream,
        )
