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
"""JT precision policy for the reusable DeepSeek MTP execution module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# This conversion targets the Torch-only MTP module.
# pylint: disable=forbidden-backend-import
import torch
from hyper_parallel.components.modules.mtp import DeepseekV3MTPExecution, MultiTokenPredictionLayer
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class JTDeepseekV3MTPExecution(DeepseekV3MTPExecution):
    """Reuse shifting, decoder execution and loss accumulation with JT rounding."""

    def __init__(self, *, module: DeepseekV3MTPExecution, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Replace a parameter-free execution policy without moving any MTP weights."""
        super().__init__()
        del module_fqn, context
        self.train(module.training)

    @staticmethod
    def fuse_inputs(layer: MultiTokenPredictionLayer, hidden: torch.Tensor,
                    embedding: torch.Tensor) -> torch.Tensor:
        """Round embedding before normalization and both normalized branches before fusion.

        Args:
            layer: Public prediction depth with JT norms and decoder.
            hidden: Previous trunk or MTP state.
            embedding: Future-token embedding before JT precision conversion.
        """
        embedding = embedding.to(torch.bfloat16)
        hidden = layer.hnorm(hidden).to(torch.bfloat16)
        embedding = layer.enorm(embedding).to(torch.bfloat16)
        return torch.cat((hidden, embedding), dim=-1)

    @staticmethod
    def recurrent_state(raw_hidden: torch.Tensor, prediction_hidden: torch.Tensor) -> torch.Tensor:
        """Carry the normalized prediction state between JT depths.

        Args:
            raw_hidden: Decoder output before prediction normalization.
            prediction_hidden: Normalized state supplied to the output head.
        """
        del raw_hidden
        return prediction_hidden
