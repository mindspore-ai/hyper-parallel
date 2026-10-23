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
"""MXFP8 local computation with optional fusion and the GroupedExperts contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import (
    fused_grouped_swiglu,
)
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    validate_npu_gmm_runtime, validate_npu_swiglu_runtime,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class MXFP8GroupedSwiGLU(GroupedExperts):
    """Retain routing and checkpoint conversion; replace only local expert math.

    Persistent weights remain [E,H,2I] and [E,I,H]. Counts name tokens per
    local expert, including zero-token experts. Output retains expert-major
    order. Scores are accepted by the inherited protocol; outer routing applies
    them exactly once after GMM2, with its original dtype policy.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Validate the supported structure and initialize a non-fused variant."""
        context = context or {}
        self.module_fqn = module_fqn
        prefix = f"{module_fqn or '<root>'}: MXFP8GroupedSwiGLU"
        if any(context.get(axis) for axis in ("tp", "cp", "ep", "pp")):
            raise NotImplementedError(f"{prefix} requires TP=CP=EP=PP=1")
        if context.get("ep_size", 1) != 1:
            raise NotImplementedError(f"{prefix} supports only EP=1")
        try:
            super().__init__(module=module, module_fqn=module_fqn, context=context)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{prefix}: {exc}") from exc
        self._validate_source_contract(module, prefix)
        try:
            validate_npu_gmm_runtime()
        except RuntimeError as exc:
            raise RuntimeError(f"{prefix}: {exc}") from exc
        self.quantizer = MXFP8Quantizer()

    def _validate_source_contract(self, module: nn.Module, prefix: str) -> None:
        """Check the initial layout and activation before any device computation."""
        hidden_act = (getattr(self.config, "hidden_act", None)
                      or getattr(self.config, "hidden_activation", None)
                      or getattr(self.config, "mlp_hidden_act", "silu"))
        source_activation = getattr(module, "act_fn", None)
        if (not self.has_gate or self.add_bias or hidden_act not in ("silu", "swiglu")
                or (source_activation is not None
                    and not isinstance(source_activation, (nn.SiLU, type(ACT2FN["silu"]))))):
            raise ValueError(f"{prefix} requires standard bias-free SwiGLU")
        if self.use_2d_experts or not self.router_gating_in_fp32:
            raise ValueError(f"{prefix} requires 3D packed weights and routing weights after GMM2")
        if self.hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError(f"{prefix} requires H and I divisible by 32")

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Validate expert groups and dispatch local math without applying scores."""
        del permuted_probs
        counts = tokens_per_expert.to(device=permuted.device)
        if counts.ndim != 1 or counts.shape[0] != gate_up_proj.shape[0]:
            raise ValueError(f"{self.module_fqn}: counts must contain one count per local expert")
        if counts.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{self.module_fqn}: counts must be integers")
        # Device counts come from routing; avoid copying them to the host per GMM.
        # CPU callers can validate values before handing counts to device kernels.
        if tokens_per_expert.device.type == "cpu":
            if bool((tokens_per_expert < 0).any()) or int(tokens_per_expert.sum()) != permuted.shape[0]:
                raise ValueError(f"{self.module_fqn}: counts must be non-negative and sum to token count")
        return self._local_compute(permuted, gate_up_proj, down_proj, counts)

    def _local_compute(self, permuted, gate_up_proj, down_proj, counts):
        """Run the non-fused local chain after common protocol validation."""
        projected = mxfp8_projection(permuted, gate_up_proj, counts, self.quantizer)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return mxfp8_projection(activated, down_proj, counts, self.quantizer)


@module_replacement
class FusedMXFP8GroupedSwiGLU(MXFP8GroupedSwiGLU):
    """Use fused activation/quantization while retaining the same family contract."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Validate both fused kernels during construction, with module diagnostics."""
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        try:
            validate_npu_swiglu_runtime()
        except RuntimeError as exc:
            raise RuntimeError(f"{module_fqn or '<root>'}: {exc}") from exc

    def _local_compute(self, permuted, gate_up_proj, down_proj, counts):
        """Use one autograd boundary without passing quantized carriers between nodes."""
        return fused_grouped_swiglu(permuted, gate_up_proj, down_proj, counts, self.quantizer)
