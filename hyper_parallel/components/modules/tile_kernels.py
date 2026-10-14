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
"""DeepSeek-V4.1 Module adapters backed by TileKernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.distributed.tensor import DTensor  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.deepseek_ascend import (
    deepseek_engram_gate,
    deepseek_mhc_post,
    deepseek_mhc_pre_apply,
    deepseek_mhc_sinkhorn,
    deepseek_mhc_split_mixes,
    deepseek_topk_gate,
)
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.mhc import PipelinedMhcModule
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class TileKernelsEngramModule(EngramModule):
    """DeepSeek-V4.1 Engram replacement using TileKernels' fused gate."""

    def _tile_fuse(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project Engram rows and invoke the fused differentiable gate."""
        key_value = self.wkv(embeddings.flatten(start_dim=-2))
        image_token_mask = None
        if token_mask is not None:
            image_token_mask = ~token_mask.to(torch.bool).reshape(-1)
        return deepseek_engram_gate(
            hidden_states,
            key_value,
            self.q_weight,
            self.k_weight,
            self.clamp_value,
            self.eps,
            image_token_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run EP1 Engram with TileKernels gate fusion."""
        hash_ids = self._aligned_hash_ids(
            hidden_states, input_ids, segment_starts, token_mask
        )
        return self._tile_fuse(hidden_states, self.embed(hash_ids), token_mask)

    def parallel_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        cp_group: Any = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> torch.Tensor:
        """Run distributed lookup followed by TileKernels gate fusion."""
        hash_ids = self._aligned_hash_ids(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        embeddings = self._sparse_lookup(
            hash_ids,
            self.embed,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            padded_num_embeddings=self.padded_num_embeddings,
        )
        return self._tile_fuse(hidden_states, embeddings, token_mask)


@module_replacement
class TileKernelsV41MhcModule(PipelinedMhcModule):
    """DeepSeek-V4.1 Manifold-Constrained Hyper-Connections (mHC) replacement."""

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the accelerated V4.1 pre path or preserve the coefficient-only API."""
        if pre_mix is None:
            return super().forward(hidden_streams)
        if isinstance(hidden_streams, DTensor):
            raise TypeError("TileKernels V4.1 mHC does not support DTensor activations")
        if hidden_streams.device.type != "npu":
            raise ValueError("TileKernels V4.1 mHC requires NPU hidden streams")
        if hidden_streams.ndim != 4:
            raise ValueError(
                "TileKernels V4.1 mHC expects hidden_streams with shape [B,S,N,D]"
            )
        if self.hc_mult != 4 or hidden_streams.shape[-2] != self.hc_mult:
            raise ValueError("TileKernels V4.1 mHC currently requires hc_mult=4")
        if hidden_streams.shape[-1] % 64:
            raise ValueError(
                "TileKernels V4.1 mHC requires hidden size divisible by 64"
            )
        if hidden_streams.dtype != torch.bfloat16:
            raise ValueError("TileKernels V4.1 mHC requires bfloat16 hidden streams")
        if pre_mix.dtype != torch.float32:
            raise ValueError("TileKernels V4.1 mHC requires float32 pre_mix")
        if pre_mix.shape != hidden_streams.shape[:3]:
            raise ValueError("TileKernels V4.1 mHC pre_mix must have shape [B,S,N]")

        def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        flattened = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        input_mixes = functional.linear(  # pylint: disable=not-callable
            flattened, _local_tensor(self.fn).float()
        )
        next_pre, post, combine = deepseek_mhc_split_mixes(
            input_mixes.contiguous(),
            _local_tensor(self.scale).float().contiguous(),
            _local_tensor(self.base).float().contiguous(),
            self.hc_mult,
            epsilon=self.hc_eps,
        )
        combine = deepseek_mhc_sinkhorn(
            combine.contiguous(), self.hc_sinkhorn_iters, self.hc_eps
        )
        collapsed = deepseek_mhc_pre_apply(hidden_streams, pre_mix)
        return collapsed, next_pre, post, combine

    @staticmethod
    def post(
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        combine: torch.Tensor,
    ) -> torch.Tensor:
        """Run TileKernels' raw post operator through Hyper's autograd bridge."""
        residual_was_3d = residual.ndim == 3
        post_was_2d = post.ndim == 2
        combine_was_3d = combine.ndim == 3
        if sublayer_output.ndim == 2:
            sublayer_output = sublayer_output.unsqueeze(0)
        if residual_was_3d:
            residual = residual.unsqueeze(0)
        if post_was_2d:
            post = post.unsqueeze(0)
        if combine_was_3d:
            combine = combine.unsqueeze(0)
        if sublayer_output.device.type != "npu":
            raise ValueError("TileKernels V4.1 mHC post requires NPU tensors")
        if sublayer_output.ndim != 3 or residual.ndim != 4:
            raise ValueError("TileKernels V4.1 mHC post expects [B,S,D] and [B,S,N,D]")
        batch, sequence, hidden = sublayer_output.shape
        if residual.shape[:2] != (batch, sequence) or residual.shape[-1] != hidden:
            raise ValueError("TileKernels V4.1 mHC post residual shape mismatch")
        num_streams = residual.shape[2]
        if post.shape != (batch, sequence, num_streams):
            raise ValueError("TileKernels V4.1 mHC post coefficient shape mismatch")
        if combine.shape != (batch, sequence, num_streams, num_streams):
            raise ValueError("TileKernels V4.1 mHC combine coefficient shape mismatch")
        if sublayer_output.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise ValueError(
                "TileKernels V4.1 mHC post requires bfloat16 output and residual"
            )
        if post.dtype != torch.float32 or combine.dtype != torch.float32:
            raise ValueError(
                "TileKernels V4.1 mHC post requires float32 post/combine coefficients"
            )
        result = deepseek_mhc_post(
            sublayer_output.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            combine.contiguous(),
        )
        return result.squeeze(0) if residual_was_3d else result


@module_replacement
class TileKernelsV41TopKRouter(nn.Module):
    """V4.1 router preserving model semantics with TileKernels Top-K."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Adopt the source router parameters and immutable configuration."""
        super().__init__()
        del module_fqn, context
        required = (
            "hidden_size",
            "num_experts",
            "top_k",
            "scoring_func",
            "routed_scaling_factor",
            "weight",
            "bias",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(
                f"V4.1 router source is missing required attributes: {missing}"
            )
        self.hidden_size = int(module.hidden_size)
        self.num_experts = int(module.num_experts)
        self.top_k = int(module.top_k)
        self.scoring_func = str(module.scoring_func)
        self.routed_scaling_factor = float(module.routed_scaling_factor)
        self.weight = module.weight
        self.bias = module.bias
        if "bias_vl" in module._parameters:  # pylint: disable=protected-access
            self.register_parameter("bias_vl", module.bias_vl)
        else:
            self.bias_vl = getattr(module, "bias_vl", None)
        self.train(module.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, normalized routing weights, and selected experts."""
        flattened = hidden_states.reshape(-1, self.hidden_size)
        logits = functional.linear(  # pylint: disable=not-callable
            flattened.float(), self.weight.float()
        )
        if self.scoring_func == "sqrtsoftplus":
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(
                f"Unsupported V4.1 router scoring function: {self.scoring_func!r}"
            )
        correction_bias = self.bias
        if image_mask is not None:
            if image_mask.shape != hidden_states.shape[:2]:
                raise ValueError("image_mask must have shape [batch, sequence]")
            if self.bias_vl is not None:
                correction_bias = torch.where(
                    image_mask.reshape(-1, 1),
                    self.bias_vl.unsqueeze(0),
                    self.bias.unsqueeze(0),
                )
        indices = deepseek_topk_gate(
            (scores + correction_bias).contiguous(), self.top_k
        ).to(torch.long)
        weights = scores.gather(1, indices)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return logits, weights * self.routed_scaling_factor, indices


__all__ = [
    "TileKernelsEngramModule",
    "TileKernelsV41MhcModule",
    "TileKernelsV41TopKRouter",
]
