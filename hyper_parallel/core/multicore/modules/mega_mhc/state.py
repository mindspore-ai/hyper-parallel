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
"""Named shifted mHC state and boundary outputs for model integration."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ..mega_mhc_common import MHC_STREAMS


class MhcState(NamedTuple):
    """Pending block output, residual streams and the mappings used by the next boundary."""

    previous_output: torch.Tensor
    residual: torch.Tensor
    pre_mix: torch.Tensor
    post_mix: torch.Tensor
    residual_mix: torch.Tensor

    @classmethod
    def initialize(cls, hidden_states: torch.Tensor) -> MhcState:
        """Expand initial activations into four streams with identity residual mixing.

        Args:
            hidden_states: Floating activations shaped [..., H], with at least one token dimension.

        Returns:
            Independent explicit state whose residual retains the input autograd connection.
        """
        if hidden_states.ndim < 2 or not hidden_states.is_floating_point():
            raise ValueError("mHC initialization requires floating activations shaped [..., H]")
        leading = hidden_states.shape[:-1]
        residual = hidden_states.unsqueeze(-2).expand(*leading, MHC_STREAMS, hidden_states.shape[-1]).contiguous()
        pre = torch.zeros((*leading, MHC_STREAMS), device=hidden_states.device, dtype=torch.float32)
        pre[..., 0] = 1.0
        post = torch.zeros_like(pre)
        identity = torch.eye(MHC_STREAMS, device=hidden_states.device, dtype=torch.float32)
        mix = identity.expand(*leading, MHC_STREAMS, MHC_STREAMS).contiguous()
        return cls(torch.zeros_like(hidden_states), residual, pre, post, mix)

    def finish(self) -> torch.Tensor:
        """Apply the final pending output and collapse streams with the pending pre mapping.

        Returns:
            Activations shaped like previous_output. This is differentiable ordinary tensor math on the state device.
        """
        if self.residual.device.type == "npu":
            residual = self.post_mix.float().unsqueeze(-1) * self.previous_output.float().unsqueeze(-2)
            for stream in range(MHC_STREAMS):
                residual = torch.addcmul(residual, self.residual[..., stream:stream + 1, :].float(),
                                        self.residual_mix[..., stream, :].float().unsqueeze(-1))
        else:
            residual = (torch.einsum("...ji,...jh->...ih", self.residual_mix.float(), self.residual.float())
                        + self.post_mix.float().unsqueeze(-1) * self.previous_output.float().unsqueeze(-2))
        residual = residual.to(self.residual.dtype)
        return (residual.float() * self.pre_mix.float().unsqueeze(-1)).sum(dim=-2).to(self.residual.dtype)


class MegaMhcOutput(NamedTuple):
    """Updated residual, next mappings and normalized input for the current block."""

    residual: torch.Tensor
    pre_mix: torch.Tensor
    post_mix: torch.Tensor
    residual_mix: torch.Tensor
    hidden_states: torch.Tensor

    def next_state(self, block_output: torch.Tensor) -> MhcState:
        """Attach the current block's output for the next boundary.

        Args:
            block_output: Block result with the same shape, dtype and device as hidden_states.

        Returns:
            Explicit state; multiple outstanding states can be held independently.
        """
        if (block_output.shape != self.hidden_states.shape or block_output.dtype != self.hidden_states.dtype
                or block_output.device != self.hidden_states.device):
            raise ValueError("block output must match the mHC block input shape, dtype and device")
        return MhcState(block_output, self.residual, self.pre_mix, self.post_mix, self.residual_mix)
