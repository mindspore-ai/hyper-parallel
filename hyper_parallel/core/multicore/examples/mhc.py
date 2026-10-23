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

"""Ordinary and Mega four-stream mHC boundaries with the same state and parameters."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.multicore import MegaMhc, MegaMhcOutput, MhcState


def _reference_residual(state: MhcState) -> torch.Tensor:
    """Evaluate the post formula without Omni for independent precision checks."""
    output, residual, _, post, mix = state
    if residual.device.type == "npu":
        # Match CANN MhcPost: start with post, then accumulate four Axpy operations.
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(-2)
        for stream in range(4):
            result = torch.addcmul(
                result, residual[..., stream:stream + 1, :].float(), mix[..., stream, :].float().unsqueeze(-1),
            )
        return result.to(residual.dtype)
    return (
        torch.einsum("...ji,...jh->...ih", mix.float(), residual.float())
        + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)
    ).to(residual.dtype)


def _mapping_logits(residual: torch.Tensor, phi: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """Use semantic CPU math and the matching CANN projection order on NPU."""
    residual_fp32 = residual.float()
    variance = residual_fp32.square().mean(dim=(-2, -1), keepdim=True) + norm_eps
    if residual.device.type != "npu":
        normalized = residual_fp32 * torch.rsqrt(variance)
        return torch.matmul(normalized.flatten(-2), phi.float().transpose(-1, -2))
    # CANN projects the unnormalized BF16 values with HF32, then scales the result.
    previous_hf32 = torch.npu.matmul.allow_hf32
    try:
        torch.npu.matmul.allow_hf32 = True
        logits = torch.matmul(residual_fp32.flatten(-2), phi.float().transpose(-1, -2))
    finally:
        torch.npu.matmul.allow_hf32 = previous_hf32
    return logits * (1.0 / torch.sqrt(variance)).squeeze(-1)


def _sinkhorn(logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
    """Show the ordinary Torch normalization used by the comparison path."""
    matrix = logits.softmax(-1) + epsilon
    matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
    return matrix


class MhcBoundary(MegaMhc):
    """Compare ordinary Torch mHC with Mega using identical parameters."""

    def __init__(self, hidden_size: int, norm_eps: float, *, use_mega_mhc: bool) -> None:
        """Initialize FP32 predictors and the BF16 input RMSNorm weight."""
        super().__init__(hidden_size, norm_eps=norm_eps)
        self.use_mega_mhc = use_mega_mhc

    def forward(self, *state: torch.Tensor | MhcState) -> MegaMhcOutput:
        """Accept explicit named state or the compatible five-tensor boundary call."""
        values = tuple(state[0]) if len(state) == 1 and isinstance(state[0], MhcState) else state
        if self.use_mega_mhc:
            return self.mega_forward(*values)
        return MegaMhcOutput(*self.native_forward(*values))

    def mega_forward(self, *state: torch.Tensor | MhcState) -> MegaMhcOutput:
        """Call managed MegaMhc with the public state contract.

        Args:
            state: Explicit MhcState or the compatible five state tensors.
        """
        return super().forward(*state)

    def _mappings(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Share coefficient transforms while keeping the Sinkhorn backend explicit."""
        pre_logits, post_logits, mix_logits = logits.split((4, 4, 16), dim=-1)
        pre_bias, post_bias, mix_bias = self.bias.float().split((4, 4, 16))
        pre = torch.sigmoid(pre_logits * self.alpha[0] + pre_bias) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_logits * self.alpha[1] + post_bias)
        mix = (mix_logits * self.alpha[2] + mix_bias).reshape(*logits.shape[:-1], 4, 4)
        return pre, post, _sinkhorn(mix, self.num_iters, self.hc_eps)

    def native_forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MhcState:
        """Evaluate ordinary mHC with the same parameters and shifted-state equations.

        Args:
            previous_output: Output of the previous block, shaped [..., H].
            residual: Four residual streams, shaped [..., 4, H].
            previous_pre_mix: Previous pre-mix coefficients, shaped [..., 4].
            previous_post_mix: Previous post-mix coefficients, shaped [..., 4].
            previous_residual_mix: Previous residual mixing matrices, shaped [..., 4, 4].
        """
        state = (previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix)
        new_residual = _reference_residual(state)
        logits = _mapping_logits(new_residual, self.phi, self.norm_eps)
        pre, post, mix = self._mappings(logits)
        if residual.device.type == "npu":
            # CANN reduces streams left to right and stores BF16 before ordinary RMSNorm.
            weighted = new_residual.float() * previous_pre_mix.float().unsqueeze(-1)
            mixed_input = weighted[..., 0, :]
            for stream in range(1, 4):
                mixed_input = mixed_input + weighted[..., stream, :]
            block_input, _ = torch.ops.npu.npu_rms_norm(
                mixed_input.to(residual.dtype), self.norm_weight, self.norm_eps,
            )
            return new_residual, pre, post, mix, block_input
        mixed_input = (new_residual.float() * previous_pre_mix.float().unsqueeze(-1)).sum(dim=-2)
        block_input = mixed_input * torch.rsqrt(mixed_input.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        block_input = (block_input * self.norm_weight.float()).to(residual.dtype)
        return new_residual, pre, post, mix, block_input

    reference_forward = native_forward
