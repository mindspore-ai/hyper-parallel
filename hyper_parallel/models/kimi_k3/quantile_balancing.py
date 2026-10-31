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
"""Kimi K3 pooled-histogram Quantile Balancing training state."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional

# Native model state uses Torch directly, as required by the repository architecture.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.dtensor.dtensor import DTensor


def _local_tensor(tensor: torch.Tensor, name: str, expert_axis: bool = False) -> torch.Tensor:
    """Cross the no-grad statistics boundary without gathering token shards."""
    if isinstance(tensor, DTensor):
        for placement in tensor.placements:
            if placement.is_partial():
                raise ValueError(f"{name} must not have a Partial placement")
            if expert_axis and placement.is_shard() and placement.dim % tensor.ndim == tensor.ndim - 1:
                raise ValueError(f"{name} requires the complete expert dimension on each rank")
        tensor = tensor.to_local()
    return tensor


class QuantileBalancer(nn.Module):
    """Accumulate required-bias histograms and commit the next step's bias.

    The router owns its FP32 ``e_score_correction_bias`` buffer. This module
    never changes scores or routing weights and never communicates during a
    forward pass. Its buffers and extra state support local or rank-local
    checkpoints between microbatches, and distributed checkpoints at completed
    optimizer-step boundaries after statistics have been committed and cleared.

    Args:
        num_experts: Number of routed experts, all present on each router rank.
        top_k: Number of selected experts; strictly smaller than num_experts.
        num_bins: Uniform histogram bins in [min(bias)-1, max(bias)+1].
        ema_decay: Previous-bias weight; zero directly uses the new estimate.
        token_chunk_size: Bound temporary token/expert tensors; zero retains the full batch.

    Note:
        Scores must be sigmoid probabilities. Bias must remain fixed throughout
        gradient accumulation. Call ``finalize_step`` once per consumed data
        step, after its backwards, including when an optimizer update is skipped.
        Callers decide which ranks own unique tokens; replicated TP/EP tokens
        must not be counted again. External checkpoint implementations must use
        ``suspend_statistics`` around recomputation. HyperParallel recomputation
        is detected automatically. Pending histograms are rank-local: ordinary
        distributed checkpoint planners deduplicate identical buffer FQNs and
        cannot preserve different ranks' in-flight statistics. Midstep distributed
        recovery requires an explicit rank-local storage/sharding protocol;
        per-rank ``torch.save`` can preserve this module's local state.
    """

    def __init__(self, num_experts: int, top_k: int, num_bins: int = 1000, ema_decay: float = 0.0,
                 token_chunk_size: int = 0) -> None:
        """Allocate persistent step statistics without owning the router bias."""
        super().__init__()
        if not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be an integer of at least two")
        if not isinstance(top_k, int) or not 0 < top_k < num_experts:
            raise ValueError("top_k must be a positive integer smaller than num_experts")
        if not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer of at least two")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not isinstance(token_chunk_size, int) or token_chunk_size < 0:
            raise ValueError("token_chunk_size must be a nonnegative integer")
        self.token_chunk_size = token_chunk_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_bins = num_bins
        self.ema_decay = float(ema_decay)
        self.register_buffer("histogram", torch.zeros(num_experts, num_bins, dtype=torch.int64))
        self.register_buffer("token_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("lower_bound", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("bin_width", torch.tensor(2.0 / num_bins, dtype=torch.float32))
        self.register_buffer("update_steps", torch.zeros((), dtype=torch.int64))
        self._step_started = False
        self._suspend_depth = 0

    def _apply(self, fn: Callable, recurse: bool = True) -> "QuantileBalancer":
        def _preserve_statistics_dtype(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if tensor.dtype == torch.float32 and converted.dtype != torch.float32:
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(_preserve_statistics_dtype, recurse=recurse)

    def get_extra_state(self) -> dict[str, Any]:
        """Serialize configuration and the host-side accumulation phase."""
        return {
            "format_version": 1,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_bins": self.num_bins,
            "ema_decay": self.ema_decay,
            "step_started": self._step_started,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint with the same quantile estimator configuration.

        Args:
            state: Extra-state dictionary from ``get_extra_state``, including the
                matching estimator configuration and the saved accumulation phase.

        Raises:
            ValueError: The format version or estimator configuration does not match.

        Note:
            Restores the step-started flag and clears statistics suspension depth.
            Persistent tensor buffers are restored separately through the state dict.
        """
        expected = self.get_extra_state()
        for key in ("format_version", "num_experts", "top_k", "num_bins", "ema_decay"):
            if state.get(key) != expected[key]:
                raise ValueError(f"QuantileBalancer checkpoint {key}={state.get(key)!r}, expected {expected[key]!r}")
        self._step_started = bool(state["step_started"])
        self._suspend_depth = 0

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize all buffers after construction or meta materialization."""
        self.histogram.zero_()
        self.token_count.zero_()
        self.lower_bound.fill_(-1.0)
        self.bin_width.fill_(2.0 / self.num_bins)
        self.update_steps.zero_()
        self._step_started = False
        self._suspend_depth = 0

    @contextmanager
    def suspend_statistics(self) -> Iterator[None]:
        """Temporarily disable statistics, including nested recomputation calls."""
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    def _validate_bias(self, correction_bias: torch.Tensor) -> torch.Tensor:
        bias = _local_tensor(correction_bias, "correction_bias", expert_axis=True)
        if bias.shape != (self.num_experts,) or bias.dtype != torch.float32:
            raise ValueError(f"correction_bias must be FP32 with shape ({self.num_experts},)")
        if bias.device != self.histogram.device:
            raise ValueError("QuantileBalancer buffers and correction_bias must be on the same device")
        return bias

    @torch.no_grad()
    def reset_statistics(self, correction_bias: torch.Tensor) -> None:
        """Discard pending counts and start fixed bins for the supplied bias.

        This does not reset ``update_steps`` or modify the router's bias.

        Args:
            correction_bias: Router-owned FP32 bias with shape ``[num_experts]`` on
                the statistics device, used to fix the next accumulation's bin range.
                DTensor inputs must expose the complete expert axis without Partial
                placements; the local bias is used without gathering token shards.
        """
        bias = self._validate_bias(correction_bias)
        self.histogram.zero_()
        self.token_count.zero_()
        self.lower_bound.copy_(bias.amin() - 1.0)
        self.bin_width.copy_((bias.amax() - bias.amin() + 2.0) / self.num_bins)
        self._step_started = True

    @torch.no_grad()
    def accumulate(
        self,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Record every valid token for every expert without retaining autograd.

        Args:
            scores: Sigmoid scores with shape ``[..., num_experts]``. DTensor
                token shards are accepted; expert shards and Partial are not.
            correction_bias: Router-owned FP32 bias, unchanged during this step.
            valid_token_mask: Boolean mask matching the local token dimensions,
                or their flattened form. This is a routing mask, not a loss mask:
                visual tokens count even when their language labels are ignored.
        """
        if not self.training or self._suspend_depth or is_recomputing():
            return
        bias = self._validate_bias(correction_bias)
        local_scores = _local_tensor(scores, "scores", expert_axis=True)
        if local_scores.ndim < 2 or local_scores.shape[-1] != self.num_experts:
            raise ValueError(f"scores must have shape [..., {self.num_experts}]")
        if not local_scores.is_floating_point() or local_scores.device != self.histogram.device:
            raise ValueError("scores must be floating point and on the QuantileBalancer device")
        local_scores = local_scores.reshape(-1, self.num_experts).float()
        mask = self._valid_mask(valid_token_mask, scores, local_scores.shape[0])
        if not self._step_started:
            self.reset_statistics(bias)
        chunk_size = self.token_chunk_size or max(local_scores.shape[0], 1)
        for start in range(0, local_scores.shape[0], chunk_size):
            chunk = local_scores[start:start + chunk_size]
            cutoff = torch.topk(chunk + bias, self.top_k + 1, dim=-1, sorted=False).values.amin(dim=-1)
            required_bias = cutoff.unsqueeze(-1) - chunk
            bin_indices = ((required_bias - self.lower_bound) / self.bin_width).floor().to(torch.int64)
            bin_indices.clamp_(min=0, max=self.num_bins - 1)
            counts = mask[start:start + chunk_size].to(torch.int64)
            self.histogram.scatter_add_(
                1, bin_indices.transpose(0, 1), counts.unsqueeze(0).expand(self.num_experts, -1))
            self.token_count.add_(counts.sum())

    def _valid_mask(self, mask: Optional[torch.Tensor], scores: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if mask is None:
            return torch.ones(num_tokens, dtype=torch.bool, device=self.histogram.device)
        mask = _local_tensor(mask, "valid_token_mask")
        score_shape = _local_tensor(scores, "scores").shape
        if mask.dtype != torch.bool or mask.device != self.histogram.device:
            raise ValueError("valid_token_mask must be boolean and on the QuantileBalancer device")
        if mask.shape not in (score_shape[:-1], (num_tokens,)):
            raise ValueError("valid_token_mask must match local score token dimensions or their flattened form")
        return mask.reshape(-1)

    @torch.no_grad()
    def finalize_step(self, correction_bias: torch.Tensor, process_groups: Sequence[Any] = ()) -> bool:
        """SUM pooled counts, update bias in place, and discard step statistics.

        Args:
            correction_bias: The same router-owned FP32 bias used for routing.
            process_groups: Orthogonal groups containing distinct token shards.
                Empty means local execution; ``(None,)`` means the world group.
                Callers must exclude groups that only replicate tokens and ensure
                all participants use the same bias and histogram configuration.

        Returns:
            Whether a nonempty global batch updated bias. Globally empty steps
            clear statistics without changing bias or ``update_steps``.
        """
        bias = self._validate_bias(correction_bias)
        if len({id(group) for group in process_groups}) != len(process_groups):
            raise ValueError("process_groups must not contain duplicate groups")
        if process_groups and (not dist.is_available() or not dist.is_initialized()):
            raise ValueError("Quantile Balancing process_groups require initialized torch.distributed")
        if not self._step_started:
            self.reset_statistics(bias)
        pooled_histogram = self.histogram.clone()
        pooled_tokens = self.token_count.clone()
        for group in process_groups:
            dist.all_reduce(pooled_histogram, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(pooled_tokens, op=dist.ReduceOp.SUM, group=group)
        # A host read is restricted to the step boundary, never a microbatch forward.
        num_tokens = int(pooled_tokens.item())
        if num_tokens:
            candidate = self._estimate_bias(pooled_histogram, num_tokens)
            candidate = self.ema_decay * bias + (1.0 - self.ema_decay) * candidate
            bias.copy_(candidate - candidate.mean())
            self.update_steps.add_(1)
        self.histogram.zero_()
        self.token_count.zero_()
        self._step_started = False
        return num_tokens > 0

    def _estimate_bias(self, histogram: torch.Tensor, num_tokens: int) -> torch.Tensor:
        numerator = num_tokens * self.top_k
        target_rank = (numerator + self.num_experts - 1) // self.num_experts
        cumulative = histogram.cumsum(dim=1)
        selected_bin = (cumulative >= target_rank).to(torch.int64).argmax(dim=1, keepdim=True)
        bin_count = histogram.gather(1, selected_bin).squeeze(1)
        preceding = cumulative.gather(1, selected_bin).squeeze(1) - bin_count
        fraction = (numerator - preceding * self.num_experts).float() / (bin_count * self.num_experts).float()
        candidate = self.lower_bound + (selected_bin.squeeze(1).float() + fraction.clamp(0.0, 1.0)) * self.bin_width
        return candidate - candidate.mean()
