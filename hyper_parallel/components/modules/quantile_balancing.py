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
"""Pooled-histogram Quantile Balancing training state."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional, Union

# Native model state uses Torch directly, as required by the repository architecture.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

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


def _validate_histogram_storage(histogram_dtype: str, max_tokens_per_rank: int) -> None:
    """Validate a local accumulation bound before choosing narrow count storage."""
    if not isinstance(max_tokens_per_rank, int) or isinstance(max_tokens_per_rank, bool) or max_tokens_per_rank < 0:
        raise ValueError("max_tokens_per_rank must be a nonnegative integer")
    if histogram_dtype not in ("int32", "int64"):
        raise ValueError("histogram_dtype must be int32 or int64")
    if histogram_dtype == "int32" and not 0 < max_tokens_per_rank <= torch.iinfo(torch.int32).max:
        raise ValueError("int32 QB requires a positive max_tokens_per_rank bounded by int32")


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
        expert_chunk_size: Bound histogram index storage along the expert axis; zero uses all experts.
        histogram_dtype: Integer count storage, int64 by default or bounded int32.
        max_tokens_per_rank: Maximum observed rows per accumulated step for int32.
            Includes padding and every microbatch; the bound is independent of token validity.

    Note:
        Scores must be sigmoid probabilities. Bias must remain fixed throughout
        gradient accumulation. Call ``finalize_step`` once after each successful
        optimizer update. Call ``discard_statistics`` when an update is skipped;
        the bias and update counter must remain unchanged.
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
                 token_chunk_size: int = 0, histogram_dtype: str = "int64",
                 max_tokens_per_rank: int = 0, expert_chunk_size: int = 0) -> None:
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
        if not isinstance(expert_chunk_size, int) or expert_chunk_size < 0:
            raise ValueError("expert_chunk_size must be a nonnegative integer")
        _validate_histogram_storage(histogram_dtype, max_tokens_per_rank)
        self.max_tokens_per_rank = max_tokens_per_rank
        self.token_chunk_size = token_chunk_size
        self.expert_chunk_size = expert_chunk_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_bins = num_bins
        self.ema_decay = float(ema_decay)
        self.register_buffer("histogram", torch.zeros(num_experts, num_bins, dtype=getattr(torch, histogram_dtype)))
        self.register_buffer("token_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("lower_bound", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("bin_width", torch.tensor(2.0 / num_bins, dtype=torch.float32))
        self.register_buffer("update_steps", torch.zeros((), dtype=torch.int64))
        self._step_started = False
        self._observed_rows = 0
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
            **({"observed_rows": self._observed_rows} if self._step_started else {}),
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
        self._observed_rows = int(state.get("observed_rows", 0))
        if self.histogram.dtype == torch.int32 and self._step_started and "observed_rows" not in state:
            raise ValueError("An int32 midstep QB restore requires observed_rows in the checkpoint")
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
        self._observed_rows = 0
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
    def discard_statistics(self) -> None:
        """Clear a skipped step's observations without advancing bias or counters."""
        self.histogram.zero_()
        self.token_count.zero_()
        self._step_started = False
        self._observed_rows = 0

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
        self._observed_rows = 0

    @torch.no_grad()
    def accumulate(
        self,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        *,
        selected_experts: Optional[torch.Tensor] = None,
    ) -> None:
        """Record every valid token for every expert without retaining autograd.

        Args:
            scores: Sigmoid scores with shape ``[..., num_experts]``. DTensor
                token shards are accepted; expert shards and Partial are not.
            correction_bias: Router-owned FP32 bias, unchanged during this step.
            valid_token_mask: Boolean mask matching the local token dimensions,
                or their flattened form. This is a routing mask, not a loss mask:
                visual tokens count even when their language labels are ignored.
            selected_experts: The router's existing top-k indices, shaped [tokens, top_k].
                Their order is retained. Excluding these entries and taking the maximum
                gives the same (k+1)-th score, including ties, without a second TopK.
                Omit this argument for standalone statistics without a routing result.
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
        self._observed_rows += local_scores.shape[0]
        if self.histogram.dtype == torch.int32 and self._observed_rows > self.max_tokens_per_rank:
            raise ValueError("QB observed rows exceed max_tokens_per_rank; use int64 or increase the declared bound")
        selected = (None if selected_experts is None else
                    _local_tensor(selected_experts, "selected_experts").reshape(-1, self.top_k))
        if selected is not None and selected.shape[0] != local_scores.shape[0]:
            raise ValueError("selected_experts must have one top-k row per local token")
        self._accumulate_chunks(local_scores, bias, mask, selected)

    def _accumulate_chunks(self, local_scores, bias, mask, selected) -> None:
        """Bound temporary histogram indices while retaining the router's cutoff."""
        expert_chunk_size = self.expert_chunk_size or self.num_experts
        chunk_size = self.token_chunk_size or max(local_scores.shape[0], 1)
        for start in range(0, local_scores.shape[0], chunk_size):
            chunk = local_scores[start:start + chunk_size]
            adjusted = chunk + bias
            if selected is None:
                cutoff = torch.topk(adjusted, self.top_k + 1, dim=-1, sorted=False).values.amin(dim=-1)
            else:
                adjusted.scatter_(1, selected[start:start + chunk_size], -torch.inf)
                cutoff = adjusted.amax(dim=-1)
            del adjusted
            counts = mask[start:start + chunk_size].to(self.histogram.dtype)
            for expert_start in range(0, self.num_experts, expert_chunk_size):
                expert_end = min(expert_start + expert_chunk_size, self.num_experts)
                required_bias = cutoff.unsqueeze(-1) - chunk[:, expert_start:expert_end]
                bin_indices = ((required_bias - self.lower_bound) / self.bin_width).floor().to(torch.int64)
                bin_indices.clamp_(min=0, max=self.num_bins - 1)
                self.histogram[expert_start:expert_end].scatter_add_(
                    1, bin_indices.transpose(0, 1), counts.unsqueeze(0).expand(expert_end - expert_start, -1))
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
    def prepare_statistics(self, correction_bias: torch.Tensor) -> torch.Tensor:
        """Validate the bias and initialize an empty step if necessary.

        Args:
            correction_bias: Router-owned bias, fixed during accumulation.

        Returns:
            The local FP32 bias used to commit this step.
        """
        bias = self._validate_bias(correction_bias)
        if not self._step_started:
            self.reset_statistics(bias)
        return bias

    @torch.no_grad()
    def commit_statistics(self, correction_bias: torch.Tensor, histogram: torch.Tensor) -> torch.Tensor:
        """Apply local or already pooled statistics and clear this step.

        Args:
            correction_bias: Validated local bias returned by prepare_statistics.
            histogram: Complete expert/bin counts, optionally reduced by a caller.

        Returns:
            Device boolean indicating a nonempty batch without a host read.
        """
        tokens = histogram[0].sum(dtype=torch.int64)
        nonempty = tokens > 0
        candidate = self._estimate_bias(histogram, tokens)
        candidate = self.ema_decay * correction_bias + (1.0 - self.ema_decay) * candidate
        candidate = candidate - candidate.mean()
        correction_bias.copy_(torch.where(nonempty, candidate, correction_bias))
        self.update_steps.add_(nonempty.to(torch.int64))
        self.discard_statistics()
        return nonempty

    @torch.no_grad()
    def finalize_step(self, correction_bias: torch.Tensor) -> bool:
        """Commit local counts; distributed callers pool before commit_statistics.

        Args:
            correction_bias: Router-owned FP32 bias.

        Returns:
            Whether a nonempty local batch updated the bias.
        """
        bias = self.prepare_statistics(correction_bias)
        return bool(self.commit_statistics(bias, self.histogram).item())

    def _estimate_bias(self, histogram: torch.Tensor, num_tokens: Union[int, torch.Tensor]) -> torch.Tensor:
        numerator = num_tokens * self.top_k
        target_rank = (numerator + self.num_experts - 1) // self.num_experts
        histogram = histogram.to(torch.int64)
        cumulative = histogram.cumsum(dim=1)
        selected_bin = (cumulative >= target_rank).to(torch.int64).argmax(dim=1, keepdim=True)
        bin_count = histogram.gather(1, selected_bin).squeeze(1)
        preceding = cumulative.gather(1, selected_bin).squeeze(1) - bin_count
        denominator = (bin_count * self.num_experts).clamp_min(1).float()
        fraction = (numerator - preceding * self.num_experts).float() / denominator
        candidate = self.lower_bound + (selected_bin.squeeze(1).float() + fraction.clamp(0.0, 1.0)) * self.bin_width
        return candidate - candidate.mean()
