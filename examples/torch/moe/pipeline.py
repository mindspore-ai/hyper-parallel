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
"""Pipeline-parallel stage builders for ``MoEDemoModel`` (see ``model.py``).

Each PP rank owns one contiguous slice of decoder layers.  Stage 0 includes
``tok_embeddings``; the last stage adds ``norm`` + ``output`` and can compute
per-micro-batch cross-entropy when targets are registered.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hyper_parallel import PipelineStage

from model import (
    MoEDemoConfig,
    MoEDemoModel,
    MoETransformerBlock,
    RMSNorm,
    precompute_freqs_cis,
)


def layer_range_for_pp_stage(
    pp_rank: int, n_layers: int, pp_size: int,
) -> Tuple[int, int]:
    """Return ``[start, end)`` layer indices owned by ``pp_rank``.

    Args:
        pp_rank: Pipeline rank in ``[0, pp_size)``.
        n_layers: Total decoder layer count in the full model.
        pp_size: Pipeline parallel width.

    Returns:
        Half-open interval of global layer indices for this stage.

    Raises:
        ValueError: If ``pp_rank`` or ``pp_size`` is out of range,
            or ``n_layers < pp_size``.
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}.")
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}.")
    if n_layers < pp_size:
        raise ValueError(f"n_layers ({n_layers}) must be >= pp_size ({pp_size}).")
    base, rem = divmod(n_layers, pp_size)
    start = pp_rank * base + min(pp_rank, rem)
    end = start + base + (1 if pp_rank < rem else 0)
    return start, end


def split_batch_dim0(
    tensor: torch.Tensor, micro_batch_num: int,
) -> List[torch.Tensor]:
    """Split ``tensor`` along batch dim 0 into ``micro_batch_num`` equal chunks."""
    if micro_batch_num < 1:
        raise ValueError("micro_batch_num must be >= 1.")
    batch = tensor.shape[0]
    if batch % micro_batch_num != 0:
        raise ValueError(
            f"batch size ({batch}) must divide micro_batch_num ({micro_batch_num})."
        )
    chunk = batch // micro_batch_num
    return [tensor[i * chunk : (i + 1) * chunk] for i in range(micro_batch_num)]


def _slice_rope_freqs(
    freqs_cis: torch.Tensor, seq_len: int, rope_seq_start: int = 0,
) -> torch.Tensor:
    """Slice RoPE table for a local sequence window (CP-aware)."""
    return freqs_cis[rope_seq_start : rope_seq_start + seq_len]


class MoEPPFirstStage(nn.Module):
    """First pipeline stage: token embedding + a slice of MoE decoder blocks."""

    def __init__(
        self, cfg: MoEDemoConfig, layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            MoETransformerBlock(cfg) for _ in layer_indices
        )
        freqs = precompute_freqs_cis(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self.rope_seq_start = 0

    def set_rope_seq_start(self, start: int) -> None:
        """Set global RoPE offset for the local CP token window."""
        self.rope_seq_start = start

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and run local decoder blocks.

        Args:
            token_ids: ``[batch, seq_len]`` token indices (stage-0 input only).

        Returns:
            Hidden states passed to the next pipeline stage.
        """
        hidden = self.tok_embeddings(token_ids)
        freqs = _slice_rope_freqs(self.freqs_cis, hidden.shape[1], self.rope_seq_start)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        return hidden


class MoEPPMiddleStage(nn.Module):
    """Middle pipeline stage: decoder blocks only (activations in / out)."""

    def __init__(
        self, cfg: MoEDemoConfig, layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            MoETransformerBlock(cfg) for _ in layer_indices
        )
        freqs = precompute_freqs_cis(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self.rope_seq_start = 0

    def set_rope_seq_start(self, start: int) -> None:
        """Set global RoPE offset for the local CP token window."""
        self.rope_seq_start = start

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run local decoder blocks on received activations."""
        freqs = _slice_rope_freqs(self.freqs_cis, hidden.shape[1], self.rope_seq_start)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        return hidden


class MoEPPLastStage(nn.Module):
    """Last pipeline stage: decoder blocks + final norm + LM head (+ optional CE loss)."""

    def __init__(
        self, cfg: MoEDemoConfig, layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            MoETransformerBlock(cfg) for _ in layer_indices
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        freqs = precompute_freqs_cis(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self._micro_index = 0
        self._micro_targets: Optional[List[torch.Tensor]] = None
        self.rope_seq_start = 0

    def set_rope_seq_start(self, start: int) -> None:
        """Set global RoPE offset for the local CP token window."""
        self.rope_seq_start = start

    def set_micro_targets(self, targets_per_mb: List[torch.Tensor]) -> None:
        """Register per-micro-batch targets for cross-entropy on the last stage."""
        self._micro_targets = targets_per_mb

    def set_micro_index(self, micro_index: int) -> None:
        """Set the active micro-batch index (called by :class:`MicrobatchLossPipelineStage`)."""
        self._micro_index = micro_index

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run local blocks, LM head, and optionally CE loss for the active micro-batch."""
        freqs = _slice_rope_freqs(self.freqs_cis, hidden.shape[1], self.rope_seq_start)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        hidden = self.norm(hidden)
        logits = self.output(hidden)
        if self._micro_targets is not None:
            targets = self._micro_targets[self._micro_index]
            return F.cross_entropy(
                logits.float().reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        return logits


class MoEPPFullStage(nn.Module):
    """Single-stage PP chunk (``pp_size == 1``): embedding through LM head."""

    def __init__(
        self, cfg: MoEDemoConfig, layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            MoETransformerBlock(cfg) for _ in layer_indices
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        freqs = precompute_freqs_cis(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self.rope_seq_start = 0
        self._micro_index = 0
        self._micro_targets: Optional[List[torch.Tensor]] = None

    def set_rope_seq_start(self, start: int) -> None:
        """Set global RoPE offset for the local CP token window."""
        self.rope_seq_start = start

    def set_micro_targets(self, targets_per_mb: List[torch.Tensor]) -> None:
        """Register per-micro-batch targets for cross-entropy."""
        self._micro_targets = targets_per_mb

    def set_micro_index(self, micro_index: int) -> None:
        """Set the active micro-batch index."""
        self._micro_index = micro_index

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full forward from tokens to logits or CE loss."""
        hidden = self.tok_embeddings(token_ids)
        freqs = _slice_rope_freqs(self.freqs_cis, hidden.shape[1], self.rope_seq_start)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        hidden = self.norm(hidden)
        logits = self.output(hidden)
        if self._micro_targets is not None:
            targets = self._micro_targets[self._micro_index]
            return F.cross_entropy(
                logits.float().reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        return logits


def build_moe_pp_chunk(
    cfg: MoEDemoConfig, pp_rank: int, pp_size: int,
) -> nn.Module:
    """Build the ``nn.Module`` chunk owned by ``pp_rank``.

    Args:
        cfg: Model hyperparameters (``n_layers`` must be >= ``pp_size``).
        pp_rank: Pipeline rank of the current process.
        pp_size: Total pipeline width.

    Returns:
        First, middle, or last stage module for ``pp_rank``.
    """
    start, end = layer_range_for_pp_stage(pp_rank, cfg.n_layers, pp_size)
    layer_indices = list(range(start, end))
    if pp_size == 1:
        return MoEPPFullStage(cfg, layer_indices)
    if pp_rank == 0:
        return MoEPPFirstStage(cfg, layer_indices)
    if pp_rank == pp_size - 1:
        return MoEPPLastStage(cfg, layer_indices)
    return MoEPPMiddleStage(cfg, layer_indices)


def _collect_keys_by_prefix(
    full_sd: Dict[str, torch.Tensor],
    prefixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Return entries from *full_sd* whose key matches any *prefixes*."""
    result: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        for k, v in full_sd.items():
            if k.startswith(prefix + ".") or k == prefix:
                result[k] = v
    return result


def _collect_layer_keys(
    full_sd: Dict[str, torch.Tensor],
    start: int,
    end: int,
) -> Dict[str, torch.Tensor]:
    """Return layer entries from *full_sd*, remapped to local indices ``[0, end-start)``."""
    result: Dict[str, torch.Tensor] = {}
    for idx in range(start, end):
        layer_prefix = f"layers.{idx}."
        for k, v in full_sd.items():
            if k.startswith(layer_prefix):
                local_key = f"layers.{idx - start}." + k[len(layer_prefix):]
                result[local_key] = v
    return result


def extract_stage_state_dict(
    full_model: MoEDemoModel,
    cfg: MoEDemoConfig,
    pp_rank: int,
    pp_size: int,
) -> Dict[str, torch.Tensor]:
    """Extract the state dict for one PP stage from a full MoEDemoModel.

    Builds a full model, then slices its state dict to keep only the
    parameters that belong to ``pp_rank``'s stage.  Useful for initializing
    PP stage modules with identical weights to a reference standalone model.

    Args:
        full_model: A fully-constructed :class:`MoEDemoModel` (used as the
            weight source; its ``state_dict()`` is copied).
        cfg: Model configuration.
        pp_rank: Pipeline rank whose parameters to extract.
        pp_size: Total pipeline width.

    Returns:
        State dict containing only the parameters for the given stage.
    """
    full_sd = copy.deepcopy(full_model.state_dict())
    start, end = layer_range_for_pp_stage(pp_rank, cfg.n_layers, pp_size)

    if pp_size == 1:
        return full_sd

    stage_sd: Dict[str, torch.Tensor] = {}

    if pp_rank == 0:
        stage_sd.update(_collect_keys_by_prefix(full_sd, ["tok_embeddings"]))
    if pp_rank == pp_size - 1:
        stage_sd.update(_collect_keys_by_prefix(full_sd, ["norm", "output"]))

    stage_sd.update(_collect_layer_keys(full_sd, start, end))

    is_boundary = pp_rank in (0, pp_size - 1)
    if pp_size > 2 and not is_boundary:
        stage_sd.update(_collect_keys_by_prefix(full_sd, ["freqs_cis"]))

    return stage_sd


class MicrobatchLossPipelineStage(PipelineStage):
    """``PipelineStage`` that forwards the micro-batch index to the submodule.

    Required when the last stage computes cross-entropy from pre-split targets
    (see :meth:`MoEPPLastStage.set_micro_targets`).
    """

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        submodule = self.submodule
        if hasattr(submodule, "set_micro_index"):
            submodule.set_micro_index(micro_index)
        return super().forward_one_chunk(micro_index, args, kwargs)


def build_pipeline_stage(
    module: nn.Module,
    *,
    pp_rank: int,
    pp_size: int,
    device: torch.device,
    pp_mesh=None,
    use_microbatch_loss: bool = False,
) -> PipelineStage:
    """Wrap ``module`` in a :class:`PipelineStage` for 1-D pipeline parallelism.

    Args:
        module: Stage-local ``nn.Module`` chunk.
        pp_rank: Pipeline rank (equals ``stage_index`` for one stage per rank).
        pp_size: Total virtual stage count.
        device: Device for P2P buffers.
        pp_mesh: Optional 1-D PP :class:`~hyper_parallel.DeviceMesh` slice.
        use_microbatch_loss: When ``True``, use :class:`MicrobatchLossPipelineStage`
            so the last stage can index per-micro-batch targets.

    Returns:
        Initialized :class:`PipelineStage` ready for a schedule ``run()`` call.
    """
    stage_cls = MicrobatchLossPipelineStage if use_microbatch_loss else PipelineStage
    return stage_cls(
        module,
        stage_index=pp_rank,
        stage_num=pp_size,
        device=device,
        mesh=pp_mesh,
    )


def count_moe_parameters(cfg: MoEDemoConfig) -> int:
    """Return parameter count of an unsharded reference ``MoEDemoModel`` (for logging)."""
    return sum(p.numel() for p in MoEDemoModel(cfg).parameters())
