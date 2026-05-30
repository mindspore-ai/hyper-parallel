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
"""Pipeline-parallel stage builders for ``tests.torch.accuracy.model.Llama3Model``.

Aligned with ``examples/torch/llama3/pipeline.py`` so PP composite tests can build
stage chunks and copy weights from a full-model reference without ``sys.path`` hacks.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hyper_parallel import PipelineStage

from tests.torch.accuracy.model import (
    Llama3DemoConfig,
    Llama3Model,
    Llama3RMSNorm,
    Llama3TransformerBlock,
    _compute_ffn_hidden_dim,
    precompute_freqs_cis,
)


def layer_range_for_pp_stage(pp_rank: int, n_layers: int, pp_size: int) -> Tuple[int, int]:
    """Return ``[start, end)`` layer indices owned by ``pp_rank``.

    Args:
        pp_rank: Pipeline rank in ``[0, pp_size)``.
        n_layers: Total decoder layer count in the full model.
        pp_size: Pipeline parallel width.

    Returns:
        Half-open interval of global layer indices for this stage.

    Raises:
        ValueError: If ``pp_rank`` or ``pp_size`` is out of range, or ``n_layers < pp_size``.
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


def split_batch_dim0(tensor: torch.Tensor, micro_batch_num: int) -> List[torch.Tensor]:
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


class Llama3PPFirstStage(nn.Module):
    """First pipeline stage: token embedding + a slice of decoder blocks."""

    def __init__(self, cfg: Llama3DemoConfig, layer_indices: Sequence[int]) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            Llama3TransformerBlock(cfg, hidden_dim) for _ in layer_indices
        )
        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self.rope_seq_start = 0

    def set_rope_seq_start(self, start: int) -> None:
        """Set global RoPE offset for the local CP token window."""
        self.rope_seq_start = start

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and run local decoder blocks."""
        hidden = self.tok_embeddings(token_ids)
        freqs = _slice_rope_freqs(self.freqs_cis, hidden.shape[1], self.rope_seq_start)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        return hidden


class Llama3PPMiddleStage(nn.Module):
    """Middle pipeline stage: decoder blocks only (activations in / out)."""

    def __init__(self, cfg: Llama3DemoConfig, layer_indices: Sequence[int]) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.layers = nn.ModuleList(
            Llama3TransformerBlock(cfg, hidden_dim) for _ in layer_indices
        )
        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
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


class Llama3PPLastStage(nn.Module):
    """Last pipeline stage: decoder blocks + final norm + LM head (+ optional CE loss)."""

    def __init__(self, cfg: Llama3DemoConfig, layer_indices: Sequence[int]) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.layers = nn.ModuleList(
            Llama3TransformerBlock(cfg, hidden_dim) for _ in layer_indices
        )
        self.norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
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
                reduction="sum",
            )
        return logits


class Llama3PPFullStage(nn.Module):
    """Single-stage PP chunk (``pp_size == 1``): embedding through LM head."""

    def __init__(self, cfg: Llama3DemoConfig, layer_indices: Sequence[int]) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            Llama3TransformerBlock(cfg, hidden_dim) for _ in layer_indices
        )
        self.norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
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
                reduction="sum",
            )
        return logits


def build_llama3_pp_chunk(cfg: Llama3DemoConfig, pp_rank: int, pp_size: int) -> nn.Module:
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
        return Llama3PPFullStage(cfg, layer_indices)
    if pp_rank == 0:
        return Llama3PPFirstStage(cfg, layer_indices)
    if pp_rank == pp_size - 1:
        return Llama3PPLastStage(cfg, layer_indices)
    return Llama3PPMiddleStage(cfg, layer_indices)


def load_pp_chunk_from_reference(
    stage_chunk: nn.Module,
    reference: Llama3Model,
    pp_rank: int,
    pp_size: int,
) -> None:
    """Copy weights from a full ``Llama3Model`` into the PP stage chunk for ``pp_rank``.

    Each local ``layers[i]`` is loaded from ``reference.layers[global_i]`` where
    ``global_i`` falls in the layer range for ``pp_rank``.  Embeddings, final norm,
    LM head, and ``freqs_cis`` are copied when present on the stage module.

    Args:
        stage_chunk: Stage module from :func:`build_llama3_pp_chunk`.
        reference: Full-model reference with matching ``cfg``.
        pp_rank: Pipeline rank of ``stage_chunk``.
        pp_size: Total pipeline width.
    """
    start, end = layer_range_for_pp_stage(pp_rank, reference.cfg.n_layers, pp_size)
    if hasattr(stage_chunk, "tok_embeddings"):
        stage_chunk.tok_embeddings.load_state_dict(reference.tok_embeddings.state_dict())
    for local_idx, global_idx in enumerate(range(start, end)):
        stage_chunk.layers[local_idx].load_state_dict(
            reference.layers[global_idx].state_dict()
        )
    if hasattr(stage_chunk, "norm"):
        stage_chunk.norm.load_state_dict(reference.norm.state_dict())
    if hasattr(stage_chunk, "output"):
        stage_chunk.output.load_state_dict(reference.output.state_dict())
    stage_chunk.freqs_cis.copy_(reference.freqs_cis)


class MicrobatchLossPipelineStage(PipelineStage):
    """``PipelineStage`` that forwards the micro-batch index to the submodule.

    Required when the last stage computes cross-entropy from pre-split targets
    (see :meth:`Llama3PPLastStage.set_micro_targets`).
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
