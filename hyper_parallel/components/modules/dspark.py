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
"""Trainable DeepSeek-V4.1 DSpark (``mtp.*``) speculative-draft stack.

DSpark is V4.1's replacement for the DeepSeek-V3 MTP module: a drafter of
``num_nextn_predict_layers`` Transformer stages that predicts
``dspark_block_size`` future tokens in one parallel pass, plus a lightweight
Markov head that models dependencies among the draft tokens and an FP32
confidence head that predicts per-position acceptance probabilities
(DeepSeek-V4.1-Flash technical report, section 2.4.3).

Training contract implemented here, following the report:

* DSpark is trained **without propagating gradients into the backbone** —
  every backbone hidden handed to this module must already be detached by the
  caller; ``main_proj`` consumes the concatenation of the
  ``dspark_target_layer_ids`` backbone hiddens.
* One forward computes base logits for all ``block_size`` draft offsets of
  every sequence position (teacher forcing). The Markov head adds a low-rank
  bias conditioned on the previous ground-truth token of each draft offset.
* The draft objective is token-weighted cross-entropy against the ``j+1``-step
  future token; the confidence objective is a BCE against greedy-hit events.
  The report does not disclose the production loss coefficients, so both are
  configuration values, mirroring the Indexer-KL precedent.

Documented v1 simplifications (structure is kept checkpoint-shaped; these are
runtime-semantic deltas to revisit):

* Draft attention lets every draft position attend the sliding
  ``dspark_window`` of backbone-projected KV entries only; intra-block
  draft-to-draft KV visibility is omitted and left to the Markov head, which
  matches the report's division of labor for draft-token dependencies.
* The published FP4/FP8 ``.scale`` companions are not materialized; the crop
  trains in BF16 like the rest of the validation model.
"""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension halves for rotary embedding."""
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _apply_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding on the trailing rope dimensions."""
    return tensor * cos + _rotate_half(tensor) * sin


class DSparkMarkovHead(nn.Module):
    """Low-rank 2-gram head biasing draft logits with the previous token."""

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        """Create the tied-rank embedding and output projections."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, markov_rank)
        self.head = nn.Linear(markov_rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the vocabulary bias and the rank-space embedding."""
        embedded = self.embed(token_ids)
        return self.head(embedded), embedded


class DSparkConfidenceHead(nn.Module):
    """FP32 acceptance-probability head over hidden and Markov features."""

    def __init__(self, input_dim: int) -> None:
        """Create the FP32 scalar projection."""
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Return per-position confidence logits in FP32."""
        joined = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(joined.float()).squeeze(-1)


class DSparkDraftAttention(nn.Module):
    """Sliding-window MQA over backbone-projected KV for draft positions.

    Queries come from the draft stream ``[batch, seq, block, dim]``; keys and
    values come from one shared low-rank projection of the stage input stream
    ``main_x`` (multi-query, matching the single ``wkv`` projection of the
    ``mtp.K.attn`` checkpoint layout), windowed causally over the last
    ``window`` backbone positions.  A learned per-head scalar sink joins the
    softmax normalization, matching V4.1 scalar-sink attention.
    """

    def __init__(self, config: Any) -> None:
        """Create the checkpoint-shaped projection set."""
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(round(self.head_dim * float(config.partial_rotary_factor)))
        self.window = int(config.v41_dspark_window)
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_groups = int(config.o_groups)
        self.o_lora_rank = int(config.o_lora_rank)
        if self.num_heads % self.o_groups:
            raise ValueError("num_attention_heads must divide into o_groups")
        self.scaling = self.head_dim ** -0.5
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        heads_per_group = self.num_heads // self.o_groups
        self.wo_a = nn.Linear(heads_per_group * self.head_dim, self.o_lora_rank * self.o_groups, bias=False)
        self.wo_b = nn.Linear(self.o_lora_rank * self.o_groups, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
            self,
            draft_states: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Attend draft queries over the windowed backbone KV stream.

        Args:
            draft_states: ``[batch, seq, block, hidden]`` collapsed draft input.
            main_states: ``[batch, seq, hidden]`` stage KV source stream.
            rope_cos: ``[batch, seq, rope_dim]`` cosine table for positions.
            rope_sin: ``[batch, seq, rope_dim]`` sine table for positions.

        Returns:
            ``[batch, seq, block, hidden]`` attention output.
        """
        batch, seq, block, _ = draft_states.shape
        query = self.wq_b(self.q_norm(self.wq_a(draft_states)))
        query = query.view(batch, seq, block, self.num_heads, self.head_dim)
        key_value = self.kv_norm(self.wkv(main_states))
        rope = self.rope_dim
        query_rot = _apply_rope(query[..., -rope:], rope_cos[:, :, None, None], rope_sin[:, :, None, None])
        query = torch.cat([query[..., :-rope], query_rot], dim=-1)
        key = torch.cat(
            [key_value[..., :-rope], _apply_rope(key_value[..., -rope:], rope_cos, rope_sin)],
            dim=-1,
        )
        # Windowed causal gather: position t sees backbone entries
        # [t-window+1, t].  Advanced indexing keeps the backward on the
        # mature gather/scatter kernels (``Tensor.unfold`` backward is
        # unreliable on the current CANN toolkit); clamped out-of-range
        # rows are masked below.
        window = self.window
        positions = torch.arange(seq, device=draft_states.device)
        offsets_back = torch.arange(window - 1, -1, -1, device=draft_states.device)
        window_index = positions.unsqueeze(1) - offsets_back.unsqueeze(0)
        valid = window_index >= 0
        window_index = window_index.clamp(min=0)
        gathered_key = key[:, window_index]
        gathered_value = key_value[:, window_index]
        scores = torch.einsum("bsjhd,bswd->bshjw", query, gathered_key.to(query.dtype)) * self.scaling
        scores = scores.masked_fill(~valid[None, :, None, None, :], torch.finfo(scores.dtype).min)
        sink = self.attn_sink.view(1, 1, self.num_heads, 1, 1).to(scores.dtype)
        joined = torch.cat([scores, sink.expand(batch, seq, -1, block, 1)], dim=-1)
        weights = joined.float().softmax(dim=-1).to(query.dtype)[..., :-1]
        context = torch.einsum("bshjw,bswd->bsjhd", weights, gathered_value.to(query.dtype))
        grouped = context.reshape(batch, seq, block, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        reduced = torch.einsum("bsjgd,grd->bsjgr", grouped, wo_a.to(context.dtype))
        return self.wo_b(reduced.flatten(3))


class DSparkStage(nn.Module):
    """One trainable DSpark Transformer stage with pipelined-mHC semantics."""

    def __init__(self, config: Any, mlp_factory: Any) -> None:
        """Create the stage from the shared crop configuration.

        Args:
            config: Crop configuration carrying ``v41_dspark_*`` fields.
            mlp_factory: Zero-argument callable building the stage MoE FFN.
        """
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.attn_hc = _DSparkHcCoefficients(config)
        self.ffn_hc = _DSparkHcCoefficients(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DSparkDraftAttention(config)
        self.mlp = mlp_factory()

    def _pre_ffn(
            self,
            hidden_streams: torch.Tensor,
            pre_mix: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the communication-free stage prefix up to the FFN input.

        Kept separate so activation checkpointing can recompute it without
        replaying the draft-MoE dispatch: collectives inside a recomputed
        region interleave with FSDP's backward collectives and deadlock
        across ranks.

        Args:
            hidden_streams: ``[batch, seq, block, streams, hidden]`` input.
            pre_mix: Previous-stage mHC pre-mix coefficients.
            main_states: Stage KV source stream.
            rope_cos: Rotary cosine table.
            rope_sin: Rotary sine table.

        Returns:
            Post-attention flat stream, FFN input, and the mHC pre/post/
            combine coefficients of the FFN half.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        flat = hidden_streams.reshape(batch, seq * block, streams, hidden)
        residual = flat
        attn_pre, attn_post, attn_comb = self.attn_hc(flat)
        collapsed = (pre_mix.unsqueeze(-1) * flat.float()).sum(dim=2).to(flat.dtype)
        attn_input = self.input_layernorm(collapsed).view(batch, seq, block, hidden)
        attn_output = self.self_attn(attn_input, main_states, rope_cos, rope_sin)
        flat = pipelined_mhc_post(attn_output.reshape(batch, seq * block, hidden), residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(flat)
        collapsed = (attn_pre.unsqueeze(-1) * flat.float()).sum(dim=2).to(flat.dtype)
        ffn_input = self.post_attention_layernorm(collapsed)
        return flat, ffn_input, ffn_pre, ffn_post, ffn_comb

    def forward(
            self,
            hidden_streams: torch.Tensor,
            pre_mix: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            input_ids: torch.Tensor,
            use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one DSpark stage over the flattened draft stream.

        Args:
            hidden_streams: ``[batch, seq, block, streams, hidden]`` input.
            pre_mix: Previous-stage mHC pre-mix coefficients.
            main_states: Stage KV source stream.
            rope_cos: Rotary cosine table.
            rope_sin: Rotary sine table.
            input_ids: Flattened draft token ids for the MoE router.
            use_checkpoint: Recompute the communication-free prefix in
                backward instead of storing its activations.

        Returns:
            Updated hidden streams and the FFN-half pre-mix coefficients.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        if use_checkpoint:
            flat, ffn_input, ffn_pre, ffn_post, ffn_comb = torch.utils.checkpoint.checkpoint(
                self._pre_ffn, hidden_streams, pre_mix, main_states,
                rope_cos, rope_sin, use_reentrant=False)
        else:
            flat, ffn_input, ffn_pre, ffn_post, ffn_comb = self._pre_ffn(
                hidden_streams, pre_mix, main_states, rope_cos, rope_sin)
        ffn_output = self.mlp(ffn_input, input_ids=input_ids)
        flat = pipelined_mhc_post(ffn_output, flat, ffn_post, ffn_comb)
        return flat.view(batch, seq, block, streams, hidden), ffn_pre


class _DSparkHcCoefficients(nn.Module):
    """Parameter-owning pipelined-mHC coefficient module for DSpark stages."""

    def __init__(self, config: Any) -> None:
        """Create the ``fn/base/scale`` coefficient layout."""
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(config.hc_eps)
        mix = (self.hc_mult + 2) * self.hc_mult
        self.input_norm = nn.RMSNorm(self.hc_mult * config.hidden_size, eps=config.rms_norm_eps)
        self.fn = nn.Parameter(
            torch.randn(mix, self.hc_mult * config.hidden_size) * float(config.initializer_range)
        )
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce pre/post/combine coefficients for the next sublayers."""
        from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps  # pylint: disable=C0415

        flattened = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = functional.linear(flattened, self.fn.float())  # pylint: disable=not-callable
        streams = self.hc_mult
        pre, post, comb = mix.split([streams, streams, streams * streams], dim=-1)
        pre_bias, post_bias, comb_bias = self.base.split([streams, streams, streams * streams])
        pre_scale, post_scale, comb_scale = self.scale[0], self.scale[1], self.scale[2]
        pre = torch.sigmoid(pre * pre_scale + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post * post_scale + post_bias)
        comb = comb.view(*comb.shape[:-1], streams, streams) * comb_scale + comb_bias.view(streams, streams)
        comb = sinkhorn_knopps(comb, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, comb


class DeepseekV41DSpark(nn.Module):
    """Trainable DSpark drafter head attached behind the V4.1 backbone."""

    def __init__(self, config: Any, mlp_factory: Any) -> None:
        """Create ``num_stages`` DSpark stages plus entry/exit components."""
        super().__init__()
        self.block_size = int(config.v41_dspark_block_size)
        self.noise_token_id = int(config.v41_dspark_noise_token_id)
        self.num_targets = len(tuple(config.v41_dspark_target_layer_ids))
        self.hc_mult = int(config.hc_mult)
        self.rope_theta = float(config.rope_theta)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(round(self.head_dim * float(config.partial_rotary_factor)))
        self.confidence_coeff = float(config.v41_dspark_confidence_coeff)
        # Sequence-chunk granularity of the draft objective: smaller chunks
        # lower the vocabulary-logits peak at the cost of more recompute
        # rounds; the loss value itself is chunking-invariant.
        self.loss_chunk_size = int(getattr(config, "v41_dspark_loss_chunk_size", 64) or 64)
        if self.block_size <= 0 or self.num_targets == 0:
            raise ValueError("DSpark requires a positive block size and target layers")
        hidden = int(config.hidden_size)
        self.main_proj = nn.Linear(hidden * self.num_targets, hidden, bias=False)
        self.main_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        depth = int(config.v41_dspark_depth)
        self.stages = nn.ModuleList(DSparkStage(config, mlp_factory) for _ in range(depth))
        # v1 deviation (recorded): the released checkpoint feeds the noise
        # placeholder through the tied backbone embedding row; a drafter-owned
        # vector is training-equivalent and avoids sharded-embedding lookups
        # from the frozen tied table.
        # An all-zero noise vector puts zero vectors through the stage
        # RMSNorms, whose backward scales like 1/sqrt(eps) and compounds
        # across stages into overflow-level gradients.
        self.noise_embedding = nn.Parameter(
            torch.randn(hidden) * float(config.initializer_range))
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(int(config.vocab_size), int(config.v41_dspark_markov_rank))
        self.confidence_head = DSparkConfidenceHead(hidden + int(config.v41_dspark_markov_rank))

    def _rope_tables(self, seq: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cosine/sine tables over backbone positions."""
        half = self.rope_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        angles = torch.arange(seq, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([angles, angles], dim=-1)
        return emb.cos()[None].to(dtype), emb.sin()[None].to(dtype)

    def init_weights(self, std: float) -> None:
        """Deterministically initialize drafter-owned bare parameters.

        Meta-device construction discards __init__-time values, and these
        parameters are replicated across data-parallel ranks, so every
        rank must draw identical values; fixed-seed CPU generators
        provide that.

        Args:
            std: Backbone initializer range for the random draws.
        """
        def _local(parameter: torch.Tensor) -> torch.Tensor:
            return parameter.to_local() if hasattr(parameter, "to_local") else parameter

        def _fill(parameter: torch.Tensor, seed: int) -> None:
            target = _local(parameter)
            # A replicated local equals the full shape and must draw the
            # same values on every rank; a dp-sharded local holds a
            # distinct slice, so an independent per-rank stream is valid.
            if tuple(target.shape) != tuple(parameter.shape):
                rank = (torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0)
                seed = seed + 7919 * (rank + 1)
            generator = torch.Generator().manual_seed(seed)
            values = torch.randn(target.shape, generator=generator) * std
            with torch.no_grad():
                target.copy_(values.to(target.device, target.dtype))

        _fill(self.noise_embedding, 0x44530001)
        for index, stage in enumerate(self.stages):
            _fill(stage.attn_hc.fn, 0x44530100 + index)
            _fill(stage.ffn_hc.fn, 0x44530200 + index)
            with torch.no_grad():
                _local(stage.attn_hc.base).zero_()
                _local(stage.ffn_hc.base).zero_()
                _local(stage.attn_hc.scale).fill_(1.0)
                _local(stage.ffn_hc.scale).fill_(1.0)
                _local(stage.self_attn.attn_sink).zero_()

    def forward(
            self,
            target_hiddens: list[torch.Tensor],
            input_ids: torch.Tensor,
            labels: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            lm_head: nn.Linear,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DSpark draft objective for one batch.

        Args:
            target_hiddens: Detached collapsed backbone hiddens, one per
                configured target layer, each ``[batch, seq, hidden]``.
            input_ids: ``[batch, seq]`` backbone input tokens.
            labels: Optional ``[batch, seq]`` supervised labels with
                ``IGNORE_INDEX`` holes; ``None`` supervises every position
                from ``input_ids`` (full LM-draft objective).
            embed_tokens: Backbone embedding tied into the drafter.
            lm_head: Backbone output head tied into the drafter.

        Returns:
            The scalar DSpark loss and a metrics dictionary.
        """
        if any(hidden.requires_grad for hidden in target_hiddens):
            raise ValueError("DSpark target hiddens must be detached from the backbone")
        batch, seq = input_ids.shape
        block = self.block_size
        device = input_ids.device
        main_states = self.main_norm(self.main_proj(torch.cat(target_hiddens, dim=-1)))

        # Tied-but-frozen backbone surfaces: the drafter reuses the backbone
        # token embeddings (already computed by the backbone forward, handed
        # over detached) and the frozen output head; the report forbids DSpark
        # gradients from reaching the backbone.
        if inputs_embeds.requires_grad:
            raise ValueError("DSpark inputs_embeds must be detached from the backbone")
        token_embeddings = inputs_embeds
        noise = self.noise_embedding.to(token_embeddings.dtype)
        draft = torch.empty(batch, seq, block, token_embeddings.shape[-1],
                            dtype=token_embeddings.dtype, device=device)
        draft[:, :, 0] = token_embeddings
        draft[:, :, 1:] = noise
        streams = draft.unsqueeze(3).expand(-1, -1, -1, self.hc_mult, -1).contiguous()
        pre_mix = streams.new_zeros(batch, seq * block, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        rope_cos, rope_sin = self._rope_tables(seq, device, token_embeddings.dtype)
        draft_ids = input_ids.unsqueeze(-1).expand(-1, -1, block).reshape(batch, seq * block)
        # The flattened draft stream (seq x block positions, hc streams) makes
        # stage activations the dominant memory term; each stage recomputes
        # its communication-free prefix in backward while the draft-MoE
        # dispatch stays outside the recomputed region (DSparkStage._pre_ffn).
        gradient_active = torch.is_grad_enabled() and self.training
        for stage in self.stages:
            streams, pre_mix = stage(streams, pre_mix, main_states, rope_cos,
                                     rope_sin, draft_ids, use_checkpoint=gradient_active)
        def _collapse(streams_in: torch.Tensor, pre_mix_in: torch.Tensor) -> torch.Tensor:
            merged = (pre_mix_in.unsqueeze(-1)
                      * streams_in.reshape(batch, seq * block, self.hc_mult, -1).float())
            merged = merged.sum(dim=2).to(streams_in.dtype).view(batch, seq, block, -1)
            return self.norm(merged)

        # The fp32 merge buffer otherwise stays resident for the norm
        # backward; recomputing it is cheap and communication-free.
        if gradient_active:
            hidden_out = torch.utils.checkpoint.checkpoint(
                _collapse, streams, pre_mix, use_reentrant=False)
        else:
            hidden_out = _collapse(streams, pre_mix)

        # Teacher-forced targets: draft offset j at position t predicts the
        # token at t + 1 + j; the Markov head is conditioned on t + j.  When
        # the trainer keeps labels on the loss-only path (the model never sees
        # them), the drafter supervises every next token from input_ids — the
        # standard LM-draft objective; labels only add the SFT ignore mask.
        pad_ids = functional.pad(input_ids, (0, block), value=self.noise_token_id)
        supervision = input_ids if labels is None else labels
        pad_labels = functional.pad(supervision, (0, block), value=IGNORE_INDEX)
        offsets = torch.arange(block, device=device)
        gather = torch.arange(seq, device=device)[:, None] + offsets[None, :]
        markov_inputs = pad_ids.gather(1, gather.reshape(1, -1).expand(batch, -1)).view(batch, seq, block)
        targets = pad_labels.gather(1, (gather + 1).reshape(1, -1).expand(batch, -1)).view(batch, seq, block)

        # Sequence-chunked objective: the [batch, seq, block, vocab] draft
        # logits are never materialized whole.  Each chunk recomputes its
        # logits inside activation checkpointing, bounding the vocabulary
        # spike to one chunk (same technique as the repository chunk loss).
        # The tied head may live as a sharded distributed tensor once it owns
        # an FSDP unit; gather one frozen full copy for the draft projection.
        raw_head_weight = lm_head.weight
        if hasattr(raw_head_weight, "full_tensor"):
            raw_head_weight = raw_head_weight.full_tensor()
        head_weight = raw_head_weight.detach()
        chunk = self.loss_chunk_size

        def _chunk_objective(hidden_chunk, markov_chunk, target_chunk):
            markov_bias, markov_embed = self.markov_head(markov_chunk)
            logits = functional.linear(hidden_chunk, head_weight) + markov_bias  # pylint: disable=not-callable
            chunk_mask = target_chunk != IGNORE_INDEX
            ce_sum = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target_chunk.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            with torch.no_grad():
                hit_sum = ((logits.argmax(dim=-1) == target_chunk) & chunk_mask).sum()
            confidence = self.confidence_head(hidden_chunk, markov_embed)
            if bool(chunk_mask.any()):
                with torch.no_grad():
                    chunk_hits = (logits.argmax(dim=-1) == target_chunk) & chunk_mask
                bce_sum = functional.binary_cross_entropy_with_logits(
                    confidence[chunk_mask], chunk_hits[chunk_mask].float(), reduction="sum"
                )
            else:
                bce_sum = confidence.sum() * 0.0
            return ce_sum, bce_sum, hit_sum.float()

        ce_total = hidden_out.new_zeros((), dtype=torch.float32)
        bce_total = hidden_out.new_zeros((), dtype=torch.float32)
        hit_total = hidden_out.new_zeros((), dtype=torch.float32)
        for start in range(0, seq, chunk):
            stop = min(start + chunk, seq)
            ce_sum, bce_sum, hit_sum = torch.utils.checkpoint.checkpoint(
                _chunk_objective,
                hidden_out[:, start:stop],
                markov_inputs[:, start:stop],
                targets[:, start:stop],
                use_reentrant=False,
            )
            ce_total = ce_total + ce_sum
            bce_total = bce_total + bce_sum
            hit_total = hit_total + hit_sum
        supervised = (targets != IGNORE_INDEX).sum()
        denominator = supervised.clamp(min=1).float()
        draft_ce = ce_total / denominator
        confidence_bce = bce_total / denominator
        loss = draft_ce + self.confidence_coeff * confidence_bce
        accuracy = (hit_total / denominator).detach()
        metrics = {
            "dspark_ce": draft_ce.detach(),
            "dspark_confidence_bce": confidence_bce.detach(),
            "dspark_draft_accuracy": accuracy,
            "dspark_supervised_tokens": supervised.detach(),
        }
        return loss, metrics


__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftAttention",
    "DSparkMarkovHead",
    "DSparkStage",
    "DeepseekV41DSpark",
]
