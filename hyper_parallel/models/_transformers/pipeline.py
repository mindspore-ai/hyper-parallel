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
"""Minimal AutoPipeline integration for the HyperParallel Trainer."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from hyper_parallel.core.pipeline_parallel import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)


@dataclass
class PipelineConfig:
    """Pipeline options; ``split[chunk][pp_rank]`` gives VPP layer counts.

    An empty split balances layers across physical PP ranks. A flat split
    specifies one layer count per rank; a nested split specifies one row per
    virtual chunk. Batch size is derived from Trainer's global batch and DP.
    """

    enabled: bool = False
    schedule: str = "1f1b"
    pp_microbatch_size: int = 1
    split: list[int] | list[list[int]] = field(default_factory=list)
    p2p_overlap: bool = True
    p2p_multi_stream: bool = False
    dynamic_shape: bool = False
    loss_fn: Callable | None = None
    # The schedule treats all tensor args/kwargs as batch-major by default.
    # These fields allow model families with non-standard layouts to override
    # that contract without changing the public AutoPipeline API.
    args_batch_dim: int | None = 0
    kwargs_batch_dim: dict[str, int | None] | None = None

    def __post_init__(self) -> None:
        """Validate schedule and split structure before model construction."""
        if type(self.pp_microbatch_size) is not int or self.pp_microbatch_size <= 0:
            raise ValueError("pipeline.pp_microbatch_size must be a positive integer")
        if self.schedule.lower() not in {"1f1b", "gpipe"}:
            raise ValueError("pipeline.schedule must be 1f1b or gpipe")
        if not isinstance(self.split, list):
            raise ValueError("pipeline.split must be a list of layer counts or a list of lists")
        if self.split:
            nested = isinstance(self.split[0], list)
            rows = self.split if nested else [self.split]
            if any(not isinstance(row, list) or not row for row in rows):
                raise ValueError("pipeline.split rows must be non-empty lists of layer counts")
            if any(type(count) is not int or count <= 0 for row in rows for count in row):
                raise ValueError("pipeline.split layer counts must be positive integers")
            if nested and self.schedule.lower() != "1f1b":
                raise ValueError("VPP (nested pipeline.split) requires schedule: 1f1b")
        if self.p2p_multi_stream and not self.p2p_overlap:
            raise ValueError("pipeline.p2p_multi_stream requires pipeline.p2p_overlap=true")


@dataclass
class PipelineInfo:
    """Runtime objects owned by one PP rank."""

    schedule: Any
    stages: list[PipelineStage]
    model_parts: list[nn.Module]
    has_first_stage: bool
    has_last_stage: bool


class PipelineModel(nn.Module):
    """Container exposing all local stage parameters to Trainer components."""

    def __init__(self, parts: list[nn.Module], config: Any) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.pipeline_config = config
        self.config = config
        # Cross-stage tied weights are not represented as a single local
        # parameter. Keep the Transformers loading/finalization contract
        # explicit instead of letting it infer aliases from ``parts``.
        self.all_tied_weights_keys: dict[str, str] = {}

    def initialize_weights(self) -> None:
        """Initialize each stage through its native HF initialization hook."""
        for part in self.parts:
            initialize = getattr(part, "initialize_weights", None)
            if callable(initialize):
                initialize()
                continue
            init_weights = getattr(part, "init_weights", None)
            if callable(init_weights):
                init_weights()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Reject accidental non-scheduled model execution."""
        raise RuntimeError("pipeline models must be executed through AutoPipeline.step()")


def _find_decoder(model: nn.Module) -> tuple[nn.Module, str]:
    """Return the decoder module and its path prefix for HF causal models."""
    decoder = getattr(model, "model", model)
    prefix = "model" if decoder is not model else ""
    if hasattr(decoder, "language_model"):
        decoder = decoder.language_model
        prefix = f"{prefix}.language_model" if prefix else "language_model"
    if not hasattr(decoder, "layers"):
        raise ValueError("PP requires a model with a decoder 'layers' container")
    return decoder, prefix


def _make_causal_mask(module: nn.Module, hidden_states: torch.Tensor, attention_mask: Any) -> torch.Tensor:
    """Build a compact causal mask without requiring a top-level model forward."""
    if isinstance(attention_mask, dict):
        layer_type = getattr(module.config, "layer_types", ["full_attention"])[0]
        return attention_mask.get(layer_type, attention_mask.get("full_attention"))
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask
    seq_len = hidden_states.shape[1]
    mask = torch.full(
        (1, 1, seq_len, seq_len),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    if attention_mask is not None:
        valid = attention_mask.to(dtype=torch.bool).view(attention_mask.shape[0], 1, 1, seq_len)
        mask = mask.masked_fill(~valid, torch.finfo(hidden_states.dtype).min)
    return mask


def _patch_stage_forward(stage_model: nn.Module, stage_index: int, stage_count: int, loss_fn: Callable | None) -> None:
    """Install a stage-aware forward on one copied HF model."""
    decoder = getattr(stage_model, "model", stage_model)
    if hasattr(decoder, "language_model"):
        decoder = decoder.language_model
    original_decoder = decoder
    original_lm_head = getattr(stage_model, "lm_head", None)

    def decoder_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        inputs_embeds=None, cache_position=None, **kwargs):
        del kwargs
        if inputs_embeds is None:
            if getattr(self, "embed_tokens", None) is not None:
                inputs_embeds = self.embed_tokens(input_ids)
            elif isinstance(input_ids, torch.Tensor) and input_ids.is_floating_point():
                inputs_embeds = input_ids
            else:
                raise ValueError("non-first PP stage requires hidden states as input")
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        mask = _make_causal_mask(self, inputs_embeds, attention_mask)
        rotary_emb = getattr(self, "rotary_emb", None)
        position_embeddings = rotary_emb(inputs_embeds, position_ids) if rotary_emb is not None else None
        hidden_states = inputs_embeds
        layers = self.layers.values() if hasattr(self.layers, "values") else self.layers
        for layer in layers:
            layer_type = getattr(layer, "self_attn", None)
            layer_type = getattr(layer_type, "layer_type", "full_attention")
            layer_mask = mask
            if isinstance(attention_mask, dict):
                layer_mask = attention_mask.get(layer_type, attention_mask.get("full_attention"))
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        if getattr(self, "norm", None) is not None:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def model_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      inputs_embeds=None, labels=None, **kwargs):
        del kwargs
        hidden_states = original_decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if stage_index != stage_count - 1 or original_lm_head is None:
            return hidden_states
        logits = original_lm_head(hidden_states)
        if labels is None:
            return logits
        if loss_fn is not None:
            return loss_fn(logits, labels)
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    original_decoder.forward = types.MethodType(decoder_forward, original_decoder)
    stage_model.forward = types.MethodType(model_forward, stage_model)


def _stage_assignments(model: nn.Module, pp_size: int, config: PipelineConfig) -> list[list[str]]:
    """Generate original-FQN assignments for a decoder-only model."""
    decoder, prefix = _find_decoder(model)
    layer_count = len(decoder.layers)
    if config.split:
        rows = config.split if isinstance(config.split[0], list) else [config.split]
        if any(len(row) != pp_size for row in rows):
            raise ValueError(f"each pipeline.split row must contain pp_size={pp_size} layer counts")
        counts = [count for row in rows for count in row]
    else:
        base, remainder = divmod(layer_count, pp_size)
        counts = [base + (index < remainder) for index in range(pp_size)]
    if sum(counts) != layer_count or any(count <= 0 for count in counts):
        raise ValueError(f"pipeline.split must assign all {layer_count} layers with at least one layer per stage")
    cuts = [0]
    for count in counts:
        cuts.append(cuts[-1] + count)
    layer_prefix = f"{prefix}.layers" if prefix else "layers"
    decoder_prefix = f"{prefix}." if prefix else ""
    assignments = []
    for stage_index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        names = [f"{layer_prefix}.{idx}" for idx in range(start, end)]
        if stage_index == 0:
            names.append(f"{decoder_prefix}embed_tokens")
        # Rotary embeddings are needed by every stage because each stage
        # reconstructs position ids and attention masks locally after the
        # activation boundary. They contain no trainable layer weights.
        names.append(f"{decoder_prefix}rotary_emb")
        if stage_index == len(cuts) - 2:
            names.extend([f"{decoder_prefix}norm", "lm_head"])
        assignments.append(names)
    return assignments


def _prune_model(model: nn.Module, keep: list[str]) -> nn.Module:
    """Deep-copy and prune a model while retaining decoder layer keys."""
    stage_model = copy.deepcopy(model)
    keep_set = set(keep)
    decoder, prefix = _find_decoder(stage_model)
    layer_prefix = f"{prefix}.layers." if prefix else "layers."
    layer_ids = {name.rsplit(".", 1)[-1] for name in keep if name.startswith(layer_prefix)}
    if isinstance(decoder.layers, nn.ModuleList):
        decoder.layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(decoder.layers) if str(i) in layer_ids})
    for name in ("embed_tokens", "rotary_emb", "norm"):
        decoder_name = f"{prefix}.{name}" if prefix else name
        if decoder_name not in keep_set:
            setattr(decoder, name, None)
    if "lm_head" not in keep_set and hasattr(stage_model, "lm_head"):
        stage_model.lm_head = None
    return stage_model


class AutoPipeline(nn.Module):
    """High-level PP wrapper used by HyperAutoModel and Trainer."""

    def __init__(self, world_mesh: Any, config: PipelineConfig, loss_fn: Callable | None = None) -> None:
        super().__init__()
        self.world_mesh = world_mesh
        mesh_names = getattr(world_mesh, "mesh_dim_names", None)
        if mesh_names is not None:
            if "pp" not in mesh_names:
                self.pp_mesh = world_mesh
            elif not hasattr(world_mesh, "__getitem__"):
                self.pp_mesh = world_mesh
            else:
                self.pp_mesh = world_mesh["pp"]
        else:
            try:
                self.pp_mesh = world_mesh["pp"]
            except (KeyError, IndexError, TypeError):
                # A one-dimensional PP mesh is also accepted directly.
                self.pp_mesh = world_mesh
        self.pipeline_config = config
        # ``config`` is populated with the Hugging Face model config after
        # build(), preserving the PreTrainedModel-facing Trainer contract.
        self.config = None
        self.loss_fn = loss_fn or config.loss_fn
        self.info: PipelineInfo | None = None
        self.parts = nn.ModuleList()
        self.model_config = None
        self.checkpoint_fqn_aliases: dict[str, str] = {}
        self.all_tied_weights_keys: dict[str, str] = {}

    def initialize_weights(self) -> None:
        """Initialize all locally owned stage modules via HF hooks."""
        for part in self.parts:
            initialize = getattr(part, "initialize_weights", None)
            if callable(initialize):
                initialize()
                continue
            init_weights = getattr(part, "init_weights", None)
            if callable(init_weights):
                init_weights()

    def build(self, model: nn.Module, *, loss_fn: Callable | None = None,
              parallelize_fn: Callable[[nn.Module], nn.Module] | None = None,
              batch_size: int | None = None) -> "AutoPipeline":
        """Split and parallelize a model for one DP-local optimizer-step batch.

        Args:
            model: Complete model to split.
            loss_fn: Optional last-stage loss function.
            parallelize_fn: Stage-local TP/CP/EP/FSDP transformation.
            batch_size: DP-local batch size, derived by Trainer. Defaults to
                one pipeline micro-batch for standalone callers.
        """
        if not self.pipeline_config.enabled:
            raise ValueError("AutoPipeline.build requires pipeline.enabled=true")
        batch_size = self.pipeline_config.pp_microbatch_size if batch_size is None else batch_size
        if (type(batch_size) is not int or batch_size <= 0
                or batch_size % self.pipeline_config.pp_microbatch_size != 0):
            raise ValueError("DP-local batch size must be positive and divisible by pipeline.pp_microbatch_size")
        self.loss_fn = loss_fn or self.loss_fn
        assignments = _stage_assignments(model, self.pp_mesh.size(), self.pipeline_config)
        pp_rank = self.pp_mesh.get_local_rank()
        local_indices = list(range(pp_rank, len(assignments), self.pp_mesh.size()))
        parts = []
        stages = []
        for stage_index in local_indices:
            part = _prune_model(model, assignments[stage_index])
            _patch_stage_forward(part, stage_index, len(assignments), self.loss_fn)
            if parallelize_fn is not None:
                part = parallelize_fn(part)
            parts.append(part)
            stages.append(PipelineStage(
                part,
                stage_index=stage_index,
                stage_num=len(assignments),
                mesh=self.pp_mesh,
                device=next(part.parameters()).device,
                dyn_shape=self.pipeline_config.dynamic_shape,
            ))
        self.parts = nn.ModuleList(parts)
        # Checkpoint loaders address the complete model by its original FQN,
        # while ``ModuleList`` adds ``parts.<local_index>`` to stage-local
        # names. Keep an explicit alias table so pretrained loading can still
        # resolve parameters without renaming the public stage FQNs.
        self.checkpoint_fqn_aliases = {}
        for local_index, _stage_index in enumerate(local_indices):
            for local_name, _ in parts[local_index].named_parameters(remove_duplicate=False):
                original_name = local_name
                self.checkpoint_fqn_aliases[original_name] = (
                    f"parts.{local_index}.{local_name}"
                )
            for local_name, _ in parts[local_index].named_buffers(remove_duplicate=False):
                self.checkpoint_fqn_aliases.setdefault(
                    local_name, f"parts.{local_index}.{local_name}"
                )
        self.model_config = model.config
        self.config = model.config
        micro_batches = batch_size // self.pipeline_config.pp_microbatch_size
        if len(parts) > 1:
            schedule_cls = ScheduleInterleaved1F1B
        else:
            schedule_cls = ScheduleGPipe if self.pipeline_config.schedule.lower() == "gpipe" else Schedule1F1B
        kwargs_batch_dim = self.pipeline_config.kwargs_batch_dim
        if kwargs_batch_dim is not None:
            kwargs_batch_dim = {
                name: (0 if dim is None else dim)
                for name, dim in kwargs_batch_dim.items()
            }
        schedule = schedule_cls(
            stages,
            micro_batches,
            args_batch_dim=self.pipeline_config.args_batch_dim,
            kwargs_batch_dim=kwargs_batch_dim,
            overlap_p2p=self.pipeline_config.p2p_overlap,
            p2p_transport="multi_stream" if self.pipeline_config.p2p_multi_stream else "auto",
        )
        self.info = PipelineInfo(
            schedule=schedule,
            stages=stages,
            model_parts=parts,
            has_first_stage=any(stage.is_first_stage for stage in stages),
            has_last_stage=any(stage.is_last_stage for stage in stages),
        )
        return self

    def step(self, model_input=None, *, target=None, losses=None, **kwargs):
        """Run one complete scheduled PP step."""
        if self.info is None:
            raise RuntimeError("AutoPipeline.build() must be called before step()")
        args = (model_input,) if self.info.has_first_stage else ()
        # ``PipelineScheduleRuntime.run`` owns the loss list.  Passing a caller
        # supplied list as a model kwarg makes every stage receive an unknown
        # argument and does not populate that list.  Keep ``losses`` as a
        # backwards-compatible ignored parameter and return the schedule list.
        del losses
        kwargs.pop("labels", None)
        if target is not None:
            kwargs["labels"] = target
        return self.info.schedule.run(*args, **kwargs)
