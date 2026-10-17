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
"""Model-specific HF Llama pipeline splitting for the Dry-run example."""
# The example targets the Torch-only HyperModels dry-run runner.
# pylint: disable=forbidden-backend-import

import copy
from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch import nn
from torch._subclasses.fake_tensor import unset_fake_temporarily

from hyper_parallel.trainer.dry_run_pipeline import (
    DryRunBoundaryLeaf,
    DryRunPipelineChunk,
)


def layer_range_for_pp_stage(
        stage_index: int,
        num_hidden_layers: int,
        stage_num: int,
        layer_split: Optional[list[int]] = None,
) -> tuple[int, int]:
    """Return the evenly balanced global decoder-layer range for one stage.

    Args:
        stage_index: Global pipeline-stage index in ``[0, stage_num)``.
        num_hidden_layers: Total number of decoder layers.
        stage_num: Number of global pipeline stages.
        layer_split: Optional decoder-layer count for every global stage.

    Returns:
        Half-open ``[start, end)`` layer interval.

    Raises:
        ValueError: If the stage index or explicit layer split is invalid.
    """
    if stage_num < 2:
        raise ValueError(f"dry-run PP stage builder requires stage_num >= 2, got {stage_num}")
    if not 0 <= stage_index < stage_num:
        raise ValueError(f"stage_index must be in [0, {stage_num}), got {stage_index}")
    if layer_split is not None:
        counts = [int(count) for count in layer_split]
        if len(counts) != stage_num or sum(counts) != num_hidden_layers or min(counts) < 0:
            raise ValueError(
                "pp_layer_split must contain one non-negative count per global stage and sum "
                f"to num_hidden_layers={num_hidden_layers}, got {counts}"
            )
        return sum(counts[:stage_index]), sum(counts[:stage_index + 1])
    base, remainder = divmod(num_hidden_layers, stage_num)
    first_larger_stage = stage_num - remainder
    start = stage_index * base + max(stage_index - first_larger_stage, 0)
    end = start + base + int(stage_index >= first_larger_stage)
    return start, end


class _PassThroughDecoderLayer(nn.Module):
    """Parameter-free placeholder preserving a non-local global layer FQN."""

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return hidden states unchanged while accepting decoder-layer kwargs."""
        del kwargs
        return hidden_states


class _HFLlamaDryRunStage(nn.Module):
    """HF Llama stage adapter with the standard pipeline tensor contract."""

    def __init__(
            self,
            full_model: nn.Module,
            loss_fn: nn.Module,
            *,
            is_first: bool,
            is_last: bool,
            num_micro_batches: int,
    ) -> None:
        """Keep only the components needed by one first, middle, or last stage."""
        super().__init__()
        self.model = full_model.model
        self.config = full_model.config
        self.is_first = is_first
        self.is_last = is_last
        self.num_micro_batches = num_micro_batches
        self._micro_index = 0
        self._micro_labels: list[torch.Tensor] = []
        object.__setattr__(self, "_loss_fn", loss_fn)
        self._causal_lm_loss = full_model.loss_function
        if is_last:
            self.lm_head = full_model.lm_head

    def set_micro_index(self, micro_index: int) -> None:
        """Select labels associated with the scheduler's current micro-batch."""
        if not 0 <= micro_index < self.num_micro_batches:
            raise ValueError(
                f"micro_index must be in [0, {self.num_micro_batches}), got {micro_index}"
            )
        self._micro_index = micro_index

    def set_micro_labels(self, labels: list[torch.Tensor]) -> None:
        """Store last-stage labels in scheduler micro-batch order."""
        if not self.is_last:
            raise ValueError("Only the last dry-run PP stage accepts labels")
        if len(labels) != self.num_micro_batches:
            raise ValueError(
                f"Expected {self.num_micro_batches} label chunks, got {len(labels)}"
            )
        self._micro_labels = labels

    def forward(
            self,
            stage_input: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            shift_labels: Optional[torch.Tensor] = None,
            **model_inputs: Any,
    ) -> torch.Tensor:
        """Run the local Llama layers and optionally produce causal-LM loss."""
        del labels, shift_labels
        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            **model_inputs,
        }
        if self.is_first:
            model_kwargs["input_ids"] = stage_input
        else:
            model_kwargs["inputs_embeds"] = stage_input
        outputs = self.model(**model_kwargs)
        hidden_states = outputs.last_hidden_state
        if not self.is_last:
            return hidden_states
        if not self._micro_labels:
            raise ValueError("Last dry-run PP stage labels were not configured")
        labels = self._micro_labels[self._micro_index]
        logits = self.lm_head(hidden_states)
        model_loss = self._causal_lm_loss(
            logits=logits,
            labels=labels,
            vocab_size=int(self.config.vocab_size),
        )
        loss_value = self._loss_fn(
            model_output=SimpleNamespace(loss=model_loss, logits=logits),
            labels=labels,
        )
        if isinstance(loss_value, dict):
            return torch.stack(list(loss_value.values())).sum()
        return loss_value


def _build_hf_llama_chunk(
        full_model: nn.Module,
        loss_fn: nn.Module,
        stage_index: int,
        stage_num: int,
        num_micro_batches: int,
        layer_split: Optional[list[int]],
        boundary_batch_size: int,
        sequence_length: int,
        boundary_dtype: torch.dtype,
) -> DryRunPipelineChunk:
    """Build one independent global HF Llama pipeline chunk."""
    # The DryRun invokes model-specific builders inside FakeTensorMode, while
    # the source model still owns meta tensors. Copy outside dispatch so the
    # meta storages remain meta instead of being wrapped as CPU FakeTensors.
    with unset_fake_temporarily():
        stage_model = copy.deepcopy(full_model)
    backbone = stage_model.model
    layers = backbone.layers
    num_hidden_layers = int(stage_model.config.num_hidden_layers)
    start, end = layer_range_for_pp_stage(
        stage_index,
        num_hidden_layers,
        stage_num,
        layer_split,
    )
    for layer_index in range(num_hidden_layers):
        if not start <= layer_index < end:
            layers[layer_index] = _PassThroughDecoderLayer()
    is_first = stage_index == 0
    is_last = stage_index == stage_num - 1
    if not is_first:
        backbone.embed_tokens = nn.Identity()
    if not is_last:
        backbone.norm = nn.Identity()
    module = _HFLlamaDryRunStage(
        stage_model,
        loss_fn,
        is_first=is_first,
        is_last=is_last,
        num_micro_batches=num_micro_batches,
    )
    boundary_shape = (
        boundary_batch_size,
        sequence_length,
        int(stage_model.config.hidden_size),
    )
    input_boundary = ()
    if not is_first:
        input_boundary = (
            DryRunBoundaryLeaf(
                boundary_shape,
                boundary_dtype,
                True,
                anchor_fqn=(
                    f"model.layers.{start}.input_layernorm" if start < end else None
                ),
                tensor_name="hidden_states",
            ),
        )
    output_boundary = ()
    if not is_last:
        output_boundary = (
            DryRunBoundaryLeaf(
                boundary_shape,
                boundary_dtype,
                True,
                anchor_fqn=f"model.layers.{end - 1}.mlp" if start < end else None,
                tensor_name="output",
            ),
        )
    fsdp_units = tuple((layers[layer_index],) for layer_index in range(start, end))
    sibling_modules = []
    if is_first:
        sibling_modules.append(backbone.embed_tokens)
    if is_last:
        sibling_modules.extend((backbone.norm, module.lm_head))
    if sibling_modules:
        fsdp_units += (tuple(sibling_modules),)
    return DryRunPipelineChunk(
        module=module,
        layer_start=start,
        layer_end=end,
        hidden_size=int(stage_model.config.hidden_size),
        stage_index=stage_index,
        input_boundary=input_boundary,
        output_boundary=output_boundary,
        fsdp_units=fsdp_units,
    )


def build_hf_llama_dry_run_stage(
        full_model: nn.Module,
        loss_fn: nn.Module,
        pp_rank: int,
        pp_size: int,
        num_micro_batches: int,
        pp_vpp: int = 1,
        pp_layer_split: Optional[list[int]] = None,
        boundary_batch_size: int = 1,
        sequence_length: int = 1,
        boundary_dtype: torch.dtype = torch.float32,
) -> tuple[DryRunPipelineChunk, ...]:
    """Split a meta-initialized HF Llama into all chunks owned by one rank.

    Args:
        full_model: Complete meta-initialized HF causal-LM model.
        loss_fn: Trainer-configured loss adapter used by the last stage.
        pp_rank: Pipeline rank of this worker.
        pp_size: Number of pipeline stages.
        num_micro_batches: Scheduler micro-batch count.
        pp_vpp: Number of virtual chunks owned by each rank.
        pp_layer_split: Optional decoder-layer counts for all global stages.
        boundary_batch_size: DP-local size of one PP micro-batch.
        sequence_length: Static logical activation sequence length.
        boundary_dtype: Activation dtype crossing PP boundaries.

    Returns:
        All rank-local modules and their global layer-range metadata.

    Raises:
        ValueError: If the model does not expose the supported HF Llama structure.
    """
    backbone = getattr(full_model, "model", None)
    required_backbone_fields = ("embed_tokens", "layers", "norm")
    missing = [name for name in required_backbone_fields if not hasattr(backbone, name)]
    if missing or not hasattr(full_model, "lm_head"):
        raise ValueError(
            "build_hf_llama_dry_run_stage requires model.embed_tokens, "
            f"model.layers, model.norm and lm_head; missing {missing}"
        )
    if bool(getattr(full_model.config, "tie_word_embeddings", False)):
        raise ValueError("Dry-run PP does not support tied input and output embeddings")
    num_hidden_layers = int(getattr(full_model.config, "num_hidden_layers", len(backbone.layers)))
    if len(backbone.layers) != num_hidden_layers:
        raise ValueError(
            "HF Llama layer count does not match config.num_hidden_layers: "
            f"{len(backbone.layers)} != {num_hidden_layers}"
        )
    if num_micro_batches < 1:
        raise ValueError("num_micro_batches must be positive")
    if not isinstance(pp_vpp, int) or isinstance(pp_vpp, bool) or pp_vpp < 1:
        raise ValueError(f"pp_vpp must be a positive integer, got {pp_vpp!r}")
    stage_num = pp_size * pp_vpp
    stage_indices = tuple(
        pp_rank + virtual_index * pp_size
        for virtual_index in range(pp_vpp)
    )
    return tuple(
        _build_hf_llama_chunk(
            full_model,
            loss_fn,
            stage_index,
            stage_num,
            num_micro_batches,
            pp_layer_split,
            boundary_batch_size,
            sequence_length,
            boundary_dtype,
        )
        for stage_index in stage_indices
    )


__all__ = ["build_hf_llama_dry_run_stage", "layer_range_for_pp_stage"]
