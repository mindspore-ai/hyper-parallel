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
"""DeepSeek-V4.1 pipeline boundaries for mHC and differentiable CSA2 state."""

from types import MethodType
from typing import Any, Optional

import torch
from torch import distributed as dist, nn
from transformers.modeling_outputs import MoeModelOutputWithPast

from hyper_parallel.components.checkpoint.load_groups import LoadReport
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import SharedCompressedAttentionState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.external_state import get_model_external_state
from hyper_parallel.models.deepseek_v41.vision import encode_image_features
from hyper_parallel.models.pipeline import PipelineAdapterSpec, PipelinePartition
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    _hc_pre, _uses_shifted_mhc,
)


def _state_layout(layers: Any, cut: int) -> tuple[tuple[str, int], ...]:
    """Select published tensors with consumers beyond this pipeline cut."""
    entries = set()
    for layer in list(layers)[cut:]:
        attention = layer.self_attn
        if not attention.compress_ratio:
            continue
        source = attention.kv_source_layer_idx
        if source is not None and source < cut:
            entries.add(("compressed_kv_by_source", source))
            entries.add(("compressed_kv_offsets", source))
            if attention.is_index_source:
                entries.add(("index_key_by_source", source))
        source = attention.index_source_layer_idx
        if source is not None and source < cut:
            entries.add(("topk_indices_by_source", source))
        source = attention.candidate_source_layer_idx
        if source is not None and source < cut and attention.is_index_source and attention.indexer.uses_candidates:
            entries.add(("candidate_blocks_by_source", source))
    return tuple(sorted(entries))


def _pack_state(state: SharedCompressedAttentionState, layout: tuple, reference: torch.Tensor,
                indexer_loss_enabled: bool) -> tuple:
    """Only the Indexer KL objective differentiates through a shared index key."""
    values = []
    for name, source in layout:
        value = getattr(state, name)[source]
        if name == "compressed_kv_offsets":
            value = torch.tensor(value, device=reference.device, dtype=torch.int64)
        if name == "index_key_by_source" and not indexer_loss_enabled:
            value = value.detach()
        values.append(value)
    return tuple(values)


def _unpack_state(values: tuple, layout: tuple) -> SharedCompressedAttentionState:
    state = SharedCompressedAttentionState()
    if len(values) != len(layout):
        raise ValueError("V4.1 pipeline shared-state tensor count does not match the partition")
    for (name, source), value in zip(layout, values):
        getattr(state, name)[source] = int(value.item()) if name == "compressed_kv_offsets" else value
    return state


class _MPipeVision(nn.Module):
    """Run the ordinary per-image ViT and aligner on an MPipe owner."""

    def __init__(self, vision: nn.Module, aligner: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.vision = vision
        self.aligner = aligner
        self.hidden_size = hidden_size

    @staticmethod
    def take_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable image geometry as one tensor alongside the owner pixels.

        Args:
            batch: Owner's model inputs. Pixels are consumed by the preprocess.
        """
        pixels = batch.pop("pixel_values", None)
        if pixels is None:
            if batch.get("image_vit_grid_hw") is not None:
                raise ValueError("MPipe image geometry requires owner pixel_values")
            # Nonempty wire tensors allow text-only and image micros on the same HCCL group.
            ids = batch["input_ids"]
            return ids.new_zeros(1, dtype=torch.float32), ids.new_zeros((1, 7))
        if batch.get("token_types") is None:
            raise ValueError("MPipe images require token_types")
        offsets = batch["image_patch_offsets"].reshape(-1)
        geometry = torch.cat((offsets[1:, None], batch["image_vit_grid_hw"], batch["image_llm_grid_hw"],
                              batch["image_batch_indices"].reshape(-1, 1), batch["image_token_starts"].reshape(-1, 1)),
                             dim=1)
        if pixels.ndim != 4 or geometry.shape[0] == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != pixels.shape[0]:
            raise ValueError("MPipe pixel_values and image_patch_offsets disagree")
        return pixels, geometry

    def forward(self, pixels: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """Encode each image independently and concatenate its LLM-width feature rows.

        Args:
            pixels: Owner-local patches or the text-only placeholder.
            geometry: Offsets, image grids and token spans packed by take_inputs.
        """
        if pixels.ndim == 1:
            return pixels.new_zeros(1, self.hidden_size)
        offsets = torch.cat((geometry.new_zeros(1), geometry[:, 0]))
        records = encode_image_features(self.vision, self.aligner, pixels, offsets, geometry[:, 1:3],
                                        geometry[:, 3:5], geometry[:, 5], geometry[:, 6])
        return torch.cat([record[-1] for record in records])


def _merge_mpipe_features(backbone, embeddings, features, token_types, image_grid, indices, starts, sequence_start):
    if image_grid is None:
        return embeddings
    sizes = image_grid.tolist()
    rows = features.split([height * width for height, width in sizes])
    records = [(index, start, height, width, feature) for index, start, (height, width), feature in
               zip(indices.tolist(), starts.tolist(), sizes, rows, strict=True)]
    return backbone._merge_image_embeddings(embeddings, token_types, records, sequence_start)  # pylint: disable=protected-access


def _backbone_forward(
        self: nn.Module, input_ids: torch.Tensor, *, pipeline_inputs: tuple = (),
        position_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False, past_key_values: Any = None, token_types: Optional[torch.Tensor] = None,
        image_llm_grid_hw: Optional[torch.Tensor] = None, image_batch_indices: Optional[torch.Tensor] = None,
        image_token_starts: Optional[torch.Tensor] = None, image_sequence_start: int = 0, **kwargs: Any,
) -> Any:
    """Execute local layers while keeping each microbatch's shared state explicit."""
    if use_cache or past_key_values is not None:
        raise ValueError("V4.1 pipeline training does not support KV cache")
    partition = self.pipeline_partition
    boundary_width = self.pipeline_boundary_width
    if partition.is_first:
        embedded = self.embed_tokens(input_ids)
        if partition.mpipe:
            embedded = _merge_mpipe_features(self, embedded, pipeline_inputs[0], token_types, image_llm_grid_hw,
                                             image_batch_indices, image_token_starts, image_sequence_start)
        hidden = embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden.new_zeros((*hidden.shape[:2], self.config.hc_mult), dtype=torch.float32)
        pre_mix[:, :, 0] = 1
        state = SharedCompressedAttentionState()
    else:
        hidden, pre_mix = pipeline_inputs[:2]
        state = _unpack_state(pipeline_inputs[boundary_width:], self.pipeline_state_in)
    position_ids, embeddings, starts = self._prepare_decoder_inputs(  # pylint: disable=protected-access
        input_ids, hidden[:, :, 0], position_ids, kwargs,
    )
    image_mask = None if token_types is None else token_types >= 0
    outputs = self._run_decoder_layers(  # pylint: disable=protected-access
        hidden, pre_mix, input_ids=input_ids, position_embeddings=embeddings, position_ids=position_ids,
        attention_mask=attention_mask, shared_state=state, segment_start_mask=starts,
        engram_token_mask=None if image_mask is None else ~image_mask, image_mask=image_mask,
        dspark_layer_ids=(), pending_mhc=pipeline_inputs[2:boundary_width],
        finish_mhc=partition.is_last, **kwargs,
    )
    if partition.is_last:
        return MoeModelOutputWithPast(last_hidden_state=self.norm(_hc_pre(*outputs)))
    indexer_loss_enabled = bool(getattr(self.config, "v41_indexer_loss_coeff", 0))
    shared = _pack_state(state, self.pipeline_state_out, hidden, indexer_loss_enabled)
    return tuple(tensor.contiguous() for tensor in (*outputs, *shared))


def _intermediate_forward(self: nn.Module, **kwargs: Any) -> tuple:
    for name in ("labels", "shift_labels", "loss_mask", "chunk_loss_targets", "chunk_loss_mask",
                 "chunk_loss_chunk_size", "chunk_loss_ignore_index"):
        kwargs.pop(name, None)
    return self.model(**kwargs)


def install_stage(model: nn.Module, partition: PipelinePartition) -> None:
    """Prune endpoint ownership and install the V4.1 tensor boundary protocol.

    Args:
        model: Full V4.1 model before sharding and parameter materialization.
        partition: Decoder range and endpoint ownership for this rank.
    """
    if getattr(model, "dspark", None) is not None:
        raise ValueError("V4.1 pipeline training does not support DSpark")
    backbone = model.model
    if partition.mpipe != (backbone.vision is not None):
        raise ValueError("V4.1 MPipe requires a vision tower, and vision pipeline training requires schedule=mpipe")
    if partition.mpipe:
        if not partition.preprocess_replica:
            backbone.vision = backbone.aligner = None
        if not partition.is_first:
            backbone.image_start = backbone.image_end = backbone.image_newline = None
    protocols = {_uses_shifted_mhc(layer) for layer in backbone.layers}
    if len(protocols) != 1:
        raise ValueError("shifted mHC must replace every V4.1 layer in a pipeline")
    backbone.pipeline_boundary_width = 5 if protocols.pop() else 2
    backbone.dspark_target_hiddens = None
    backbone.pipeline_state_in = _state_layout(backbone.layers, partition.start)
    backbone.pipeline_state_out = _state_layout(backbone.layers, partition.stop)
    backbone.pipeline_partition = partition
    backbone.forward = MethodType(_backbone_forward, backbone)
    if not partition.is_first:
        backbone.embed_tokens = None
    if not partition.is_last:
        backbone.norm = None
        model.lm_head = None
        model.forward = MethodType(_intermediate_forward, model)


class _LocalChunks(nn.Module):
    """Register stage-owned layers/endpoints under their original checkpoint FQNs.

    The schedule runs the independent FSDP roots in `parts`. Keeping them in a
    tuple avoids registering a second, chunk-prefixed name for every parameter.
    """

    def __init__(self, parts: tuple[nn.Module, ...], pp_group: dist.ProcessGroup) -> None:
        super().__init__()
        self.parts = parts
        self.config = parts[0].config
        self._hp_pipeline_partition = parts[-1]._hp_pipeline_partition  # pylint: disable=protected-access
        self.model = nn.Module()
        self.model.embed_tokens = parts[0].model.embed_tokens
        self.model.layers = nn.ModuleDict({
            key: layer for part in parts for key, layer in part.model.layers.items()
        })
        self.model.norm = parts[-1].model.norm
        self.lm_head = parts[-1].lm_head
        first = parts[0].model
        self._replica_keys = set()
        chunk_parameters = [(name, param) for part in parts for name, param in part.named_parameters()]
        if first.pipeline_partition.mpipe:
            self._detach_visual_preprocess(first)
            chunk_parameters = [(n, p) for n, p in chunk_parameters if n not in self._replica_keys]
        owned = dict(self.named_parameters())
        if len(owned) != len(chunk_parameters) or any(owned.get(name) is not param for name, param in chunk_parameters):
            raise ValueError("V4.1 virtual chunks must partition the rank's model parameters under unique global names")
        self._collect_load_metadata(pp_group)
        states = [get_model_external_state(part) for part in parts]
        if states[0] is not None:
            self._hp_external_state = DeepseekV41HostState(
                self, {name: table for state in states for name, table in state.tables.items()}, states[0].mesh_context,
            )
            self._hp_external_state.check_identities()

    def _detach_visual_preprocess(self, first: nn.Module) -> None:
        """Give the visual FSDP units their own execution root and stage-zero checkpoint ownership."""
        self._visual_modules = (first.vision, first.aligner)
        if first.pipeline_partition.is_first:
            self.model.vision, self.model.aligner = first.vision, first.aligner
            self.model.image_start, self.model.image_end = first.image_start, first.image_end
            self.model.image_newline = first.image_newline
        else:
            # Only stage zero checkpoints and optimizes the visual weights.
            self._replica_keys = {f"model.{name}.{key}" for name in ("vision", "aligner")
                                  for key in getattr(first, name).state_dict()}
        # The transpose prefix runs first. Avoid making a visual FSDP unit a child
        # of a decoder execution root that will initialize its own HSDP scheduler.
        first.vision = first.aligner = None

    def build_mpipe_preprocess(self) -> nn.Module:
        """Bind the visual execution replica after FSDP and weight loading."""
        return _MPipeVision(*self._visual_modules, self.config.hidden_size)

    def _collect_load_metadata(self, pp_group: dist.ProcessGroup) -> None:
        reports = [getattr(part, "_hp_checkpoint_load_report", None) for part in self.parts]
        if all(report is None for report in reports):
            return
        if any(report is None for report in reports):
            raise ValueError("Virtual stages must use the same pretrained-weight loading policy")
        # A source key is unexpected only when no virtual stage on any PP rank consumed it.
        unused = set.intersection(*(set(report.unexpected_keys) for report in reports))
        unused_by_rank = [None] * dist.get_world_size(pp_group)
        dist.all_gather_object(unused_by_rank, unused, group=pp_group)
        self._hp_checkpoint_load_report = LoadReport(
            loaded_keys=tuple(sorted({key for report in reports for key in report.loaded_keys} - self._replica_keys)),
            missing_keys=tuple(sorted({key for report in reports for key in report.missing_keys} - self._replica_keys)),
            unexpected_keys=tuple(sorted(set.intersection(*unused_by_rank))),
        )
        self._weight_conversions = [
            transform for part in self.parts for transform in (getattr(part, "_weight_conversions", None) or ())
        ]

    def train(self, mode: bool = True) -> nn.Module:
        """Set training mode on the execution roots as well as their shared children.

        Args:
            mode: Whether to enable training.
        """
        for part in self.parts:
            part.train(mode)
        if self._replica_keys:
            for module in self._visual_modules:
                module.train(mode)
        return super().train(mode)


def get_pipeline_spec() -> PipelineAdapterSpec:
    """Return the V4.1 decoder partition provider."""
    return PipelineAdapterSpec("model.layers", install_stage, _LocalChunks)
