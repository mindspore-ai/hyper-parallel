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
"""Per-batch visual encoder planning and packed execution."""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.core.dtensor._utils import create_group, differentiable_variable_all_gather
from hyper_parallel.distributed.encoder_parallel.attention import (
    EncoderAttentionContext,
    activate_encoder_attention,
)
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderFeature,
    EncoderLocalInputs,
    EncoderParallelAdapterSpec,
    EncoderSegment,
    EncoderWorkItem,
)
from hyper_parallel.distributed.encoder_parallel.exchange import (
    FeatureExchangeMode,
    route_features,
    select_feature_exchange,
    unpack_records,
)
from hyper_parallel.distributed.encoder_parallel.mesh import (
    collect_decoder_metadata,
    derive_reshaped_submesh,
    select_encoder_domain_size,
)
from hyper_parallel.distributed.encoder_parallel.planning import (
    DecoderSequenceLayout,
    assign_lpt,
    build_destinations,
    build_segments,
)


def _invalid_head_count(count: Any) -> bool:
    """Reject booleans, nonintegers, and zero-sized QKV head axes."""
    return isinstance(count, bool) or not isinstance(count, int) or count < 1


def validate_ulysses_head_divisibility(*, encoder_cp_size: int, head_counts: Mapping[str, int],
                                       attention_pattern: str, model: nn.Module) -> None:
    """Check every Q/K/V head count before deriving any Encoder-CP group."""
    required = ("query", "key", "value")
    missing = [name for name in required if name not in head_counts]
    invalid = {name: head_counts[name] for name in required if name in head_counts
               and _invalid_head_count(head_counts[name])}
    non_divisible = {name: head_counts[name] for name in required if name in head_counts
                     and name not in invalid and head_counts[name] % encoder_cp_size}
    if missing or invalid or non_divisible:
        matched = sorted(name for name, _ in model.named_modules()
                         if fnmatch.fnmatchcase(name, attention_pattern))
        raise ValueError(
            "Encoder-CP/Ulysses head divisibility check failed; "
            "constraint=head_count % encoder_cp_size == 0 for every Q/K/V; "
            "config_path=accelerator.encoder_parallel.encoder_cp_size; "
            f"encoder_cp_size={encoder_cp_size}; head_counts={dict(head_counts)}; "
            f"missing={missing}; invalid_values={invalid}; non_divisible={non_divisible}; "
            f"attention_pattern={attention_pattern!r}; matched_modules={matched}"
        )


def validate_attention_replacement(model: nn.Module, spec: EncoderParallelAdapterSpec) -> frozenset[str]:
    """Require an exact structural replacement for every visual attention.

    Args:
        model: Final model after module replacements.
        spec: Model-owned attention matching and replacement contract.
    """
    expected = frozenset(name for name, _ in model.named_modules()
                         if fnmatch.fnmatchcase(name, spec.attention_pattern))
    recorded = set(getattr(model, "_hp_replaced_module_fqns", ()) or ())
    recorded_visual = {name for name in recorded if fnmatch.fnmatchcase(name, spec.attention_pattern)}

    def owned_by_complete_family(name: str) -> bool:
        """Accept class-owned attention inside a declared parameter-owning family."""
        prefix = name.rsplit(".", 1)[0]
        while prefix:
            info = get_family_build_info(model.get_submodule(prefix))
            if info is not None and info.resolved.family.sharding is not None:
                return True
            prefix = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return False

    missing = sorted(name for name in expected - recorded_visual if not owned_by_complete_family(name))
    unexpected = sorted(recorded_visual - expected)
    wrong_type = sorted(name for name in expected
                        if not isinstance(model.get_submodule(name), spec.replacement_attention_type))
    if not expected or missing or unexpected or wrong_type:
        raise RuntimeError(
            "incomplete vision attention replacement: "
            f"expected={len(expected)}; missing={missing}; unexpected={unexpected}; wrong_type={wrong_type}"
        )
    return expected


def validate_encoder_execution_options(options: Mapping[str, Any], bridge: Any) -> None:
    """Reject unsupported execution modes before creating Encoder groups.

    Args:
        options: Trainer microbatch, compile, and placement settings.
        bridge: Model-bound validator for vision-specific restrictions.
    """
    bridge.validate_training(options)
    if options.get("compile_enabled"):
        raise NotImplementedError("v1 requires compile.enabled=false")
    if options.get("validate_placement"):
        raise NotImplementedError("v1 requires model.validate_placement=false")


@dataclass
class EncoderParallelPlan:
    """One microbatch of derived visual work and feature routing."""

    runtime: "EncoderParallelRuntime"
    active: bool
    decoder_layout: DecoderSequenceLayout
    buckets: tuple[tuple[EncoderWorkItem, ...], ...]
    destinations: dict[int, tuple[int, ...]]
    local_segments: tuple[EncoderSegment, ...]
    k_fsdp_max: int
    decoder_items: tuple[EncoderWorkItem, ...] = ()

    def _gather_features(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """All-gather each local segment and trim dummy padding."""
        gathered = []
        offset = 0
        for segment in self.local_segments:
            local = hidden[offset:offset + segment.local_padded_length]
            offset += segment.local_padded_length
            full = differentiable_variable_all_gather(
                local, output_splits=[segment.local_padded_length] * self.runtime.config.encoder_cp_size,
                group=self.runtime.encoder_cp_mesh.get_group(),
            ) if self.runtime.config.encoder_cp_size > 1 else local
            gathered.append(full if segment.is_dummy else full[:segment.item.sequence_length])
        return tuple(gathered)

    def _execute_local(self, image_inputs: Mapping[str, Any] | EncoderLocalInputs
                       ) -> tuple[tuple[EncoderFeature, ...], torch.Tensor, torch.Tensor]:
        """Run the local ViT pass and build routed feature records."""
        bridge = self.runtime.bridge
        context = EncoderAttentionContext(
            self.local_segments,
            self.runtime.encoder_cp_mesh,
            bridge.head_counts,
            self.runtime.expected,
        )
        with activate_encoder_attention(context):
            hidden = bridge.encode_local(image_inputs, self.local_segments)
        context.assert_exactly_once()
        full_features = self._gather_features(hidden)
        records = []
        anchors = [feature.sum() * 0.0 for feature in full_features]
        prototype = None
        for slot in range(self.k_fsdp_max):
            segment = self.local_segments[slot] if slot < len(self.local_segments) else None
            real = segment is not None and not segment.is_dummy and self.runtime.encoder_cp_rank == 0
            if real:
                feature = bridge.project(full_features[slot], segment.item)
                if feature.ndim != 2 or feature.shape[0] != segment.item.output_rows:
                    raise ValueError(
                        "aligner output does not match planned rows; "
                        f"item_id={segment.item.item_id}; actual_shape={tuple(feature.shape)}; "
                        f"expected_rows={segment.item.output_rows}"
                    )
                records.append((segment.item, feature))
            else:
                feature = bridge.project(hidden, None)
                anchors.append(feature.sum() * 0.0)
            prototype = feature
        if prototype is None:
            raise RuntimeError("active plan must have an aligner slot")
        anchor = torch.stack(anchors).sum() if anchors else prototype.sum() * 0.0
        return tuple(records), prototype, anchor

    def encode_and_route(self, image_inputs: Mapping[str, Any] | EncoderLocalInputs
                         ) -> tuple[list[EncoderFeature], torch.Tensor | None]:
        """Run packed ViT and route aligned features to this decoder owner.

        Args:
            image_inputs: Prepared local patches, or a native image bundle for direct callers.

        Returns:
            Routed image records and a zero-valued gradient anchor, or no work.
        """
        if not self.active:
            return [], None
        if self.runtime.cross_decoder_parents and not isinstance(image_inputs, EncoderLocalInputs):
            raise ValueError("Cross-parent encoder execution requires transported EncoderLocalInputs")
        records, prototype, local_anchor = self._execute_local(image_inputs)
        mode = self.runtime.feature_exchange_mode
        exchange_mesh = self.runtime.tp_mesh if mode in (
            FeatureExchangeMode.TP_VARIABLE_ALL_GATHER, FeatureExchangeMode.TP_VARIABLE_A2A
        ) else self.runtime.parent_mesh
        rows, received_items = route_features(
            mode=mode, producer_records=records, prototype=prototype,
            parent_rank=self.runtime.parent_rank, destinations=self.destinations,
            buckets=self.buckets, encoder_cp_size=self.runtime.config.encoder_cp_size, exchange_mesh=exchange_mesh,
        )
        unpacked = unpack_records(rows, received_items)
        return list(unpacked), local_anchor + rows.sum() * 0.0


class EncoderParallelRuntime:
    """Own static topology and create one plan per Omni microbatch."""

    def __init__(self, *, config: EncoderParallelConfig, mesh_context: Any,
                 model: nn.Module, spec: EncoderParallelAdapterSpec,
                 execution_options: Mapping[str, Any]) -> None:
        """Validate the static contract and derive Encoder-CP groups."""
        self.config = config
        self.mesh_context = mesh_context
        self.model = model
        self.spec = spec
        self.bridge = spec.bind(model)
        validate_encoder_execution_options(
            {**execution_options, "loss_parallel": mesh_context.loss_parallel}, self.bridge,
        )
        if mesh_context.pp_size != 1:
            raise NotImplementedError("v1 requires pp_size=1")
        validate_ulysses_head_divisibility(encoder_cp_size=config.encoder_cp_size,
                                           head_counts=self.bridge.head_counts,
                                           attention_pattern=spec.attention_pattern, model=model)
        decoder_size = mesh_context.cp_size * mesh_context.tp_size
        parent_size = select_encoder_domain_size(
            decoder_size=decoder_size, job_size=mesh_context.device_mesh.size(),
            encoder_cp_size=config.encoder_cp_size, resource_domain=config.resource_domain,
        )
        self.cross_decoder_parents = parent_size > decoder_size
        if dist.is_initialized():
            # Reuse the job control group before deriving independent resource groups.
            create_group(mesh_context.device_mesh.mesh.flatten().tolist())
        self.global_mesh = mesh_context.device_mesh.flatten("encoder_global")
        self.decoder_mesh = mesh_context.device_mesh[("cp", "tp")].flatten("encoder_decoder_parent")
        self.expected = validate_attention_replacement(model, spec)
        self.encoder_dp_size = parent_size // config.encoder_cp_size
        self._bind_resource_domain(parent_size, decoder_size)
        self.parent_rank = self.parent_mesh.get_local_rank()
        self.encoder_cp_rank = self.parent_rank % config.encoder_cp_size
        self.encoder_dp_rank = self.parent_rank // config.encoder_cp_size
        self.tp_mesh = mesh_context.device_mesh["tp"]
        self.feature_exchange_mode = select_feature_exchange(mesh_context, parent_size=parent_size)

    def _bind_resource_domain(self, parent_size: int, decoder_size: int) -> None:
        """Derive resource groups without coupling the planning protocol to topology."""
        root = self.mesh_context.device_mesh
        if parent_size == decoder_size:
            encoder_mesh = derive_reshaped_submesh(
                root, source_dims=("cp", "tp"),
                target_sizes=(self.encoder_dp_size, self.config.encoder_cp_size),
                target_names=("encoder_dp", "encoder_cp"),
            )
        else:
            dimensions = tuple(root.mesh_dim_names)
            if dimensions[-2:] != ("cp", "tp"):
                raise ValueError("Independent encoder domains require canonical trailing CP/TP rank order")
            encoder_mesh = derive_reshaped_submesh(
                root, source_dims=dimensions,
                target_sizes=(root.size() // parent_size, self.encoder_dp_size, self.config.encoder_cp_size),
                target_names=("encoder_domain", "encoder_dp", "encoder_cp"),
            )
        self.parent_mesh = encoder_mesh[("encoder_dp", "encoder_cp")].flatten(f"encoder_parent_{parent_size}")
        self.encoder_cp_mesh = encoder_mesh["encoder_cp"]

    def preflight_batch(self, batch: Mapping[str, Any]
                        ) -> tuple[tuple[EncoderWorkItem, ...], DecoderSequenceLayout]:
        """Validate local metadata and SP geometry before control collectives.

        Args:
            batch: Device batch with decoder tokens and optional image fields.
        """
        layout = DecoderSequenceLayout.build(
            mesh_context=self.mesh_context, cp_local_length=int(batch["input_ids"].shape[1])
        )
        present = [batch.get(name) is not None for name in self.bridge.input_fields]
        if any(present) and not all(present):
            missing = [name for name in self.bridge.input_fields if batch.get(name) is None]
            raise ValueError(f"incomplete image metadata bundle: missing={missing}")
        return tuple(self.bridge.parse_images(batch)), layout

    def finalize_plan(self, items: tuple[EncoderWorkItem, ...],
                      layout: DecoderSequenceLayout) -> EncoderParallelPlan:
        """Agree on active mode, then assign images and exact decoder owners.

        Args:
            items: Locally validated image jobs.
            layout: Locally validated decoder token ownership.
        """
        marker = torch.tensor(int(bool(items)), device=self.model.device if hasattr(self.model, "device")
                              else next(self.model.parameters()).device)
        metadata_invalid = False
        if dist.is_initialized() and self.decoder_mesh.size() > 1:
            signature = tuple(
                (item.item_id, item.batch_index, item.sequence_length, item.output_rows,
                 item.decoder_token_start, item.decoder_token_length, item.payload_start, item.payload_stop,
                 tuple(sorted(item.metadata.items())))
                for item in items
            )
            digest = hashlib.sha256(repr((layout.cp_local_length, signature)).encode()).digest()
            digest_value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            local_digest = torch.tensor(digest_value, dtype=torch.int64, device=marker.device)
            peer_digests = [torch.empty_like(local_digest) for _ in range(self.decoder_mesh.size())]
            dist.all_gather(peer_digests, local_digest, group=self.decoder_mesh.get_group())
            metadata_invalid = any(int(peer.item()) != digest_value for peer in peer_digests)
        invalid = torch.tensor(int(metadata_invalid), device=marker.device)
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.global_mesh.get_group())
        if bool(invalid.item()):
            raise ValueError("image metadata differs across decoder parent ranks")
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(marker, op=dist.ReduceOp.MAX, group=self.global_mesh.get_group())
        active = bool(marker.item())
        decoder_items = items
        items, destinations = self._collect_decoder_items(items, layout, device=marker.device)
        buckets = assign_lpt(items, self.encoder_dp_size, self.config.encoder_cp_size)
        segments = build_segments(buckets[self.encoder_dp_rank], self.encoder_cp_rank,
                                  self.config.encoder_cp_size)
        slot_count = torch.tensor(len(buckets[self.encoder_dp_rank]), device=marker.device)
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(slot_count, op=dist.ReduceOp.MAX, group=self.global_mesh.get_group())
        return EncoderParallelPlan(
            runtime=self,
            active=active, decoder_layout=layout, buckets=buckets,
            destinations=destinations, local_segments=segments,
            k_fsdp_max=max(1, int(slot_count.item())),
            decoder_items=decoder_items,
        )

    def _collect_decoder_items(self, items: tuple[EncoderWorkItem, ...],
                               layout: DecoderSequenceLayout, *, device: torch.device) -> tuple:
        """Normalize source identity and token ownership for every resource domain."""
        decoder_size = self.mesh_context.cp_size * self.mesh_context.tp_size
        peers = collect_decoder_metadata(
            (layout.cp_local_length, items), decoder_size=decoder_size, parent_mesh=self.parent_mesh, device=device,
        )
        global_items, destinations = [], {}
        for index, metadata in enumerate(peers):
            if metadata is None:
                raise ValueError("encoder domain is missing a decoder leader's metadata")
            source_length, source_items = metadata
            source_rank = index * decoder_size
            layouts = {
                source_rank + cp * self.mesh_context.tp_size + tp: DecoderSequenceLayout.build(
                    mesh_context=self.mesh_context, cp_local_length=source_length, cp_rank=cp, tp_rank=tp,
                )
                for cp in range(self.mesh_context.cp_size) for tp in range(self.mesh_context.tp_size)
            }
            scoped = tuple(replace(item, item_id=len(global_items) + offset, source_rank=source_rank)
                           for offset, item in enumerate(source_items))
            destinations.update(build_destinations(scoped, layouts))
            global_items.extend(scoped)
        return tuple(global_items), destinations



def build_encoder_parallel_runtime(*, config: EncoderParallelConfig, mesh_context: Any,
                                   model: nn.Module | None, model_adapter_spec: Any,
                                   execution_options: Mapping[str, Any] | None) -> EncoderParallelRuntime | None:
    """Create the model-neutral runtime only when the feature is enabled."""
    if not config.enabled:
        return None
    if model is None or model_adapter_spec is None or model_adapter_spec.encoder_parallel is None:
        raise ValueError("enabled encoder parallel requires a model adapter provider")
    spec = model_adapter_spec.encoder_parallel()
    if not isinstance(spec, EncoderParallelAdapterSpec):
        raise TypeError("encoder_parallel provider returned an invalid spec")
    return EncoderParallelRuntime(config=config, mesh_context=mesh_context, model=model,
                                  spec=spec, execution_options=execution_options or {})
