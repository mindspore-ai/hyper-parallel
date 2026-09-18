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
"""Tensor summaries, input identity, and parameter probe recorders."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.distributed as dist

from hyper_parallel.models.validation_spec import (
    ModelValidationSpec,
    ParameterProbeSpec,
)
from hyper_parallel.tools.model_integration.evidence_store import EvidenceStore
from hyper_parallel.tools.model_integration.optimizer_layout import (
    optimizer_state_for_parameter,
    tensor_layout,
)

_SMALL_TENSOR_NUMEL = 8192

def _tensor_numel(tensor: Any) -> int:
    shape = getattr(tensor, "shape", ())
    return math.prod(int(size) for size in shape)


def _local_tensor(tensor: Any) -> torch.Tensor:
    """Return one detached local shard without requesting redistribution."""
    to_local = getattr(tensor, "to_local", None)
    value = to_local() if callable(to_local) else tensor
    return value.detach().float().cpu().contiguous()


def _shard_identity(tensor: Any) -> tuple[tuple[int, int], ...]:
    """Identify the logical shard while excluding replicated mesh axes."""
    layout = getattr(tensor, "layout", None)
    mesh = getattr(layout, "mesh", getattr(tensor, "device_mesh", None))
    placements = getattr(layout, "placements", getattr(tensor, "placements", ()))
    get_coordinate = getattr(mesh, "get_coordinate", None)
    coordinate = get_coordinate() if callable(get_coordinate) else None
    if coordinate is None:
        return ()
    shard_coordinates = []
    for mesh_dim, placement in enumerate(placements or ()):
        is_shard = getattr(placement, "is_shard", None)
        is_ragged = getattr(placement, "is_ragged_shard", None)
        if (callable(is_shard) and is_shard()) or (
            callable(is_ragged) and is_ragged()
        ):
            shard_coordinates.append((mesh_dim, int(coordinate[mesh_dim])))
    return tuple(shard_coordinates)


def _local_shards_follow_flattened_order(tensor: Any) -> bool:
    """Return whether rank-ordered shard values reconstruct the flattened tensor."""
    layout = getattr(tensor, "layout", None)
    placements = getattr(layout, "placements", getattr(tensor, "placements", ()))
    shard_placements = []
    for placement in placements or ():
        is_shard = getattr(placement, "is_shard", None)
        is_ragged = getattr(placement, "is_ragged_shard", None)
        if (callable(is_shard) and is_shard()) or (
            callable(is_ragged) and is_ragged()
        ):
            shard_placements.append(placement)
    if not shard_placements:
        return True
    if len(shard_placements) != 1:
        return False
    placement = shard_placements[0]
    return getattr(placement, "dim", None) == 0 and not hasattr(
        placement,
        "split_factor",
    )


def _local_tensor_summary(tensor: Any) -> dict[str, Any]:
    """Summarize one local shard for cross-rank aggregation."""
    value = _local_tensor(tensor).reshape(-1)
    digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return {
        "rank": dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        "shard": _shard_identity(tensor),
        "sha256": digest,
        "numel": value.numel(),
        "finite": bool(torch.isfinite(value).all()) if value.numel() else True,
        "sum": float(value.double().sum()),
        "squared_l2": float(torch.square(value.double()).sum()),
        "min": float(value.min()) if value.numel() else None,
        "max": float(value.max()) if value.numel() else None,
        "values": value.tolist() if value.numel() <= _SMALL_TENSOR_NUMEL else None,
        "global_shape": [int(size) for size in getattr(tensor, "shape", ())],
        "dtype": str(getattr(tensor, "dtype", None)),
        "layout": tensor_layout(tensor),
    }


def _gather_shard_summaries(
    local_summary: Optional[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Optional[dict[str, Any]]]]:
    """Gather optional metadata records and retain one value per logical shard."""
    rank_summaries = [local_summary]
    if dist.is_available() and dist.is_initialized():
        rank_summaries = [None] * dist.get_world_size()
        dist.all_gather_object(rank_summaries, local_summary)
    by_shard: dict[tuple[tuple[int, int], ...], list[dict[str, Any]]] = {}
    for summary in rank_summaries:
        if summary is None:
            continue
        shard = tuple(tuple(item) for item in summary["shard"])
        by_shard.setdefault(shard, []).append(summary)
    shards = [
        min(replicas, key=lambda item: item["rank"])
        for replicas in by_shard.values()
    ]
    return shards, rank_summaries


def _tensor_summary(name: str, tensor: Any) -> Optional[dict[str, Any]]:
    """Build one global summary while allowing rank-local lazy state absence."""
    local_summary = None if tensor is None else _local_tensor_summary(tensor)
    shards, rank_summaries = _gather_shard_summaries(local_summary)
    if not shards:
        return None
    shards.sort(key=lambda item: item["shard"])
    replica_groups: dict[tuple[Any, ...], set[str]] = {}
    for rank_summary in rank_summaries:
        if rank_summary is None:
            continue
        shard = tuple(tuple(item) for item in rank_summary["shard"])
        replica_groups.setdefault(shard, set()).add(rank_summary["sha256"])
    canonical_digest = hashlib.sha256(
        json.dumps(
            [(summary["shard"], summary["sha256"]) for summary in shards],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    minima = [summary["min"] for summary in shards if summary["min"] is not None]
    maxima = [summary["max"] for summary in shards if summary["max"] is not None]
    reference = shards[0]
    summary = {
        "name": name,
        "global_shape": reference["global_shape"],
        "canonical_shape": reference["global_shape"],
        "dtype": reference["dtype"],
        "sha256": canonical_digest,
        "finite": all(shard["finite"] for shard in shards),
        "sum": sum(shard["sum"] for shard in shards),
        "l2": math.sqrt(sum(shard["squared_l2"] for shard in shards)),
        "min": min(minima) if minima else None,
        "max": max(maxima) if maxima else None,
        "layout": reference["layout"],
        "replica": {
            "logical_shards": len(replica_groups),
            "replicas_equal": all(len(digests) == 1 for digests in replica_groups.values()),
        },
        "missing_ranks": [
            rank for rank, rank_summary in enumerate(rank_summaries) if rank_summary is None
        ],
    }
    if (
        _local_shards_follow_flattened_order(tensor)
        and sum(shard["numel"] for shard in shards) <= _SMALL_TENSOR_NUMEL
        and all(shard["values"] is not None for shard in shards)
    ):
        summary["values"] = [value for shard in shards for value in shard["values"]]
    return summary


def _global_tensor_state_names(local_names: Iterable[str]) -> tuple[str, ...]:
    """Return a rank-consistent tensor-state key order for collective probes."""
    local_names = tuple(sorted(local_names))
    rank_names = [local_names]
    if dist.is_available() and dist.is_initialized():
        rank_names = [None] * dist.get_world_size()
        dist.all_gather_object(rank_names, local_names)
    return tuple(sorted({name for names in rank_names for name in (names or ())}))


class InputIdentityRecorder:
    """Hash global step inputs from deterministic per-rank tensor payloads."""

    def __init__(
            self,
            mesh_context: Any = None,
            cp_replicated_forward_fields: Iterable[str] = (),
    ) -> None:
        """Configure mesh-aware canonicalization for one Trainer run."""
        self.mesh_context = mesh_context
        self.cp_replicated_forward_fields = set(cp_replicated_forward_fields)
        self._entries: list[dict[str, Any]] = []
        self._micro_steps = 0

    def begin_step(self) -> None:
        """Reset the current optimizer-step identity."""
        self._entries = []
        self._micro_steps = 0

    @staticmethod
    def _tensor_leaves(value: Any) -> list[Any]:
        """Find tensors inside mappings, sequences, and metadata dataclasses."""
        if torch.is_tensor(value):
            return [value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [
                tensor
                for dataclass_field in dataclasses.fields(value)
                for field_value in (getattr(value, dataclass_field.name),)
                for tensor in InputIdentityRecorder._tensor_leaves(field_value)
            ]
        if isinstance(value, Mapping):
            return [
                tensor
                for name in sorted(value)
                for tensor in InputIdentityRecorder._tensor_leaves(value[name])
            ]
        if isinstance(value, (list, tuple)):
            return [
                tensor
                for child in value
                for tensor in InputIdentityRecorder._tensor_leaves(child)
            ]
        return []

    def record(self, model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> None:
        """Add model/loss tensor fields without changing their device or values."""
        for owner, values in (("model", model_inputs), ("loss", loss_inputs)):
            for field_name in sorted(values):
                flattened = self._tensor_leaves(values[field_name])
                for value_index, value in enumerate(flattened):
                    local = value.detach()
                    to_local = getattr(local, "to_local", None)
                    if callable(to_local):
                        local = to_local()
                    self._entries.append(
                        {
                            "micro_step": self._micro_steps,
                            "owner": owner,
                            "field": field_name,
                            "value_index": value_index,
                            "cp_replicated": (
                                owner == "model"
                                and field_name in self.cp_replicated_forward_fields
                            ),
                            "tensor": local.cpu().contiguous(),
                        }
                    )
        self._micro_steps += 1

    def _coordinates(self) -> dict[str, int]:
        return {
            name: int(getattr(self.mesh_context, f"{name}_rank", 0))
            for name in ("dp", "cp", "tp", "pp")
        }

    @staticmethod
    def _hash_tensor(value: torch.Tensor) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{tuple(value.shape)}:{value.dtype}".encode())
        byte_view = value.contiguous().view(torch.uint8)
        hasher.update(byte_view.numpy().tobytes())
        return hasher.hexdigest()

    @classmethod
    def _canonical_global_hash(cls, rank_payloads: list[dict[str, Any]]) -> str:
        malformed_ranks = [
            rank
            for rank, payload in enumerate(rank_payloads)
            if not isinstance(payload, Mapping)
            or "coordinates" not in payload
            or "entries" not in payload
        ]
        if malformed_ranks:
            raise RuntimeError(
                "input-identity collective received non-input payloads from ranks "
                f"{malformed_ranks}; parameter probes must execute a rank-consistent "
                "collective sequence"
            )
        selected = [
            payload
            for payload in rank_payloads
            if payload["coordinates"]["tp"] == 0 and payload["coordinates"]["pp"] == 0
        ]
        grouped: dict[tuple[Any, ...], list[tuple[int, torch.Tensor, bool]]] = {}
        for payload in selected:
            coordinates = payload["coordinates"]
            for entry in payload["entries"]:
                key = (
                    entry["micro_step"],
                    coordinates["dp"],
                    entry["owner"],
                    entry["field"],
                    entry["value_index"],
                )
                grouped.setdefault(key, []).append(
                    (
                        coordinates["cp"],
                        entry["tensor"],
                        bool(entry.get("cp_replicated", False)),
                    )
                )
        component_hashes = []
        for key, shards in grouped.items():
            ordered_shards = sorted(shards, key=lambda item: item[0])
            ordered = [tensor for _, tensor, _ in ordered_shards]
            tensor = ordered[0]
            cp_replicated = all(replicated for _, _, replicated in ordered_shards)
            if cp_replicated and any(
                    not torch.equal(tensor, replica) for replica in ordered[1:]
            ):
                cp_replicated = False
            if len(ordered) > 1 and tensor.ndim > 0 and not cp_replicated:
                try:
                    tensor = torch.cat(ordered, dim=-1)
                except RuntimeError:
                    tensor = torch.cat([value.reshape(-1) for value in ordered])
            field_identity = f"{key[2]}:{key[3]}:{key[4]}"
            if tensor.ndim > 0:
                values = tensor.reshape(tensor.shape[0], -1)
                component_hashes.extend(
                    f"{field_identity}:{cls._hash_tensor(value)}" for value in values
                )
            else:
                component_hashes.append(f"{field_identity}:{cls._hash_tensor(tensor)}")
        hasher = hashlib.sha256()
        for component_hash in sorted(component_hashes):
            hasher.update(component_hash.encode())
        return hasher.hexdigest()

    def finish_step(self) -> dict[str, Any]:
        """Return rank and global digests for the optimizer step."""
        local_payload = {
            "coordinates": self._coordinates(),
            "entries": self._entries,
        }
        rank_payloads = [local_payload]
        if dist.is_available() and dist.is_initialized():
            rank_payloads = [None] * dist.get_world_size()
            dist.all_gather_object(rank_payloads, local_payload)
        return {
            "global_sha256": self._canonical_global_hash(rank_payloads),
            "rank_count": len(rank_payloads),
            "micro_steps": self._micro_steps,
        }


class ParameterProbeRecorder:
    """Capture canonical parameter/main-gradient/update evidence by semantic watchlist."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        evidence_store: EvidenceStore,
        *,
        rank: int,
        validation_spec: Optional[ModelValidationSpec] = None,
        include_small_parameters_numel_le: int = _SMALL_TENSOR_NUMEL,
        representative_parameter_names: Iterable[str] = (),
    ) -> None:
        """Resolve a stable watchlist from the final model and adapter declarations."""
        self.model = model
        self.optimizer = optimizer
        self.evidence_store = evidence_store
        self.rank = rank
        self.validation_spec = validation_spec
        self.include_small_parameters_numel_le = include_small_parameters_numel_le
        self.representative_parameter_names = set(representative_parameter_names)
        self._named_parameters = dict(model.named_parameters(remove_duplicate=False))
        self._probes = self._resolve_probes()
        self._validate_required_patterns()
        self._step = 0

    def _resolve_probes(self) -> dict[str, ParameterProbeSpec]:
        probes = {}
        declared = self.validation_spec.parameter_probes if self.validation_spec else ()
        replaced_module_fqns = tuple(
            getattr(self.model, "_hp_replaced_module_fqns", ()) or ()
        )
        for parameter_name, parameter in self._named_parameters.items():
            main_parameter = getattr(parameter, "main_param", parameter)
            replacement_parameter = any(
                parameter_name == module_fqn
                or parameter_name.startswith(f"{module_fqn}.")
                for module_fqn in replaced_module_fqns
            )
            if (
                replacement_parameter
                or parameter_name in self.representative_parameter_names
                or _tensor_numel(main_parameter) <= self.include_small_parameters_numel_le
            ):
                probes[parameter_name] = ParameterProbeSpec(
                    match=parameter_name,
                    required=True,
                    stages=(
                        "model_before",
                        "main_before",
                        "main_grad_before_clip",
                        "main_grad_after_clip",
                        "main_after_optimizer",
                        "model_after_copy_back",
                        "optimizer_state",
                    ),
                )
            for probe in declared:
                if fnmatch.fnmatchcase(parameter_name, probe.match):
                    probes[parameter_name] = probe
        return dict(sorted(probes.items()))

    def _validate_required_patterns(self) -> None:
        for probe in self.validation_spec.parameter_probes if self.validation_spec else ():
            if probe.required and not any(
                fnmatch.fnmatchcase(name, probe.match) for name in self._named_parameters
            ):
                raise ValueError(
                    f"required parameter probe {probe.match!r} matched no final parameter"
                )

    @property
    def watched_parameter_names(self) -> tuple[str, ...]:
        """Return stable final FQNs in capture order."""
        return tuple(self._probes)

    def begin_step(self, step: int) -> None:
        """Start evidence for one optimizer step."""
        self._step = step
        self.capture("model_before")
        self.capture("main_before")

    @staticmethod
    def _stage_tensor(parameter: Any, stage: str) -> Any:
        main_parameter = getattr(parameter, "main_param", parameter)
        if stage in ("model_before", "model_after_copy_back"):
            return parameter
        if stage in ("main_before", "main_after_optimizer"):
            return main_parameter
        if stage in ("main_grad_before_clip", "main_grad_after_clip"):
            main_gradient = getattr(parameter, "main_grad", None)
            return main_gradient if main_gradient is not None else getattr(main_parameter, "grad", None)
        return None

    def _optimizer_state(self, parameter: Any) -> dict[str, Any]:
        main_parameter = (
            None if parameter is None else getattr(parameter, "main_param", parameter)
        )
        state = (
            {} if main_parameter is None
            else optimizer_state_for_parameter(self.optimizer, main_parameter)
        )
        state_names = _global_tensor_state_names(
            name for name, value in state.items() if torch.is_tensor(value)
        )
        summaries = {}
        for name in state_names:
            value = state.get(name)
            summary = _tensor_summary(name, value if torch.is_tensor(value) else None)
            if summary is not None:
                summaries[name] = summary
        return summaries

    def capture(self, stage: str) -> None:
        """Capture global summaries and persist one canonical rank-zero copy."""
        records = []
        for parameter_name, probe in self._probes.items():
            if stage not in probe.stages:
                continue
            parameter = self._named_parameters.get(parameter_name)
            if stage == "optimizer_state":
                summary = self._optimizer_state(parameter)
                if not summary:
                    continue
                record = {"name": parameter_name, "optimizer_state": summary}
            else:
                tensor = None if parameter is None else self._stage_tensor(parameter, stage)
                record = _tensor_summary(parameter_name, tensor)
                if record is None:
                    continue
                if not probe.check_replica_consistency:
                    record.pop("replica", None)
            record.update(
                {
                    "step": self._step,
                    "stage": stage,
                    "rank": self.rank,
                    "comparison": probe.compare,
                }
            )
            records.append(record)
        if self.rank != 0:
            return
        for record in records:
            self.evidence_store.append_jsonl(
                f"parameter_probes/rank{self.rank}.jsonl",
                record,
            )


__all__ = [
    "InputIdentityRecorder",
    "ParameterProbeRecorder",
]
