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
"""Model-owned mappings between Trainer, canonical, and rollout weights."""

from hashlib import sha256
import json
import re
from typing import Any, Mapping, Optional

from rl.roles.model import VLLMModelRegistration
from rl.roles.weight_sync.layout import (
    TensorRegion,
    describe_source_tensor,
    intersect_regions,
)


_PACKED_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<projection>gate_up_proj|down_proj)(?:\.weight)?$"
)


def direct_fragment_record(
    name: str,
    starts: tuple[int, ...],
    lengths: tuple[int, ...],
    dtype_name: str,
    values: bytes,
) -> tuple[str, dict[str, Any]]:
    """Hash one bounded canonical fragment independently of physical storage."""
    header = json.dumps(
        [name, list(starts), list(lengths), dtype_name],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256()
    digest.update(header)
    digest.update(values)
    key = header.decode("utf-8")
    return key, {"sha256": digest.hexdigest(), "num_bytes": len(values)}


def aggregate_direct_content_identity(
    fragments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate bounded fragment hashes into one canonical content identity."""
    if not fragments:
        raise ValueError("Direct content identity requires at least one fragment")
    digest = sha256()
    total_bytes = 0
    normalized = {}
    for key, record in sorted(fragments.items()):
        tensor_digest = str(record["sha256"])
        num_bytes = int(record["num_bytes"])
        if num_bytes <= 0:
            raise ValueError(f"Direct content fragment {key!r} has no bytes")
        digest.update(
            json.dumps(
                [key, tensor_digest, num_bytes],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        normalized[key] = {"sha256": tensor_digest, "num_bytes": num_bytes}
        total_bytes += num_bytes
    return {
        "algorithm": "sha256-canonical-fragments-v1",
        "fragment_count": len(normalized),
        "total_bytes": total_bytes,
        "digest": digest.hexdigest(),
        "fragments": normalized,
    }


class ModelWeightAdapter:
    """Map one registered model family without coupling transport to model names."""

    def __init__(self, model: VLLMModelRegistration) -> None:
        """Resolve the canonical parameter mapping for this model family."""
        self._model = model

    def _actor_name(self, name: str) -> Optional[str]:
        mapper = getattr(self._model, "actor_weight_name", None)
        return mapper(name) if callable(mapper) else name

    def map_local_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Map Trainer state names without changing their physical storage."""
        mapped = {}
        for name, tensor in state_dict.items():
            mapped_name = self._actor_name(name)
            if mapped_name is None:
                continue
            if mapped_name in mapped:
                raise ValueError(
                    f"vLLM policy-name mapping collision: {name!r} maps to {mapped_name!r}"
                )
            mapped[mapped_name] = tensor
        return mapped

    def direct_source_descriptions(
        self,
        state_dict: Mapping[str, Any],
        source_rank: int,
    ) -> list[dict[str, Any]]:
        """Describe canonical tensors while retaining physical source ownership."""
        descriptions = []
        for name, tensor in sorted(state_dict.items()):
            mapped_name = self._actor_name(name)
            if mapped_name is None:
                continue
            description = describe_source_tensor(mapped_name, tensor, source_rank)
            description["source_name"] = mapped_name
            description["source_starts"] = [0] * len(description["global_shape"])
            descriptions.append(description)
        return descriptions


class PackedExpertWeightAdapter(ModelWeightAdapter):
    """Map Transformers packed experts to canonical grouped expert tensors."""

    @staticmethod
    def _packed_slices(
        name: str,
        global_shape: tuple[int, ...],
    ) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
        match = _PACKED_EXPERT_PATTERN.match(name)
        if match is None:
            return ((name, (0,) * len(global_shape), global_shape),)
        if len(global_shape) != 3:
            raise ValueError(
                f"Packed expert {name!r} must be rank 3, got {global_shape}"
            )
        prefix = match.group("prefix")
        projection = match.group("projection")
        if projection == "down_proj":
            return (
                (
                    f"{prefix}.down_proj.weight",
                    (0, 0, 0),
                    global_shape,
                ),
            )
        if global_shape[1] % 2 != 0:
            raise ValueError(
                f"Packed gate/up dimension must be even: {name!r} {global_shape}"
            )
        intermediate_size = global_shape[1] // 2
        canonical_shape = (global_shape[0], intermediate_size, global_shape[2])
        return (
            (f"{prefix}.gate_proj.weight", (0, 0, 0), canonical_shape),
            (
                f"{prefix}.up_proj.weight",
                (0, intermediate_size, 0),
                canonical_shape,
            ),
        )

    @staticmethod
    def _slice_description(
        description: Mapping[str, Any],
        canonical_name: str,
        slice_starts: tuple[int, ...],
        canonical_shape: tuple[int, ...],
    ) -> Optional[dict[str, Any]]:
        physical_shape = tuple(int(value) for value in description["global_shape"])
        if len(slice_starts) != len(physical_shape) or len(canonical_shape) != len(physical_shape):
            raise ValueError(
                f"Canonical expert slice rank differs from physical tensor: {canonical_name!r}"
            )
        if "region_starts" not in description:
            raise ValueError(
                "Packed expert mapping requires explicit source regions"
            )
        physical_region_starts = tuple(
            int(value) for value in description["region_starts"]
        )
        physical_region_lengths = tuple(
            int(value) for value in description["local_shape"]
        )
        intersection = intersect_regions(
            TensorRegion(physical_region_starts, physical_region_lengths),
            TensorRegion(slice_starts, canonical_shape),
        )
        if intersection is None:
            return None
        intersection_starts = intersection.starts
        intersection_lengths = intersection.lengths
        canonical_starts = tuple(
            start - base for start, base in zip(intersection_starts, slice_starts)
        )
        source_starts = tuple(
            start - base
            for start, base in zip(intersection_starts, physical_region_starts)
        )
        return {
            "name": canonical_name,
            "source_name": str(description["name"]),
            "source_starts": list(source_starts),
            "dtype_name": str(description["dtype_name"]),
            "element_size": int(description["element_size"]),
            "global_shape": list(canonical_shape),
            "local_shape": list(intersection_lengths),
            "source_rank": int(description["source_rank"]),
            "shard_dim": None,
            "region_starts": list(canonical_starts),
        }

    def direct_source_descriptions(
        self,
        state_dict: Mapping[str, Any],
        source_rank: int,
    ) -> list[dict[str, Any]]:
        """Describe packed expert slices in canonical grouped coordinates."""
        descriptions = []
        for name, tensor in sorted(state_dict.items()):
            mapped_name = self._actor_name(name)
            if mapped_name is None:
                continue
            physical = describe_source_tensor(mapped_name, tensor, source_rank)
            global_shape = tuple(int(value) for value in physical["global_shape"])
            for canonical_name, starts, canonical_shape in self._packed_slices(
                mapped_name,
                global_shape,
            ):
                mapped = self._slice_description(
                    physical,
                    canonical_name,
                    starts,
                    canonical_shape,
                )
                if mapped is not None:
                    descriptions.append(mapped)
        return descriptions


# Preserve the existing internal import name while both families share one implementation.
DeepseekV3WeightAdapter = PackedExpertWeightAdapter


def build_model_weight_adapter(
    model: VLLMModelRegistration,
) -> ModelWeightAdapter:
    """Return the adapter owned by one registered model family."""
    if model.family in ("deepseek_v3", "qwen3_moe"):
        return PackedExpertWeightAdapter(model)
    if model.family == "qwen3":
        return ModelWeightAdapter(model)
    raise ValueError(f"Unsupported weight-sync model family: {model.family!r}")


__all__ = [
    "DeepseekV3WeightAdapter",
    "ModelWeightAdapter",
    "PackedExpertWeightAdapter",
    "aggregate_direct_content_identity",
    "build_model_weight_adapter",
    "direct_fragment_record",
]
