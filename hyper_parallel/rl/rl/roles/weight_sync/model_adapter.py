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

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

from rl.roles.model_setup import VLLMModelRegistration
from rl.roles.weight_sync.layout import describe_source_tensor


@dataclass(frozen=True)
class _WeightRows:
    """One fused QKV row interval and its canonical HF location."""

    name: str
    source_start: int
    target_start: int
    length: int
    target_rows: int


def _canonical_weight_rows(name: str, config: Any) -> tuple[_WeightRows, ...]:
    """Map fused QKV rows to canonical Qwen3 parameter names."""
    groups = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    sizes = (int(config.num_attention_heads) // groups * head_dim, head_dim, head_dim)
    prefix = name.removesuffix("linear_qkv.weight")
    rows = []
    for group in range(groups):
        offset = group * sum(sizes)
        for projection, size in zip(("q_proj", "k_proj", "v_proj"), sizes):
            rows.append(_WeightRows(prefix + projection + ".weight", offset, group * size, size, groups * size))
            offset += size
    return tuple(rows)


class ModelWeightAdapter:
    """Map one registered model family without coupling transport to model names."""

    def __init__(self, model: VLLMModelRegistration) -> None:
        """Resolve the canonical parameter mapping for this model family."""
        self._model = model
        self._source_config = None

    def bind_source(self, payload: Any) -> None:
        """Read model configuration without retaining the Actor or its tensors."""
        config = getattr(payload, "config", None)
        if config is not None:
            self._source_config = config

    def _canonical_rows(self, name: str) -> tuple:
        if not name.endswith("self_attn.linear_qkv.weight"):
            return ()
        if self._source_config is None:
            config_path = Path(self._model.model.weights_path) / "config.json"
            self._source_config = SimpleNamespace(**json.loads(config_path.read_text(encoding="utf-8")))
        return _canonical_weight_rows(name, self._source_config)

    def _expert_slices(self, name: str, shape: Any) -> list[dict[str, Any]]:
        """Validate the public GroupedExperts layout before exposing its slices."""
        if self._model.family != "qwen3_moe" or not _EXPERT_WEIGHT_PATTERN.fullmatch(name):
            return []
        if self._source_config is None:
            config_path = Path(self._model.model.weights_path) / "config.json"
            self._source_config = SimpleNamespace(**json.loads(config_path.read_text(encoding="utf-8")))
        return _expert_weight_slices(name, shape, self._source_config)

    def packed_metadata(self, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach model-owned row conversion to whole-parameter transfer units."""
        result = []
        for entry in metadata:
            expert_slices = self._expert_slices(entry["name"], entry["shape"])
            if expert_slices:
                entry = {**entry, "canonical_experts": expert_slices}
            rows = self._canonical_rows(entry["name"])
            if rows:
                if len(entry["shape"]) != 2 or sum(row.length for row in rows) != entry["shape"][0]:
                    raise ValueError(f"Fused Qwen3 weight {entry['name']!r} has an incompatible shape")
                entry = {**entry, "canonical_rows": [asdict(row) for row in rows]}
            result.append(entry)
        return result

    def map_local_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Map Trainer state names without changing their physical storage."""
        mapped = {}
        for name, tensor in state_dict.items():
            mapped_name = self._model.actor_weight_name(name)
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
            mapped_name = self._model.actor_weight_name(name)
            if mapped_name is None:
                continue
            description = describe_source_tensor(mapped_name, tensor, source_rank)
            description["source_name"] = mapped_name
            description["source_starts"] = [0] * len(description["global_shape"])
            expert_slices = self._expert_slices(mapped_name, description["global_shape"])
            if expert_slices:
                descriptions.extend(_expert_source_regions(description, expert_slices))
                continue
            rows = self._canonical_rows(mapped_name)
            if rows:
                descriptions.extend(_canonical_source_regions(description, rows))
            else:
                descriptions.append(description)
        return descriptions


def _canonical_source_regions(description: dict[str, Any], rows: tuple) -> list[dict[str, Any]]:
    """Intersect physical FSDP/TP shards with the model's canonical row ranges."""
    shape = description["global_shape"]
    starts = description.get("region_starts")
    if len(shape) != 2 or starts is None or sum(row.length for row in rows) != shape[0]:
        raise ValueError(f"Fused Qwen3 source {description['name']!r} requires explicit two-dimensional regions")
    local_rows, local_columns = description["local_shape"]
    result = []
    for row in rows:
        begin = max(starts[0], row.source_start)
        end = min(starts[0] + local_rows, row.source_start + row.length)
        if begin >= end or local_columns == 0:
            continue
        result.append({
            **description,
            "name": row.name,
            "global_shape": [row.target_rows, shape[1]],
            "local_shape": [end - begin, local_columns],
            "region_starts": [row.target_start + begin - row.source_start, starts[1]],
            "source_starts": [begin - starts[0], 0],
            "shard_dim": None,
        })
    return result


_EXPERT_WEIGHT_PATTERN = re.compile(
    r"(?P<prefix>model\.layers\.\d+\.mlp\.experts)\.(?P<projection>gate_up_proj|down_proj)"
)


def _expert_weight_slices(name: str, shape: Any, config: Any) -> list[dict[str, Any]]:
    """Describe individual expert projections in GroupedExperts physical axes."""
    match = _EXPERT_WEIGHT_PATTERN.fullmatch(name)
    if match is None:
        return []
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    is_gate_up = match["projection"] == "gate_up_proj"
    expected = (experts, hidden, 2 * intermediate) if is_gate_up else (experts, intermediate, hidden)
    if tuple(shape) != expected:
        raise ValueError(f"GroupedExperts {name!r} has shape {tuple(shape)}, expected {expected}")
    slices = []
    for expert in range(experts):
        projections = (("gate_proj", 0), ("up_proj", intermediate)) if is_gate_up else (("down_proj", 0),)
        for projection, offset in projections:
            slices.append({
                "name": f"{match['prefix']}.{expert}.{projection}.weight",
                "starts": [expert, 0, offset],
                "shape": [1, hidden, intermediate] if is_gate_up else [1, intermediate, hidden],
            })
    return slices


def _expert_source_regions(description: dict[str, Any], slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Intersect EP/EDP shards without changing physical source tensor axes."""
    physical_starts = description.get("region_starts")
    if physical_starts is None:
        raise ValueError("GroupedExperts publication requires explicit source mesh regions")
    result = []
    for entry in slices:
        starts = [max(left, right) for left, right in zip(physical_starts, entry["starts"])]
        ends = [min(left + length, right + size) for left, length, right, size in zip(
            physical_starts, description["local_shape"], entry["starts"], entry["shape"],
        )]
        lengths = [end - start for start, end in zip(starts, ends)]
        if any(length <= 0 for length in lengths):
            continue
        result.append({
            **description,
            "name": entry["name"],
            "global_shape": entry["shape"],
            "local_shape": lengths,
            "region_starts": [start - base for start, base in zip(starts, entry["starts"])],
            "source_starts": [start - base for start, base in zip(starts, physical_starts)],
            "shard_dim": None,
        })
    return result


def build_model_weight_adapter(
    model: VLLMModelRegistration,
) -> ModelWeightAdapter:
    """Return the adapter owned by one registered model family."""
    if model.family in ("qwen3", "qwen3_moe"):
        return ModelWeightAdapter(model)
    raise ValueError(f"Unsupported weight-sync model family: {model.family!r}")


__all__ = [
    "ModelWeightAdapter",
    "build_model_weight_adapter",
]


def alias_tied_embeddings(
    state_dict: dict[str, Any],
    model: VLLMModelRegistration,
) -> dict[str, Any]:
    """Expose both tied checkpoint names without allocating another tensor."""
    if not model.model.tie_word_embeddings:
        return state_dict
    embedding_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    if embedding_name in state_dict and lm_head_name not in state_dict:
        state_dict[lm_head_name] = state_dict[embedding_name]
    elif lm_head_name in state_dict and embedding_name not in state_dict:
        state_dict[embedding_name] = state_dict[lm_head_name]
    return state_dict


def _direct_tensor_description(
    source_name: str,
    destination_name: str,
    parameter: Any,
    local_shape: tuple[int, ...],
    placement: str,
    shard_dim: Optional[int],
    destination_starts: tuple[int, ...],
    destination_permutation: Optional[tuple[int, ...]] = None,
    accepted_source_dtypes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe one logical Actor tensor region inside a physical parameter."""
    if len(local_shape) != len(parameter.shape):
        raise ValueError(
            f"Native direct tensor {source_name!r} rank mismatch: "
            f"logical={local_shape}, destination={tuple(parameter.shape)}"
        )
    if len(destination_starts) != len(parameter.shape):
        raise ValueError(
            f"Native direct tensor {source_name!r} offset rank mismatch: "
            f"offset={destination_starts}, destination={tuple(parameter.shape)}"
        )
    permutation = destination_permutation or tuple(range(len(local_shape)))
    if sorted(permutation) != list(range(len(local_shape))):
        raise ValueError(
            f"Native direct tensor {source_name!r} has invalid permutation {permutation}"
        )
    physical_lengths = tuple(local_shape[axis] for axis in permutation)
    if any(
        start < 0 or start + length > int(limit)
        for start, length, limit in zip(
            destination_starts,
            physical_lengths,
            parameter.shape,
        )
    ):
        raise ValueError(
            f"Native direct tensor {source_name!r} exceeds {destination_name!r}: "
            f"offset={destination_starts}, logical={local_shape}, "
            f"destination={tuple(parameter.shape)}"
        )
    return {
        "name": source_name,
        "destination_name": destination_name,
        "dtype_name": str(parameter.dtype).rsplit(".", maxsplit=1)[-1],
        "element_size": int(parameter.element_size()),
        "local_shape": list(local_shape),
        "placement": placement,
        "shard_dim": shard_dim,
        "destination_starts": list(destination_starts),
        "destination_permutation": list(permutation),
        "accepted_source_dtypes": list(accepted_source_dtypes),
    }


def _native_qwen3_qkv_descriptions(
    name: str,
    parameter: Any,
    hf_config: Any,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Map native fused QKV storage to the three canonical Actor tensors."""
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(hf_config.num_key_value_heads)
    hidden_size = int(hf_config.hidden_size)
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_heads))
    if num_heads % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 query heads {num_heads} are not divisible by TP {tp_size}"
        )
    q_size = num_heads * head_dim
    q_local_size = q_size // tp_size
    kv_size = num_kv_heads * head_dim
    if num_kv_heads < tp_size:
        if tp_size % num_kv_heads != 0:
            raise ValueError(
                f"Native Qwen3 TP {tp_size} cannot replicate {num_kv_heads} KV heads"
            )
        raise ValueError(
            "Native Qwen3 direct reshard does not support grouped KV-head replication: "
            f"kv_heads={num_kv_heads}, tp_size={tp_size}"
        )
    if num_kv_heads % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 KV heads {num_kv_heads} are not divisible by TP {tp_size}"
        )
    kv_local_size = kv_size // tp_size
    tail_shape = tuple(int(size) for size in parameter.shape[1:])
    expected_shape = (q_local_size + 2 * kv_local_size,) + tail_shape
    if tuple(int(size) for size in parameter.shape) != expected_shape:
        raise ValueError(
            f"Native Qwen3 fused QKV parameter {name!r} has shape "
            f"{tuple(parameter.shape)}, expected {expected_shape}"
        )
    source_suffixes = ("q_proj", "k_proj", "v_proj")
    local_sizes = (q_local_size, kv_local_size, kv_local_size)
    descriptions = []
    destination_offset = 0
    for source_suffix, local_size in zip(source_suffixes, local_sizes):
        source_name = name.replace("qkv_proj", source_suffix)
        local_shape = (local_size,) + tail_shape
        destination_starts = (destination_offset,) + (0,) * len(tail_shape)
        descriptions.append(
            _direct_tensor_description(
                source_name,
                name,
                parameter,
                local_shape,
                "shard",
                0,
                destination_starts,
            )
        )
        destination_offset += local_size
    return descriptions


def _native_qwen3_gate_up_descriptions(
    name: str,
    parameter: Any,
    hf_config: Any,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Map native fused gate/up storage to canonical Actor MLP tensors."""
    intermediate_size = int(hf_config.intermediate_size)
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 intermediate size {intermediate_size} is not divisible by TP {tp_size}"
        )
    local_size = intermediate_size // tp_size
    tail_shape = tuple(int(size) for size in parameter.shape[1:])
    expected_shape = (2 * local_size,) + tail_shape
    if tuple(int(size) for size in parameter.shape) != expected_shape:
        raise ValueError(
            f"Native Qwen3 fused gate/up parameter {name!r} has shape "
            f"{tuple(parameter.shape)}, expected {expected_shape}"
        )
    descriptions = []
    for source_suffix, destination_offset in (
        ("gate_proj", 0),
        ("up_proj", local_size),
    ):
        descriptions.append(
            _direct_tensor_description(
                name.replace("gate_up_proj", source_suffix),
                name,
                parameter,
                (local_size,) + tail_shape,
                "shard",
                0,
                (destination_offset,) + (0,) * len(tail_shape),
            )
        )
    return descriptions


def _native_qwen3_direct_tensors(
    model: Any,
    hf_config: Any,
    tp_rank: int,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Describe native vLLM Qwen3 storage in canonical Actor coordinates."""
    tensors = []
    vocab_size = int(hf_config.vocab_size)
    modules = dict(model.named_modules()) if getattr(hf_config, "model_type", "") == "qwen3_moe" else {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if name.endswith((".experts.w13_weight", ".experts.w2_weight")):
            module_name = name.rsplit(".", 1)[0]
            tensors.extend(_native_expert_descriptions(name, parameter, modules[module_name], hf_config))
            continue
        if ".qkv_proj." in name:
            tensors.extend(
                _native_qwen3_qkv_descriptions(
                    name,
                    parameter,
                    hf_config,
                    tp_size,
                )
            )
            continue
        if ".gate_up_proj." in name:
            tensors.extend(
                _native_qwen3_gate_up_descriptions(
                    name,
                    parameter,
                    hf_config,
                    tp_size,
                )
            )
            continue
        parameter_shape = tuple(int(size) for size in parameter.shape)
        destination_starts = (0,) * len(parameter_shape)
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            partition_size = parameter_shape[0]
            source_start = tp_rank * partition_size
            local_size = max(0, min(partition_size, vocab_size - source_start))
            if local_size <= 0:
                raise ValueError(
                    f"Native Qwen3 vocabulary shard {tp_rank} contains no Actor rows"
                )
            local_shape = (local_size,) + parameter_shape[1:]
            placement = "shard"
            shard_dim = 0
        elif name.endswith(".self_attn.q_proj.weight"):
            local_shape = parameter_shape
            placement = "shard"
            shard_dim = 0
        elif name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight")):
            local_shape = parameter_shape
            placement = "shard"
            shard_dim = 1
        else:
            local_shape = parameter_shape
            placement = "replicate"
            shard_dim = None
        tensors.append(
            _direct_tensor_description(
                name,
                name,
                parameter,
                local_shape,
                placement,
                shard_dim,
                destination_starts,
            )
        )
    return tensors


def _owned_experts(layer: Any, experts: int) -> list[tuple[int, int]]:
    """Validate the global-to-local map and return each physical expert owner."""
    ep_size, tp_size = int(layer.ep_size), int(layer.tp_size)
    if ep_size > 1:
        expert_map = layer.expert_map
        if expert_map is None or int(expert_map.numel()) != experts:
            raise ValueError("Native EP requires a complete global-to-local expert map")
        mapping = [int(value) for value in expert_map.tolist()]
        owned = [(global_id, local_id) for global_id, local_id in enumerate(mapping) if local_id >= 0]
        if any(value < -1 for value in mapping) or sorted(local_id for _, local_id in owned) != list(range(len(owned))):
            raise ValueError("Native expert map must uniquely cover local expert storage")
        if tp_size != 1:
            raise ValueError("Native EP expert tensors must not retain an expert TP dimension")
    else:
        owned = list(enumerate(range(experts)))
    return owned


def _native_expert_descriptions(name: str, parameter: Any, layer: Any, config: Any) -> list[dict[str, Any]]:
    """Map fixed-image Ascend expert runtime storage using the actual expert map."""
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    ep_size = int(layer.ep_size)
    tp_size = int(layer.tp_size)
    tp_rank = int(layer.tp_rank)
    if ep_size <= 0 or tp_size <= 0 or intermediate % tp_size or not 0 <= tp_rank < tp_size:
        raise ValueError("Native expert TP coordinates do not divide the intermediate dimension")
    local_intermediate = intermediate // tp_size
    owned = _owned_experts(layer, experts)
    is_gate_up = name.endswith(".w13_weight")
    expected = ((len(owned), hidden, 2 * local_intermediate) if is_gate_up
                else (len(owned), local_intermediate, hidden))
    if tuple(parameter.shape) != expected:
        raise ValueError(
            f"Native Ascend expert runtime {name!r} has shape {tuple(parameter.shape)}, expected {expected}"
        )
    prefix = name.rsplit(".", 1)[0]
    descriptions = []
    shard_dim = 2 if is_gate_up else 1
    placement = "replicate" if tp_size == 1 else "shard"
    for global_id, local_id in owned:
        projections = (("gate_proj", 0), ("up_proj", local_intermediate)) if is_gate_up else (("down_proj", 0),)
        for projection, offset in projections:
            logical_shape = ((1, hidden, local_intermediate) if is_gate_up else (1, local_intermediate, hidden))
            description = _direct_tensor_description(
                f"{prefix}.{global_id}.{projection}.weight", name, parameter,
                logical_shape, placement, None if tp_size == 1 else shard_dim,
                (local_id, 0, offset),
            )
            description["region_starts"] = ([0, 0, tp_rank * local_intermediate] if is_gate_up
                                             else [0, tp_rank * local_intermediate, 0])
            descriptions.append(description)
    return descriptions


def _hyper_tp_placement(model: Any, name: str, tp_size: int) -> tuple[str, Optional[int]]:
    """Read the public apply pass's parameter placement for the Hyper Qwen3 model."""
    placements = tuple(getattr(model, "_tp_placements", {}).get(name, ()))
    if tp_size == 1 and not placements:
        return "replicate", None
    if len(placements) != 1:
        raise ValueError(f"Direct reshard parameter {name!r} requires one TP placement, got {placements}")
    placement = placements[0]
    if callable(getattr(placement, "is_shard", None)) and placement.is_shard():
        return "shard", int(placement.dim)
    if callable(getattr(placement, "is_replicate", None)) and placement.is_replicate():
        return "replicate", None
    raise ValueError(f"Direct reshard parameter {name!r} has unsupported placement {placement!r}")


def rollout_tensor_descriptions(
    model: Any, hf_config: Any, *, is_hyper: bool, tp_rank: int, tp_size: int,
) -> list[dict[str, Any]]:
    """Describe Native fused storage or public Hyper TP placements."""
    if not is_hyper:
        if hf_config is None:
            raise ValueError("Native Qwen3 direct reshard requires an HF config")
        return _native_qwen3_direct_tensors(model, hf_config, tp_rank, tp_size)
    if not hasattr(model, "_tp_placements"):
        raise ValueError("Hyper direct reshard requires public TP placements")
    tensors = []
    for name, parameter in sorted(model.named_parameters()):
        placement, shard_dim = _hyper_tp_placement(model, name, tp_size)
        tensors.append({
            "name": name,
            "dtype_name": str(parameter.dtype).rsplit(".", maxsplit=1)[-1],
            "element_size": int(parameter.element_size()),
            "local_shape": list(parameter.shape),
            "placement": placement,
            "shard_dim": shard_dim,
        })
    return tensors
