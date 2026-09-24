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
"""Configured model and exported weights adapted to Hyper's model build pipeline."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import DeepseekV32Config, PreTrainedModel

from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models._transformers.model_builder import (
    _build_replacement_context,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM,
)
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.jt_expert_parallel import (
    jt_deepseek_v3_ep_compute,
)
from hyper_parallel.models.jt_deepseek_v3.adapter.conversion.checkpoint_mapping import convert_reference


def _configured_moe_fqns(config: DeepseekV32Config) -> tuple[str, ...]:
    """Return the configured routed-MoE parents, excluding dense trunk layers."""
    trunk = tuple(
        f"model.layers.{index}.mlp"
        for index, layer_type in enumerate(config.mlp_layer_types) if layer_type == "sparse"
    )
    mtp = tuple(
        f"mtp.layers.{index}.transformer_layer.mlp"
        for index in range(config.num_nextn_predict_layers)
    )
    return trunk + mtp


def _has_explicit_ep_override(overrides: dict[str, Any], fqn: str) -> bool:
    """Let a user-selected EP compute factory win over the reference default."""
    return any(
        (key == fqn or (any(char in key for char in "*?[") and fnmatchcase(fqn, key)))
        and getattr(spec, "local_compute_fn", None) is not None
        for key, spec in overrides.items()
    )


def _with_model_ep_overrides(distributed_setup: Any, config: DeepseekV32Config) -> Any:
    """Add only missing configured MoE EP factories as explicit model FQNs."""
    overrides = dict(getattr(distributed_setup, "plan_overrides", None) or {})
    for fqn in _configured_moe_fqns(config):
        if _has_explicit_ep_override(overrides, fqn):
            continue
        if fqn in overrides:
            overrides[fqn] = replace(
                overrides[fqn],
                local_compute_fn=jt_deepseek_v3_ep_compute,
                region_dispatch=(
                    False if overrides[fqn].region_dispatch is None
                    else overrides[fqn].region_dispatch
                ),
            )
            continue
        overrides[fqn] = ModuleShardingSpec(
            local_compute_fn=jt_deepseek_v3_ep_compute,
            region_dispatch=False,
        )
    return replace(distributed_setup, plan_overrides=overrides)


def _load_reference_state(model: PreTrainedModel, arrays: dict[str, np.ndarray]) -> tuple[dict, list]:
    """Fuse only replaced MLA projections and verify complete lossless loading."""
    logical_groups = []
    for target in model.state_dict():
        suffix = "linear_qkv.weight"
        if not target.endswith(f".{suffix}"):
            continue
        prefix = target[:-len(suffix)]
        source_names = [f"{prefix}q_a_proj.weight", f"{prefix}kv_a_proj_with_mqa.weight"]
        values = [arrays.pop(name) for name in source_names]
        arrays[target] = np.concatenate(values, axis=0)
        restored = np.split(arrays[target], [values[0].shape[0]], axis=0)
        if any(left.tobytes() != right.tobytes() for left, right in zip(values, restored)):
            raise ValueError(f"Non-invertible MLA fusion: {target}")
        logical_groups.append({
            "storage": target,
            "logical_parameters": source_names,
            "sections": [value.shape[0] for value in values],
        })

    expected = model.state_dict()
    if set(expected) != set(arrays):
        raise ValueError(
            f"State coverage mismatch: missing={set(expected) - set(arrays)}, "
            f"unexpected={set(arrays) - set(expected)}",
        )
    for name, value in arrays.items():
        if tuple(expected[name].shape) != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {expected[name].shape} != {value.shape}")
        if torch.from_numpy(value).dtype != expected[name].dtype:
            raise ValueError(
                f"Reference dtype mismatch for {name}: "
                f"{torch.from_numpy(value).dtype} != {expected[name].dtype}",
            )
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        strict=True,
    )
    for name, value in model.state_dict().items():
        if value.detach().numpy().tobytes() != arrays[name].tobytes():
            raise ValueError(f"Loaded tensor differs: {name}")
    if model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError("JT embedding and LM head must not be tied")

    return expected, logical_groups


def build_jt_model(*, config: dict[str, Any], reference_weights: str | Path, source_tp_size: int,
                    distributed_setup: Any, **infrastructure_options: Any) -> PreTrainedModel:
    """Load the complete model before applying Hyper's parallel infrastructure.

    Args:
        config: Native HF/JT fields supplied directly by the Hyper recipe.
        source_tp_size: Number of tensor-parallel shards in the exported weights.
        reference_weights: Directory of exported initial parameter arrays.
        distributed_setup: Resolved mesh, replacements and sharding declarations.
    """
    if infrastructure_options.get("model_init_dtype") not in (None, "float32"):
        raise ValueError("JT precision requires FP32 master parameters")
    infrastructure_options["model_init_dtype"] = "float32"

    config = DeepseekV32Config(**config)
    arrays, _, conversion = convert_reference(reference_weights, config, source_tp_size=source_tp_size)
    setup = _with_model_ep_overrides(distributed_setup, config)
    mesh = setup.mesh_context
    if (mesh.tp_size, mesh.ep_size, mesh.cp_size, mesh.dp_size, mesh.pp_size) != (8, 8, 1, 1, 1):
        raise ValueError("JT recipe requires TP8/EP8 and DP/CP/PP1")
    if not mesh.sequence_parallel or not mesh.loss_parallel:
        raise ValueError("JT recipe requires sequence_parallel and loss_parallel")

    with torch.device("meta"):
        model = JTDeepseekV3ForCausalLM(config)
        model, _ = _apply_module_replacement_actions(
            model,
            getattr(setup, "module_replacements", None),
            weights_mapping=get_model_conversion_mapping(model),
            context=_build_replacement_context(setup, None),
        )
    model.to_empty(device="cpu")
    # Rotary buffers are nonpersistent; restore their deterministic reference state after to_empty().
    model.model.rotary_emb = type(model.model.rotary_emb)(config)

    expected, logical_groups = _load_reference_state(model, arrays)

    framework_setup = replace(setup, module_replacements=())

    # Replacements have already shaped the reference state, so do not apply them a second time.
    planner, fsdp = instantiate_infrastructure(distributed_setup=framework_setup)
    device = torch.device(mesh.device_mesh.device_type, torch.distributed.get_rank() % mesh.tp_size)
    model.to(device)
    model.loss_group = mesh.device_mesh["tp"].get_group()
    global_shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    model = apply_model_infrastructure(
        model,
        mesh=mesh,
        sharding_planner=planner,
        fsdp2_manager=fsdp,
        distributed_setup=framework_setup,
        device=device,
        is_meta_device=False,
        is_hf_model=True,
        **infrastructure_options,
    )
    model.jt_replicated_names = tuple(
        name for name, value in model.named_parameters() if tuple(value.shape) == global_shapes[name]
    )
    model.build_report = {
        **conversion,
        "model_class": type(model).__name__,
        "loaded_state_tensors": len(expected),
        "logical_optimizer_groups": logical_groups,
        "all_loaded_values_exact": True,
    }
    return model
