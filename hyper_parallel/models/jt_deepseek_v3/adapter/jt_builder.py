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

from functools import partial
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu
from transformers import PreTrainedModel

from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, local_compute
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.models.build_options import FSDP2Config, get_device_id
from hyper_parallel.models._transformers.model_builder import (
    _build_replacement_context,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models.jt_deepseek_v3.configuration_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import configure_context_parallel
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM,
)
from hyper_parallel.models.replacement import _apply_module_replacement_actions


@local_compute
def build_jt_ep(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind public EP execution using the JT model routing contract."""
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError("JT requires an EP mesh")
    module.ep_group = ep_mesh.get_group("ep")
    module.ep_world = ep_mesh["ep"].size()
    executor = build_ep_compute(
        module,
        ep_mesh,
        router_fn=type(module).route,
        archetype_key="jt_deepseek_v3_hf",
        expected_attrs=["gate", "experts", "shared_experts", "config"],
        combine=module.combine_routed,
        use_grouped_gemm=True,
    )
    module.ep_compute = partial(executor, module)
    return type(module).forward


def _configured_moe_fqns(config: JTDeepseekV3Config) -> tuple[str, ...]:
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


def _with_model_ep_overrides(distributed_setup: Any, config: JTDeepseekV3Config) -> Any:
    """Add only missing configured MoE EP factories as explicit model FQNs."""
    overrides = dict(getattr(distributed_setup, "plan_overrides", None) or {})
    for fqn in _configured_moe_fqns(config):
        if _has_explicit_ep_override(overrides, fqn):
            continue
        if fqn in overrides:
            overrides[fqn] = replace(
                overrides[fqn],
                local_compute_fn=build_jt_ep,
                region_dispatch=(
                    False if overrides[fqn].region_dispatch is None
                    else overrides[fqn].region_dispatch
                ),
            )
            continue
        overrides[fqn] = ModuleShardingSpec(
            local_compute_fn=build_jt_ep,
            region_dispatch=False,
        )
    return replace(distributed_setup, plan_overrides=overrides)


def _load_reference_state(model: PreTrainedModel, arrays: dict[str, np.ndarray]) -> dict:
    """Load already-converted recipe weights without renaming or reshaping tensors."""
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
    return expected


def build_jt_model(*, config: dict[str, Any], reference_weights: str | Path,
                    distributed_setup: Any, **infrastructure_options: Any) -> PreTrainedModel:
    """Load the native JT model and an offline-converted model.npz artifact."""

    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    config = JTDeepseekV3Config(**config)
    setup = _with_model_ep_overrides(distributed_setup, config)
    mesh = setup.mesh_context
    # Source-layout FSDP owns parameters and gradient synchronization even at DP1.
    framework_setup = replace(
        setup, module_replacements=(), strategy_config=setup.strategy_config or FSDP2Config(),
    )
    planner, fsdp = instantiate_infrastructure(distributed_setup=framework_setup)
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
    with np.load(Path(reference_weights) / "model.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    expected = _load_reference_state(model, arrays)

    device = torch.device(mesh.device_mesh.device_type, get_device_id())
    model.to(device)
    model.loss_group = mesh.device_mesh["tp"].get_group()
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
    configure_context_parallel(model, mesh)
    dp_cp_mesh = mesh.dp_cp_mesh
    # QK clipping must use one maximum on every rank that holds the same attention heads.
    model.qk_clip_group = None if dp_cp_mesh is None or dp_cp_mesh.size() == 1 else dp_cp_mesh.get_group()
    model.build_report = {
        "model_class": type(model).__name__,
        "loaded_state_tensors": len(expected),
        "all_loaded_values_exact": True,
    }
    return model
