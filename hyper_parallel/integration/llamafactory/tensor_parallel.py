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
"""Tensor Parallel support for LlamaFactory models via HyperParallel shard_module.

This module translates Transformers-style ``base_model_tp_plan`` (e.g. the
colwise / rowwise annotations used in qwen3_vl_moe) into HyperParallel's
``ShardingPlan`` and applies it with ``shard_module``.

Typical colwise / rowwise TP for an ``nn.Linear(in, out)`` whose PyTorch
weight shape is ``[out, in]``:

    colwise  – weight Shard(0) on tp dim, input Replicate, output Shard(-1)
    rowwise  – weight Shard(1) on tp dim, input Shard(-1),  output Replicate (via Partial→reduce)
"""
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from functools import partial

import torch
from torch import nn

from hyper_parallel import DeviceMesh, init_device_mesh, shard_module
from hyper_parallel.core.tensor_parallel import ParallelStyle
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.core.tensor_parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformers-style tp_plan → HyperParallel ShardingPlan translation
# ---------------------------------------------------------------------------

def _match_tp_plan(param_fqn: str, tp_plan: Dict[str, str]) -> Optional[str]:
    """Match a fully-qualified parameter/module name against a wildcard tp_plan.

    Transformers uses ``*`` as a wildcard for layer indices, e.g.
    ``"layers.*.self_attn.q_proj"`` matches ``"layers.0.self_attn.q_proj"``.
    """
    generic_param_name = re.sub(r"\d+", "*", param_fqn)
    for name in tp_plan:
        if generic_param_name.endswith(name):
            return tp_plan[name]
    return None


def build_tp_sharding_plan(
    model: nn.Module,
    tp_plan: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Translate a Transformers-style tp_plan into HyperParallel ShardingPlan components.

    Returns:
        (param_plan, input_plan, output_plan)
    """
    param_plan: Dict[str, Any] = {}
    input_plan: Dict[str, Any] = {}
    output_plan: Dict[str, Any] = {}

    for name, param in model.named_parameters():
        module_name = name.replace(".weight", "").replace(".bias", "")
        if re.sub(r"\d+", "*", module_name) in tp_plan:
            continue
        if name not in param_plan:
            param_plan[name] = (Replicate(),)
            if name.endswith(".weight") or name.endswith(".bias"):
                input_plan[f"{module_name}.input"] = (Replicate(),)
                output_plan[f"{module_name}.output"] = (Replicate(),)

    return param_plan, input_plan, output_plan

# ---------------------------------------------------------------------------
# Activation conversion hooks (local Tensor ↔ DTensor at TP boundaries)
# ---------------------------------------------------------------------------


def build_tp_device_mesh(
    device_type: str,
    dp_size: int,
    tp_size: int,
) -> DeviceMesh:
    """Create a 2-D ``(dp, tp)`` DeviceMesh."""
    return init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))


def _infer_replicate_activation_paths(
    model: nn.Module,
    string_tp_plan: Dict[str, str],
) -> List[str]:
    """Infer non-TP module paths that should carry replicated DTensor activations.

    Instead of enumerating known module classes (which is easy to miss for custom
    architectures), we conservatively hook all *leaf* modules, except TP-plan
    matched Linear modules that already have Colwise/Rowwise styles.
    """
    paths: List[str] = []
    for name, mod in model.named_modules():
        # Skip non-leaf modules; hooks on leaves are enough and avoid redundancy.
        if any(True for _ in mod.children()):
            continue
        # TP-targeted Linear modules already use Colwise/Rowwise wrappers.
        if isinstance(mod, nn.Linear) and _match_tp_plan(name, string_tp_plan) is not None:
            continue
        paths.append(name)
    return paths


def apply_tensor_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    *,
    replicate_activation_paths: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Apply tensor parallelism to a model using HyperParallel's shard_module.

    Args:
        model: The nn.Module (HuggingFace model) to parallelize.
        device_mesh: A DeviceMesh that contains the TP dimension.
        tp_plan: Transformers-style tp_plan dict. If None, auto-detected from
            ``model.config.base_model_tp_plan``.
        tp_mesh_dim: Index of the TP dimension in the device_mesh.

    Returns:
        The model with TP sharding applied (weights distributed, hooks registered).
    """
    parallel_styles: Dict[str, ParallelStyle] = {
        "model.language_model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.language_model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.language_model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.language_model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.language_model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.language_model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.language_model.layers.*.mlp.down_proj": RowwiseParallel(),
    }
    param_plan, input_plan, output_plan = build_tp_sharding_plan(
        model, parallel_styles
    )
    if not param_plan:
        logger.warning("TP plan matched zero parameters; skipping tensor parallelism.")
        return model

    logger.info(
        "Applying tensor parallelism: %d parameters sharded, %d forward hooks.",
        len(param_plan),
        len(input_plan),
    )

    sharding_plan = ShardingPlan(
        plan=param_plan or None,
        input_plan=input_plan or None,
        output_plan=output_plan or None,
    )
    tp_mesh = device_mesh['tp']
    model = shard_module(model, tp_mesh, sharding_plan)
    parallelize_module(model, tp_mesh, parallel_styles)

    return model


def _apply_tensor_parallel(model: nn.Module, hp_args, device_type: str) -> nn.Module:
    """Apply tensor parallelism if configured in hp_args."""
    tp_size = getattr(hp_args, "tp_size", 1)

    if tp_size <= 1:
        return model

    world_size = get_platform().get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({tp_size}).")
    dp_size = world_size // tp_size

    root_mesh = build_tp_device_mesh(device_type, dp_size, tp_size)

    model = apply_tensor_parallel(model, root_mesh)

    return model