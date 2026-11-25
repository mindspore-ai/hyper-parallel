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
"""Construct the existing HP model using the shared explicit replacement plan."""

from copy import deepcopy
from fnmatch import fnmatchcase
from types import SimpleNamespace
from typing import Any

from rl.unified_execution.execution_plan import EXPERTS_TARGET, ROLLOUT_EXPERTS_TARGET
from rl.unified_execution.models import get_model_family, resolve_runtime_symbol


def normalized_compute_setup(plan_overrides: list[dict], mesh_context: Any = None) -> Any:
    """Normalize complete training declarations against the inference mesh."""
    # Optional model runtime imports must not enter config-only processes.
    from hyper_parallel.distributed.mesh import DistributedSetup  # pylint: disable=C0415
    from hyper_parallel.trainer.config import (  # pylint: disable=C0415
        PlanOverride,
        normalize_distributed_setup_overrides,
    )
    from hyper_parallel.trainer.config.resolver import resolve_component  # pylint: disable=C0415

    entries = resolve_component(plan_overrides, annotation=list[PlanOverride], path="train.plan_overrides")
    setup = DistributedSetup() if mesh_context is None else DistributedSetup(mesh_context=mesh_context)
    normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=entries))
    return setup


def build_inference_model(hf_config: Any, plan_overrides: list[dict], *,
                          attn_implementation: str = "eager", parallel_context: Any = None) -> Any:
    """Build the training model through its complete HP parallel construction path.

    Args:
        hf_config: vLLM's resolved Qwen3 configuration.
        plan_overrides: Complete Actor YAML plan, including parameter and activation layouts.
        attn_implementation: Initial HF attention selection before the cache bridge is installed.
        parallel_context: Worker-local HP compute meshes.

    Returns:
        Independent causal LM with the training-side HP sharding and redistribution implementation.
    """
    # HP model construction is optional in config and plugin-registration processes.
    import torch  # pylint: disable=C0415
    family = get_model_family(getattr(hf_config, "model_type", None))
    model_class = resolve_runtime_symbol(family.builder_target)
    if getattr(hf_config, "attention_dropout", 0.0) != 0.0:
        raise ValueError("Unified Qwen3 requires attention_dropout=0")
    entries = deepcopy(plan_overrides)
    replicated_expert_parents = []
    for entry in entries:
        action = entry.get("replace", {})
        implementation = action.get("implementation")
        target = implementation.get("_target_") if isinstance(implementation, dict) else implementation
        if target == EXPERTS_TARGET:
            action["implementation"] = {"_target_": ROLLOUT_EXPERTS_TARGET}
            if (getattr(parallel_context, "ep_size", 1) > 1
                    and not getattr(parallel_context, "sequence_parallel", False)):
                patterns = entry["match"] if isinstance(entry["match"], list) else [entry["match"]]
                # HP's training TP-extend-EP template splits router input rows even
                # without dense SP. Rollout's no-SP experts consume replicated rows.
                # Match only the parents of explicitly selected routed experts.
                for index in range(hf_config.num_hidden_layers):
                    if (index in hf_config.mlp_only_layers
                            or (index + 1) % hf_config.decoder_sparse_step):
                        continue
                    expert_path = f"model.layers.{index}.mlp.experts"
                    if any(fnmatchcase(expert_path, pattern) for pattern in patterns):
                        replicated_expert_parents.append(expert_path.removesuffix(".experts"))
    for parent in replicated_expert_parents:
        entries.append({
            "match": parent,
            "in_dst": {"hidden_states": {"tp": "replicate", "cp": "shard(1)"}},
            "out_src": {"output": {"tp": "replicate", "cp": "shard(1)"}},
        })
    setup = normalized_compute_setup(entries, getattr(parallel_context, "mesh_context", None))
    # The public builder uses the device already selected by this worker and
    # shares replacements, parameter shards and forward wrappers with training.
    # vLLM supplies checkpoint tensors later through its load_weights protocol.
    model = model_class.from_config(
        deepcopy(hf_config), distributed_setup=setup,
        attn_implementation=attn_implementation, torch_dtype=torch.get_default_dtype(),
        validate_placement=False,
    )
    model.eval()
    return model
