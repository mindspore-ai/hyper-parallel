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
from types import SimpleNamespace
from typing import Any

from rl.unified_execution.models import get_model_family, resolve_runtime_symbol


def normalized_compute_setup(plan_overrides: list[dict], mesh_context: Any = None) -> Any:
    """Normalize complete training declarations against the inference mesh."""
    # Optional model runtime imports must not enter config-only processes.
    from hyper_parallel.distributed.mesh import DistributedSetup  # pylint: disable=C0415
    from hyper_parallel.trainer.config import (  # pylint: disable=C0415
        PlanOverride, normalize_distributed_setup_overrides,
    )
    from hyper_parallel.trainer.config.resolver import resolve_component  # pylint: disable=C0415

    entries = deepcopy(plan_overrides)
    entries = resolve_component(entries, annotation=list[PlanOverride], path="train.plan_overrides")
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
    setup = normalized_compute_setup(plan_overrides, getattr(parallel_context, "mesh_context", None))
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
