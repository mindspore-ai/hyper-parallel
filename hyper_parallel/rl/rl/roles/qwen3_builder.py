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
"""RL-owned Qwen3 construction compatibility over the shared model builder.

Actor, Reference, Critic and Hyper-vLLM retain their existing tied-weight
handling here. Main-project callers use HyperAutoModelForCausalLM with the
Qwen3 adapter and explicit DistributedSetup declarations.
"""

from dataclasses import replace
from functools import wraps
from typing import Any

from hyper_parallel.distributed import ShardingPlanner, apply_sharding_plan
# The atomic loader accepts these infrastructure objects at its build boundary.
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.models.registry import get_model_adapter


def _require_qwen3(model: Any) -> None:
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        raise ValueError("Qwen3 runtime adaptation requires config.model_type='qwen3'")


def restore_tied_parameter(model: Any) -> None:
    """Bind Qwen3's tied head to the embedding before FSDP discovers owners."""
    _require_qwen3(model)
    if not model.config.tie_word_embeddings:
        return
    embedding = model.model.embed_tokens.weight
    head = model.lm_head.weight
    if tuple(embedding.shape) != tuple(head.shape):
        raise ValueError("Qwen3 tied parameters must have matching shapes")
    if embedding.requires_grad != head.requires_grad:
        raise ValueError("Qwen3 tied parameters must have matching requires_grad")
    model.lm_head.register_parameter("weight", embedding)


def adapt_materialization(model: Any) -> Any:
    """Preserve tied aliases only on this model instance's to_empty calls."""
    _require_qwen3(model)
    if not model.config.tie_word_embeddings:
        return model
    original_to_empty = model.to_empty

    @wraps(original_to_empty)
    def to_empty(*args: Any, **kwargs: Any) -> Any:
        """Materialize this instance, then restore its tied embedding alias."""
        result = original_to_empty(*args, **kwargs)
        restore_tied_parameter(model)
        return result

    model.to_empty = to_empty
    return model


class _PlannerModelView:
    """Expose alias-inclusive parameter metadata without changing the model."""

    def __init__(self, model: Any) -> None:
        """Store the real model without registering or changing its modules."""
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """Enumerate both Qwen3 embedding and head for the planner's contracts."""
        kwargs.setdefault("remove_duplicate", False)
        return self.model.named_parameters(*args, **kwargs)


class Qwen3ShardingPlanner(ShardingPlanner):
    """Run the shared planner with a Qwen3-only alias-inclusive model view."""

    def plan(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Preserve every parameter FQN when deriving Qwen3's parallel plan."""
        _require_qwen3(model)
        return super().plan(_PlannerModelView(model), *args, **kwargs)


class _Qwen3FSDP2Manager(FSDP2Manager):
    """Restore Qwen3 parameter identity at the existing FSDP entry boundary."""

    def parallelize(self, model: Any, source_shard_info: Any = None) -> Any:
        """Retie this Qwen3 instance before the shared FSDP manager wraps it."""
        restore_tied_parameter(model)
        return super().parallelize(model, source_shard_info=source_shard_info)


def apply_qwen3_sharding_plan(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply the shared plan and restore tied identity for the vLLM model."""
    _require_qwen3(model)
    result, source_info = apply_sharding_plan(model, *args, **kwargs)
    restore_tied_parameter(result)
    return result, source_info


class Qwen3AutoModel(HyperAutoModelForCausalLM):
    """Select scoped adapters while inheriting the complete shared loader."""

    @classmethod
    def _from_pretrained_parent_class(cls, *args: Any, **kwargs: Any) -> Any:
        transform = kwargs.pop("model_transform", None)
        model = super()._from_pretrained_parent_class(*args, **kwargs)
        model = adapt_materialization(model)
        return model if transform is None else transform(model)

    @classmethod
    def _from_config_parent_class(cls, *args: Any, **kwargs: Any) -> Any:
        transform = kwargs.pop("model_transform", None)
        model = adapt_materialization(super()._from_config_parent_class(*args, **kwargs))
        return model if transform is None else transform(model)

    @classmethod
    def _build_model(cls, *args: Any, **kwargs: Any) -> Any:
        """Build Qwen3 with scoped sharding and tied-parameter FSDP adapters."""
        setup = kwargs["distributed_setup"]
        kwargs["sharding_planner"] = Qwen3ShardingPlanner(
            plan_overrides=setup.plan_overrides,
            allow_uncovered_params=getattr(setup, "allow_uncovered_params", False),
        )
        manager = kwargs.get("fsdp2_manager")
        if manager is not None:
            kwargs["fsdp2_manager"] = _Qwen3FSDP2Manager(
                config=manager.config,
                mesh_context=manager.mesh_context,
                fp32_main_params=manager.fp32_main_params,
            )
        return super()._build_model(*args, **kwargs)


def get_module_replacements() -> tuple[ModuleReplacementSpec, ...]:
    """Select shared Qwen3 kernels while keeping the native TP-safe dense MLP."""
    # Resolve optional HF classes only when constructing a Qwen3 model.
    from transformers.models.qwen3.modeling_qwen3 import (  # pylint: disable=C0415
        Qwen3Attention, Qwen3RMSNorm,
    )
    factories = get_model_adapter("qwen3").replacements()
    return (
        ModuleReplacementSpec(
            match=("*.input_layernorm", "*.post_attention_layernorm", "model.norm"),
            module_type=Qwen3RMSNorm, factory=factories.replace_qwen3_rms_norm,
        ),
        ModuleReplacementSpec(
            match=("*.self_attn",), module_type=Qwen3Attention,
            factory=factories.replace_qwen3_flash_attention,
        ),
    )


def get_parallel_overrides() -> dict[str, ModuleShardingSpec]:
    """Declare the fused attention's local-kernel boundary."""
    return {"*.self_attn": ModuleShardingSpec(region_dispatch=False)}


def from_pretrained(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Build an RL Qwen3 using shared model kernels and scoped infrastructure."""
    setup = kwargs.pop("distributed_setup", None) or DistributedSetup()
    setup = replace(
        setup,
        module_replacements=get_module_replacements() + tuple(setup.module_replacements or ()),
        plan_overrides={**get_parallel_overrides(), **(setup.plan_overrides or {})},
    )
    model = Qwen3AutoModel.from_pretrained(pretrained_model_name_or_path, distributed_setup=setup, **kwargs)
    # Live HF weights are converted by replacement factories on a single rank.
    if not hasattr(model, "_hp_used_replacement_weight_conversions"):
        # The shared HF exporter exposes conversion provenance only through these internal attributes.
        # pylint: disable=protected-access
        model._hp_used_replacement_weight_conversions = model._hp_replacement_weight_conversions
    return model


def build_causal_lm(pretrained_model_name_or_path: str, *, fused: bool = True, **kwargs: Any) -> Any:
    """Build a causal Qwen3, optionally retaining the native HF attention interface."""
    loader = from_pretrained if fused else Qwen3AutoModel.from_pretrained
    return loader(pretrained_model_name_or_path, **kwargs)
