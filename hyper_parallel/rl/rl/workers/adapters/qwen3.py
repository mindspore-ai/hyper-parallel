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

Actor, Reference and Critic retain their existing tied-weight
handling here. Main-project callers use HyperAutoModelForCausalLM with the
Qwen3 adapter and explicit DistributedSetup declarations.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from hyper_parallel.components.checkpoint.load_groups import target_layout
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.distributed import ShardingPlanner
# The atomic loader accepts these infrastructure objects at its build boundary.
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed._builder.source_shard import build_source_shard_info
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup


def _require_qwen3(model: Any) -> None:
    if getattr(getattr(model, "config", None), "model_type", None) not in ("qwen3", "qwen3_moe"):
        raise ValueError("Qwen3 runtime adaptation requires a supported Qwen3 family config")


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
    """Register RL-owned derived buffers and preserve this instance's tied aliases."""
    from rl.workers.engine.materialization import register_qwen3_materialization  # pylint: disable=C0415

    _require_qwen3(model)
    register_qwen3_materialization(model)
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep the final shared plan for parameter metadata after materialization."""
        super().__init__(*args, **kwargs)
        self.finalized_plan = None

    def plan(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Preserve every parameter FQN when deriving Qwen3's parallel plan."""
        _require_qwen3(model)
        # Preserve logical dimensions before production TP replaces parameters
        # by local tensors. Architecture audits must not depend on FSDP residency.
        model._rl_logical_parameter_shapes = {  # pylint: disable=protected-access
            name: tuple(parameter.shape) for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        for module in model.modules():
            make_transforms = getattr(module, "make_transforms", None)
            if callable(make_transforms):
                # TP may divide module attributes such as KV head count. Weight
                # conversion remains a mapping of full checkpoint coordinates.
                module._rl_logical_weight_transforms = make_transforms()  # pylint: disable=protected-access
        plan = super().plan(_PlannerModelView(model), *args, **kwargs)
        self.finalized_plan = plan
        modules = dict(model.named_modules())
        for name, spec in plan.modules.items():
            # External schedulers enter/leave the original model at these
            # boundaries. Keep the planner's declarations, never infer SP.
            modules[name]._rl_input_layouts = dict(spec.in_src or {})  # pylint: disable=protected-access
            outputs = spec.out_dst or spec.out_src or {}
            modules[name]._rl_output_layout = outputs.get("output")  # pylint: disable=protected-access
        return plan

    def retain_parameter_layouts(self, model: Any, mesh_context: Any) -> None:
        """Retain shared source layouts when no FSDP manager consumes them."""
        mesh = getattr(mesh_context, "device_mesh", None)
        if self.finalized_plan is None or mesh is None:
            return
        sources = build_source_shard_info(
            self.finalized_plan, dense_source_mesh=mesh,
            expert_source_mesh=getattr(mesh_context, "fsdp_moe_mesh", None),
        )
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if target_layout(parameter) is not None or name not in sources:
                continue
            placements, source_mesh = sources[name]
            if source_mesh is None:
                continue
            layout = Layout.from_device_mesh(source_mesh)
            layout.set_placements(placements)
            layout.placement_to_tensor_map(parameter.ndim)
            parameter._sharding_spec = layout  # pylint: disable=protected-access


class _Qwen3FSDP2Manager(FSDP2Manager):
    """Restore Qwen3 parameter identity at the existing FSDP entry boundary."""

    def parallelize(self, model: Any, source_shard_info: Any = None,
                    ignored_params: set[Any] | None = None) -> Any:
        """Retie this Qwen3 instance before the shared FSDP manager wraps it."""
        restore_tied_parameter(model)
        return super().parallelize(model, source_shard_info=source_shard_info, ignored_params=ignored_params)


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
        planner = Qwen3ShardingPlanner(
            plan_overrides=setup.plan_overrides,
            allow_uncovered_params=getattr(setup, "allow_uncovered_params", False),
        )
        kwargs["sharding_planner"] = planner
        manager = kwargs.get("fsdp2_manager")
        if manager is not None:
            kwargs["fsdp2_manager"] = _Qwen3FSDP2Manager(
                config=manager.config,
                mesh_context=manager.mesh_context,
                fp32_main_params=manager.fp32_main_params,
            )
        model = super()._build_model(*args, **kwargs)
        if manager is None:
            planner.retain_parameter_layouts(model, setup.mesh_context)
        return model


def from_pretrained(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Build an RL Qwen3 using shared model kernels and scoped infrastructure."""
    setup = kwargs.pop("distributed_setup", None) or DistributedSetup()
    model = Qwen3AutoModel.from_pretrained(pretrained_model_name_or_path, distributed_setup=setup, **kwargs)
    # Live HF weights are converted by replacement factories on a single rank.
    if not hasattr(model, "_hp_used_replacement_weight_conversions"):
        # The shared HF exporter exposes conversion provenance only through these internal attributes.
        # pylint: disable=protected-access
        model._hp_used_replacement_weight_conversions = getattr(model, "_hp_replacement_weight_conversions", [])
    return model


def build_causal_lm(pretrained_model_name_or_path: str, *, fused: bool = True, **kwargs: Any) -> Any:
    """Build a causal Qwen3, optionally retaining the native HF attention interface."""
    loader = from_pretrained if fused else Qwen3AutoModel.from_pretrained
    return loader(pretrained_model_name_or_path, **kwargs)
