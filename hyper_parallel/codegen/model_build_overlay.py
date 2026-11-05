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
"""Codegen-owned model observation layered over the native AutoModel API."""
# pylint: disable=forbidden-backend-import

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import random
from typing import Any, Iterator, Optional

import torch

from hyper_parallel.codegen.manager import ensure_codegen_artifact_from_snapshot
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.distributed._builder.applier import (
    _get_active_mesh,
    _preflight_compute_injection,
)
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models import replacement as replacement_api
from hyper_parallel.models.registry import get_model_adapter


def _prepare_model_source_compat(hf_config: Any) -> None:
    """Run an optional model-owned compatibility hook before construction."""
    model_type = getattr(hf_config, "model_type", None)
    adapter = get_model_adapter(model_type) if model_type else None
    loader = getattr(adapter, "codegen", None) if adapter is not None else None
    provider = loader() if loader is not None else None
    prepare = getattr(provider, "prepare_model_import", None)
    if prepare is not None:
        prepare()


@dataclass
class CodegenModelBuildResult:
    """Result populated by one scoped offline model observation."""

    artifact_layout: Optional[Any] = None
    snapshot: Optional[Any] = None


@contextmanager
def _codegen_model_init_context() -> Iterator[None]:
    """Build codegen's observation model with parameters and buffers on meta."""
    with init_empty_weights(include_buffers=True), auto_model._model_init_context(  # pylint: disable=protected-access
        True
    ):
        yield


def _record_source(
    session: CodegenBuildSession,
    model: Any,
    hf_config: Any,
    pretrained_model_name_or_path: Any,
    is_hf_model: bool,
) -> None:
    """Record deterministic source identity from the constructed meta model."""
    model_type = type(model)
    source_path = inspect.getsourcefile(model_type)
    architectures = list(getattr(hf_config, "architectures", None) or ())
    architecture = architectures[0] if architectures else model_type.__name__
    session.record_source({
        "model_type": getattr(hf_config, "model_type", None),
        "architecture": architecture,
        "architectures": architectures,
        "model_class": f"{model_type.__module__}:{model_type.__qualname__}",
        "module_name": model_type.__module__,
        "file_path": source_path,
        "pretrained_model_name_or_path": (
            None if pretrained_model_name_or_path is None else str(pretrained_model_name_or_path)
        ),
        "is_hf_model": bool(is_hf_model),
    })


def _observe_replacements(
    model: Any,
    distributed_setup: Any,
    low_precision_config: Any,
    load_base_model: bool,
    session: CodegenBuildSession,
) -> Any:
    """Apply native replacement declarations while recording their result."""
    weights_mapping = get_model_conversion_mapping(model)
    specs = tuple(getattr(distributed_setup, "module_replacements", None) or ())
    try:
        with _observe_replacement_installations(model, specs) as applications:
            model, _ = replacement_api._apply_module_replacement_actions(  # pylint: disable=protected-access
                model,
                specs,
                weights_mapping=weights_mapping,
                context=model_builder._build_replacement_context(  # pylint: disable=protected-access
                    distributed_setup,
                    low_precision_config,
                ),
                capture_checkpoint_metadata=load_base_model,
            )
    except replacement_api.ModuleReplacementMatchError as exc:
        session.record_replacements(
            declarations=replacement_api.describe_replacement_declarations(specs),
            applications=(),
            unmatched=exc.unmatched,
        )
        raise
    session.record_replacements(
        declarations=replacement_api.describe_replacement_declarations(specs),
        applications=applications,
        unmatched=(),
    )
    return model


@contextmanager
def _observe_replacement_installations(
    model: Any,
    specs: tuple[Any, ...],
) -> Iterator[list[dict[str, Any]]]:
    """Observe native replacement installation without changing its API."""
    applications: list[dict[str, Any]] = []
    spec_indexes = {id(spec): index for index, spec in enumerate(specs)}
    original_install = replacement_api._install_replacement_target  # pylint: disable=protected-access

    def _observe_install(*args: Any, **kwargs: Any) -> None:
        """Record the source and result of the one native installation call."""
        target = args[1]
        transforms = args[3]
        transform_count = len(transforms)
        source_type = _symbol_path(type(target.source))
        original_install(*args, **kwargs)
        replacement = model.get_submodule(target.module_fqns[0])
        applications.append({
            "rule_id": replacement_api._replacement_rule_id(  # pylint: disable=protected-access
                target.spec,
                spec_indexes.get(id(target.spec), 0),
            ),
            "module_fqns": list(target.module_fqns),
            "primary_fqn": target.module_fqns[0],
            "source_type": source_type,
            "replacement_type": _symbol_path(type(replacement)),
            "weight_transforms": [
                _symbol_path(type(transform))
                for transform in transforms[transform_count:]
            ],
        })

    replacement_api._install_replacement_target = _observe_install  # pylint: disable=protected-access
    try:
        yield applications
    finally:
        replacement_api._install_replacement_target = original_install  # pylint: disable=protected-access


def _symbol_path(symbol: Any) -> str:
    """Return a stable import path for an observed runtime type."""
    return f"{symbol.__module__}:{symbol.__qualname__}"


def _observe_plan(
    model: Any,
    mesh: Any,
    sharding_planner: Any,
    is_hf_model: bool,
    session: CodegenBuildSession,
) -> None:
    """Derive a native plan without applying distributed runtime operations."""
    model_sharding_requested = (
        mesh is not None
        and any(size > 1 for size in (mesh.tp_size, mesh.cp_size, mesh.ep_size))
    )
    if sharding_planner is None:
        session.mark_not_required("planner_unavailable")
        return
    if mesh is None:
        session.mark_not_required("mesh_unavailable")
        return
    if mesh.device_mesh is None:
        session.mark_not_required("device_mesh_unavailable")
        return
    if is_hf_model and not model_sharding_requested:
        if session.has_replacement_applications:
            _record_replacement_only_plan(mesh, session)
        else:
            session.mark_not_required("pure_data_parallel")
        return
    plan = sharding_planner.plan(
        model,
        mesh.device_mesh,
        tp_size=mesh.tp_size,
        cp_size=mesh.cp_size,
        ep_size=mesh.ep_size,
        sequence_parallel=mesh.sequence_parallel,
        loss_parallel=mesh.loss_parallel,
    )
    active_mesh = _get_active_mesh(mesh.device_mesh, plan.mesh_dim_names)
    _preflight_compute_injection(plan, active_mesh, model=model)
    session.observe_plan(plan, model, mesh)


def _record_replacement_only_plan(
    mesh: Any,
    session: CodegenBuildSession,
) -> None:
    """Derive an emitter snapshot when replacement is the only source patch."""
    session.record_plan({
        "param_plan": {},
        "boundary_classes": {},
        "boundary_types": {},
        "nested_components": [],
        "mesh_dim_names": [],
        "injections": [],
        "parallel_dims": {
            "dp_size": getattr(mesh, "dp_size", 1) if mesh is not None else 1,
            "tp_size": getattr(mesh, "tp_size", 1) if mesh is not None else 1,
            "cp_size": getattr(mesh, "cp_size", 1) if mesh is not None else 1,
            "pp_size": getattr(mesh, "pp_size", 1) if mesh is not None else 1,
            "ep_size": getattr(mesh, "ep_size", 1) if mesh is not None else 1,
            "sequence_parallel": bool(
                getattr(mesh, "sequence_parallel", False)
            ),
            "loss_parallel": bool(getattr(mesh, "loss_parallel", False)),
        },
    })


def _publish_session(
    session: CodegenBuildSession,
    artifact_dir: Optional[str],
    result: CodegenModelBuildResult,
) -> Any:
    """Publish an offline snapshot while preserving host RNG state."""
    cpu_rng_state = torch.get_rng_state()
    python_rng_state = random.getstate()

    def _publish(snapshot: Any) -> None:
        """Publish one snapshot and restore RNG state afterward."""
        try:
            result.artifact_layout = ensure_codegen_artifact_from_snapshot(snapshot, artifact_dir)
        finally:
            torch.set_rng_state(cpu_rng_state)
            random.setstate(python_rng_state)

    return session.finish(_publish)


def _codegen_build_model(session: CodegenBuildSession, artifact_dir: Optional[str], result: CodegenModelBuildResult):
    """Create the classmethod used only while the CLI constructs a meta model."""
    def _build_model(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model,
        hf_config,
        mesh,
        sharding_planner,
        fsdp2_manager,
        peft_config,
        torch_dtype,
        attn_implementation,
        validate_placement,
        load_base_model,
        distributed_setup=None,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        activation_checkpoint=None,
        activation_checkpoint_selection=None,
        swap_inputs=False,
        allow_cache_entry_mutation=False,
        activation_swap="none",
        model_init_dtype=None,
        **kwargs,
    ) -> Any:
        """Construct and observe a meta model without runtime infrastructure."""
        del (
            fsdp2_manager,
            validate_placement,
            compile_config,
            activation_checkpoint,
            activation_checkpoint_selection,
            swap_inputs,
            allow_cache_entry_mutation,
            activation_swap,
            model_init_dtype,
        )
        init_kwargs = auto_model._model_init_kwargs(  # pylint: disable=protected-access
            dict(kwargs),
            load_base_model=False,
        )
        if "trust_remote_code" in kwargs:
            init_kwargs["trust_remote_code"] = kwargs["trust_remote_code"]
        _prepare_model_source_compat(hf_config)
        with _codegen_model_init_context():
            _, model = auto_model._init_model(  # pylint: disable=protected-access
                cls,
                None,
                hf_config,
                attn_implementation,
                torch_dtype,
                is_hf_model,
                *model_args,
                **init_kwargs,
            )
        _record_source(
            session,
            model,
            hf_config,
            pretrained_model_name_or_path,
            is_hf_model,
        )
        model_builder._apply_pre_sharding_features(  # pylint: disable=protected-access
            peft_config,
            qat_config,
            fp8_config,
        )
        model = _observe_replacements(
            model,
            distributed_setup,
            getattr(distributed_setup, "low_precision_config", None),
            load_base_model,
            session,
        )
        if freeze_config is not None:
            model_builder._apply_parameter_freezing(  # pylint: disable=protected-access
                model,
                freeze_config,
            )
        method_context = model_builder._build_replacement_context(  # pylint: disable=protected-access
            distributed_setup,
            getattr(distributed_setup, "low_precision_config", None),
        )
        method_context["mesh_context"] = getattr(distributed_setup, "mesh_context", None)
        model_builder.apply_method_replacements(
            model,
            getattr(distributed_setup, "method_replacements", None) or (),
            context=method_context,
        )
        _observe_plan(model, mesh, sharding_planner, is_hf_model, session)
        snapshot = _publish_session(session, artifact_dir, result)
        result.snapshot = snapshot
        model._hp_codegen_snapshot = snapshot  # pylint: disable=protected-access
        model.train()
        return model

    return _build_model


@contextmanager
def use_codegen_model_build_overlay(
    session: CodegenBuildSession,
    artifact_dir: Optional[str] = None,
) -> Iterator[CodegenModelBuildResult]:
    """Temporarily replace AutoModel's final build step for offline codegen."""
    model_class = auto_model._BaseHyperAutoModelClass  # pylint: disable=protected-access
    original = model_class.__dict__["_build_model"]
    result = CodegenModelBuildResult()
    model_class._build_model = classmethod(  # pylint: disable=protected-access
        _codegen_build_model(session, artifact_dir, result)
    )
    try:
        yield result
    finally:
        model_class._build_model = original  # pylint: disable=protected-access


__all__ = ["CodegenModelBuildResult", "use_codegen_model_build_overlay"]
