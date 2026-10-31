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

from hyper_parallel.codegen.hash import file_line_count, sha256_file
from hyper_parallel.codegen.manager import ensure_codegen_artifact_from_snapshot
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.replacement import _apply_module_replacement_actions


@dataclass
class CodegenModelBuildResult:
    """Result populated by one scoped offline model observation."""

    artifact_layout: Optional[Any] = None
    snapshot: Optional[Any] = None


def _record_source(
    session: CodegenBuildSession,
    model: Any,
    hf_config: Any,
    pretrained_model_name_or_path: Any,
    backend: Any,
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
        "sha256": sha256_file(source_path) if source_path else None,
        "line_count": file_line_count(source_path) if source_path else None,
        "pretrained_model_name_or_path": (
            None if pretrained_model_name_or_path is None else str(pretrained_model_name_or_path)
        ),
        "is_hf_model": bool(is_hf_model),
        "backend": str(getattr(backend, "value", backend) or "native"),
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
    model, _ = _apply_module_replacement_actions(
        model,
        getattr(distributed_setup, "module_replacements", None),
        weights_mapping=weights_mapping,
        context=model_builder._build_replacement_context(  # pylint: disable=protected-access
            distributed_setup,
            low_precision_config,
        ),
        capture_checkpoint_metadata=load_base_model,
        observation_callback=session.record_replacements,
    )
    return model


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
    if is_hf_model and not model_sharding_requested:
        session.mark_not_required("pure_data_parallel")
        return
    if mesh.device_mesh is None:
        session.mark_not_required("device_mesh_unavailable")
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
    session.observe_plan(plan, model, mesh)
    session.observe_plan_application(plan, None, validate_mode=False)


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
        backend,
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
        with auto_model._model_init_context(True):  # pylint: disable=protected-access
            _, model = auto_model._init_model(  # pylint: disable=protected-access
                cls,
                None,
                hf_config,
                attn_implementation,
                torch_dtype,
                is_hf_model,
                *model_args,
                backend=backend,
                **init_kwargs,
            )
        _record_source(
            session,
            model,
            hf_config,
            pretrained_model_name_or_path,
            backend,
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
