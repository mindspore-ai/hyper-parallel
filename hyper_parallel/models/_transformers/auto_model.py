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
"""HyperAutoModel classes — HF-compatible model interface.

Following design doc 01_hf_compatibility_layer.md §6.
Stub — provides from_pretrained/from_config as entry points.
"""

import logging
from dataclasses import is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)

from hyper_parallel.models._transformers.model_builder import (
    _init_model,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models._transformers.config_resolver import get_hf_config
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.codegen.modeling_backend import ModelingBackend, resolve_modeling_backend
from hyper_parallel.models.build_options import get_device_id, get_device_type  # pylint: disable=syntax-error
from hyper_parallel.models.build_options import CompileConfig

logger = logging.getLogger(__name__)


def _has_module_replacements(distributed_setup: Any) -> bool:
    """Return whether the distributed setup declares module replacement."""
    rules = getattr(distributed_setup, "module_replacements", None) or ()
    return bool(tuple(rules))


def _current_device() -> torch.device:
    """Return the current accelerator device, or CPU when no accelerator exists."""
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_device_id())


_CONFIG_ONLY_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
    "use_auth_token",
}


def _model_init_kwargs(
    kwargs: dict[str, Any],
    *,
    load_base_model: bool,
) -> dict[str, Any]:
    """Drop config/download-only kwargs when constructing directly from config."""
    if load_base_model:
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key not in _CONFIG_ONLY_KWARGS
    }


def _with_codegen_model_args(
    config: Any,
    *,
    pretrained_model_name_or_path: str,
    modeling_backend: str,
    codegen_artifact_dir: Optional[str],
    config_overrides: Optional[dict[str, Any]],
) -> Any:
    """Return a config-like object carrying model args needed by generation."""
    yaml_path = getattr(config, "_yaml_path", None)
    model_changes: dict[str, Any] = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
    }
    if codegen_artifact_dir is not None:
        model_changes["codegen_artifact_dir"] = codegen_artifact_dir
    if config_overrides is not None:
        model_changes["config_overrides"] = config_overrides

    if config is None:
        result = SimpleNamespace(
            codegen=True,
            modeling_backend=modeling_backend,
            model=SimpleNamespace(**model_changes),
        )
        if yaml_path is not None:
            result._yaml_path = yaml_path  # pylint: disable=protected-access
        return result

    model = getattr(config, "model", None)
    if hasattr(model, "replace"):
        model = model.replace(**model_changes)
    else:
        model = SimpleNamespace(**model_changes)

    if is_dataclass(config):
        result = replace(
            config,
            model=model,
            codegen=True,
            modeling_backend=modeling_backend,
        )
        if yaml_path is not None:
            result._yaml_path = yaml_path  # pylint: disable=protected-access
        return result
    values = dict(vars(config))
    values.update(
        model=model,
        codegen=True,
        modeling_backend=modeling_backend,
    )
    result = SimpleNamespace(**values)
    if yaml_path is not None:
        result._yaml_path = yaml_path  # pylint: disable=protected-access
    return result


def _prepare_codegen_artifact(
    config: Any,
    *,
    pretrained_model_name_or_path: str,
    modeling_backend: str,
    codegen_artifact_dir: Optional[str],
    config_overrides: Optional[dict[str, Any]],
    hf_config: Any,
) -> str:
    """Generate or reuse the artifact bundle for the gen backend."""
    from hyper_parallel.codegen.manager import (  # pylint: disable=import-outside-toplevel
        ensure_codegen_artifact,
        preflight_integrity_check,
    )

    codegen_config = _with_codegen_model_args(
        config,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        modeling_backend=modeling_backend,
        codegen_artifact_dir=codegen_artifact_dir,
        config_overrides=config_overrides,
    )
    layout = ensure_codegen_artifact(
        codegen_config,
        artifact_dir=codegen_artifact_dir,
        hf_config=hf_config,
    )
    if layout is None:
        raise RuntimeError("codegen is enabled but no artifact layout was produced")
    preflight_integrity_check(
        codegen_config,
        artifact_dir=layout.artifact_dir,
        hf_config=hf_config,
    )
    return layout.artifact_dir


class _BaseHyperAutoModelClass:
    """Shared from_pretrained / from_config logic.

    Following design doc 01 §6.1-6.2.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        """Delegate to the parent HuggingFace AutoModel class.

        Used by the HF-native path in _init_model so that the model is loaded
        through the standard transformers checkpoint logic.
        """
        # Mixin: `super()` resolves to the HF AutoModel base at runtime (see
        # HyperAutoModelForCausalLM etc. below), which provides from_pretrained.
        return super().from_pretrained(*args, **kwargs)  # pylint: disable=E1101

    @classmethod
    def _from_config_parent_class(cls, *args, **kwargs):
        """Delegate to the parent Hugging Face AutoModel ``from_config`` path."""
        return super().from_config(*args, **kwargs)  # pylint: disable=no-member

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        backend: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        force_hf: bool = False,
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        codegen_artifact_dir: Optional[str] = None,
        modeling_backend: Optional[str] = None,
        codegen: bool = False,
        config_overrides: Optional[dict[str, Any]] = None,
        load_base_model: bool = True,
        codegen_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """HF-compatible from_pretrained entry point.

        Following design doc 01 §6.2:
        ① resolve distributed_setup → MeshContext
        ② instantiate_infrastructure → ShardingPlanner + FSDP2Manager
        ③ AutoConfig.from_pretrained → hf_config
        ④ get_is_hf_model → custom/HF path
        ⑤ _build_model → meta + shard + load
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        # ② Instantiate infrastructure
        sharding_planner, fsdp2_manager = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )

        # ③ Get HF config
        config_kwargs = dict(kwargs)
        config_kwargs.update(config_overrides or {})
        hf_config = get_hf_config(
            pretrained_model_name_or_path, attn_implementation, torch_dtype, **config_kwargs
        )
        if modeling_backend is None and backend is not None:
            modeling_backend = getattr(backend, "value", backend)

        # ④ Determine model path
        resolved_backend = resolve_modeling_backend(
            hf_config,
            modeling_backend=modeling_backend,
            force_hf=force_hf,
            codegen_enabled=codegen,
        )
        if resolved_backend is ModelingBackend.GEN and codegen_artifact_dir is None:
            codegen_artifact_dir = _prepare_codegen_artifact(
                codegen_config,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                modeling_backend=resolved_backend.value,
                codegen_artifact_dir=codegen_artifact_dir,
                config_overrides=config_overrides,
                hf_config=hf_config,
            )

        # ⑤ Build model
        return cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=resolved_backend is ModelingBackend.HF,
            hf_config=hf_config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            backend=resolved_backend,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=load_base_model,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            codegen_artifact_dir=codegen_artifact_dir,
            **kwargs,
        )

    @classmethod
    def from_config(  # pylint: disable=unused-argument
        cls,
        config: PretrainedConfig,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        device_mesh: Optional[Any] = None,
        backend: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        modeling_backend: Optional[str] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Build model from PretrainedConfig (no weight loading).

        Following design doc 01 §6.1.
        """
        if modeling_backend in (ModelingBackend.GEN, ModelingBackend.GEN.value):
            raise ValueError(
                "modeling_backend='gen' is only supported by from_pretrained(); "
                "generated models require a prepared codegen artifact"
            )

        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        sharding_planner, fsdp2_manager = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )
        if modeling_backend is None and backend is not None:
            modeling_backend = getattr(backend, "value", backend)

        resolved_backend = resolve_modeling_backend(
            config,
            modeling_backend=modeling_backend,
            force_hf=False,
            codegen_enabled=False,
        )

        return cls._build_model(
            None,
            *model_args,
            is_hf_model=resolved_backend is ModelingBackend.HF,
            hf_config=config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            backend=resolved_backend,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=False,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            **kwargs,
        )

    @classmethod
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
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        codegen_artifact_dir: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Core model building orchestration.

        Following design doc 01 §6.3:
        Step 1: Determine meta device
        Step 2: Build model (meta or real device)
        Step 3-12: apply_model_infrastructure (PEFT, QAT, ShardingPlan,
        activation checkpoint, FSDP2, load, layer compile)
        """
        # Lazy imports: no_init_weights moved between transformers submodules
        # across versions (hence the ImportError fallback).
        # pylint: disable=import-outside-toplevel
        from contextlib import nullcontext
        from transformers.modeling_utils import ContextManagers
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights
        from hyper_parallel import init_empty_weights
        # pylint: enable=import-outside-toplevel

        # Step 1: Determine meta device (inline world-size probe: auto_models
        # must not import the trainer runtime).
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        is_meta_device = (
            world_size > 1
            or backend is ModelingBackend.CUSTOM
            or _has_module_replacements(distributed_setup)
        ) and kwargs.get("quantization_config") is None

        init_ctx = (
            ContextManagers([no_init_weights(), init_empty_weights()])
            if is_meta_device
            else nullcontext()
        )

        # Step 2: Build model
        init_pretrained_path = (
            pretrained_model_name_or_path if load_base_model else None
        )
        init_kwargs = _model_init_kwargs(
            dict(kwargs),
            load_base_model=load_base_model,
        )
        with init_ctx:
            _, model = _init_model(
                cls,
                init_pretrained_path,
                hf_config,
                attn_implementation,
                torch_dtype,
                is_hf_model,
                *model_args,
                backend=backend,
                codegen_artifact_dir=codegen_artifact_dir,
                **init_kwargs,
            )

        # Step 3-12: Apply infrastructure
        infrastructure_is_hf_model = is_hf_model or backend is ModelingBackend.GEN
        model = apply_model_infrastructure(
            model,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            peft_config=peft_config,
            qat_config=qat_config,
            fp8_config=fp8_config,
            freeze_config=freeze_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=infrastructure_is_hf_model,
            device=_current_device(),
            load_base_model=load_base_model,
            pretrained_path=pretrained_model_name_or_path,
            validate_placement=validate_placement,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            codegen_artifact_dir=(
                codegen_artifact_dir if backend is ModelingBackend.GEN else None
            ),
            hf_config=hf_config,
        )

        model.train()
        return model


class HyperAutoModelForCausalLM(_BaseHyperAutoModelClass, AutoModelForCausalLM):
    """Hyper-Parallel CausalLM — equivalent to AutoModelForCausalLM.

    Following design doc 01 §6.1.
    """


class HyperAutoModelForImageTextToText(_BaseHyperAutoModelClass, AutoModelForImageTextToText):
    """Hyper-Parallel VLM — equivalent to AutoModelForImageTextToText."""


class HyperAutoModelForSequenceClassification(_BaseHyperAutoModelClass, AutoModelForSequenceClassification):
    """Hyper-Parallel SequenceClassification."""
