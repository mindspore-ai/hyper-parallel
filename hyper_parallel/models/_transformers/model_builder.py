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
"""Internal model construction pipeline (05 §15.2.1).

``instantiate_infrastructure`` builds the ShardingPlanner/FSDP2 adapter from
a resolved ``DistributedSetup``; ``apply_model_infrastructure`` runs the
fixed build order — pre-sharding features -> module replacement ->
plan/apply sharding -> activation checkpoint/swap -> FSDP2 ->
materialize/load -> compile. This module consumes only normalized
AutoModels objects and never imports trainer config (05 §15.2.6).
"""

import logging
from typing import Any, Dict, Literal, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from hyper_parallel import DTensor
from hyper_parallel.core.fully_shard.hsdp_utils import get_hsdp_state
from hyper_parallel.components.checkpoint.weight_conversion import (
    get_model_conversion_mapping,
)

from hyper_parallel.models._transformers.checkpoint_loader import (
    CheckpointManager,
    _finalize_model_loading,
)
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
)
from hyper_parallel.distributed.attention_swap import (
    apply_attention_swap,
    validate_attention_swap,
)
from hyper_parallel.distributed.compile import (
    _resolve_compile_config,
    apply_compile,
)
from hyper_parallel.distributed._builder.fsdp_adapter import (
    FSDP2Manager,
    _apply_fsdp2,
    _instantiate_fsdp2,
)
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.models.registry import _resolve_custom_model_cls
from hyper_parallel.models.replacement import _apply_module_replacement_actions

logger = logging.getLogger(__name__)


def instantiate_infrastructure(
    distributed_setup: Optional[DistributedSetup] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Instantiate distributed infrastructure components.

    Following design doc 01 §8.2.

    Returns:
        (sharding_planner, fsdp2_manager) tuple.
    """
    del kwargs, device
    # ShardingPlanner — already implemented in distributed/_builder.
    # DistributedSetup.plan_overrides is the NORMALIZED
    # ``{match: ModuleShardingSpec}`` mapping: the Trainer desugars its raw
    # YAML PlanOverride entries onto the setup before model construction
    # (05 §15.2.6), so the planner has exactly one override interface.
    plan_overrides = getattr(distributed_setup, "plan_overrides", None)
    sharding_planner = ShardingPlanner(
        plan_overrides=plan_overrides,
        # F4b escape hatch (accuracy_fix_plan.md §2): exploratory debugging
        # only — downgrades the uncovered-trainable-param hard error to a
        # warning. Defaults to fail-fast.
        allow_uncovered_params=getattr(
            distributed_setup, "allow_uncovered_params", False))

    # FSDP2Manager: build from strategy config if available
    fsdp2_manager = None
    mesh = distributed_setup.mesh_context if distributed_setup is not None else None
    strategy_cfg = distributed_setup.strategy_config if distributed_setup is not None else None
    if strategy_cfg is not None:
        fsdp2_manager = _instantiate_fsdp2(
            config=strategy_cfg,
            mesh_context=mesh,
            fp32_main_params=distributed_setup.fp32_main_params,
        )

    if fsdp2_manager is None:
        logger.info("FSDP2Manager: no strategy_config provided; skipping FSDP2 wrap")
    else:
        logger.info("FSDP2Manager instantiated with %s", type(fsdp2_manager.config).__name__)

    return sharding_planner, fsdp2_manager


def _init_model(
    cls,                          # HyperAutoModelForCausalLM etc.
    pretrained_model_name_or_path: Optional[str],
    hf_config,                    # AutoConfig.from_pretrained() result
    attn_implementation: str,
    torch_dtype,
    is_hf_model: bool,
    *model_args,
    backend=None,
    **kwargs,
) -> tuple[bool, PreTrainedModel]:
    """Initialize model — dispatching to custom or HF path.

    Following design doc 01 §7.

    Args:
        cls: The HyperAutoModel* class.
        pretrained_model_name_or_path: HF hub repo ID or local path (None for from_config).
        hf_config: PretrainedConfig from AutoConfig.
        attn_implementation: "sdpa" / "flash_attention_2" / "eager".
        torch_dtype: "auto" / "bfloat16" / etc.
        is_hf_model: True = HF native, False = custom implementation.
        *model_args: Extra positional args for model constructor.
        backend: Backend configuration (reserved for interface compatibility
            with HyperAutoModel.from_pretrained; not used yet).
        **kwargs: Extra keyword args.

    Returns:
        (is_custom_model, model)
    """
    _ = backend  # Reserved for interface compatibility; not used yet.
    architectures = getattr(hf_config, "architectures", []) or []
    arch_name = architectures[0] if architectures else ""

    # ── Path A: HF native ──
    if is_hf_model:
        if pretrained_model_name_or_path is None:
            config_kwargs = dict(kwargs)
            if torch_dtype != "auto":
                config_kwargs["dtype"] = torch_dtype
            config_kwargs["attn_implementation"] = attn_implementation
            model = getattr(cls, "_from_config_parent_class")(
                hf_config,
                **config_kwargs,
            )
        else:
            model = getattr(cls, "_from_pretrained_parent_class")(
                pretrained_model_name_or_path,
                *model_args,
                config=hf_config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        return False, model

    # ── Path B: Custom model implementation ──
    custom_model_cls = _resolve_custom_model_cls(arch_name)
    if custom_model_cls is None:
        # Fallback to HF native
        logger.warning(
            "Custom model class for %s not found; falling back to HF native.",
            arch_name,
        )
        if pretrained_model_name_or_path is None:
            config_kwargs = dict(kwargs)
            if torch_dtype != "auto":
                config_kwargs["dtype"] = torch_dtype
            config_kwargs["attn_implementation"] = attn_implementation
            model = getattr(cls, "_from_config_parent_class")(
                hf_config, **config_kwargs
            )
        else:
            model = getattr(cls, "_from_pretrained_parent_class")(
                pretrained_model_name_or_path,
                *model_args,
                config=hf_config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        return False, model

    # Instantiate custom model
    if pretrained_model_name_or_path is not None:
        model = custom_model_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    else:
        model = custom_model_cls.from_config(
            hf_config,
            *model_args,
            **kwargs,
        )

    return True, model


def _plan_and_apply_sharding(
    model: nn.Module,
    mesh,
    sharding_planner,
    is_hf_model: bool,
    validate_placement: bool,
) -> tuple[nn.Module, Optional[dict]]:
    """Plan parameter layouts and apply dual-mode sharding when requested."""
    model_sharding_requested = (
        mesh is not None
        and any(size > 1 for size in (mesh.tp_size, mesh.cp_size, mesh.ep_size))
    )
    if (
        sharding_planner is None
        or mesh is None
        or (is_hf_model and not model_sharding_requested)
    ):
        return model, None
    if mesh.device_mesh is None:
        logger.warning("MeshContext has no device_mesh; skipping sharding")
        return model, None

    logger.info(
        "Running ShardingPlanner.plan(tp=%d, cp=%d, ep=%d, "
        "sequence_parallel=%s, loss_parallel=%s)",
        mesh.tp_size,
        mesh.cp_size,
        mesh.ep_size,
        mesh.sequence_parallel,
        mesh.loss_parallel,
    )
    plan = sharding_planner.plan(
        model,
        mesh.device_mesh,
        tp_size=mesh.tp_size,
        cp_size=mesh.cp_size,
        ep_size=mesh.ep_size,
        sequence_parallel=mesh.sequence_parallel,
        loss_parallel=mesh.loss_parallel,
    )
    model, source_shard_info = apply_sharding_plan(
        model,
        plan,
        mesh,
        validate_mode=validate_placement,
    )
    logger.info("Sharding plan applied; source_shard_info keys=%d", len(source_shard_info or {}))
    return model, source_shard_info


def _move_model_to_device(
    model: nn.Module,
    is_meta_device: bool,
    device,
) -> nn.Module:
    """Materialize meta parameters or move an initialized model to its device."""
    if device is None:
        return model
    if not is_meta_device:
        model.to(device)
        logger.info("Model moved to %s", device)
        return model

    model.to_empty(device=device)
    return model


def _initialize_model_weights(model: nn.Module) -> None:
    """Initialize materialized state through the model's native contract."""
    for module in model.modules():
        module._is_hf_initialized = False  # pylint: disable=W0212
    for tensor in (*model.parameters(), *model.buffers()):
        tensor._is_hf_initialized = False  # pylint: disable=W0212

    initialize_weights = getattr(model, "initialize_weights", None)
    native_initialization = callable(initialize_weights)
    if callable(initialize_weights):
        initialize_weights()
    else:
        init_weights = getattr(model, "init_weights", None)
        native_initialization = callable(init_weights)
        if callable(init_weights):
            init_weights()
        else:
            for module in model.modules():
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
    if native_initialization:
        for module in model.modules():
            if not getattr(module, "_hp_reset_after_materialization", False):
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
    logger.info("Initialized model state with model-native random initialization")


def _build_replacement_context(
    distributed_setup: Any,
    low_precision_config: Any,
) -> dict[str, Any]:
    """Build the replacement-factory context from the resolved setup."""
    if low_precision_config is None:
        low_precision_config = getattr(
            distributed_setup,
            "low_precision_config",
            None,
        )
    low_precision_enabled = bool(
        getattr(low_precision_config, "enabled", False)
    )
    mesh_context = getattr(distributed_setup, "mesh_context", None)
    return {
        "low_precision": low_precision_config if low_precision_enabled else None,
        "tp": getattr(mesh_context, "tp_size", 1) > 1,
        "cp": getattr(mesh_context, "cp_size", 1) > 1,
        "ep": getattr(mesh_context, "ep_size", 1) > 1,
        "pp": getattr(mesh_context, "pp_size", 1) > 1,
    }


def _apply_pre_sharding_features(
    model: nn.Module,
    peft_config: Optional[Any],
    qat_config: Optional[Any],
    fp8_config: Optional[Any],
) -> None:
    """Apply or report optional features that precede sharding."""
    if peft_config is not None:
        logger.warning("PEFT injection not implemented in stub")
    if qat_config is not None:
        logger.warning("QAT not implemented in stub")
    if fp8_config is not None:
        logger.warning("FP8 not implemented in stub")


def _apply_activation_features(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    activation_swap: str,
    compile_for_execution: bool,
    mesh: Optional[MeshContext],
    swap_inputs: bool = False,
) -> nn.Module:
    """Apply activation checkpointing and attention swap in execution order."""
    if activation_checkpoint not in (None, "off"):
        model = _apply_activation_checkpointing(
            model,
            activation_checkpoint,
            enable_compile=compile_for_execution,
            swap_inputs=swap_inputs,
        )
    validate_attention_swap(
        activation_swap,
        activation_checkpoint=activation_checkpoint,
        enable_compile=compile_for_execution,
        pp_size=getattr(mesh, "pp_size", 1),
    )
    return apply_attention_swap(model, activation_swap) if activation_swap != "none" else model


def _materialize_and_load_model(
    model: nn.Module,
    *,
    is_meta_device: bool,
    device: Optional[torch.device],
    load_base_model: bool,
    pretrained_path: Optional[str],
    weights_mapping: Any,
) -> nn.Module:
    """Materialize model storage and load or initialize meta-device weights."""
    model = _move_model_to_device(model, is_meta_device, device)
    if not is_meta_device:
        return model
    if load_base_model:
        load_report = CheckpointManager(model).load_checkpoint(
            pretrained_path, strict=False, weights_mapping=weights_mapping
        )
        _finalize_model_loading(model, load_report, strict=True)
    else:
        _initialize_model_weights(model)
    return model


def apply_model_infrastructure(
    model: nn.Module,
    mesh: Optional[MeshContext] = None,
    sharding_planner: Optional[ShardingPlanner] = None,
    fsdp2_manager: Optional[FSDP2Manager] = None,
    peft_config: Optional[Any] = None,
    qat_config: Optional[Any] = None,
    fp8_config: Optional[Any] = None,
    freeze_config: Optional[Any] = None,
    compile_config: Optional[Union[CompileConfig, dict]] = None,
    activation_checkpoint: Optional[str] = None,
    activation_swap: str = "none",
    swap_inputs: bool = False,
    is_meta_device: bool = False,
    is_hf_model: bool = False,
    device: Optional[torch.device] = None,
    load_base_model: bool = False,
    pretrained_path: Optional[str] = None,
    validate_placement: bool = False,
    low_precision_config: Optional[Any] = None,
    model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Apply model infrastructure (sharding, recompute, FSDP2, and compile).

    The execution order is: parallel layout -> recompute wrappers -> FSDP2 ->
    materialization/loading -> per-layer compile. Placement validation keeps
    the DTensor placement path and skips compile, while FSDP2 consumes DTensor
    parameter layouts in both modes.
    """

    distributed_setup = kwargs.get("distributed_setup")

    compile_config, compile_for_execution = _resolve_compile_config(
        compile_config, validate_placement, fsdp2_manager
    )
    _apply_pre_sharding_features(
        model, peft_config, qat_config, fp8_config
    )

    # Step 5.5: structure-preserving replacement before plan derivation.
    weights_mapping = get_model_conversion_mapping(model)
    model, weights_mapping = _apply_module_replacement_actions(
        model,
        getattr(distributed_setup, "module_replacements", None),
        weights_mapping=weights_mapping,
        context=_build_replacement_context(distributed_setup, low_precision_config),
        capture_checkpoint_metadata=load_base_model,
    )

    if freeze_config is not None:
        logger.warning("Parameter freezing not implemented in stub")

    # Steps 7-8: plan and apply parameter/activation layouts.
    model, source_shard_info = _plan_and_apply_sharding(
        model,
        mesh,
        sharding_planner,
        is_hf_model,
        validate_placement,
    )

    model = _apply_activation_features(
        model,
        activation_checkpoint,
        activation_swap,
        compile_for_execution,
        mesh,
        swap_inputs=swap_inputs,
    )
    # Step 10: both dual modes use FSDP2. In validate mode the parameters stay
    # as DTensors, and FSDP derives their source layouts directly.
    model = _apply_fsdp2(
        model,
        fsdp2_manager,
        source_shard_info,
    )

    # Steps 11-12: materialize model storage, then load or initialize weights.
    model = _materialize_and_load_model(
        model,
        is_meta_device=is_meta_device,
        device=device,
        load_base_model=load_base_model,
        pretrained_path=pretrained_path,
        weights_mapping=weights_mapping,
    )

    # Final dtype conversion belongs to the atomic build (05 stage-5 item
    # 5): the Trainer never patches model dtype after construction.
    apply_model_init_dtype(model, model_init_dtype)

    # Step 13: compile only the execution model, after FSDP and loading.
    if compile_for_execution:
        model = apply_compile(model, compile_config)

    return model


_MODEL_INIT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_model_init_dtype(
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]],
) -> Optional[torch.dtype]:
    """Resolve the configured final model initialization dtype."""
    if model_init_dtype is None:
        return None
    if model_init_dtype not in _MODEL_INIT_DTYPES:
        raise ValueError(
            "model_init_dtype must be one of float16, bfloat16, float32, "
            f"or null; got {model_init_dtype!r}"
        )
    return _MODEL_INIT_DTYPES[model_init_dtype]


def _model_tensor_identities(model: nn.Module) -> Dict[str, int]:
    """Capture Parameter and DTensor-buffer identities by FQN."""
    identities = {
        f"parameter:{name}": id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    identities.update({
        f"buffer:{name}": id(buffer)
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if isinstance(buffer, DTensor)
    })
    return identities


def _dtensor_layouts(model: nn.Module) -> Dict[int, tuple[Any, tuple[Any, ...]]]:
    """Capture DTensor mesh and placements before dtype conversion."""
    return {
        id(tensor): (tensor.device_mesh, tuple(tensor.placements))
        for tensor in list(model.parameters()) + list(model.buffers())
        if isinstance(tensor, DTensor)
    }


def _refresh_hsdp_precision_state(model: nn.Module) -> None:
    """Refresh FSDP storage and dtype metadata after model conversion."""
    visited_states = set()
    for module in model.modules():
        hsdp_state = get_hsdp_state(module)
        if hsdp_state is None or id(hsdp_state) in visited_states:
            continue
        visited_states.add(id(hsdp_state))
        for hsdp_param in hsdp_state.hsdp_params:
            hsdp_param.reset_sharded_param()
            hsdp_param.init_dtype_attrs(hsdp_state.mp_policy)


def _validate_model_init_dtype(
        model: nn.Module,
        target_dtype: torch.dtype,
) -> None:
    """Validate floating model parameters and buffers after conversion."""
    mismatched = [
        name
        for name, tensor in (
            list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False))
        )
        if tensor.is_floating_point() and tensor.dtype != target_dtype
    ]
    if mismatched:
        raise RuntimeError(
            "Model initialization dtype conversion failed for: "
            f"{', '.join(sorted(mismatched))}"
        )


def apply_model_init_dtype(
        model: nn.Module,
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]],
) -> None:
    """Convert initialized model floating state to the configured final dtype.

    Moved from ``trainer/model_init_dtype.py`` (05 stage-5 item 5): the
    final dtype is part of the atomic model construction, not a Trainer
    post-build patch.

    Args:
        model: Model whose loaded or newly initialized floating state is converted.
        model_init_dtype: Final initialization dtype, or ``None`` for no conversion.
    """
    target_dtype = _resolve_model_init_dtype(model_init_dtype)
    if target_dtype is None:
        return

    identities_before = _model_tensor_identities(model)
    layouts_before = _dtensor_layouts(model)
    swap_on_conversion = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        model.to(dtype=target_dtype)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(swap_on_conversion)

    if _model_tensor_identities(model) != identities_before:
        raise RuntimeError(
            "Model initialization dtype conversion replaced a Parameter or DTensor identity"
        )
    for tensor in list(model.parameters()) + list(model.buffers()):
        previous_layout = layouts_before.get(id(tensor))
        if previous_layout is None:
            continue
        if (
            tensor.device_mesh is not previous_layout[0]
            or tuple(tensor.placements) != previous_layout[1]
        ):
            raise RuntimeError(
                "Model initialization dtype conversion changed a DTensor layout"
            )

    _refresh_hsdp_precision_state(model)
    _validate_model_init_dtype(model, target_dtype)
