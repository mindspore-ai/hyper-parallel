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
"""instantiate_infrastructure + apply_model_infrastructure.

Following design doc 01_hf_compatibility_layer.md §8.
Stub — creates ShardingPlanner, FSDP2Manager, and AutoPipeline.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming

from hyper_models._transformers.checkpoint_loader import CheckpointManager, LoadReport
from hyper_models.components.activation_checkpoint import (
    _apply_activation_checkpointing as _apply_activation_checkpointing_impl,
)
from hyper_models.components.activation_swap.attention_swap import (
    apply_attention_swap,
    validate_attention_swap,
)
from hyper_models.components.compile import apply_compile
from hyper_models.components.distributed.fsdp2 import FSDP2Manager, _instantiate_fsdp2
from hyper_models.components.distributed.infrastructure import DistributedSetup, MeshContext
from hyper_models.components.distributed.pipelining import AutoPipeline, _instantiate_pipeline
from hyper_models.components.distributed.sharding_applier import apply_sharding_plan
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_models.trainer.config import (
    CompileConfig,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)
from hyper_parallel import DTensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FinalizeTarget:
    """Registered model tensor considered during pretrained finalization."""

    fqn: str
    module: nn.Module
    tensor_name: str
    tensor: torch.Tensor
    is_parameter: bool
    is_non_persistent: bool


@dataclass(frozen=True)
class _TargetSnapshot:
    """Identity and layout invariants for a loaded model tensor."""

    tensor_id: int
    local_tensor_id: int
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout_id: int | None


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    enable_compile: bool = False,
) -> nn.Module:
    """Forward activation checkpointing to the distributed implementation."""
    return _apply_activation_checkpointing_impl(
        model,
        activation_checkpoint,
        enable_compile=enable_compile,
    )


def instantiate_infrastructure(
    distributed_setup: Optional[DistributedSetup] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Instantiate distributed infrastructure components.

    Following design doc 01 §8.2.

    Returns:
        (sharding_planner, fsdp2_manager, autopipeline) tuple.
    """
    del kwargs, device
    # ShardingPlanner — already implemented in components/distributed.
    # plan_overrides come from the resolved TrainerConfig (YAML
    # plan_overrides → List[PlanOverride]) via DistributedSetup; they are
    # desugared HERE (placement DSL → objects, when-filtered against the
    # accelerator topology) so the planner has exactly one override
    # interface.
    entries = getattr(distributed_setup, "plan_overrides", None) or None
    if entries is not None:
        mesh_ctx = getattr(distributed_setup, "mesh_context", None)
        plan_overrides = entries_to_plan_overrides(
            entries,
            cp_size=getattr(mesh_ctx, "cp_size", 1),
            ep_size=getattr(mesh_ctx, "ep_size", 1),
            low_precision_enabled=bool(
                getattr(
                    getattr(distributed_setup, "low_precision_config", None),
                    "enabled",
                    False,
                )
            ),
        )
    else:
        plan_overrides = None
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
        fsdp2_manager = _instantiate_fsdp2(config=strategy_cfg, mesh_context=mesh)

    if fsdp2_manager is None:
        logger.info("FSDP2Manager: no strategy_config provided; skipping FSDP2 wrap")
    else:
        logger.info("FSDP2Manager instantiated with %s", type(fsdp2_manager.config).__name__)

    # AutoPipeline: only when pp_size > 1
    autopipeline = None
    pipeline_cfg = distributed_setup.pipeline_config if distributed_setup is not None else None
    if mesh is not None and mesh.pp_size > 1:
        autopipeline = _instantiate_pipeline(pipeline_cfg, mesh)
        if autopipeline is not None:
            logger.info("AutoPipeline instantiated for pp_size=%d", mesh.pp_size)

    return sharding_planner, fsdp2_manager, autopipeline


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


def _apply_fsdp2(
    model: nn.Module,
    fsdp2_manager,
    source_shard_info,
) -> nn.Module:
    """Apply FSDP2; Torch scheduler hooks remain outside Dynamo by design."""
    if fsdp2_manager is None:
        return model
    if not isinstance(fsdp2_manager, FSDP2Manager):
        logger.warning("fsdp2_manager is not an FSDP2Manager instance")
        return model
    model = fsdp2_manager.parallelize(
        model,
        source_shard_info=source_shard_info,
    )
    logger.info("FSDP2 wrap applied")
    return model


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
    if callable(initialize_weights):
        initialize_weights()
    else:
        init_weights = getattr(model, "init_weights", None)
        if callable(init_weights):
            init_weights()
        else:
            for module in model.modules():
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
    logger.info("Initialized model state with model-native random initialization")


def _join_fqn(module_name: str, tensor_name: str) -> str:
    """Join one module path and direct tensor name."""
    return f"{module_name}.{tensor_name}" if module_name else tensor_name


def _build_finalize_targets(model: nn.Module) -> dict[str, _FinalizeTarget]:
    """Build a registry including persistent and non-persistent model state."""
    targets = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for tensor_name, parameter in module._parameters.items():  # pylint: disable=W0212
            if parameter is None:
                continue
            fqn = _join_fqn(module_name, tensor_name)
            targets[fqn] = _FinalizeTarget(fqn, module, tensor_name, parameter, True, False)

        non_persistent = module._non_persistent_buffers_set  # pylint: disable=W0212
        for tensor_name, buffer in module._buffers.items():  # pylint: disable=W0212
            if buffer is None:
                continue
            fqn = _join_fqn(module_name, tensor_name)
            targets[fqn] = _FinalizeTarget(
                fqn,
                module,
                tensor_name,
                buffer,
                False,
                tensor_name in non_persistent,
            )
    return targets


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local storage tensor used by one model tensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _validate_materialized(targets: list[_FinalizeTarget]) -> None:
    """Reject model state that remains on meta after storage materialization."""
    meta_keys = [target.fqn for target in targets if _local_tensor(target.tensor).is_meta]
    if meta_keys:
        preview = ", ".join(meta_keys[:10])
        raise ValueError(
            f"Model finalization found {len(meta_keys)} tensors on meta device; first keys: {preview}"
        )


def _snapshot_target(target: _FinalizeTarget) -> _TargetSnapshot:
    """Capture identity and shape invariants without copying tensor data."""
    local_tensor = _local_tensor(target.tensor)
    layout = getattr(target.tensor, "layout", None)
    if layout is None:
        layout = getattr(target.tensor, "_sharding_spec", None)
    return _TargetSnapshot(
        tensor_id=id(target.tensor),
        local_tensor_id=id(local_tensor),
        global_shape=tuple(target.tensor.shape),
        local_shape=tuple(local_tensor.shape),
        layout_id=id(layout) if layout is not None else None,
    )


def _mark_loaded_targets_initialized(
    targets: dict[str, _FinalizeTarget],
    loaded_keys: set[str],
) -> dict[str, _TargetSnapshot]:
    """Mark loaded tensors initialized and retain their structural invariants."""
    snapshots = {}
    unknown_keys = sorted(loaded_keys - targets.keys())
    if unknown_keys:
        preview = ", ".join(unknown_keys[:10])
        raise ValueError(
            f"Load report contains {len(unknown_keys)} model keys not registered after wrapping; first keys: {preview}"
        )
    loaded_targets = [targets[key] for key in loaded_keys]
    _validate_materialized(loaded_targets)
    for target in loaded_targets:
        target.tensor._is_hf_initialized = True  # pylint: disable=W0212
        snapshots[target.fqn] = _snapshot_target(target)
    return snapshots


def _shares_local_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two tensors represent the same local parameter storage."""
    if first is second:
        return True
    first_local = _local_tensor(first)
    second_local = _local_tensor(second)
    return (
        first_local.device == second_local.device
        and first_local.untyped_storage().data_ptr() == second_local.untyped_storage().data_ptr()
    )


def _resolve_tied_aliases(
    model: nn.Module,
    targets: dict[str, _FinalizeTarget],
    loaded_keys: set[str],
) -> set[str]:
    """Mark and validate loaded aliases of tied model parameters."""
    tied_mapping = getattr(model, "all_tied_weights_keys", {}) or {}
    initialized_aliases = set()
    for target_name, source_name in tied_mapping.items():
        target = targets.get(target_name)
        source = targets.get(source_name)
        if target is None or source is None:
            continue
        if target_name not in loaded_keys and source_name not in loaded_keys:
            continue
        if tuple(target.tensor.shape) != tuple(source.tensor.shape):
            raise ValueError(
                f"Tied parameters must have matching shapes: {target_name}={tuple(target.tensor.shape)} vs "
                f"{source_name}={tuple(source.tensor.shape)}"
            )
        if not _shares_local_storage(target.tensor, source.tensor):
            raise ValueError(
                f"Tied parameters no longer share local storage after distributed wrapping: "
                f"{target_name} and {source_name}"
            )
        target.tensor._is_hf_initialized = True  # pylint: disable=W0212
        source.tensor._is_hf_initialized = True  # pylint: disable=W0212
        initialized_aliases.update((target_name, source_name))
    return initialized_aliases


def _matches_any_pattern(key: str, patterns: set[str]) -> bool:
    """Return whether a model state key matches any configured regex pattern."""
    return any(re.search(pattern, key) is not None for pattern in patterns)


def _adjust_loading_keys(
    model: nn.Module,
    missing_keys: set[str],
    unexpected_keys: set[str],
) -> tuple[set[str], set[str], int, int]:
    """Apply Transformers-compatible ignore patterns to loading results."""
    missing_patterns = set(getattr(model, "_keys_to_ignore_on_load_missing", None) or set())
    unexpected_patterns = set(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or set())
    adjusted_missing = {
        key for key in missing_keys if not _matches_any_pattern(key, missing_patterns)
    }
    adjusted_unexpected = {
        key for key in unexpected_keys if not _matches_any_pattern(key, unexpected_patterns)
    }
    return (
        adjusted_missing,
        adjusted_unexpected,
        len(missing_keys) - len(adjusted_missing),
        len(unexpected_keys) - len(adjusted_unexpected),
    )


def _prepare_initialization_targets(targets: list[_FinalizeTarget]) -> None:
    """Clear stale initialization flags only for state that must be rebuilt."""
    owner_modules = {}
    for target in targets:
        target.tensor._is_hf_initialized = False  # pylint: disable=W0212
        owner_modules[id(target.module)] = target.module
    for module in owner_modules.values():
        module._is_hf_initialized = False  # pylint: disable=W0212


def _initialize_model_state_after_loading(model: nn.Module) -> None:
    """Run the guarded Transformers initialization entry point."""
    initialize_weights = getattr(model, "initialize_weights", None)
    if not callable(initialize_weights):
        raise ValueError(
            "Deferred pretrained loading requires a callable model.initialize_weights()"
        )
    initialize_weights()


def _validate_loaded_target_snapshots(
    targets: dict[str, _FinalizeTarget],
    snapshots: dict[str, _TargetSnapshot],
) -> None:
    """Ensure finalization did not replace or reshape loaded distributed state."""
    for key, expected in snapshots.items():
        target = targets[key]
        actual = _snapshot_target(target)
        if actual != expected:
            raise ValueError(
                f"Model finalization replaced or reshaped loaded tensor {key}: "
                f"expected={expected}, actual={actual}"
            )
        if not getattr(target.tensor, "_is_hf_initialized", False):
            raise ValueError(f"Loaded tensor lost its initialized marker during finalization: {key}")


def _validate_initialization_targets(targets: list[_FinalizeTarget]) -> None:
    """Ensure requested model state was initialized by the model contract."""
    uninitialized = [
        target.fqn
        for target in targets
        if not getattr(target.tensor, "_is_hf_initialized", False)
    ]
    if uninitialized:
        preview = ", ".join(uninitialized[:10])
        raise ValueError(
            f"Model initialize_weights() left {len(uninitialized)} tensors uninitialized; first keys: {preview}"
        )


def _finalize_model_loading(
    model: nn.Module,
    load_report: LoadReport,
    *,
    strict: bool,
) -> LoadReport:
    """Finalize deferred pretrained loading without replacing distributed parameters."""
    targets = _build_finalize_targets(model)
    loaded_keys = set(load_report.loaded_keys)
    missing_keys = set(load_report.missing_keys)
    unexpected_keys = set(load_report.unexpected_keys)
    loaded_snapshots = _mark_loaded_targets_initialized(targets, loaded_keys)

    tied_aliases = _resolve_tied_aliases(model, targets, loaded_keys)
    missing_keys.difference_update(tied_aliases)
    missing_keys, unexpected_keys, ignored_missing, ignored_unexpected = _adjust_loading_keys(
        model,
        missing_keys,
        unexpected_keys,
    )
    if strict and missing_keys:
        preview = ", ".join(sorted(missing_keys)[:10])
        raise RuntimeError(
            f"Checkpoint did not load {len(missing_keys)} owned model tensors after finalization; "
            f"first keys: {preview}"
        )

    initialization_targets = [
        target
        for target in targets.values()
        if target.is_non_persistent or target.fqn in missing_keys
    ]
    missing_sharded = [
        target.fqn
        for target in initialization_targets
        if target.fqn in missing_keys and isinstance(target.tensor, DTensor)
    ]
    if missing_sharded:
        preview = ", ".join(sorted(missing_sharded)[:10])
        raise ValueError(
            f"Missing distributed parameters require DTensor-aware initialization; first keys: {preview}"
        )

    _validate_materialized(initialization_targets)
    _prepare_initialization_targets(initialization_targets)
    if initialization_targets:
        _initialize_model_state_after_loading(model)
        for target in initialization_targets:
            target.tensor._is_hf_initialized = True  # pylint: disable=W0212
    _validate_loaded_target_snapshots(targets, loaded_snapshots)
    _validate_initialization_targets(initialization_targets)
    _validate_materialized(list(targets.values()))
    _resolve_tied_aliases(model, targets, loaded_keys)

    finalized_report = LoadReport(
        loaded_keys=tuple(sorted(loaded_keys)),
        missing_keys=tuple(sorted(missing_keys)),
        unexpected_keys=tuple(sorted(unexpected_keys)),
    )
    logger.info(
        "Finalized pretrained model state: loaded=%d initialized=%d ignored_missing=%d "
        "ignored_unexpected=%d unresolved_missing=%d unexpected=%d",
        len(finalized_report.loaded_keys),
        len(initialization_targets),
        ignored_missing,
        ignored_unexpected,
        len(finalized_report.missing_keys),
        len(finalized_report.unexpected_keys),
    )
    return finalized_report


def _apply_module_replacement_actions(
    model: nn.Module,
    distributed_setup,
    weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
    low_precision_config=None,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter]]:
    """Apply explicit replacement rules before sharding sees the model."""

    if weights_mapping is None:
        weights_mapping = get_model_conversion_mapping(model)
    entries = getattr(distributed_setup, "plan_overrides", None) or []
    mesh_context = getattr(distributed_setup, "mesh_context", None)
    if low_precision_config is None:
        low_precision_config = getattr(
            distributed_setup,
            "low_precision_config",
            None,
        )
    low_precision_enabled = bool(
        getattr(low_precision_config, "enabled", False)
    )
    context = {
        "low_precision": low_precision_config if low_precision_enabled else None,
        "tp": getattr(mesh_context, "tp_size", 1) > 1,
        "cp": getattr(mesh_context, "cp_size", 1) > 1,
        "ep": getattr(mesh_context, "ep_size", 1) > 1,
        "pp": getattr(mesh_context, "pp_size", 1) > 1,
    }
    active_model_parallel_axes = [
        axis.upper()
        for axis in ("tp", "cp", "ep", "pp")
        if context[axis]
    ]
    if low_precision_enabled and active_model_parallel_axes:
        raise NotImplementedError(
            "Low-precision online training currently requires TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}."
        )
    from hyper_models.components.model_transform import (  # pylint: disable=import-outside-toplevel
        apply_module_replacements,
        compile_module_replacements,
    )

    rules = entries_to_module_replacements(
        entries,
        low_precision_enabled=low_precision_enabled,
    )
    if low_precision_enabled and not any(
        entry.replace_module is not None and entry.when == "low_precision"
        for entry in entries
    ):
        raise ValueError(
            "low_precision is enabled but no low-precision module replacement "
            "is configured"
        )
    plan = compile_module_replacements(model, rules)
    return apply_module_replacements(
        model,
        plan,
        weights_mapping=weights_mapping,
        context=context,
    )


def apply_model_infrastructure(
    model: nn.Module,
    mesh: Optional[MeshContext] = None,
    sharding_planner: Optional[ShardingPlanner] = None,
    fsdp2_manager: Optional[FSDP2Manager] = None,
    autopipeline: Optional[AutoPipeline] = None,
    peft_config: Optional[Any] = None,
    qat_config: Optional[Any] = None,
    fp8_config: Optional[Any] = None,
    freeze_config: Optional[Any] = None,
    compile_config: Optional[Union[CompileConfig, dict]] = None,
    activation_checkpoint: Optional[str] = None,
    activation_swap: str = "none",
    is_meta_device: bool = False,
    is_hf_model: bool = False,
    device: Optional[torch.device] = None,
    load_base_model: bool = False,
    pretrained_path: Optional[str] = None,
    validate_placement: bool = False,
    low_precision_config: Optional[Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """Apply model infrastructure (sharding, recompute, FSDP2, and compile).

    The execution order is: parallel layout -> recompute wrappers -> FSDP2 ->
    materialization/loading -> per-layer compile. Placement validation keeps
    the DTensor placement path and skips compile, while FSDP2 consumes DTensor
    parameter layouts in both modes.
    """

    distributed_setup = kwargs.get("distributed_setup")

    if isinstance(compile_config, dict):
        compile_config = CompileConfig(enabled=True, **compile_config)
    if compile_config is not None and not isinstance(compile_config, CompileConfig):
        raise TypeError("compile_config must be a CompileConfig, mapping, or None")
    compile_for_execution = bool(
        not validate_placement
        and compile_config is not None
        and compile_config.enabled
    )
    if validate_placement and compile_config is not None and compile_config.enabled:
        logger.info("Skipping decoder-layer compile during placement validation")
    if (
        compile_for_execution
        and compile_config.fullgraph
        and isinstance(fsdp2_manager, FSDP2Manager)
    ):
        raise ValueError(
            "compile.fullgraph=True is incompatible with FSDP hooks kept eager "
            "by _dynamo_disable; set compile.fullgraph=False"
        )

    # Step 3: PP split (if autopipeline)
    if autopipeline is not None:
        autopipeline.build(model)

    # Step 4: PEFT injection (before sharding)
    if peft_config is not None:
        logger.warning("PEFT injection not implemented in stub")

    # Step 5: QAT / FP8 (before sharding)
    if qat_config is not None:
        logger.warning("QAT not implemented in stub")
    if fp8_config is not None:
        logger.warning("FP8 not implemented in stub")

    # Step 5.5: structure-preserving replacement before plan derivation.
    weights_mapping = get_model_conversion_mapping(model)
    model, weights_mapping = _apply_module_replacement_actions(
        model,
        distributed_setup,
        weights_mapping,
        low_precision_config=low_precision_config,
    )

    # Step 6: Parameter freezing (before sharding)
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

    # Step 9-1: activation checkpointing remains inside the FSDP boundary.
    if activation_checkpoint not in (None, "off"):
        model = _apply_activation_checkpointing(
            model,
            activation_checkpoint,
            enable_compile=compile_for_execution,
        )

    # Step 9-2:activation swap.
    validate_attention_swap(
        activation_swap,
        activation_checkpoint=activation_checkpoint,
        enable_compile=compile_for_execution,
        pp_size=getattr(mesh, "pp_size", 1),
    )
    if activation_swap != "none":
        model = apply_attention_swap(model, activation_swap)


    # Step 10: both dual modes use FSDP2. In validate mode the parameters stay
    # as DTensors, and FSDP derives their source layouts directly.
    model = _apply_fsdp2(
        model,
        fsdp2_manager,
        source_shard_info,
    )

    # Steps 11-12: materialize model storage, then load or initialize weights.
    model = _move_model_to_device(model, is_meta_device, device)
    if is_meta_device:
        if load_base_model:
            load_report = CheckpointManager(model).load_checkpoint(
                pretrained_path,
                strict=False,
                weights_mapping=weights_mapping,
            )
            _finalize_model_loading(model, load_report, strict=True)
        else:
            _initialize_model_weights(model)

    # Step 13: compile only the execution model, after FSDP and loading.
    if compile_for_execution:
        model = apply_compile(model, compile_config)

    return model
