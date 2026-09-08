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
"""Eight-card dual-mode Trainer FSDP main_param accuracy check.

The distributed model follows the Trainer infrastructure chain used by the new
dual-mode path: ``ShardingPlanner`` applies TP and returns source-layout
metadata, ``FSDP2Manager`` resolves parameter identities and wraps nested
FSDP2 units, and ``fully_shard`` consumes the per-unit metadata.  A full
single-process model is kept as the standalone reference.  Every step compares
the local-batch loss and every managed parameter gradient before both optimizers
step.

The configuration uses DP(2), CP(2), TP(2), EP(8), dense HSDP(2x2), and
expert FSDP shard size 1 in one eight-card run. Both paths use bfloat16 model
parameters and the same ``Float16OptimizerWithFloat16Params`` wrapper around
AdamW. The run compares loss, gradients, fp32 main_param updates, and model
copyback while also covering the multi-dimensional parallel implementation.
"""
# pylint: disable=C0413,C9002,E1123

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)

from hyper_parallel.models._transformers.model_builder import instantiate_infrastructure
from hyper_parallel.data.parallel.batch_parallel import shard_batch_for_cp
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.trainer.runtime.distributed import (
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.trainer.runtime.device import get_device_type
from hyper_parallel.trainer.config.manager import parse_training_args
from hyper_parallel.trainer.config import (
    TrainerConfig,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.models._transformers.model_builder import apply_model_init_dtype
from hyper_parallel import (
    DTensor,
    DeviceMesh,
    HSDPModule,
    SkipDTensorDispatch,
)
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate, Shard
from hyper_parallel.core.utils import clip_grad_norm_


_TEST_YAML_DIRECTORY = Path(__file__).with_name("test_yamls")
_LOG_DIRECTORY = Path("outputs/dualmode_trainer/fsdp_accuracy/adamw_fp32_main_params")
_INIT_SEED = 31415
_DATA_SEED = 27182
_TRAINING_STEPS = 20
_BATCH_SIZE = 8
_SEQUENCE_LENGTH = 8
_VOCAB_SIZE = 128
_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_QWEN2_NUM_LAYERS = 2
_NUM_HEADS = 4
_NUM_EXPERTS = 8
_NUM_EXPERTS_PER_TOKEN = 2
_RTOL = 5.0e-3
_ATOL = 5.0e-3


@dataclass(frozen=True)
class _BatchLayout:
    """Local DP and CP ranges represented by one distributed rank."""

    local_batch_size: int
    local_batch_start: int
    local_batch_end: int
    local_sequence_start: int
    local_sequence_end: int


@dataclass(frozen=True)
class _AccuracyState:
    """Models, optimizers, and distributed metadata used across all steps."""

    standalone_model: torch.nn.Module
    distributed_model: torch.nn.Module
    source_shard_info_by_fqn: dict[str, tuple[tuple[Placement, ...], DeviceMesh]]
    mesh_context: MeshContext
    replicate_parameter_name: str
    standalone_optimizer: Float16OptimizerWithFloat16Params
    distributed_optimizer: Float16OptimizerWithFloat16Params
    standalone_log: Optional[TextIO]
    distributed_log: Optional[TextIO]


@dataclass(frozen=True)
class _StepResult:
    """Numerical errors collected from one optimizer step."""

    standalone_loss: float
    distributed_loss: float
    maximum_gradient_error: float
    grad_norm_error: float


@dataclass(frozen=True)
class _AccuracySummary:
    """Maximum numerical errors observed across all training steps."""

    maximum_loss_error: float
    maximum_gradient_error: float
    maximum_grad_norm_error: float


def _parse_config(config_name: str) -> TrainerConfig:
    """Resolve one accuracy-test YAML through the normal Trainer parser."""
    return parse_training_args([str(_TEST_YAML_DIRECTORY / config_name)])


def _build_model_config() -> Qwen2MoeConfig:
    """Return the small Qwen2-MoE configuration used by both copies."""
    return Qwen2MoeConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        moe_intermediate_size=_INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=_QWEN2_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=2,
        num_experts=_NUM_EXPERTS,
        num_experts_per_tok=_NUM_EXPERTS_PER_TOKEN,
        num_experts_shared=1,
        max_position_embeddings=_SEQUENCE_LENGTH,
        qkv_bias=True,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _build_standalone_model(
    device: torch.device,
    model_init_dtype: Optional[str],
) -> Qwen2MoeForCausalLM:
    """Build the unsharded reference model from the fixed initialization seed."""
    torch.manual_seed(_INIT_SEED)
    model = Qwen2MoeForCausalLM(_build_model_config()).to(device=device)
    _install_qwen2_rowwise_bias(model)
    apply_model_init_dtype(model, model_init_dtype)
    return model.train()


def _install_qwen2_rowwise_bias(model: torch.nn.Module) -> None:
    """Add deterministic nonzero RowWise attention biases before sharding."""
    for layer_index, layer in enumerate(model.model.layers):
        projection = layer.self_attn.o_proj
        values = torch.linspace(
            -0.1,
            0.1,
            projection.out_features,
            device=projection.weight.device,
            dtype=projection.weight.dtype,
        ) + layer_index * 1.0e-3
        projection.bias = torch.nn.Parameter(values)


def _build_dual_mode_model(
    distributed_setup: DistributedSetup,
    device: torch.device,
    model_init_dtype: Optional[str],
) -> tuple[torch.nn.Module, dict[str, tuple[Placement, DeviceMesh]]]:
    """Build and wrap the model through the dual-mode TP/FSDP infrastructure."""
    torch.manual_seed(_INIT_SEED)
    model = Qwen2MoeForCausalLM(_build_model_config()).to(device=device)
    _install_qwen2_rowwise_bias(model)
    apply_model_init_dtype(model, model_init_dtype)
    model.train()

    mesh_context = distributed_setup.mesh_context
    sharding_planner, fsdp_manager = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=device,
    )
    if fsdp_manager is None or mesh_context.device_mesh is None:
        raise RuntimeError("The accuracy test requires an initialized FSDP + TP mesh")

    sharding_plan = sharding_planner.plan(
        model,
        mesh_context.device_mesh,
        tp_size=mesh_context.tp_size,
        cp_size=mesh_context.cp_size,
        ep_size=mesh_context.ep_size,
        sequence_parallel=mesh_context.sequence_parallel,
        loss_parallel=mesh_context.loss_parallel,
    )
    model, source_shard_info_by_fqn = apply_sharding_plan(
        model,
        sharding_plan,
        mesh_context,
    )
    # This parameter is intentionally FSDP-managed but remains replicated on
    # the HSDP shard mesh. The FQN is configured under fsdp_config and resolved
    # by FSDP2Manager before it applies the root fully_shard call.
    replicate_parameter = model.model.norm.weight
    fsdp_manager.parallelize(model, source_shard_info_by_fqn)
    _assert_runtime_fsdp_configuration(
        model,
        replicate_parameter,
        forward_prefetch_depth=fsdp_manager.config.forward_prefetch_depth,
        backward_prefetch_depth=fsdp_manager.config.backward_prefetch_depth,
        expected_nested_unit_count=_QWEN2_NUM_LAYERS * 2,
    )
    return model, source_shard_info_by_fqn


def _assert_runtime_fsdp_configuration(
    model: torch.nn.Module,
    replicate_parameter: torch.nn.Parameter,
    forward_prefetch_depth: int,
    backward_prefetch_depth: int,
    expected_nested_unit_count: int,
) -> None:
    """Verify configured prefetch depths, replicate params, and root policy."""
    nested_fsdp_units = [
        module
        for module in model.modules()
        if module is not model and isinstance(module, HSDPModule)
    ]
    if len(nested_fsdp_units) != expected_nested_unit_count:
        raise AssertionError(
            f"expected {expected_nested_unit_count} nested FSDP units, "
            f"got {len(nested_fsdp_units)}"
        )
    for unit_index, fsdp_unit in enumerate(nested_fsdp_units):
        expected_forward = nested_fsdp_units[
            unit_index + 1:unit_index + 1 + forward_prefetch_depth
        ]
        expected_backward = list(reversed(
            nested_fsdp_units[
                max(0, unit_index - backward_prefetch_depth):unit_index
            ]
        ))
        if fsdp_unit.hsdp_scheduler.forward_prefetch_cells != expected_forward:
            raise AssertionError(
                f"FSDP unit {unit_index}: forward prefetch does not match depth "
                f"{forward_prefetch_depth}"
            )
        if fsdp_unit.hsdp_scheduler.backward_prefetch_cells != expected_backward:
            raise AssertionError(
                f"FSDP unit {unit_index}: backward prefetch does not match depth "
                f"{backward_prefetch_depth}"
            )
    root_scheduler = model.hsdp_scheduler
    if root_scheduler.hsdp_state.raw_replicate_params != {replicate_parameter}:
        raise AssertionError("root replicate_params does not contain model.norm.weight")
    if root_scheduler.reshard_after_forward:
        raise AssertionError("FSDP root must use reshard_after_forward=False")


def _build_global_batch(step_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic global token batch shared by both model copies."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_DATA_SEED + step_index)
    tokens = torch.randint(
        _VOCAB_SIZE,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
    )
    targets = torch.randint(
        _VOCAB_SIZE,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
    )
    return tokens.to(device=device), targets.to(device=device)


def _cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the summed token loss used by both paths."""
    return F.cross_entropy(
        logits.float().reshape(-1, _VOCAB_SIZE),
        targets.reshape(-1),
        reduction="sum",
    )


def _validate_global_token_coverage(
    global_targets: torch.Tensor,
    local_targets: torch.Tensor,
    mesh_context: MeshContext,
) -> None:
    """Validate world execution and logical DP×CP token coverage."""
    local_token_count = local_targets.numel()
    world_token_count = torch.tensor(
        local_token_count,
        dtype=torch.int64,
        device=local_targets.device,
    )
    dist.all_reduce(world_token_count, op=dist.ReduceOp.SUM)
    world_token_count_value = int(world_token_count.item())
    topology_world_token_count = (
        local_token_count
        * mesh_context.dp_size
        * mesh_context.cp_size
        * mesh_context.tp_size
    )
    if world_token_count_value != topology_world_token_count:
        raise AssertionError(
            "distributed token coverage does not match DP×CP×TP topology: "
            f"local={local_token_count}, dp={mesh_context.dp_size}, "
            f"cp={mesh_context.cp_size}, tp={mesh_context.tp_size}, "
            f"world={world_token_count_value}, expected={topology_world_token_count}"
        )

    logical_global_token_count = (
        local_token_count * mesh_context.dp_size * mesh_context.cp_size
    )
    standalone_global_token_count = global_targets.numel()
    if logical_global_token_count != standalone_global_token_count:
        raise AssertionError(
            "distributed logical global token count differs from standalone: "
            f"distributed={logical_global_token_count}, "
            f"standalone={standalone_global_token_count}, "
            f"dp={mesh_context.dp_size}, cp={mesh_context.cp_size}, "
            f"tp={mesh_context.tp_size}"
        )


def _get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return one contiguous local shard while preserving its collective device."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().contiguous()


def _all_gather_cat(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    dimension: int,
) -> torch.Tensor:
    """Gather equal-sized shards on one mesh axis and concatenate by mesh rank."""
    mesh_size = mesh.mesh_shape[0]
    if mesh_size == 1:
        return local_tensor
    gathered = [torch.empty_like(local_tensor) for _ in range(mesh_size)]
    dist.all_gather(gathered, local_tensor, group=mesh.get_group())
    normalized_dimension = dimension if dimension >= 0 else local_tensor.ndim + dimension
    return torch.cat(gathered, dim=normalized_dimension)


def _all_gather_shards(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    dimension: int,
    global_dimension_size: int,
) -> torch.Tensor:
    """Gather potentially empty or uneven shards and trim collective padding."""
    mesh_size = mesh.mesh_shape[0]
    if mesh_size == 1:
        return local_tensor
    normalized_dimension = dimension if dimension >= 0 else local_tensor.ndim + dimension
    padded_size = (global_dimension_size + mesh_size - 1) // mesh_size
    padded_shape = list(local_tensor.shape)
    padded_shape[normalized_dimension] = padded_size
    padded_tensor = local_tensor.new_zeros(padded_shape)
    if local_tensor.numel() > 0:
        padded_tensor.narrow(
            normalized_dimension,
            0,
            local_tensor.shape[normalized_dimension],
        ).copy_(local_tensor)
    gathered_tensor = _all_gather_cat(padded_tensor, mesh, normalized_dimension)
    return gathered_tensor.narrow(normalized_dimension, 0, global_dimension_size)


def _reconstruct_global_tensor(
    local_tensor: torch.Tensor,
    expected_shape: torch.Size,
    parameter_name: str,
    source_shard_info_by_fqn: dict[str, tuple[tuple[Placement, ...], DeviceMesh]],
    mesh_context: MeshContext,
    replicate_parameter_name: str,
) -> torch.Tensor:
    """Invert FSDP and then TP/EP placement to recover a canonical Tensor."""
    fsdp_non_moe_mesh = mesh_context.fsdp_non_moe_mesh
    if fsdp_non_moe_mesh is None:
        raise RuntimeError("global reconstruction requires a dense FSDP mesh")
    placements, source_mesh = source_shard_info_by_fqn.get(
        parameter_name,
        ((Replicate(),), fsdp_non_moe_mesh["tp"]),
    )
    placement = placements[0]
    global_tensor = _get_local_tensor(local_tensor)
    post_fsdp_shape = list(expected_shape)
    if isinstance(placement, Shard):
        source_mesh_size = source_mesh.mesh_shape[0]
        post_fsdp_shape[placement.dim] = (
            post_fsdp_shape[placement.dim] + source_mesh_size - 1
        ) // source_mesh_size
    if parameter_name != replicate_parameter_name:
        fsdp_shard_mesh = fsdp_non_moe_mesh["fsdp_shard"]
        if mesh_context.fsdp_moe_mesh is not None and ".experts." in parameter_name:
            fsdp_shard_mesh = mesh_context.fsdp_moe_mesh["edp_shard"]
        global_tensor = _all_gather_shards(
            global_tensor,
            fsdp_shard_mesh,
            0,
            post_fsdp_shape[0],
        )
    if isinstance(placement, Shard):
        global_tensor = _all_gather_shards(
            global_tensor,
            source_mesh,
            placement.dim,
            expected_shape[placement.dim],
        )
    elif not isinstance(placement, Replicate):
        raise ValueError(
            f"unsupported source placement {placement!r} for global parameter "
            f"{parameter_name}"
        )
    return global_tensor


def _compare_global_parameter_view(
    stage: str,
    standalone_model: torch.nn.Module,
    distributed_model: torch.nn.Module,
    source_shard_info_by_fqn: dict[str, tuple[tuple[Placement, ...], DeviceMesh]],
    mesh_context: MeshContext,
    replicate_parameter_name: str,
    *,
    gradients: bool,
    main_params: bool = False,
) -> float:
    """Reconstruct and compare model or main parameters and their gradients."""
    reference_parameters = dict(standalone_model.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    maximum_error = 0.0
    for parameter_name, reference_parameter in reference_parameters.items():
        distributed_parameter = distributed_parameters[parameter_name]
        reference_parameter_view = (
            reference_parameter.main_param if main_params else reference_parameter
        )
        distributed_parameter_view = (
            distributed_parameter.main_param if main_params else distributed_parameter
        )
        reference_tensor = (
            reference_parameter_view.grad if gradients else reference_parameter_view
        )
        distributed_tensor = (
            distributed_parameter_view.grad if gradients else distributed_parameter_view
        )
        if reference_tensor is None or distributed_tensor is None:
            raise AssertionError(
                f"{stage}, parameter {parameter_name}: missing "
                f"{'gradient' if gradients else 'value'}"
            )
        actual_global = _reconstruct_global_tensor(
            distributed_tensor,
            reference_tensor.shape,
            parameter_name,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
        )
        actual_global = actual_global.float()
        expected_global = reference_tensor.detach().float()
        if actual_global.shape != expected_global.shape:
            raise AssertionError(
                f"{stage}, parameter {parameter_name}: reconstructed global shape "
                f"{tuple(actual_global.shape)} != reference shape {tuple(expected_global.shape)}"
            )
        try:
            torch.testing.assert_close(
                actual_global,
                expected_global,
                rtol=_RTOL if gradients else 1.0e-5,
                atol=_ATOL if gradients else 1.0e-5,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{stage}, parameter {parameter_name} global-view mismatch; "
                f"actual norm={float(actual_global.norm()):.8f}, "
                f"reference norm={float(expected_global.norm()):.8f}\n{error}"
            ) from error
        maximum_error = max(
            maximum_error,
            float(torch.max(torch.abs(actual_global - expected_global))),
        )
    return maximum_error


def _build_accuracy_optimizer(
    config: TrainerConfig,
    model: torch.nn.Module,
) -> Float16OptimizerWithFloat16Params:
    """Build the configured AdamW runtime and its fp32 main_param wrapper."""
    inner_optimizer = config.optimizer.target.build(model=model).get_optimizer()
    optimizer = Float16OptimizerWithFloat16Params(inner_optimizer, model)
    for parameter_name, parameter in model.named_parameters():
        if parameter.main_param.dtype != torch.float32:
            raise AssertionError(
                f"parameter {parameter_name}: main_param must be float32, "
                f"got {parameter.main_param.dtype}"
            )
    return optimizer


def _assert_model_param_copyback(model: torch.nn.Module, stage: str) -> None:
    """Verify every local model shard is the dtype-cast fp32 main_param value."""
    for parameter_name, parameter in model.named_parameters():
        model_local = _get_local_tensor(parameter)
        main_param_local = _get_local_tensor(parameter.main_param)
        try:
            torch.testing.assert_close(
                model_local,
                main_param_local.to(dtype=model_local.dtype),
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{stage}, parameter {parameter_name}: model copyback mismatch\n{error}"
            ) from error


def _assert_distributed_main_grads(
    step_index: int,
    distributed_parameters: dict[str, torch.nn.Parameter],
) -> None:
    """Verify FSDP stores every reduced gradient in fp32 main_grad."""
    for parameter_name, parameter in distributed_parameters.items():
        if parameter.grad is not None:
            raise AssertionError(
                f"step {step_index}, parameter {parameter_name}: FSDP grad must be empty"
            )
        if parameter.main_grad is None:
            raise AssertionError(
                f"step {step_index}, parameter {parameter_name}: FSDP main_grad is missing"
            )
        if parameter.main_grad.dtype != torch.float32:
            raise AssertionError(
                f"step {step_index}, parameter {parameter_name}: main_grad must be "
                f"float32, got {parameter.main_grad.dtype}"
            )


def _write_step_log(
    log_file: TextIO,
    step_index: int,
    loss: float,
    grad_norm: float,
    **errors: float,
) -> None:
    """Write one parseable optimizer accuracy record and flush it immediately."""
    error_fields = "".join(f" {name}={value:.9e}" for name, value in errors.items())
    log_file.write(
        f"step={step_index} loss={loss:.9e} grad_norm={grad_norm:.9e}{error_fields}\n"
    )
    log_file.flush()


def _forward_and_backward(
    state: _AccuracyState,
    global_tokens: torch.Tensor,
    global_targets: torch.Tensor,
    local_tokens: torch.Tensor,
    local_targets: torch.Tensor,
    local_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run normalized standalone and distributed forward/backward passes."""
    state.standalone_optimizer.zero_grad(set_to_none=True)
    standalone_logits = state.standalone_model(input_ids=global_tokens).logits
    _validate_global_token_coverage(
        global_targets,
        local_targets,
        state.mesh_context,
    )
    standalone_full_loss = (
        _cross_entropy_sum(standalone_logits, global_targets) / global_targets.numel()
    )
    standalone_full_loss.backward()

    state.distributed_optimizer.zero_grad(set_to_none=True)
    distributed_logits = state.distributed_model(
        input_ids=local_tokens,
        position_ids=local_position_ids,
    ).logits
    local_token_count = local_targets.numel()
    distributed_local_mean = (
        _cross_entropy_sum(distributed_logits, local_targets) / local_token_count
    )
    token_counts = {
        "foundation_tokens": local_targets.new_tensor(local_token_count),
    }
    distributed_loss = mean_global_loss(
        distributed_local_mean,
        token_counts,
        token_counts,
        state.mesh_context,
    )["foundation_loss"]
    distributed_loss.backward()
    return standalone_full_loss.detach(), distributed_loss


def _compare_step(
    step_index: int,
    state: _AccuracyState,
    global_tokens: torch.Tensor,
    global_targets: torch.Tensor,
    local_tokens: torch.Tensor,
    local_targets: torch.Tensor,
    local_position_ids: torch.Tensor,
) -> _StepResult:
    """Run one reference/distributed step and compare loss plus every gradient."""
    standalone_local_loss, distributed_loss = _forward_and_backward(
        state,
        global_tokens,
        global_targets,
        local_tokens,
        local_targets,
        local_position_ids,
    )

    standalone_parameters = dict(state.standalone_model.named_parameters())
    distributed_parameters = dict(state.distributed_model.named_parameters())
    if standalone_parameters.keys() != distributed_parameters.keys():
        raise AssertionError("standalone and distributed parameter names differ")
    _assert_distributed_main_grads(step_index, distributed_parameters)

    standalone_loss_value = float(standalone_local_loss.detach().cpu())
    distributed_loss_value = float(distributed_loss.detach().cpu())
    np.testing.assert_allclose(
        distributed_loss_value,
        standalone_loss_value,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"step {step_index} local loss mismatch",
    )

    standalone_grad_norm = float(torch.nn.utils.clip_grad_norm_(
        state.standalone_model.parameters(),
        max_norm=float("inf"),
    ))
    distributed_grad_norm = float(clip_grad_norm_(
        state.distributed_model,
        max_norm=float("inf"),
    ))
    grad_norm_error = abs(distributed_grad_norm - standalone_grad_norm)
    np.testing.assert_allclose(
        distributed_grad_norm,
        standalone_grad_norm,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"step {step_index} gradient norm mismatch",
    )

    with SkipDTensorDispatch():
        state.standalone_optimizer.prepare_grads()
        state.distributed_optimizer.prepare_grads()
    maximum_gradient_error = _compare_global_parameter_view(
        f"step {step_index} fp32 main_grad values",
        state.standalone_model,
        state.distributed_model,
        state.source_shard_info_by_fqn,
        state.mesh_context,
        state.replicate_parameter_name,
        gradients=True,
        main_params=True,
    )
    if state.standalone_log is not None and state.distributed_log is not None:
        _write_step_log(
            state.standalone_log,
            step_index,
            standalone_loss_value,
            standalone_grad_norm,
        )
        _write_step_log(
            state.distributed_log,
            step_index,
            distributed_loss_value,
            distributed_grad_norm,
            loss_error=abs(distributed_loss_value - standalone_loss_value),
            grad_norm_error=grad_norm_error,
            max_parameter_grad_error=maximum_gradient_error,
        )

    with SkipDTensorDispatch():
        state.standalone_optimizer.step_with_ready_grads()
        state.distributed_optimizer.step_with_ready_grads()
    _assert_model_param_copyback(
        state.standalone_model,
        f"step {step_index} standalone",
    )
    _assert_model_param_copyback(
        state.distributed_model,
        f"step {step_index} distributed",
    )
    return _StepResult(
        standalone_loss=standalone_loss_value,
        distributed_loss=distributed_loss_value,
        maximum_gradient_error=maximum_gradient_error,
        grad_norm_error=grad_norm_error,
    )


def _validate_accuracy_topology(mesh_context: MeshContext, world_size: int) -> None:
    """Validate the exact DP, CP, TP, EP, and FSDP topology under test."""
    expected_world_size = (
        mesh_context.dp_size
        * mesh_context.cp_size
        * mesh_context.tp_size
        * mesh_context.pp_size
    )
    if world_size != expected_world_size or mesh_context.pp_size != 1:
        raise ValueError(
            "accuracy test requires world_size == dp * cp * tp with pp_size=1; "
            f"got world_size={world_size}, expected={expected_world_size}, "
            f"pp_size={mesh_context.pp_size}"
        )
    topology = (
        mesh_context.dp_size,
        mesh_context.cp_size,
        mesh_context.tp_size,
        mesh_context.ep_size,
        mesh_context.edp_shard_size,
    )
    if topology != (2, 2, 2, 8, 1):
        raise AssertionError(
            "accuracy test requires dp=2, cp=2, tp=2, ep=8, and edp_shard=1; "
            f"got {topology}"
        )
    if mesh_context.fsdp_moe_mesh is None:
        raise RuntimeError("MoE accuracy test requires an expert FSDP mesh")
    if mesh_context.fsdp_moe_mesh.mesh_shape[-1] != mesh_context.ep_size:
        raise AssertionError("MoE expert mesh EP dimension does not match ep_size")


def _build_batch_layout(
    mesh_context: MeshContext,
    device_mesh: DeviceMesh,
) -> _BatchLayout:
    """Resolve the local DP batch and CP sequence ranges from the mesh."""
    if _BATCH_SIZE % mesh_context.dp_size != 0:
        raise ValueError(
            f"batch size {_BATCH_SIZE} must be divisible by DP size {mesh_context.dp_size}"
        )
    if _SEQUENCE_LENGTH % mesh_context.cp_size != 0:
        raise ValueError(
            f"sequence length {_SEQUENCE_LENGTH} must be divisible by CP size "
            f"{mesh_context.cp_size}"
        )
    local_batch_size = _BATCH_SIZE // mesh_context.dp_size
    local_batch_start = device_mesh.get_local_rank("dp") * local_batch_size
    local_batch_end = local_batch_start + local_batch_size
    local_sequence_size = _SEQUENCE_LENGTH // mesh_context.cp_size
    local_sequence_start = device_mesh.get_local_rank("cp") * local_sequence_size
    local_sequence_end = local_sequence_start + local_sequence_size
    return _BatchLayout(
        local_batch_size=local_batch_size,
        local_batch_start=local_batch_start,
        local_batch_end=local_batch_end,
        local_sequence_start=local_sequence_start,
        local_sequence_end=local_sequence_end,
    )


def _open_accuracy_logs(case_name: str) -> tuple[Optional[TextIO], Optional[TextIO]]:
    """Open rank-zero standalone and distributed accuracy logs."""
    if dist.get_rank() != 0:
        return None, None
    _LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parallel_mode = case_name.removesuffix("_accuracy")
    standalone_log = (_LOG_DIRECTORY / f"{parallel_mode}_standalone_adamw.log").open(
        "w", encoding="utf-8"
    )
    distributed_log = (_LOG_DIRECTORY / f"{parallel_mode}_dist_adamw.log").open(
        "w", encoding="utf-8"
    )
    return standalone_log, distributed_log


def _print_step_result(
    case_name: str,
    step_index: int,
    step_result: _StepResult,
    mesh_context: MeshContext,
) -> None:
    """Print one rank-zero accuracy and topology record."""
    if dist.get_rank() != 0:
        return
    loss_error = abs(step_result.standalone_loss - step_result.distributed_loss)
    print(
        f"[{case_name} step {step_index}] standalone_loss={step_result.standalone_loss:.6f} "
        f"distributed_loss={step_result.distributed_loss:.6f} "
        f"loss_error={loss_error:.6e} "
        f"grad_norm_error={step_result.grad_norm_error:.6e} "
        f"max_grad_error={step_result.maximum_gradient_error:.6e} "
        f"(dp={mesh_context.dp_size}, cp={mesh_context.cp_size}, "
        f"fsdp_replicate={mesh_context.dp_replicate_size}, "
        f"fsdp_shard={mesh_context.dp_shard_size}, "
        f"edp_shard={mesh_context.edp_shard_size}, "
        f"tp={mesh_context.tp_size}, ep={mesh_context.ep_size})"
    )


def _run_training_steps(
    case_name: str,
    device: torch.device,
    state: _AccuracyState,
    layout: _BatchLayout,
) -> _AccuracySummary:
    """Run all optimizer steps and collect the largest numerical errors."""
    maximum_loss_error = 0.0
    maximum_gradient_error = 0.0
    maximum_grad_norm_error = 0.0
    for step_index in range(_TRAINING_STEPS):
        global_tokens, global_targets = _build_global_batch(step_index, device)
        local_batch = shard_batch_for_cp(
            {
                "input_ids": global_tokens[layout.local_batch_start:layout.local_batch_end],
                "labels": global_targets[layout.local_batch_start:layout.local_batch_end],
            },
            state.mesh_context.cp_mesh,
        )
        local_position_ids = torch.arange(
            layout.local_sequence_start,
            layout.local_sequence_end,
            device=device,
        ).unsqueeze(0).expand(layout.local_batch_size, -1)
        step_result = _compare_step(
            step_index,
            state,
            global_tokens,
            global_targets,
            local_batch["input_ids"],
            local_batch["labels"],
            local_position_ids,
        )
        step_loss_error = abs(
            step_result.standalone_loss - step_result.distributed_loss
        )
        maximum_loss_error = max(maximum_loss_error, step_loss_error)
        maximum_gradient_error = max(
            maximum_gradient_error,
            step_result.maximum_gradient_error,
        )
        maximum_grad_norm_error = max(
            maximum_grad_norm_error,
            step_result.grad_norm_error,
        )
        _print_step_result(case_name, step_index, step_result, state.mesh_context)
    return _AccuracySummary(
        maximum_loss_error=maximum_loss_error,
        maximum_gradient_error=maximum_gradient_error,
        maximum_grad_norm_error=maximum_grad_norm_error,
    )


def _run_accuracy_case(
    config_name: str,
    case_name: str,
) -> None:
    """Compare one eight-card dual-mode FSDP case with standalone training."""
    config = _parse_config(config_name)
    initialize_distributed(backend=config.training.backend)
    device = torch.device(
        get_device_type(),
        int(os.environ.get("LOCAL_RANK", "0")),
    )
    distributed_setup = create_distributed_setup_from_config(config)
    normalize_distributed_setup_overrides(distributed_setup, config)
    mesh_context = distributed_setup.mesh_context
    _validate_accuracy_topology(mesh_context, dist.get_world_size())
    device_mesh = mesh_context.device_mesh
    if device_mesh is None or mesh_context.fsdp_non_moe_mesh is None:
        raise RuntimeError("dual-mode accuracy test requires device and dense FSDP meshes")
    layout = _build_batch_layout(mesh_context, device_mesh)
    if not config.optimizer.fp32_main_params:
        raise AssertionError("accuracy test requires optimizer.fp32_main_params=true")

    standalone_model = _build_standalone_model(device, config.model_init_dtype)
    distributed_model, source_shard_info_by_fqn = _build_dual_mode_model(
        distributed_setup,
        device,
        config.model_init_dtype,
    )
    replicate_parameter_name = "model.norm.weight"

    _compare_global_parameter_view(
        "initial weights",
        standalone_model,
        distributed_model,
        source_shard_info_by_fqn,
        mesh_context,
        replicate_parameter_name,
        gradients=False,
    )

    standalone_optimizer = _build_accuracy_optimizer(config, standalone_model)
    distributed_optimizer = _build_accuracy_optimizer(config, distributed_model)
    standalone_log, distributed_log = _open_accuracy_logs(case_name)
    state = _AccuracyState(
        standalone_model=standalone_model,
        distributed_model=distributed_model,
        source_shard_info_by_fqn=source_shard_info_by_fqn,
        mesh_context=mesh_context,
        replicate_parameter_name=replicate_parameter_name,
        standalone_optimizer=standalone_optimizer,
        distributed_optimizer=distributed_optimizer,
        standalone_log=standalone_log,
        distributed_log=distributed_log,
    )
    summary = _run_training_steps(case_name, device, state, layout)
    if state.standalone_log is not None and state.distributed_log is not None:
        state.standalone_log.close()
        state.distributed_log.close()

    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"[{case_name}] passed all loss and parameter-gradient checks; "
            f"max_loss_error={summary.maximum_loss_error:.6e}, "
            f"max_grad_norm_error={summary.maximum_grad_norm_error:.6e}, "
            f"max_grad_error={summary.maximum_gradient_error:.6e}"
        )
    destroy_process_group()


def test_qwen2_tp_cp_ep_fsdp_global_accuracy() -> None:
    """Validate Qwen2-MoE mixed parallelism with fp32 main_param values."""
    _run_accuracy_case(
        "qwen2_tp_cp_ep_fsdp_global_accuracy.yaml",
        "qwen2_tp_cp_ep_fsdp_global_accuracy",
    )
