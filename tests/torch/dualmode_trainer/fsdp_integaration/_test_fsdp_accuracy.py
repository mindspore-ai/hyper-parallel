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
"""Eight-card dual-mode Trainer accuracy checks for FSDP and TP combinations.

The distributed model follows the Trainer infrastructure chain used by the new
dual-mode path: ``ShardingPlanner`` applies TP and returns source-layout
metadata, ``FSDP2Manager`` resolves parameter identities and wraps nested
FSDP2 units, and ``fully_shard`` consumes the per-unit metadata.  A full
single-process model is kept as the standalone reference.  Every step compares
the local-batch loss and every managed parameter gradient before both optimizers
step.

The HSDP+TP configuration uses ``dp_replicate=2``, ``dp_shard=2`` and
``tp=2``. The DP+CP+TP configuration uses a basic ``(2, 2, 2)`` mesh and
reshapes the DP+CP domain into an FSDP shard axis of size four. The
``model.norm.weight`` FQN is configured under ``fsdp_config`` so it
remains FSDP-managed but is not sharded on the HSDP shard axis.  Forward and
backward prefetch are both enabled with depth 1.
"""
# pylint: disable=C0413,C9002,E1123
from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from hyper_models._transformers.infrastructure import instantiate_infrastructure
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.components.distributed.infrastructure import (
    DistributedSetup,
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_models.components.distributed.sharding_applier import apply_sharding_plan
from hyper_models.components.utils.device import get_device_type
from hyper_models.config.manager import parse_training_args
from hyper_models.trainer.config import TrainerConfig
from hyper_parallel import (
    DTensor,
    DeviceMesh,
    HSDPModule,
    SkipDTensorDispatch,
    hsdp_sync_stream,
)
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate, Shard
from hyper_parallel.core.utils import clip_grad_norm_


_TEST_YAML_DIRECTORY = Path(__file__).with_name("test_yamls")
_LOG_DIRECTORY = Path("outputs/dualmode_trainer/fsdp_accuracy/adamw")
_INIT_SEED = 31415
_DATA_SEED = 27182
_TRAINING_STEPS = 100
_BATCH_SIZE = 8
_SEQUENCE_LENGTH = 8
_VOCAB_SIZE = 128
_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_NUM_LAYERS = 20
_QWEN2_NUM_LAYERS = 2
_NUM_HEADS = 4
_NUM_KEY_VALUE_HEADS = 4
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOKEN = 2
_RTOL = 5.0e-3
_ATOL = 5.0e-3


def _parse_config(config_name: str) -> TrainerConfig:
    """Resolve one accuracy-test YAML through the normal Trainer parser."""
    return parse_training_args([str(_TEST_YAML_DIRECTORY / config_name)])


def _build_model_config(
    use_moe: bool = False,
    use_qwen2_moe: bool = False,
) -> LlamaConfig | Qwen2MoeConfig | Qwen3MoeConfig:
    """Return the small HF dense or MoE configuration used by both copies."""
    if use_qwen2_moe:
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
    if use_moe:
        return Qwen3MoeConfig(
            vocab_size=_VOCAB_SIZE,
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            moe_intermediate_size=_INTERMEDIATE_SIZE,
            num_hidden_layers=_NUM_LAYERS,
            num_attention_heads=_NUM_HEADS,
            num_key_value_heads=_NUM_KEY_VALUE_HEADS,
            num_experts=_NUM_EXPERTS,
            num_experts_per_tok=_NUM_EXPERTS_PER_TOKEN,
            max_position_embeddings=_SEQUENCE_LENGTH,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            use_cache=False,
        )
    return LlamaConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KEY_VALUE_HEADS,
        max_position_embeddings=_SEQUENCE_LENGTH,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _build_standalone_model(
    device: torch.device,
    use_moe: bool = False,
    use_qwen2_moe: bool = False,
) -> LlamaForCausalLM | Qwen2MoeForCausalLM | Qwen3MoeForCausalLM:
    """Build the unsharded reference model from the fixed initialization seed."""
    torch.manual_seed(_INIT_SEED)
    model_config = _build_model_config(use_moe, use_qwen2_moe)
    if use_qwen2_moe:
        model_class = Qwen2MoeForCausalLM
    else:
        model_class = Qwen3MoeForCausalLM if use_moe else LlamaForCausalLM
    model = model_class(model_config)
    if use_qwen2_moe:
        _install_qwen2_rowwise_bias(model)
    return model.to(device=device).train()


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
    use_moe: bool = False,
    use_qwen2_moe: bool = False,
) -> tuple[torch.nn.Module, dict[str, tuple[Placement, DeviceMesh]]]:
    """Build and wrap the model through the dual-mode TP/FSDP infrastructure."""
    torch.manual_seed(_INIT_SEED)
    model_config = _build_model_config(use_moe, use_qwen2_moe)
    if use_qwen2_moe:
        model_class = Qwen2MoeForCausalLM
    else:
        model_class = Qwen3MoeForCausalLM if use_moe else LlamaForCausalLM
    model = model_class(model_config)
    if use_qwen2_moe:
        _install_qwen2_rowwise_bias(model)
    model = model.to(device=device).train()

    mesh_context = distributed_setup.mesh_context
    sharding_planner, fsdp_manager, _ = instantiate_infrastructure(
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
        expected_nested_unit_count=(
            _QWEN2_NUM_LAYERS if use_qwen2_moe else _NUM_LAYERS
        ) * (2 if use_moe or use_qwen2_moe else 1),
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


def _slice_along_dimension(
    tensor: torch.Tensor,
    dimension: int,
    index: int,
    size: int,
) -> torch.Tensor:
    """Select one contiguous shard along a tensor dimension."""
    normalized_dimension = dimension if dimension >= 0 else tensor.ndim + dimension
    if normalized_dimension < 0 or normalized_dimension >= tensor.ndim:
        raise ValueError(f"invalid gradient shard dimension {dimension} for shape {tuple(tensor.shape)}")
    if tensor.shape[normalized_dimension] % size != 0:
        raise ValueError(
            f"gradient shape {tuple(tensor.shape)} is not divisible by shard size {size} "
            f"on dimension {dimension}"
        )
    shard_length = tensor.shape[normalized_dimension] // size
    start = index * shard_length
    slices = [slice(None)] * tensor.ndim
    slices[normalized_dimension] = slice(start, start + shard_length)
    return tensor[tuple(slices)]


def _expected_local_gradient(
    full_gradient: torch.Tensor,
    parameter_name: str,
    source_shard_info_by_fqn: dict[str, tuple[tuple[Placement, ...], DeviceMesh]],
    mesh_context: MeshContext,
    replicate_parameter_name: str,
) -> torch.Tensor:
    """Map a standalone full gradient to the current HSDP + TP local shard."""
    fsdp_non_moe_mesh = mesh_context.fsdp_non_moe_mesh
    if fsdp_non_moe_mesh is None:
        raise RuntimeError("gradient comparison requires a dense FSDP mesh")
    placements, source_mesh = source_shard_info_by_fqn.get(
        parameter_name,
        ((Replicate(),), fsdp_non_moe_mesh["tp"]),
    )
    placement = placements[0]
    local_gradient = full_gradient
    if isinstance(placement, Shard):
        source_mesh_dim = source_mesh.mesh_dim_names[0]
        local_gradient = _slice_along_dimension(
            local_gradient,
            placement.dim,
            source_mesh.get_local_rank(source_mesh_dim),
            source_mesh.mesh_shape[0],
        )
    elif not isinstance(placement, Replicate):
        raise ValueError(
            f"unsupported source placement {placement!r} for parameter {parameter_name}"
        )
    if parameter_name == replicate_parameter_name:
        return local_gradient
    fsdp_shard_mesh = fsdp_non_moe_mesh["fsdp_shard"]
    fsdp_shard_size = mesh_context.dp_shard_size
    if (
        mesh_context.fsdp_moe_mesh is not None
        and source_mesh is mesh_context.fsdp_moe_mesh["ep"]
    ):
        fsdp_shard_mesh = mesh_context.fsdp_moe_mesh["edp_shard"]
        fsdp_shard_size = mesh_context.edp_shard_size
    return _slice_along_dimension(
        local_gradient,
        0,
        fsdp_shard_mesh.get_local_rank(),
        fsdp_shard_size,
    )


def _get_local_gradient(parameter_name: str, parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return the optimizer parameter's local gradient shard for comparison."""
    gradient = parameter.grad
    if gradient is None:
        raise AssertionError(f"parameter {parameter_name} has no gradient")
    if isinstance(gradient, DTensor):
        gradient = gradient.to_local()
    return gradient.detach().cpu()


def _get_local_parameter(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return the optimizer parameter's current local value."""
    parameter_data = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    return parameter_data.detach().cpu()


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
) -> float:
    """Reconstruct and compare every distributed parameter or gradient."""
    reference_parameters = dict(standalone_model.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    maximum_error = 0.0
    for parameter_name, reference_parameter in reference_parameters.items():
        reference_tensor = reference_parameter.grad if gradients else reference_parameter
        distributed_parameter = distributed_parameters[parameter_name]
        distributed_tensor = distributed_parameter.grad if gradients else distributed_parameter
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
        expected_global = reference_tensor.detach()
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
                atol=_ATOL if gradients else 1.0e-6,
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


def _compare_step(
    step_index: int,
    standalone_model: torch.nn.Module,
    distributed_model: torch.nn.Module,
    global_tokens: torch.Tensor,
    global_targets: torch.Tensor,
    local_tokens: torch.Tensor,
    local_targets: torch.Tensor,
    local_position_ids: torch.Tensor,
    local_batch_start: int,
    local_batch_end: int,
    local_sequence_start: int,
    local_sequence_end: int,
    source_shard_info_by_fqn: dict[str, tuple[tuple[Placement, ...], DeviceMesh]],
    mesh_context: MeshContext,
    replicate_parameter_name: str,
    standalone_optimizer: torch.optim.Optimizer,
    distributed_optimizer: torch.optim.Optimizer,
    standalone_log: TextIO | None,
    distributed_log: TextIO | None,
    global_view_only: bool = False,
) -> tuple[float, float, float, float]:
    """Run one reference/distributed step and compare loss plus every gradient."""
    activation_records: dict[str, dict[str, torch.Tensor]] = {
        "standalone": {},
        "distributed": {},
    }
    activation_hooks = []

    def _record_activation(
        path: str,
        module: torch.nn.Module,
        records: dict[str, torch.Tensor],
    ) -> None:
        def _pre_hook(unused_module, inputs, kwargs):
            del unused_module
            hidden_states = inputs[0] if inputs else kwargs["hidden_states"]
            records[f"{path}.input"] = hidden_states.detach().clone()

        def _forward_hook(unused_module, unused_inputs, unused_kwargs, output):
            del unused_module, unused_inputs, unused_kwargs
            main_output = output[0] if isinstance(output, (tuple, list)) else output
            records[f"{path}.output"] = main_output.detach().clone()

        activation_hooks.append(module.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        activation_hooks.append(module.register_forward_hook(_forward_hook, with_kwargs=True))

    trace_qwen2 = step_index == 0 and hasattr(
        standalone_model.model.layers[0].mlp,
        "shared_expert_gate",
    )
    if trace_qwen2:
        for model_name, model in (
            ("standalone", standalone_model),
            ("distributed", distributed_model),
        ):
            layer = model.model.layers[0]
            records = activation_records[model_name]
            _record_activation("attention", layer.self_attn, records)
            _record_activation("moe", layer.mlp, records)

    standalone_optimizer.zero_grad(set_to_none=True)
    standalone_logits = standalone_model(input_ids=global_tokens).logits
    standalone_full_loss = _cross_entropy_sum(standalone_logits, global_targets)
    with torch.no_grad():
        standalone_local_loss = _cross_entropy_sum(
            standalone_logits[
                local_batch_start:local_batch_end,
                local_sequence_start:local_sequence_end,
            ],
            local_targets,
        )
    standalone_full_loss.backward()

    distributed_optimizer.zero_grad(set_to_none=True)
    distributed_logits = distributed_model(
        input_ids=local_tokens,
        position_ids=local_position_ids,
    ).logits
    for hook in activation_hooks:
        hook.remove()
    if trace_qwen2:
        for activation_name, full_activation in activation_records["standalone"].items():
            actual_activation = activation_records["distributed"][activation_name]
            activation_sequence_start = (
                local_sequence_start + mesh_context.tp_rank * actual_activation.shape[1]
            )
            activation_sequence_end = activation_sequence_start + actual_activation.shape[1]
            expected_activation = full_activation[
                local_batch_start:local_batch_end,
                activation_sequence_start:activation_sequence_end,
            ]
            absolute_error = torch.abs(actual_activation - expected_activation)
            maximum_error = float(torch.max(absolute_error))
            mean_error = float(torch.mean(absolute_error))
            if maximum_error > 1.0e-3:
                raise AssertionError(
                    f"layer 0 {activation_name} is the first traced activation mismatch: "
                    f"max_abs={maximum_error:.8e}, mean_abs={mean_error:.8e}"
                )
    distributed_loss = _cross_entropy_sum(distributed_logits, local_targets)
    distributed_loss.backward(
        torch.tensor(
            1.0 / mesh_context.tp_size,
            dtype=distributed_loss.dtype,
            device=distributed_loss.device,
        )
    )
    hsdp_sync_stream()

    standalone_parameters = dict(standalone_model.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    if standalone_parameters.keys() != distributed_parameters.keys():
        raise AssertionError("standalone and distributed parameter names differ")

    maximum_gradient_error = 0.0
    for parameter_name, standalone_parameter in standalone_parameters.items():
        if global_view_only:
            continue
        if ".experts." in parameter_name:
            replica_size = mesh_context.fsdp_moe_mesh["edp_replicate"].mesh_shape[0]
        else:
            replica_size = mesh_context.fsdp_non_moe_mesh["fsdp_replicate"].mesh_shape[0]
        if replica_size > 1:
            continue
        expected_gradient = _expected_local_gradient(
            standalone_parameter.grad.detach().cpu(),
            parameter_name,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
        )
        actual_gradient = _get_local_gradient(
            parameter_name,
            distributed_parameters[parameter_name],
        )
        if actual_gradient.shape != expected_gradient.shape:
            raise AssertionError(
                f"step {step_index}, parameter {parameter_name}: expected gradient shape "
                f"{tuple(expected_gradient.shape)}, got {tuple(actual_gradient.shape)}"
            )
        np.testing.assert_allclose(
            actual_gradient.numpy(),
            expected_gradient.numpy(),
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"step {step_index}, parameter {parameter_name} gradient mismatch",
        )
        maximum_gradient_error = max(
            maximum_gradient_error,
            float(torch.max(torch.abs(actual_gradient - expected_gradient))),
        )

    if step_index == 0 or global_view_only:
        global_gradient_error = _compare_global_parameter_view(
            f"step {step_index} gradients",
            standalone_model,
            distributed_model,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
            gradients=True,
        )
        maximum_gradient_error = max(maximum_gradient_error, global_gradient_error)

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
        standalone_model.parameters(),
        max_norm=float("inf"),
    ))
    distributed_grad_norm = float(clip_grad_norm_(
        distributed_model,
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
    if standalone_log is not None and distributed_log is not None:
        _write_step_log(
            standalone_log,
            step_index,
            standalone_loss_value,
            standalone_grad_norm,
        )
        _write_step_log(
            distributed_log,
            step_index,
            distributed_loss_value,
            distributed_grad_norm,
            loss_error=abs(distributed_loss_value - standalone_loss_value),
            grad_norm_error=grad_norm_error,
            max_parameter_grad_error=maximum_gradient_error,
        )

    with SkipDTensorDispatch():
        standalone_optimizer.step()
        distributed_optimizer.step()

    if step_index == 0 or global_view_only:
        _compare_global_parameter_view(
            f"step {step_index} updated weights",
            standalone_model,
            distributed_model,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
            gradients=False,
        )

    for parameter_name, standalone_parameter in standalone_parameters.items():
        if global_view_only:
            continue
        expected_parameter = _expected_local_gradient(
            standalone_parameter.detach().cpu(),
            parameter_name,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
        )
        actual_parameter = _get_local_parameter(distributed_parameters[parameter_name])
        np.testing.assert_allclose(
            actual_parameter.numpy(),
            expected_parameter.numpy(),
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"step {step_index}, parameter {parameter_name} optimizer update mismatch",
        )
    return standalone_loss_value, distributed_loss_value, maximum_gradient_error, grad_norm_error


def _run_accuracy_case(
    config_name: str,
    case_name: str,
    use_moe: bool = False,
    use_qwen2_moe: bool = False,
) -> None:
    """Compare one eight-card dual-mode FSDP case with standalone training."""
    config = _parse_config(config_name)
    initialize_distributed(backend=config.training.backend)
    device_type = get_device_type()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(device_type, local_rank)
    world_size = dist.get_world_size()

    distributed_setup = create_distributed_setup_from_config(config)
    mesh_context = distributed_setup.mesh_context
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

    standalone_model = _build_standalone_model(device, use_moe, use_qwen2_moe)
    distributed_model, source_shard_info_by_fqn = _build_dual_mode_model(
        distributed_setup,
        device,
        use_moe,
        use_qwen2_moe,
    )
    device_mesh = mesh_context.device_mesh
    fsdp_non_moe_mesh = mesh_context.fsdp_non_moe_mesh
    if device_mesh is None or fsdp_non_moe_mesh is None:
        raise RuntimeError("dual-mode accuracy test requires device and dense FSDP meshes")

    tp_size = mesh_context.tp_size
    if use_moe or use_qwen2_moe:
        if mesh_context.fsdp_moe_mesh is None:
            raise RuntimeError("MoE accuracy test requires an expert FSDP mesh")
        if mesh_context.fsdp_moe_mesh.mesh_shape[-1] != mesh_context.ep_size:
            raise AssertionError(
                "MoE expert mesh EP dimension does not match ep_size"
            )
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
    replicate_parameter_name = "model.norm.weight"

    if use_qwen2_moe:
        _compare_global_parameter_view(
            "initial weights",
            standalone_model,
            distributed_model,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
            gradients=False,
        )

    if use_qwen2_moe:
        standalone_optimizer = torch.optim.SGD(standalone_model.parameters(), lr=1.0e-4)
        distributed_optimizer = torch.optim.SGD(distributed_model.parameters(), lr=1.0e-4)
    else:
        standalone_optimizer = config.optimizer.build(model=standalone_model).get_optimizer()
        distributed_optimizer = config.optimizer.build(model=distributed_model).get_optimizer()
    standalone_log = None
    distributed_log = None
    if dist.get_rank() == 0:
        _LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        parallel_mode = case_name.removesuffix("_accuracy")
        standalone_log = (_LOG_DIRECTORY / f"{parallel_mode}_standalone_adamw.log").open(
            "w", encoding="utf-8"
        )
        distributed_log = (_LOG_DIRECTORY / f"{parallel_mode}_dist_adamw.log").open(
            "w", encoding="utf-8"
        )
    maximum_loss_error = 0.0
    maximum_gradient_error = 0.0
    maximum_grad_norm_error = 0.0
    for step_index in range(_TRAINING_STEPS):
        global_tokens, global_targets = _build_global_batch(step_index, device)
        local_batch = shard_batch_for_cp(
            {
                "input_ids": global_tokens[local_batch_start:local_batch_end],
                "labels": global_targets[local_batch_start:local_batch_end],
            },
            mesh_context.cp_mesh,
        )
        local_tokens = local_batch["input_ids"]
        local_targets = local_batch["labels"]
        local_position_ids = torch.arange(
            local_sequence_start,
            local_sequence_end,
            device=device,
        ).unsqueeze(0).expand(local_batch_size, -1)
        standalone_loss, distributed_loss, step_gradient_error, step_grad_norm_error = _compare_step(
            step_index,
            standalone_model,
            distributed_model,
            global_tokens,
            global_targets,
            local_tokens,
            local_targets,
            local_position_ids,
            local_batch_start,
            local_batch_end,
            local_sequence_start,
            local_sequence_end,
            source_shard_info_by_fqn,
            mesh_context,
            replicate_parameter_name,
            standalone_optimizer,
            distributed_optimizer,
            standalone_log,
            distributed_log,
            global_view_only=use_qwen2_moe,
        )
        step_loss_error = abs(standalone_loss - distributed_loss)
        maximum_loss_error = max(maximum_loss_error, step_loss_error)
        maximum_gradient_error = max(maximum_gradient_error, step_gradient_error)
        maximum_grad_norm_error = max(maximum_grad_norm_error, step_grad_norm_error)
        if dist.get_rank() == 0:
            print(
                f"[{case_name} step {step_index}] standalone_loss={standalone_loss:.6f} "
                f"distributed_loss={distributed_loss:.6f} "
                f"loss_error={step_loss_error:.6e} "
                f"grad_norm_error={step_grad_norm_error:.6e} "
                f"max_grad_error={step_gradient_error:.6e} "
                f"(dp={mesh_context.dp_size}, cp={mesh_context.cp_size}, "
                f"fsdp_replicate={mesh_context.dp_replicate_size}, "
                f"fsdp_shard={mesh_context.dp_shard_size}, "
                f"edp_shard={mesh_context.edp_shard_size}, "
                f"tp={tp_size}, ep={mesh_context.ep_size})"
            )
        standalone_model.zero_grad(set_to_none=True)

    if standalone_log is not None and distributed_log is not None:
        standalone_log.close()
        distributed_log.close()

    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"[{case_name}] passed all loss and parameter-gradient checks; "
            f"max_loss_error={maximum_loss_error:.6e}, "
            f"max_grad_norm_error={maximum_grad_norm_error:.6e}, "
            f"max_grad_error={maximum_gradient_error:.6e}"
        )
    destroy_process_group()


def test_hsdp_tp_accuracy() -> None:
    """Validate the eight-card HSDP(2x2)+TP(2) Trainer path."""
    _run_accuracy_case("hsdp_tp_accuracy.yaml", "hsdp_tp_accuracy")


def test_dp_cp_tp_accuracy() -> None:
    """Validate FSDP shard four over a basic DP(2)+CP(2)+TP(2) mesh."""
    _run_accuracy_case("dp_cp_tp_accuracy.yaml", "dp_cp_tp_accuracy")


def test_hsdp_tp_ep_moe_accuracy() -> None:
    """Validate HF MoE with HSDP, TP, and derived EDP/EP meshes."""
    _run_accuracy_case(
        "hsdp_tp_ep_moe_accuracy.yaml",
        "hsdp_tp_ep_moe_accuracy",
        use_moe=True,
    )


def test_qwen2_tp_cp_ep_fsdp_global_accuracy() -> None:
    """Validate Qwen2-MoE TP×CP×EP×FSDP using reconstructed global views."""
    _run_accuracy_case(
        "qwen2_tp_cp_ep_fsdp_global_accuracy.yaml",
        "qwen2_tp_cp_ep_fsdp_global_accuracy",
        use_qwen2_moe=True,
    )
