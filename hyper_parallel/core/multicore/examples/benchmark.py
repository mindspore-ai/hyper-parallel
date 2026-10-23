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

"""Eight-NPU common versus three-Mega-kernel MoE Transformer optimizer benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.components.optim import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.multicore import MegaMoeExperts, shmem
from hyper_parallel.core.multicore.examples.model import (
    MoeTransformerConfig,
    MoeTransformerModel,
)
from hyper_parallel.core.multicore.examples.moe import NativeExperts, _local_tensor
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _resolve_capacity_factors,
)
from hyper_parallel.core.optimizer import get_hyper_optimizer

_WORLD_SIZE = 8
_BATCH_SIZE = 1
_SEQUENCE_LENGTH = MoeTransformerConfig.local_num_tokens
_DTYPE = torch.bfloat16
_RTOL = 2e-2
_ATOL = 2e-3


@dataclass(frozen=True)
class _Workload:
    """Objects needed by one complete optimizer step."""

    model: MoeTransformerModel
    optimizer: Float16OptimizerWithFloat16Params
    input_ids: torch.Tensor
    labels: torch.Tensor
    world_size: int
    device: torch.device


@dataclass
class _TensorComparison:
    """Device-side pass state and per-tensor comparison diagnostics."""

    passed: torch.Tensor
    tensor_names: list[str] = field(default_factory=list)
    absolute_errors: list[torch.Tensor] = field(default_factory=list)
    normalized_errors: list[torch.Tensor] = field(default_factory=list)


@dataclass
class _BenchmarkContext:
    """Runtime values and owned models shared by all benchmark phases."""

    args: argparse.Namespace
    rank: int
    world_size: int
    device: torch.device
    config: MoeTransformerConfig
    initialized_here: bool
    models: dict[str, MoeTransformerModel] = field(default_factory=dict)


def _parse_mega_kernels(value: str) -> tuple[str, ...]:
    """Parse explicit comma-separated component names without changing model structure."""
    names = tuple(part.strip() for part in value.split(","))
    if not names or any(name not in ("moe", "gate", "mhc") for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("use a comma-separated subset of moe,gate,mhc without duplicates")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the minimal optimizer benchmark interface.

    Args:
        argv: Optional command-line arguments; ``None`` reads the process arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare common and selected Mega kernels in an eight-NPU MoE Transformer demo.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--initial-capacity-factor", type=float, default=None,
                        help="push only: initial receive factor (default: 1.25)")
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--capacity-growth-factor", type=float, default=None,
                        help="push only: growth multiplier (default: 1.25; 1.0 fits actual demand)")
    parser.add_argument(
        "--mega-kernels", type=_parse_mega_kernels, default=("moe", "gate", "mhc"),
        help="optimized backend kernels; common always uses the same architecture with Torch references",
    )
    parser.add_argument(
        "--accuracy-mode", choices=("components", "full_step"), default="components",
        help="components isolates mHC rounding from discrete TopK; full_step checks all enabled kernels together",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="output/moe_transformer_benchmark.json",
        help="rank-zero JSON result path",
    )
    args = parser.parse_args(argv)
    args.initial_capacity_factor, args.capacity_growth_factor = _resolve_capacity_factors(
        args.dispatch_mode, args.initial_capacity_factor, args.capacity_growth_factor)
    if args.warmup_steps < 1:
        raise ValueError(f"warmup_steps must be positive, got {args.warmup_steps}.")
    if args.measured_steps <= 0:
        raise ValueError(f"measured_steps must be positive, got {args.measured_steps}.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be finite and positive, got {args.learning_rate}."
        )
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError(
            f"weight_decay must be finite and nonnegative, got {args.weight_decay}."
        )
    return args


def _init_runtime() -> tuple[int, int, torch.device, bool]:
    """Bind the NPU and return rank, world size, device, and group ownership."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != _WORLD_SIZE:
        if initialized_here:
            dist.destroy_process_group()
        raise ValueError(
            f"MoE Transformer benchmark requires {_WORLD_SIZE} ranks, got {world_size}."
        )
    return rank, world_size, torch.device("npu", local_rank), initialized_here


def _build_model(
    config: MoeTransformerConfig, device: torch.device, mega_kernels: tuple[str, ...],
) -> MoeTransformerModel:
    """Build BF16 dense weights while preserving FP32 mHC predictors and rotary buffers."""
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(_DTYPE)
        model = MoeTransformerModel(config, ep_group=dist.group.WORLD, mega_kernels=mega_kernels)
    finally:
        torch.set_default_dtype(default_dtype)
    return model.to(device=device)


def _reset_seed(seed: int) -> None:
    """Reset CPU and NPU initializers before constructing one backend."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _build_backend_model(
    config: MoeTransformerConfig,
    device: torch.device,
    seed: int,
    backend: str,
    mega_kernels: tuple[str, ...],
) -> MoeTransformerModel:
    """Build equal architectures and select common or requested Mega execution paths."""
    if backend not in ("common", "mega"):
        raise ValueError(f"unsupported MoE Transformer backend {backend!r}.")
    selected = mega_kernels if backend == "mega" else ()
    _reset_seed(seed)
    model = _build_model(config, device, selected)
    if "moe" in selected:
        return model
    ep_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(config.ep_size,),
        mesh_dim_names=("ep",),
    )
    for layer in model.layers:
        layer.mlp.use_native_experts(ep_mesh, device)
    return model


def _build_workload(
    model: MoeTransformerModel,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    world_size: int,
    device: torch.device,
) -> _Workload:
    """Attach an independent FP32-main-parameter AdamW optimizer."""
    chained_optimizer = get_hyper_optimizer(
        model=model,
        muon_params=[],
        adamw_params=[
            {
                "params": list(model.parameters()),
                "weight_decay": args.weight_decay,
            }
        ],
        adamw_kwargs={
            "lr": args.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
    )
    return _Workload(
        model=model,
        optimizer=Float16OptimizerWithFloat16Params(
            chained_optimizer,
            model,
        ),
        input_ids=input_ids,
        labels=labels,
        world_size=world_size,
        device=device,
    )


def _build_batch(
    config: MoeTransformerConfig,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic rank-local token IDs and labels."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000 + rank)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
        dtype=torch.int64,
    ).to(device)
    return input_ids, input_ids.clone()


def _dense_gradients(model: MoeTransformerModel) -> list[torch.Tensor]:
    """Return gradients of replicated parameters, excluding expert shards."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) not in expert_parameters and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("MoE Transformer optimizer step produced no dense gradients.")
    return gradients


def _synchronize_dense_gradients(model: MoeTransformerModel, world_size: int) -> None:
    """Average replicated gradients in dtype buckets, leaving EP expert shards local."""
    buckets: dict[torch.dtype, list[torch.Tensor]] = {}
    for gradient in _dense_gradients(model):
        buckets.setdefault(gradient.dtype, []).append(gradient)
    for gradients in buckets.values():
        dist.all_reduce_coalesced(gradients, op=dist.ReduceOp.SUM)
        torch._foreach_div_(gradients, world_size)  # pylint: disable=protected-access


def _prepare_expert_main_gradients(model: MoeTransformerModel) -> None:
    """Attach EP layout to local expert gradients for the mixed-precision optimizer."""
    for parameter in model.expert_parameters():
        if isinstance(parameter, DTensor) and parameter.grad is not None:
            parameter.main_grad = DTensor.from_local(
                parameter.grad.detach(), parameter.device_mesh, parameter.placements, shape=tuple(parameter.shape),
            )


def _optimizer_step(
    workload: _Workload,
    *,
    capture_gradients: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Run one AdamW step and optionally retain first-step gradients."""
    with SkipDTensorDispatch():
        workload.optimizer.zero_grad(set_to_none=True)
    result = workload.model(workload.input_ids, workload.labels)
    loss = result["loss"]
    logits = result["logits"]
    if loss is None:
        raise RuntimeError("MoE Transformer model did not return a training loss.")
    if logits is None:
        raise RuntimeError("MoE Transformer model did not return logits.")
    loss.backward()
    _synchronize_dense_gradients(workload.model, workload.world_size)
    gradients = None
    if capture_gradients:
        gradients = {}
        canonical_gradients = _canonical_tensors(workload.model, gradients=True)
        for name, tensor in canonical_gradients.items():
            gradients[name] = tensor.detach().clone()
    _prepare_expert_main_gradients(workload.model)
    with SkipDTensorDispatch(no_skip={torch.zeros_like}):
        workload.optimizer.step()
    return loss.detach(), logits.detach(), gradients


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    """Reduce a host predicate across all benchmark ranks."""
    status = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return bool(status.cpu().item())


def _all_gradients_finite(
    gradients: dict[str, torch.Tensor],
    device: torch.device,
) -> bool:
    """Return whether every captured model gradient is finite."""
    finite = torch.ones((), dtype=torch.int32, device=device)
    for gradient in gradients.values():
        finite = torch.minimum(
            finite,
            torch.isfinite(gradient).all().to(torch.int32),
        )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.cpu().item())


def _rank_max(value: float, device: torch.device) -> float:
    """Return a floating-point measurement reduced by maximum across ranks."""
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu().item())


def _timed_step(workload: _Workload) -> tuple[torch.Tensor, float]:
    """Run one optimizer step from a synchronized, untimed boundary."""
    dist.barrier()
    torch.npu.synchronize(workload.device)
    start = time.perf_counter()
    loss, _, _ = _optimizer_step(workload)
    torch.npu.synchronize(workload.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return loss, _rank_max(elapsed_ms, workload.device)


def _validate_first_step(
    workload: _Workload,
) -> tuple[
    dict[str, Any],
    float,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Validate finite work and dense/expert updates on the first step."""
    dense_parameter = workload.model.lm_head.weight
    expert_parameter = _local_tensor(workload.model.expert_parameters()[0])
    dense_before = dense_parameter.detach().reshape(-1)[:4096].clone()
    expert_before = expert_parameter.detach().reshape(-1)[:4096].clone()
    dist.barrier()
    torch.npu.synchronize(workload.device)
    start = time.perf_counter()
    loss, logits, gradients = _optimizer_step(
        workload,
        capture_gradients=True,
    )
    torch.npu.synchronize(workload.device)
    if gradients is None:
        raise RuntimeError("MoE Transformer first optimizer step did not capture gradients.")
    first_step_ms = _rank_max(
        (time.perf_counter() - start) * 1000.0,
        workload.device,
    )
    validation = {
        "loss": float(loss.float().cpu().item()),
        "loss_finite": _all_ranks_true(
            bool(torch.isfinite(loss).cpu().item()),
            workload.device,
        ),
        "gradients_finite": _all_gradients_finite(gradients, workload.device),
        "dense_parameter_updated": _all_ranks_true(
            not torch.equal(dense_before, dense_parameter.detach().reshape(-1)[:4096]),
            workload.device,
        ),
        "expert_parameter_updated": _all_ranks_true(
            not torch.equal(
                expert_before, expert_parameter.detach().reshape(-1)[:4096]
            ),
            workload.device,
        ),
    }
    if not all(value for name, value in validation.items() if name != "loss"):
        raise RuntimeError(f"optimizer-step validation failed: {validation}.")
    return validation, first_step_ms, loss, logits, gradients


def _parameter_value(
    parameter: torch.Tensor,
    *,
    gradients: bool,
) -> torch.Tensor:
    """Return one local parameter or its required gradient."""
    value = parameter.grad if gradients else parameter
    if value is None:
        raise RuntimeError("MoE Transformer accuracy comparison found a missing gradient.")
    return _local_tensor(value)


def _canonical_tensors(
    model: MoeTransformerModel,
    *,
    gradients: bool,
) -> dict[str, torch.Tensor]:
    """Return dense and packed local-expert tensors under backend-neutral names."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    tensors = {
        f"dense.{name}": _parameter_value(parameter, gradients=gradients)
        for name, parameter in model.named_parameters()
        if id(parameter) not in expert_parameters and (not gradients or parameter.requires_grad)
    }
    for layer_index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        if isinstance(experts, MegaMoeExperts):
            gate_up = _parameter_value(
                experts.gate_up_weight,
                gradients=gradients,
            )
            down = _parameter_value(experts.down_weight, gradients=gradients)
        elif isinstance(experts, NativeExperts):
            grouped = experts.module
            gate = _parameter_value(grouped.w1, gradients=gradients)
            up = _parameter_value(grouped.w3, gradients=gradients)
            down = _parameter_value(grouped.w2, gradients=gradients)
            gate_up = torch.cat(
                (gate.transpose(1, 2), up.transpose(1, 2)),
                dim=-1,
            )
            down = down.transpose(1, 2)
        else:
            raise TypeError(
                f"unsupported MoE Transformer expert backend {type(experts).__name__}."
            )
        tensors[f"experts.{layer_index}.gate_up_weight"] = gate_up
        tensors[f"experts.{layer_index}.down_weight"] = down
    return tensors


def _compare_tensor_maps(
    name: str,
    common_tensors: dict[str, torch.Tensor],
    mega_tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare tensor maps on device and reduce diagnostics across ranks."""
    if common_tensors.keys() != mega_tensors.keys():
        raise RuntimeError(
            f"{name} tensor names differ: common={sorted(common_tensors)}, "
            f"mega={sorted(mega_tensors)}."
        )
    comparison = _TensorComparison(
        passed=torch.ones((), dtype=torch.int32, device=device)
    )
    for tensor_name, common_tensor in common_tensors.items():
        _compare_tensor_pair(
            name,
            tensor_name,
            common_tensor,
            mega_tensors[tensor_name],
            comparison,
            rtol=rtol,
            atol=atol,
        )
    return _finalize_tensor_comparison(name, comparison, rtol=rtol, atol=atol)


def _compare_tensor_pair(
    comparison_name: str,
    tensor_name: str,
    common_tensor: torch.Tensor,
    mega_tensor: torch.Tensor,
    comparison: _TensorComparison,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Accumulate device-side diagnostics for one corresponding tensor pair."""
    comparison.tensor_names.append(tensor_name)
    if mega_tensor.shape != common_tensor.shape:
        raise RuntimeError(
            f"{comparison_name}.{tensor_name} shape differs: common={tuple(common_tensor.shape)}, "
            f"mega={tuple(mega_tensor.shape)}."
        )
    common_float = common_tensor.detach().float()
    mega_float = mega_tensor.detach().float()
    difference = (mega_float - common_float).abs()
    comparison.passed = torch.minimum(
        comparison.passed,
        torch.isclose(mega_float, common_float, rtol=rtol, atol=atol)
        .all()
        .to(torch.int32),
    )
    comparison.absolute_errors.append(difference.max())
    tolerance = atol + rtol * common_float.abs()
    normalized_error = difference / tolerance.clamp_min(torch.finfo(torch.float32).tiny)
    comparison.normalized_errors.append(normalized_error.max())


def _finalize_tensor_comparison(
    name: str,
    comparison: _TensorComparison,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Reduce accumulated diagnostics and return the public result mapping."""
    absolute_by_tensor = torch.stack(comparison.absolute_errors)
    normalized_by_tensor = torch.stack(comparison.normalized_errors)
    dist.all_reduce(comparison.passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(absolute_by_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(normalized_by_tensor, op=dist.ReduceOp.MAX)
    maximum_absolute, absolute_index = absolute_by_tensor.max(dim=0)
    maximum_normalized, normalized_index = normalized_by_tensor.max(dim=0)
    result = {
        "passed": bool(comparison.passed.cpu().item()),
        "tensor_count": len(comparison.tensor_names),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(maximum_absolute.cpu().item()),
        "max_absolute_error_tensor": comparison.tensor_names[int(absolute_index.cpu().item())],
        "max_normalized_error": float(maximum_normalized.cpu().item()),
        "max_normalized_error_tensor": comparison.tensor_names[int(normalized_index.cpu().item())],
    }
    if not result["passed"]:
        raise RuntimeError(f"MoE Transformer common/Mega {name} comparison failed: {result}.")
    return result


def _measure_backend(
    workload: _Workload,
    validation: dict[str, Any],
    first_step_ms: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Finish warmup and measure one backend after its compared first step."""
    for _ in range(args.warmup_steps - 1):
        warmup_loss, warmup_latency = _timed_step(workload)
        del warmup_loss, warmup_latency
    latencies = []
    losses = []
    for _ in range(args.measured_steps):
        loss, latency = _timed_step(workload)
        latencies.append(latency)
        losses.append(float(loss.float().cpu().item()))
    managers = {}
    for layer in workload.model.layers:
        if isinstance(layer.mlp.experts, MegaMoeExperts):
            # Capacity-growth records belong to the internal shared execution-resource group.
            # pylint: disable-next=protected-access
            manager = layer.mlp.experts._resource_group.resources.heap_manager
            managers[id(manager)] = manager
    state = shmem.debug_state() if managers else {}
    return {
        "shmem_heap_bytes": state.get("config", {}).get("heap_size_bytes", 0),
        "heap_growth": [record for manager in managers.values() for record in manager.growth_records],
        "validation": validation,
        "validation_optimizer_step_ms": first_step_ms,
        "validation_optimizer_step_index": (
            2 if args.accuracy_mode == "components" and "mhc" in args.mega_kernels else 1
        ),
        "steady_state_optimizer_step_ms": {
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "measured_losses": losses,
    }


def _write_result(
    args: argparse.Namespace,
    config: MoeTransformerConfig,
    rank: int,
    accuracy: dict[str, Any],
    backends: dict[str, dict[str, Any]],
) -> None:
    """Write common-versus-MegaMoe accuracy and performance from rank zero."""
    common_median = backends["common"]["steady_state_optimizer_step_ms"]["median"]
    mega_median = backends["mega"]["steady_state_optimizer_step_ms"]["median"]
    ratio = mega_median / common_median
    if rank == 0:
        result = {
            "architecture": "moe_transformer",
            "mega_kernels": list(args.mega_kernels),
            "accuracy_mode": args.accuracy_mode,
            "common_mhc_backend": "torch",
            "gate_route_backend": {"common": "torch",
                                   "mega": "native" if "gate" in args.mega_kernels else "torch"},
            "mhc_precision_reference": "torch",
            "topology": {"world_size": _WORLD_SIZE, "tp": 1, "ep": config.ep_size},
            "shape": {
                "batch_size": _BATCH_SIZE,
                "sequence_length": _SEQUENCE_LENGTH,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_experts": config.num_experts,
                "top_k": config.top_k,
                "initial_capacity_factor": config.initial_capacity_factor,
                "dispatch_mode": config.dispatch_mode,
                "capacity_growth_factor": config.capacity_growth_factor,
                "dtype": "bfloat16",
                "mhc_streams": 4,
                "scoring_func": "sqrtsoftplus",
            },
            "optimizer": {
                "name": "HyperParallel AdamW",
                "wrapper": "Float16OptimizerWithFloat16Params",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "model_parameter_dtype": "mixed_bfloat16_float32",
                "main_parameter_dtype": "float32",
            },
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "comparison": {
                "accuracy": accuracy,
                "performance": {
                    "order": ["common", "mega"],
                    "metric": "rank-max complete optimizer-step milliseconds",
                    "common_median_ms": common_median,
                    "mega_median_ms": mega_median,
                    "mega_over_common": ratio,
                    "mega_reduction_percent": (1.0 - ratio) * 100.0,
                },
            },
            "backends": backends,
            "runtime": {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
            },
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RESULT_JSON={path}")
        print(json.dumps(result, sort_keys=True))
    dist.barrier()


def _compare_first_step_accuracy(
    models: dict[str, MoeTransformerModel],
    workloads: dict[str, _Workload],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], float]]]:
    """Compare initial state, first forward/backward, and optimizer updates."""
    accuracy = {
        "initial_parameters": _compare_tensor_maps(
            "initial parameters",
            _canonical_tensors(models["common"], gradients=False),
            _canonical_tensors(models["mega"], gradients=False),
            device,
            rtol=0.0,
            atol=0.0,
        )
    }
    (
        common_validation,
        common_first_ms,
        common_loss,
        common_logits,
        common_gradients,
    ) = _validate_first_step(workloads["common"])
    (
        mega_validation,
        mega_first_ms,
        mega_loss,
        mega_logits,
        mega_gradients,
    ) = _validate_first_step(workloads["mega"])
    accuracy.update(
        {
            "forward": _compare_tensor_maps(
                "forward",
                {"loss": common_loss, "logits": common_logits},
                {"loss": mega_loss, "logits": mega_logits},
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
            "gradients": _compare_tensor_maps(
                "gradients",
                common_gradients,
                mega_gradients,
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
            "updated_parameters": _compare_tensor_maps(
                "updated parameters",
                _canonical_tensors(models["common"], gradients=False),
                _canonical_tensors(models["mega"], gradients=False),
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
        }
    )
    del common_gradients, mega_gradients
    del common_loss, mega_loss, common_logits, mega_logits
    torch.npu.empty_cache()
    first_steps = {
        "common": (common_validation, common_first_ms),
        "mega": (mega_validation, mega_first_ms),
    }
    return accuracy, first_steps


def _compare_mhc_component(context: _BenchmarkContext) -> dict[str, Any]:
    """Check fused mHC outputs and all input/parameter gradients on identical inputs."""
    hidden = context.config.hidden_size
    tokens = _SEQUENCE_LENGTH
    generator = torch.Generator(device="cpu").manual_seed(context.args.seed + 20_000 + context.rank)
    shapes = ((1, tokens, hidden), (1, tokens, 4, hidden), (1, tokens, 4),
              (1, tokens, 4), (1, tokens, 4, 4))
    dtypes = (_DTYPE, _DTYPE, torch.float32, torch.float32, torch.float32)
    inputs = tuple(torch.randn(shape, generator=generator).to(device=context.device, dtype=dtype)
                   for shape, dtype in zip(shapes, dtypes))
    probes = tuple(torch.randn(shape, generator=generator).to(device=context.device, dtype=dtype) / tokens
                   for shape, dtype in zip((shapes[1], shapes[2], shapes[3], shapes[4], shapes[0]),
                                           (_DTYPE, torch.float32, torch.float32, torch.float32, _DTYPE)))
    forwards, gradients = {}, {}
    for backend in ("reference", "common", "mega"):
        boundary = context.models["common" if backend == "reference" else backend].layers[0].attention_mhc
        state = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        outputs = boundary.reference_forward(*state) if backend == "reference" else boundary(*state)
        values = torch.autograd.grad(outputs, (*state, *boundary.parameters()), grad_outputs=probes)
        forwards[backend] = {str(index): output.detach() for index, output in enumerate(outputs)}
        gradients[backend] = {str(index): value.detach() for index, value in enumerate(values)}
    comparisons = {}
    for backend in ("common", "mega"):
        comparisons[f"{backend}_forward"] = _compare_tensor_maps(
            f"mHC {backend} forward", forwards["reference"], forwards[backend],
            context.device, rtol=_RTOL, atol=_ATOL,
        )
        comparisons[f"{backend}_gradients"] = _compare_tensor_maps(
            f"mHC {backend} gradients", gradients["reference"], gradients[backend],
            context.device, rtol=_RTOL, atol=_ATOL,
        )
    return comparisons


def _validate_accuracy(
    context: _BenchmarkContext, workloads: dict[str, _Workload],
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], float]]]:
    """Separate mHC component precision from complete-network discrete routing."""
    if context.args.accuracy_mode == "full_step" or "mhc" not in context.args.mega_kernels:
        accuracy, first_steps = _compare_first_step_accuracy(context.models, workloads, context.device)
        accuracy["scope"] = "full_step_with_selected_kernels"
        return accuracy, first_steps
    component = _compare_mhc_component(context)
    boundaries = [boundary for layer in context.models["mega"].layers
                  for boundary in (layer.attention_mhc, layer.ffn_mhc)]
    try:
        for boundary in boundaries:
            boundary.use_mega_mhc = False
        accuracy, first_steps = _compare_first_step_accuracy(context.models, workloads, context.device)
    finally:
        for boundary in boundaries:
            boundary.use_mega_mhc = True
    accuracy["scope"] = "mhc_component_and_full_step_with_common_mhc"
    accuracy["mhc_component"] = component
    for backend in ("common", "mega"):
        validation, latency, *captured = _validate_first_step(workloads[backend])
        del captured
        first_steps[backend] = (validation, latency)
    accuracy["selected_kernels_training_step"] = {
        backend: validation for backend, (validation, _) in first_steps.items()
    }
    torch.npu.empty_cache()
    return accuracy, first_steps


def _prepare_benchmark(argv: list[str] | None) -> _BenchmarkContext:
    """Parse arguments and initialize the distributed benchmark runtime."""
    args = parse_args(argv)
    rank, world_size, device, initialized_here = _init_runtime()
    config = replace(
        MoeTransformerConfig(),
        initial_capacity_factor=args.initial_capacity_factor,
        dispatch_mode=args.dispatch_mode,
        capacity_growth_factor=args.capacity_growth_factor,
    )
    return _BenchmarkContext(args, rank, world_size, device, config, initialized_here)


def _build_benchmark_workloads(context: _BenchmarkContext) -> dict[str, _Workload]:
    """Construct deterministic common and MegaMoe workloads."""
    input_ids, labels = _build_batch(
        context.config,
        context.rank,
        context.args.seed,
        context.device,
    )
    for backend in ("common", "mega"):
        context.models[backend] = _build_backend_model(
            context.config,
            context.device,
            context.args.seed,
            backend,
            tuple(context.args.mega_kernels),
        )
    return {
        backend: _build_workload(
            model,
            context.args,
            input_ids,
            labels,
            context.world_size,
            context.device,
        )
        for backend, model in context.models.items()
    }


def _measure_backends(
    workloads: dict[str, _Workload],
    first_steps: dict[str, tuple[dict[str, Any], float]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Measure both backends after their compared first steps."""
    measurements = {}
    for backend in ("common", "mega"):
        validation, latency_ms = first_steps[backend]
        measurements[backend] = _measure_backend(workloads[backend], validation, latency_ms, args)
    return measurements


def main(argv: list[str] | None = None) -> int:
    """Compare common and selected Mega kernels for MoE Transformer accuracy, then time A and B.

    Args:
        argv: Optional command-line arguments for tests or direct invocation.

    Returns:
        Zero after the distributed comparison completes successfully.
    """
    context = _prepare_benchmark(argv)
    try:
        workloads = _build_benchmark_workloads(context)
        accuracy, first_steps = _validate_accuracy(context, workloads)
        backends = _measure_backends(workloads, first_steps, context.args)
        _write_result(
            context.args,
            context.config,
            context.rank,
            accuracy,
            backends,
        )
        return 0
    finally:
        if sys.exc_info()[0] is None:
            for model in context.models.values():
                model.close()
            if context.initialized_here and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
