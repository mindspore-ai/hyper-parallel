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

"""Minimal eight-NPU Qwen MoE optimizer-step benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates CANN and the native payload before framework import.
# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu
from qwen_moe_model import QwenMoeConfig, QwenMoeModel

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.auto_models.components.optim import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.platform.torch.common import GroupedExperts

# pylint: enable=wrong-import-position

_WORLD_SIZE = 8
_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 1024
_DTYPE = torch.bfloat16
_RTOL = 2e-2
_ATOL = 2e-3


@dataclass(frozen=True)
class _Workload:
    """Objects needed by one complete optimizer step."""

    model: QwenMoeModel
    optimizer: Float16OptimizerWithFloat16Params
    input_ids: torch.Tensor
    labels: torch.Tensor
    world_size: int
    device: torch.device


class _CommonExpertsAdapter(torch.nn.Module):
    """Expose standard EP routed experts through the Qwen expert interface."""

    def __init__(
        self,
        config: QwenMoeConfig,
        source: MegaMoeExperts,
        ep_mesh: Any,
        device: torch.device,
    ) -> None:
        """Shard common experts and copy one local MegaMoe expert shard."""
        super().__init__()
        self.module = GroupedExperts(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_experts,
            use_grouped_mm=True,
        ).to(device=device, dtype=_DTYPE)
        ExpertParallel().apply(self.module, ep_mesh)
        gate_weight, up_weight = source.gate_up_weight.split(
            config.intermediate_size,
            dim=-1,
        )
        with torch.no_grad():
            _local_tensor(self.module.w1).copy_(gate_weight.transpose(1, 2))
            _local_tensor(self.module.w3).copy_(up_weight.transpose(1, 2))
            _local_tensor(self.module.w2).copy_(
                source.down_weight.transpose(1, 2)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the common routed-expert implementation."""
        if tokens_per_expert is None:
            tokens_per_expert = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64), minlength=self.module.num_experts,
            )
        routed_input, mapping = torch_npu.npu_moe_token_permute(
            hidden_states.reshape(-1, hidden_states.shape[-1]), topk_ids,
        )
        expert_output = self.module(routed_input, tokens_per_expert)
        output = torch_npu.npu_moe_token_unpermute(
            expert_output, mapping, probs=topk_weights.float().contiguous(),
        )
        return output.reshape(hidden_states.shape)

    @staticmethod
    def close() -> None:
        """Match the managed expert lifecycle without owning SHMEM."""


def _capacity_factor(value: str) -> float | None:
    """Parse ``none`` or a finite factor of at least one."""
    if value.lower() == "none":
        return None
    try:
        factor = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be 'none' or a number, got {value!r}."
        ) from error
    if not math.isfinite(factor) or factor < 1.0:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be finite and at least 1.0, got {value!r}."
        )
    return factor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the minimal optimizer benchmark interface."""
    parser = argparse.ArgumentParser(
        description="Run an eight-NPU Qwen MoE optimizer-step benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--expert-capacity-factor", type=_capacity_factor, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="output/qwen_moe_benchmark.json",
        help="rank-zero JSON result path",
    )
    args = parser.parse_args(argv)
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


def _init_runtime() -> tuple[int, int, torch.device]:
    """Bind the local NPU and initialize the fixed EP world."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != _WORLD_SIZE:
        raise ValueError(
            f"Qwen benchmark requires {_WORLD_SIZE} ranks, got {world_size}."
        )
    return rank, world_size, torch.device("npu", local_rank)


def _build_model(config: QwenMoeConfig, device: torch.device) -> QwenMoeModel:
    """Build BF16 model weights and keep rotary buffers in their native dtype."""
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(_DTYPE)
        model = QwenMoeModel(config, ep_group=dist.group.WORLD)
    finally:
        torch.set_default_dtype(default_dtype)
    return model.to(device=device)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local value of a plain tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _reset_seed(seed: int) -> None:
    """Reset CPU and NPU initializers before constructing one backend."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _build_backend_model(
    config: QwenMoeConfig,
    device: torch.device,
    seed: int,
    backend: str,
) -> QwenMoeModel:
    """Build one deterministically initialized MegaMoe or common model."""
    _reset_seed(seed)
    model = _build_model(config, device)
    if backend == "mega_moe":
        return model
    if backend != "common":
        raise ValueError(f"unsupported Qwen expert backend {backend!r}.")
    ep_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(config.ep_size,),
        mesh_dim_names=("ep",),
    )
    for layer in model.layers:
        source = layer.mlp.experts
        common = _CommonExpertsAdapter(config, source, ep_mesh, device)
        source.close()
        layer.mlp.experts = common
    return model


def _build_workload(
    model: QwenMoeModel,
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
    config: QwenMoeConfig,
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


def _dense_gradients(model: QwenMoeModel) -> list[torch.Tensor]:
    """Return gradients of replicated parameters, excluding expert shards."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) not in expert_parameters and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Qwen optimizer step produced no dense gradients.")
    return gradients


def _synchronize_dense_gradients(model: QwenMoeModel, world_size: int) -> None:
    """Average replicated dense gradients before the optimizer step."""
    gradients = _dense_gradients(model)
    dist.all_reduce_coalesced(gradients, op=dist.ReduceOp.SUM)
    torch._foreach_div_(gradients, world_size)  # pylint: disable=protected-access


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
        raise RuntimeError("Qwen model did not return a training loss.")
    if logits is None:
        raise RuntimeError("Qwen model did not return logits.")
    loss.backward()
    _synchronize_dense_gradients(workload.model, workload.world_size)
    gradients = None
    if capture_gradients:
        gradients = {
            name: tensor.detach().clone()
            for name, tensor in _canonical_tensors(
                workload.model,
                gradients=True,
            ).items()
        }
    with SkipDTensorDispatch():
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
        raise RuntimeError("Qwen first optimizer step did not capture gradients.")
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
        raise RuntimeError("Qwen accuracy comparison found a missing gradient.")
    return _local_tensor(value)


def _canonical_tensors(
    model: QwenMoeModel,
    *,
    gradients: bool,
) -> dict[str, torch.Tensor]:
    """Return dense and packed local-expert tensors under backend-neutral names."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    tensors = {
        f"dense.{name}": _parameter_value(parameter, gradients=gradients)
        for name, parameter in model.named_parameters()
        if id(parameter) not in expert_parameters
    }
    for layer_index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        if isinstance(experts, MegaMoeExperts):
            gate_up = _parameter_value(
                experts.gate_up_weight,
                gradients=gradients,
            )
            down = _parameter_value(experts.down_weight, gradients=gradients)
        elif isinstance(experts, _CommonExpertsAdapter):
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
                f"unsupported Qwen expert backend {type(experts).__name__}."
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
    passed = torch.ones((), dtype=torch.int32, device=device)
    tensor_names = []
    absolute_errors = []
    normalized_errors = []
    for tensor_name, common_tensor in common_tensors.items():
        tensor_names.append(tensor_name)
        mega_tensor = mega_tensors[tensor_name]
        if mega_tensor.shape != common_tensor.shape:
            raise RuntimeError(
                f"{name}.{tensor_name} shape differs: common={tuple(common_tensor.shape)}, "
                f"mega={tuple(mega_tensor.shape)}."
            )
        common_float = common_tensor.detach().float()
        mega_float = mega_tensor.detach().float()
        difference = (mega_float - common_float).abs()
        passed = torch.minimum(
            passed,
            torch.isclose(
                mega_float,
                common_float,
                rtol=rtol,
                atol=atol,
            )
            .all()
            .to(torch.int32),
        )
        absolute_errors.append(difference.max())
        tolerance = atol + rtol * common_float.abs()
        normalized_errors.append(
            (difference / tolerance.clamp_min(torch.finfo(torch.float32).tiny)).max()
        )
    absolute_by_tensor = torch.stack(absolute_errors)
    normalized_by_tensor = torch.stack(normalized_errors)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(absolute_by_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(normalized_by_tensor, op=dist.ReduceOp.MAX)
    maximum_absolute, absolute_index = absolute_by_tensor.max(dim=0)
    maximum_normalized, normalized_index = normalized_by_tensor.max(dim=0)
    result = {
        "passed": bool(passed.cpu().item()),
        "tensor_count": len(common_tensors),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(maximum_absolute.cpu().item()),
        "max_absolute_error_tensor": tensor_names[int(absolute_index.cpu().item())],
        "max_normalized_error": float(maximum_normalized.cpu().item()),
        "max_normalized_error_tensor": tensor_names[int(normalized_index.cpu().item())],
    }
    if not result["passed"]:
        raise RuntimeError(f"Qwen common/MegaMoe {name} comparison failed: {result}.")
    return result


def _measure_backend(
    workload: _Workload,
    validation: dict[str, Any],
    first_step_ms: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Finish warmup and measure one backend after its compared first step."""
    for _ in range(args.warmup_steps - 1):
        _timed_step(workload)
    latencies = []
    losses = []
    for _ in range(args.measured_steps):
        loss, latency = _timed_step(workload)
        latencies.append(latency)
        losses.append(float(loss.float().cpu().item()))
    return {
        "validation": validation,
        "first_optimizer_step_ms": first_step_ms,
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
    config: QwenMoeConfig,
    rank: int,
    accuracy: dict[str, Any],
    backends: dict[str, dict[str, Any]],
) -> None:
    """Write common-versus-MegaMoe accuracy and performance from rank zero."""
    common_median = backends["common"]["steady_state_optimizer_step_ms"]["median"]
    mega_median = backends["mega_moe"]["steady_state_optimizer_step_ms"]["median"]
    ratio = mega_median / common_median
    if rank == 0:
        result = {
            "topology": {"world_size": _WORLD_SIZE, "tp": 1, "ep": config.ep_size},
            "shape": {
                "batch_size": _BATCH_SIZE,
                "sequence_length": _SEQUENCE_LENGTH,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_experts": config.num_experts,
                "top_k": config.top_k,
                "expert_capacity_factor": config.expert_capacity_factor,
                "dtype": "bfloat16",
            },
            "optimizer": {
                "name": "HyperParallel AdamW",
                "wrapper": "Float16OptimizerWithFloat16Params",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "model_parameter_dtype": "bfloat16",
                "main_parameter_dtype": "float32",
            },
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "comparison": {
                "accuracy": accuracy,
                "performance": {
                    "order": ["common", "mega_moe"],
                    "metric": "rank-max complete optimizer-step milliseconds",
                    "common_median_ms": common_median,
                    "mega_moe_median_ms": mega_median,
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
    models: dict[str, QwenMoeModel],
    workloads: dict[str, _Workload],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], float]]]:
    """Compare initial state, first forward/backward, and optimizer updates."""
    accuracy = {
        "initial_parameters": _compare_tensor_maps(
            "initial parameters",
            _canonical_tensors(models["common"], gradients=False),
            _canonical_tensors(models["mega_moe"], gradients=False),
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
    ) = _validate_first_step(workloads["mega_moe"])
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
                _canonical_tensors(models["mega_moe"], gradients=False),
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
        "mega_moe": (mega_validation, mega_first_ms),
    }
    return accuracy, first_steps


def main(argv: list[str] | None = None) -> int:
    """Compare common and MegaMoe Qwen accuracy, then time A and B.

    Args:
        argv: Optional command-line arguments for tests or direct invocation.

    Returns:
        Zero after the distributed comparison completes successfully.
    """
    args = parse_args(argv)
    rank, world_size, device = _init_runtime()
    config = replace(
        QwenMoeConfig(),
        expert_capacity_factor=args.expert_capacity_factor,
    )
    models: dict[str, QwenMoeModel] = {}
    try:
        input_ids, labels = _build_batch(config, rank, args.seed, device)
        for backend in ("common", "mega_moe"):
            models[backend] = _build_backend_model(
                config,
                device,
                args.seed,
                backend,
            )
        workloads = {
            backend: _build_workload(
                model,
                args,
                input_ids,
                labels,
                world_size,
                device,
            )
            for backend, model in models.items()
        }
        accuracy, first_steps = _compare_first_step_accuracy(
            models,
            workloads,
            device,
        )
        common_validation, common_first_ms = first_steps["common"]
        mega_validation, mega_first_ms = first_steps["mega_moe"]
        backends = {
            "common": _measure_backend(
                workloads["common"],
                common_validation,
                common_first_ms,
                args,
            ),
            "mega_moe": _measure_backend(
                workloads["mega_moe"],
                mega_validation,
                mega_first_ms,
                args,
            ),
        }
        _write_result(
            args,
            config,
            rank,
            accuracy,
            backends,
        )
        return 0
    finally:
        for model in models.values():
            model.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
