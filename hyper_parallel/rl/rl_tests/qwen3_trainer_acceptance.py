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
"""Run the Qwen3 Trainer NPU acceptance gates without starting rollout."""

import argparse
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

from rl.algorithm.loss import build_algorithm
from rl.config import build_runtime_config
from rl.dataset.contracts import ExperienceBatch
from rl.roles.model import build_role_model, build_role_optimizer
from rl.roles.policy.actor import Actor

from hyper_parallel import get_platform
from hyper_parallel.auto_models.components.distributed.infrastructure import (
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo

platform = get_platform()
logger = logging.getLogger(__name__)


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the production runtime subset needed by a Trainer-only gate."""
    return {
        "model": {
            "name": "qwen3",
            "weights_path": args.model_path,
            "tokenizer_path": args.model_path,
            "attn_implementation": "sdpa",
            "config_overrides": None,
        },
        "train": {
            "max_steps": args.steps,
            "seed": args.seed,
            "prompt_batch_size": args.batch_size,
            "micro_batch_size": args.batch_size,
            "response_mini_batch_size": args.batch_size,
            "policy_update_epochs": 1,
            "init_device": "meta",
            "comm_backend": "hccl",
            "accelerator": {
                "dp_replicate": 1,
                "dp_shard": args.dp_shard,
                "tp": args.tp,
                "cp": 1,
                "pp": 1,
                "activation_checkpoint": "off",
                "reshard_after_forward": True,
                "comm_fusion": True,
                "cpu_offload": False,
            },
            "mixed_precision": {
                "enabled": True,
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "output_dtype": None,
            },
            "optimizer": {
                "lr": args.learning_rate,
                "lr_min": 0.0,
                "lr_decay_style": "constant",
                "lr_warmup_ratio": 0.0,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "eps": 1.0e-8,
                "betas": [0.9, 0.999],
                "foreach": None,
            },
            "checkpoint": {
                "output_dir": str(Path(args.output_dir) / "unused-checkpoints"),
                "save_steps": 0,
                "save_final": False,
            },
        },
    }


def _local_tensor(value: Any) -> Any:
    """Return the physical tensor backing a plain Tensor or DTensor."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _tensor_digest(value: Any) -> tuple[str, tuple[int, ...]]:
    """Hash one rank-local tensor without gathering the complete model."""
    local = _local_tensor(value).detach().to("cpu").contiguous()
    serialized = platform.tensor_type_cast(local, "float32")
    payload = platform.tensor_to_numpy(serialized).tobytes()
    return sha256(payload).hexdigest(), tuple(int(size) for size in local.shape)


def _find_parameter(model: Any, suffix: str) -> tuple[str, Any]:
    """Find one parameter by its stable Qwen3 suffix, retaining tied aliases."""
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Qwen3 parameter ending in {suffix!r}, got "
            f"{[name for name, _ in matches]}"
        )
    return matches[0]


def _placement_description(parameter: Any) -> dict[str, Any]:
    """Describe the local parameter layout used by checkpoint and clipping."""
    layout = getattr(parameter, "_sharding_spec", None)
    if layout is None:
        layout = getattr(parameter, "layout", None)
    placements = getattr(layout, "placements", None)
    if placements is None:
        placements = getattr(parameter, "placements", ())
    mesh = getattr(layout, "mesh", None)
    mesh_names = getattr(mesh, "mesh_dim_names", ())
    return {
        "mesh_dim_names": list(mesh_names or ()),
        "placements": [repr(placement) for placement in placements or ()],
    }


def _validate_tied_qwen3(model: Any, tp_size: int) -> dict[str, Any]:
    """Require shared Qwen3 embedding/head storage, values, and TP metadata."""
    embed_name, embed = _find_parameter(model, "model.embed_tokens.weight")
    head_name, head = _find_parameter(model, "lm_head.weight")
    embed_local = _local_tensor(embed)
    head_local = _local_tensor(head)
    same_parameter = embed is head
    same_storage = (
        embed_local.untyped_storage().data_ptr()
        == head_local.untyped_storage().data_ptr()
    )
    if not same_storage:
        raise RuntimeError("Qwen3 embedding and lm_head no longer share local storage")
    same_values = bool((embed_local == head_local).all().item())
    if not same_values:
        raise RuntimeError("Qwen3 embedding and lm_head values diverged")
    embed_layout = _placement_description(embed)
    head_layout = _placement_description(head)
    if tp_size > 1:
        expected_shard = "Shard(dim=0)"
        if expected_shard not in embed_layout["placements"]:
            raise RuntimeError(
                f"Qwen3 embedding is missing TP vocab sharding: {embed_layout}"
            )
        if expected_shard not in head_layout["placements"]:
            raise RuntimeError(
                f"Qwen3 lm_head is missing TP vocab sharding: {head_layout}"
            )
    return {
        "embedding_name": embed_name,
        "lm_head_name": head_name,
        "same_parameter": same_parameter,
        "same_storage": same_storage,
        "same_values": same_values,
        "embedding_layout": embed_layout,
        "lm_head_layout": head_layout,
    }


def _build_experience(actor: Actor, device: Any, batch_size: int) -> ExperienceBatch:
    """Create fixed token-aligned GRPO inputs and their pre-update log-probs."""
    sequence_length = 24
    sequences = []
    for row in range(batch_size):
        sequences.append(
            [101 + row * 31 + position for position in range(sequence_length)]
        )
    sequence_tensor = platform.tensor(
        sequences,
        dtype=platform.tensor_dtype.long,
        device=device,
    )
    attention_mask = sequence_tensor.new_ones(
        sequence_tensor.shape,
        dtype=platform.tensor_dtype.bool,
    )
    action_mask = sequence_tensor.new_zeros(
        sequence_tensor.shape,
        dtype=platform.tensor_dtype.bool,
    )
    action_mask[:, sequence_length // 2:] = True
    old_log_probs = actor.compute_log_probs(
        ExperienceBatch(
            trajectories=(),
            sequences=sequence_tensor,
            attention_mask=attention_mask,
            action_mask=action_mask,
            rewards=sequence_tensor.new_zeros(
                (batch_size,), dtype=platform.tensor_dtype.float32
            ),
            old_log_probs=sequence_tensor.new_zeros(
                (batch_size, sequence_length - 1),
                dtype=platform.tensor_dtype.float32,
            ),
            responses=tuple("acceptance" for _ in range(batch_size)),
            generation_seconds=0.0,
        )
    )
    advantages = old_log_probs.new_ones(old_log_probs.shape)
    return ExperienceBatch(
        trajectories=(),
        sequences=sequence_tensor,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=sequence_tensor.new_zeros(
            (batch_size,), dtype=platform.tensor_dtype.float32
        ),
        old_log_probs=old_log_probs.detach(),
        responses=tuple("acceptance" for _ in range(batch_size)),
        generation_seconds=0.0,
        advantages=advantages,
        reference_log_probs=old_log_probs.detach().clone(),
    )


def _optimizer_steps(optimizer: Any) -> list[int]:
    """Return every populated AdamW parameter-group step."""
    steps = []
    for chained_optimizer in optimizer:
        for group in chained_optimizer.param_groups:
            if "step" in group:
                steps.append(int(group["step"]))
    return steps


def _validate_rank_results(results: list[Mapping[str, Any]]) -> None:
    """Validate synchronized loss, gradient, update, and layout evidence."""
    if not results:
        raise RuntimeError("Trainer acceptance gathered no rank results")
    expected_world_size = int(results[0]["world_size"])
    if len(results) != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} rank results, got {len(results)}"
        )
    steps = int(results[0]["steps"])
    if any(int(result["steps"]) != steps for result in results):
        raise RuntimeError(f"Trainer ranks disagree on step count: {results}")
    for step_index in range(steps):
        step_metrics = [result["step_metrics"][step_index] for result in results]
        losses = [float(metrics["total_loss"]) for metrics in step_metrics]
        gradient_norms = [float(metrics["gradient_norm"]) for metrics in step_metrics]
        if not all(math.isfinite(value) for value in losses + gradient_norms):
            raise RuntimeError(f"Trainer produced non-finite metrics: {step_metrics}")
        if not all(value > 0.0 for value in gradient_norms):
            raise RuntimeError(f"Trainer produced a zero gradient norm: {gradient_norms}")
        if max(losses) - min(losses) > 1.0e-6:
            raise RuntimeError(f"Trainer ranks disagree on step {step_index + 1} loss: {losses}")
        if max(gradient_norms) - min(gradient_norms) > 1.0e-6:
            raise RuntimeError(
                "Trainer ranks disagree on step "
                f"{step_index + 1} global gradient norm: {gradient_norms}"
            )
    for result in results:
        tied_qwen3 = result["tied_qwen3"]
        if not tied_qwen3["same_storage"] or not tied_qwen3["same_values"]:
            raise RuntimeError(f"Tied Qwen3 storage failed: {result}")
        if result["parameter_before"] == result["parameter_after"]:
            raise RuntimeError(f"Optimizer did not update the probe parameter: {result}")
        if not result["optimizer_steps"] or set(result["optimizer_steps"]) != {steps}:
            raise RuntimeError(f"Unexpected AdamW step state: {result}")


def run_acceptance(args: argparse.Namespace) -> None:
    """Run one distributed Qwen3 Trainer topology and write rank-zero evidence."""
    runtime_config = build_runtime_config(_build_config(args))
    initialize_distributed(runtime_config.training.backend)
    rank = int(platform.get_rank())
    world_size = int(platform.get_world_size())
    expected_world_size = args.dp_shard * args.tp
    if world_size != expected_world_size:
        raise ValueError(
            f"world_size must equal dp_shard*tp={expected_world_size}, got {world_size}"
        )
    setup = create_distributed_setup_from_config(runtime_config)
    parallel_dims = setup.mesh_context
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = platform.device(local_rank)
    device_handle = platform.get_device_handle(platform.device_type())
    device_handle.set_device(local_rank)
    random.seed(args.seed)
    platform.manual_seed(args.seed)

    model = build_role_model(runtime_config, setup, frozen=False)
    _validate_tied_qwen3(model, args.tp)
    probe_name, probe_parameter = _find_parameter(
        model,
        "model.layers.0.self_attn.q_proj.weight",
    )
    parameter_before, probe_shape_before = _tensor_digest(probe_parameter)
    optimizer, lr_scheduler = build_role_optimizer(runtime_config, model)
    dp_mesh = parallel_dims.dp_cp_mesh
    dp_group = None
    if dp_mesh is not None and parallel_dims.dp_size > 1:
        dp_group = dp_mesh.get_group()
    dp_group_info = GroupInfo(
        group_name="rl_trainer_acceptance_dp",
        group=dp_group,
        rank_size=parallel_dims.dp_size,
    )
    algorithm = build_algorithm(
        {
            "name": "grpo",
            "loss_aggregation": "token-mean",
            "kl_coef": 0.001,
        }
    )
    actor = Actor(
        actor_model=model,
        algorithm=algorithm,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        dp_group_info=dp_group_info,
        dp_size=parallel_dims.dp_size,
        micro_batch_size=args.batch_size,
        response_mini_batch_size=args.batch_size,
        update_epochs=1,
        max_grad_norm=1.0,
    )
    step_metrics = []
    for _ in range(args.steps):
        experience = _build_experience(actor, device, args.batch_size)
        metrics = actor.update(experience)
        step_metrics.append(
            {
                "total_loss": metrics.total_loss,
                "policy_loss": metrics.policy_loss,
                "kl_loss": metrics.kl_loss,
                "gradient_norm": metrics.gradient_norm,
                "valid_tokens": metrics.valid_tokens,
                "optimizer_steps": metrics.optimizer_steps,
            }
        )
    _, probe_parameter = _find_parameter(
        model,
        "model.layers.0.self_attn.q_proj.weight",
    )
    tied_qwen3 = _validate_tied_qwen3(model, args.tp)
    parameter_after, probe_shape_after = _tensor_digest(probe_parameter)
    result = {
        "rank": rank,
        "world_size": world_size,
        "dp_rank": int(parallel_dims.dp_rank),
        "tp_rank": int(parallel_dims.tp_rank),
        "dp_size": int(parallel_dims.dp_size),
        "tp_size": int(parallel_dims.tp_size),
        "steps": args.steps,
        "tied_qwen3": tied_qwen3,
        "probe_parameter": probe_name,
        "probe_shape_before": list(probe_shape_before),
        "probe_shape_after": list(probe_shape_after),
        "parameter_before": parameter_before,
        "parameter_after": parameter_after,
        "optimizer_steps": _optimizer_steps(optimizer),
        "step_metrics": step_metrics,
        "metrics": step_metrics[-1],
    }
    gathered: list[Any] = [None] * world_size
    platform.all_gather_object(gathered, result)
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"trainer-dp{args.dp_shard}-tp{args.tp}.json"
        output_path.write_text(
            json.dumps({"status": "pending", "ranks": gathered}, indent=2) + "\n",
            encoding="utf-8",
        )
        _validate_rank_results(gathered)
        output_path.write_text(
            json.dumps({"status": "passed", "ranks": gathered}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Qwen3 Trainer acceptance passed: %s", output_path)
    platform.barrier()
    destroy_process_group()


def main() -> None:
    """Parse the Trainer topology and execute the NPU acceptance gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dp-shard", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.dp_shard <= 0 or args.tp <= 0 or args.steps <= 0:
        raise ValueError("--dp-shard, --tp, and --steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    run_acceptance(args)


if __name__ == "__main__":
    main()
