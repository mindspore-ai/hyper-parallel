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
"""RL configuration resolve."""
from copy import deepcopy
from typing import Any, Mapping
import torch
from rl.workers.engine.model import model_trust_remote_code, trainer_attention_implementation
from rl.workers.critic import build_value_model
from rl.workers.adapters.qwen3 import build_causal_lm
from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.components.optim import AdamW, MultiLRScheduler
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.trainer.config import AcceleratorConfig, ActivationCheckpointConfig, MixedPrecisionConfig, OptimizerConfig, PlanOverride, Target, TrainerConfig, TrainingConfig
from hyper_parallel.trainer.config.resolver import resolve_component
from rl.config.schema import _activation_checkpoint_mode, _normalize_dtype_name, _optimizer_fp32_main_params, _trainer_topology, build_model_registration, optional_mapping, required_mapping, router_replay_mode
from rl.config.validate import _validate_model_scope


def resolve_router_replay_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive rollout capture from the replay switch without mutating the recipe."""
    resolved = deepcopy(dict(config))
    enabled = router_replay_mode(resolved) == "R3"
    vllm = optional_mapping(required_mapping(resolved, "rollout"), "vllm")
    if ("enable_return_routed_experts" in vllm
            and vllm["enable_return_routed_experts"] is not enabled):
        raise ValueError("rollout.vllm.enable_return_routed_experts must match train.router_replay.mode")
    if enabled:
        resolved["rollout"].setdefault("vllm", {})["enable_return_routed_experts"] = True
    return resolved


def _build_optimizer_target(optimizer_config: Mapping[str, Any]) -> Target:
    """Build the HyperAutoModel AdamW target."""
    optimizer_kwargs = {
        "adamw_lr": float(optimizer_config.get("lr", 1.0e-6)),
        "adamw_weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        "adamw_betas": tuple(optimizer_config.get("betas", (0.9, 0.999))),
        "adamw_eps": float(optimizer_config.get("eps", 1.0e-8)),
    }
    if optimizer_config.get("foreach") is not None:
        optimizer_kwargs["foreach"] = bool(optimizer_config["foreach"])
    return Target(
        AdamW,
        target_path="hyper_parallel.components.optim.AdamW",
        adamw_config=optimizer_kwargs,
        no_decay_params=optimizer_config.get("no_decay_params", ["bias", "norm", "ln_"]),
    )


def _build_model_target(
    model_config: Mapping[str, Any],
    param_dtype_name: str,
    mixed_precision_enabled: bool,
    *,
    value_model: bool = False,
    family: str = "qwen3",
) -> Target:
    """Build the pretrained causal-language-model target."""
    attention_implementation = trainer_attention_implementation(model_config)
    if family in ("qwen3_moe", "qwen3_5"):
        builder = HyperAutoModelForCausalLM.from_pretrained
        target_path = "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained"
        builder_options = {}
    elif value_model:
        builder = build_value_model
        target_path = "rl.workers.critic.build_value_model"
        builder_options = {}
    else:
        builder = build_causal_lm
        target_path = "rl.workers.adapters.qwen3.build_causal_lm"
        builder_options = {"fused": True}
    return Target(
        builder,
        target_path=target_path,
        **builder_options,
        pretrained_model_name_or_path=str(model_config["weights_path"]),
        torch_dtype=_normalize_dtype_name(param_dtype_name) if mixed_precision_enabled else "float32",
        attn_implementation=attention_implementation,
        force_hf=True,
        local_files_only=True,
        trust_remote_code=model_trust_remote_code(model_config),
    )


def _build_lr_scheduler_target(optimizer_config: Mapping[str, Any]) -> Target:
    """Build the configured learning-rate scheduler target."""
    return Target(
        MultiLRScheduler,
        target_path=(
            "hyper_parallel.components.optim.MultiLRScheduler"
        ),
        lr_decay_style=str(optimizer_config.get("lr_decay_style", "constant")),
        lr_config={
            "lr_warmup_ratio": float(optimizer_config.get("lr_warmup_ratio", 0.0)),
            "min_lr": float(optimizer_config.get("lr_min", 0.0)),
        },
    )


def _build_training_config(
    train_config: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    trainer_dp_size: int,
    prompt_batch_size: int,
    backend: str,
) -> TrainingConfig:
    """Build core training-loop settings."""
    return TrainingConfig(
        train_iters=int(train_config["max_steps"]),
        global_batch_size=trainer_dp_size * prompt_batch_size,
        micro_batch_size=prompt_batch_size,
        backend=backend,
        max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
        init_device=str(train_config.get("init_device", "meta")),
        loss_aggregation="token_weighted",
        seed=int(train_config.get("seed", 1234)),
    )


def _build_accelerator_config(
    accelerator_config: Mapping[str, Any],
) -> AcceleratorConfig:
    """Build the supported parallel-dimension configuration."""
    return AcceleratorConfig(
        tp_size=int(accelerator_config.get("tp", 1)),
        cp_size=int(accelerator_config.get("cp", 1)),
        ep_size=int(accelerator_config.get("ep", 1)),
        pp_size=int(accelerator_config.get("pp", 1)),
        sequence_parallel=bool(accelerator_config.get("sequence_parallel", False)),
        loss_parallel=bool(accelerator_config.get("loss_parallel", False)),
    )


def _build_fsdp_config(
    accelerator_config: Mapping[str, Any],
    mixed_precision_config: Mapping[str, Any],
    dp_shard: int,
) -> FSDP2Config:
    """Build FSDP, mixed-precision, and CPU-offload settings."""
    mixed_precision_enabled = bool(mixed_precision_config.get("enabled", True))
    return FSDP2Config(
        dp_shard_size=dp_shard,
        edp_shard_size=int(accelerator_config.get("edp_shard", 1)),
        mix_precision=FSDP2MixedPrecisionConfig(
            param_dtype=(
                _normalize_dtype_name(
                    mixed_precision_config.get("param_dtype", "bfloat16")
                )
                if mixed_precision_enabled
                else None
            ),
            reduce_dtype=(
                _normalize_dtype_name(
                    mixed_precision_config.get("reduce_dtype", "float32")
                )
                if mixed_precision_enabled
                else None
            ),
            output_dtype=(
                _normalize_dtype_name(mixed_precision_config.get("output_dtype"))
                if mixed_precision_enabled
                else None
            ),
        ),
        enable_offload=bool(accelerator_config.get("cpu_offload", False)),
        reshard_after_forward=bool(
            accelerator_config.get("reshard_after_forward", True)
        ),
        reshard_after_backward=bool(accelerator_config.get("reshard_after_backward", False)),
        requires_grad_sync=bool(accelerator_config.get("requires_grad_sync", True)),
        comm_fusion=bool(accelerator_config.get("comm_fusion", True)),
        comm_fusion_zero_copy=accelerator_config.get("comm_fusion_zero_copy"),
    )


def _build_checkpoint_config(
    checkpoint_config: Mapping[str, Any],
) -> CheckpointingConfig:
    """Build checkpoint save settings."""
    return CheckpointingConfig(
        save_ckpt=(
            bool(checkpoint_config.get("save_final", True))
            or int(checkpoint_config.get("save_steps", 0)) > 0
        ),
        checkpoint_dir=str(checkpoint_config["output_dir"]),
        save_consolidated="none",
    )


def _runtime_backend(train_config: Mapping[str, Any], accelerator_config: Mapping[str, Any]) -> str:
    """Include CPU communication when optimizer state is offloaded."""
    backend = str(train_config.get("comm_backend") or "hccl")
    if (accelerator_config.get("cpu_offload", False)
            or accelerator_config.get("reference_cpu_offload", False)) and ":" not in backend:
        device_type = (torch.accelerator.current_accelerator() or torch.device("cpu")).type
        backend = f"cpu:gloo,{device_type}:{backend}"
    return backend


def build_runtime_config(
    config: Mapping[str, Any], *, critic: bool = False, reference: bool = False,
) -> TrainerConfig:
    """Translate Hyper-RL YAML into the HyperAutoModel runtime configuration."""
    if "consistency" in config:
        raise ValueError("consistency was removed; remove the field and use native-vLLM")
    if critic and reference:
        raise ValueError("A role cannot be both Critic and Reference")
    registration = build_model_registration(config)
    _validate_model_scope(config, registration)
    if critic and registration.family == "qwen3_moe":
        raise ValueError("Qwen3-MoE does not support a Critic model")
    if critic and registration.family == "qwen3_5":
        raise ValueError("Qwen3.5/Qwen3.8 does not support a Critic model")
    model_config = required_mapping(config, "model")
    train_config = required_mapping(config, "train")
    accelerator_config = required_mapping(train_config, "accelerator")
    if reference:
        role_config = optional_mapping(train_config, "reference")
        train_config = {**train_config, **role_config}
        accelerator_config = {
            **accelerator_config,
            "cpu_offload": accelerator_config.get(
                "reference_cpu_offload", accelerator_config.get("cpu_offload", False)
            ),
        }
    if "plan_overrides" not in train_config:
        raise ValueError("train.plan_overrides is required; declare model replacement rules explicitly")
    optimizer_config = required_mapping(train_config, "optimizer")
    mixed_precision_config = required_mapping(train_config, "mixed_precision")
    checkpoint_config = required_mapping(train_config, "checkpoint")
    if critic:
        critic_config = optional_mapping(train_config, "critic")
        train_config = {**train_config, **critic_config}
        model_config = {**model_config, "weights_path": critic_config.get("weights_path", model_config["weights_path"])}
        optimizer_config = {**optimizer_config, **optional_mapping(critic_config, "optimizer")}
    if model_config.get("config_overrides") not in (None, {}):
        raise ValueError("model.config_overrides is not supported by HyperAutoModel")

    topology = _trainer_topology(accelerator_config)
    dp_shard = topology["dp_shard"]
    trainer_dp_size = topology["dp_replicate"] * topology["dp_shard"]
    param_dtype_name = str(mixed_precision_config.get("param_dtype", "bfloat16"))
    mixed_precision_enabled = bool(mixed_precision_config.get("enabled", True))
    backend = _runtime_backend(train_config, accelerator_config)

    return TrainerConfig(
        model=_build_model_target(
            model_config, param_dtype_name, mixed_precision_enabled,
            value_model=critic,
            family=registration.family,
        ),
        plan_overrides=resolve_component(
            train_config["plan_overrides"], annotation=list[PlanOverride], path="train.plan_overrides",
        ),
        optimizer=OptimizerConfig(
            target=_build_optimizer_target(optimizer_config),
            fp32_main_params=_optimizer_fp32_main_params(optimizer_config),
        ),
        lr_scheduler=_build_lr_scheduler_target(optimizer_config),
        training=_build_training_config(
            train_config,
            optimizer_config,
            trainer_dp_size,
            int(train_config.get("prompt_batch_size", 1)),
            backend,
        ),
        accelerator=_build_accelerator_config(accelerator_config),
        fsdp_config=_build_fsdp_config(
            accelerator_config, mixed_precision_config, dp_shard
        ),
        mixed_precision=MixedPrecisionConfig(enabled=mixed_precision_enabled),
        activation_checkpoint=ActivationCheckpointConfig(
            mode=_activation_checkpoint_mode(accelerator_config)
        ),
        checkpoint=_build_checkpoint_config(checkpoint_config),
    )
