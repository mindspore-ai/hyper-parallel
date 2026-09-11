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
"""CPU acceptance test for one complete successful hyperparallel-RL configuration."""
# Test fixtures inspect stable runtime config attributes and local registry state.
# pylint: disable=missing-public-docstring,protected-access

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rl.config as config_module
from rl.algorithm import build_algorithm
from rl.config import (
    build_model_registration,
    build_runtime_config,
    resolve_vllm_automatic_limits,
    uses_colocated_vllm,
    validate_config,
)
from rl.roles.model import QWEN3_30B_A3B_CONFIG, resolve_vllm_model
from hyper_parallel.platform.platform import PlatformType


def _complete_config(tmp_path: Path) -> dict:
    model_dir = tmp_path / "model"
    tokenizer_dir = tmp_path / "tokenizer"
    model_dir.mkdir()
    tokenizer_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "tie_word_embeddings": True,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "head_dim": 4,
                "layer_types": ["full_attention", "full_attention"],
            }
        ),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_path.touch()
    test_path.touch()
    return {
        "model": {
            "name": "qwen3",
            "registry_name": "qwen",
            "weights_path": str(model_dir),
            "tokenizer_path": str(tokenizer_dir),
            "attn_implementation": "sdpa",
        },
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "max_prompt_length": 16,
        },
        "rollout": {
            "engine": "vllm",
            "num_return_sequences": 2,
            "max_new_tokens": 8,
            "seed": 7,
            "ignore_eos": False,
            "vllm": {
                "deployment": "disjoint",
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "visible_devices": "4",
                "host": "127.0.0.1",
                "port": 8200,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.8,
                "kv_cache_memory_bytes": 2 * 1024 * 1024,
                "max_model_len": 64,
                "max_num_seqs": None,
                "max_num_batched_tokens": 64,
                "block_size": 16,
                "model_implementation": "native",
                "weight_sync": {
                    "strategy": "full_gather",
                    "fallback_strategy": "none",
                    "bucket_size_mb": 1,
                },
            },
        },
        "agentic": {
            "environment": "ut-config-environment",
            "max_turns": 1,
            "max_observation_tokens": 0,
            "interaction_mode": "single_turn",
            "apply_chat_template": False,
        },
        "algorithm": {"name": "grpo", "loss_aggregation": "token-mean"},
        "evaluation": {
            "enabled": True,
            "batch_size": 1,
            "max_new_tokens": 8,
            "max_samples": 2,
            "log_samples": 0,
            "progress_steps": 0,
            "ignore_eos": False,
        },
        "train": {
            "max_steps": 2,
            "prompt_batch_size": 1,
            "micro_batch_size": 1,
            "response_mini_batch_size": 2,
            "policy_update_epochs": 1,
            "seed": 1234,
            "comm_backend": "hccl",
            "init_device": "meta",
            "learning_gate": {"enabled": False},
            "accelerator": {
                "dp_replicate": 1,
                "dp_shard": 1,
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "cpu_offload": False,
                "reshard_after_forward": True,
                "comm_fusion": True,
                "activation_checkpoint": "selective",
            },
            "optimizer": {
                "lr": 1.0e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "foreach": False,
                "max_grad_norm": 1.0,
                "lr_decay_style": "constant",
                "lr_warmup_ratio": 0.1,
                "lr_min": 0.0,
            },
            "mixed_precision": {
                "enabled": True,
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "output_dtype": None,
            },
            "checkpoint": {
                "output_dir": str(tmp_path / "checkpoints"),
                "save_steps": 0,
                "save_final": True,
                "verify_reload": False,
            },
        },
        "logging": {
            "backends": ["console"],
            "wandb": {"mode": "disabled"},
        },
        "consistency": {"enabled": False},
    }


def test_complete_config_resolves_validates_and_builds_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One real disjoint Qwen3 recipe reaches validated HyperAutoModel runtime objects."""
    config = _complete_config(tmp_path)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        config_module,
        "platform",
        SimpleNamespace(
            platform_type=PlatformType.PYTORCH,
            device_type=lambda: "npu",
        ),
    )
    monkeypatch.setitem(
        config_module.ENVIRONMENTS._items,
        "ut-config-environment",
        lambda _context: object(),
    )

    resolved = resolve_vllm_automatic_limits(config)
    algorithm = build_algorithm(resolved["algorithm"])
    validate_config(resolved, algorithm)
    registration = build_model_registration(resolved)
    runtime = build_runtime_config(resolved)

    assert config["rollout"]["vllm"]["max_num_seqs"] is None
    assert resolved["rollout"]["vllm"]["max_num_seqs"] == 2
    assert not uses_colocated_vllm(resolved)
    assert registration.name == "qwen"
    assert registration.family == "qwen3"
    assert registration.tie_word_embeddings
    assert runtime.training.train_iters == 2
    assert runtime.training.global_batch_size == 1
    assert runtime.training.backend == "hccl"
    assert runtime.accelerator.tp_size == 1
    assert runtime.fsdp_config.dp_shard_size == 1
    assert runtime.fsdp_config.mix_precision.param_dtype == "bfloat16"
    assert runtime.fsdp_config.mix_precision.reduce_dtype == "float32"
    assert runtime.activation_checkpoint.mode == "selective"
    assert runtime.checkpoint.save_ckpt


def test_moe_model_registrations_use_checkpoint_identity(tmp_path: Path) -> None:
    """Qwen3-MoE and Moonlight resolve family-specific Trainer and rollout identities."""
    qwen_dir = tmp_path / "qwen-moe"
    deepseek_dir = tmp_path / "moonlight"
    qwen_dir.mkdir()
    deepseek_dir.mkdir()
    qwen_config = dict(QWEN3_30B_A3B_CONFIG)
    qwen_config.update(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        tie_word_embeddings=False,
    )
    (qwen_dir / "config.json").write_text(json.dumps(qwen_config), encoding="utf-8")
    (deepseek_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "q_lora_rank": None,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    qwen_model = {
        "registry_name": "qwen3-30b-a3b",
        "name": "qwen3_moe",
        "weights_path": str(qwen_dir),
        "tokenizer_path": str(qwen_dir),
    }
    deepseek_model = {
        "registry_name": "moonlight",
        "name": "deepseek_v3",
        "weights_path": str(deepseek_dir),
        "tokenizer_path": str(deepseek_dir),
        "trust_remote_code": False,
        "tokenizer_trust_remote_code": True,
        "attention_implementation": "transformers_builtin",
    }

    qwen = build_model_registration({"model": qwen_model})
    deepseek = build_model_registration({"model": deepseek_model})

    assert qwen.family == "qwen3_moe"
    assert resolve_vllm_model(qwen, "hyper").architecture == "HyperQwen3MoeForCausalLM"
    assert resolve_vllm_model(qwen, "native").architecture == "Qwen3MoeForCausalLM"
    assert deepseek.family == "deepseek_v3"
    assert deepseek.q_lora_rank is None
    assert resolve_vllm_model(deepseek, "hyper").architecture == "HyperDeepseekV3ForCausalLM"
    assert resolve_vllm_model(deepseek, "native").architecture == "DeepseekV3ForCausalLM"


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_moe_four_rank_topology_and_weight_sync_are_supported(family: str) -> None:
    """Supported MoE families share the colocated TP2/EP4 synchronization contract."""
    accelerator = {
        "dp_replicate": 1,
        "dp_shard": 2,
        "tp": 2,
        "ep": 4,
        "cp": 1,
        "pp": 1,
    }
    vllm = {
        "deployment": "colocated",
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "enable_eplb": False,
        "weight_sync": {
            "strategy": "direct_reshard",
            "fallback_strategy": "full_gather",
            "bucket_size_mb": 128,
        },
    }
    rollout_model = SimpleNamespace(family=family, is_hyper=True)

    config_module._validate_moe_ep1_topology(vllm, rollout_model, accelerator)
    config_module._validate_vllm_weight_sync(
        vllm,
        "colocated",
        rollout_model,
        accelerator,
    )

    assert config_module._trainer_ep_size(accelerator, family) == 4
