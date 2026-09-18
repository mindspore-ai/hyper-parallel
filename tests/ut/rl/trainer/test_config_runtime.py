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
"""CPU acceptance test for one complete successful Hyper-RL configuration."""
# Test fixtures inspect stable runtime config attributes and local registry state.
# pylint: disable=missing-public-docstring,protected-access

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rl.config as config_module
from rl.algorithm import build_algorithm
from rl.config import (
    build_model_registration,
    build_runtime_config,
    resolve_vllm_automatic_limits,
    uses_colocated_vllm,
    validate_config,
)
from rl.roles.model_setup import resolve_vllm_model


@pytest.fixture(name="validated_recipe")
def _validated_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Provide a valid baseline so each negative case isolates one configuration error."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("npu"))
    monkeypatch.setitem(config_module.ENVIRONMENTS._items, "ut-config-environment", lambda _context: object())
    config = resolve_vllm_automatic_limits(_complete_config(tmp_path))
    validate_config(config, build_algorithm(config["algorithm"]))
    return config


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("train", "max_steps"), 0, "max_steps must be positive"),
        (("train", "micro_batch_size"), 0, "micro_batch_size must be positive"),
        (("train", "response_mini_batch_size"), 3, "cannot exceed the local rollout"),
        (("train", "policy_update_epochs"), 0, "policy_update_epochs must be positive"),
        (("rollout", "num_return_sequences"), 1, "at least 2"),
        (("rollout", "ignore_eos"), "false", "ignore_eos must be a boolean"),
        (("train", "accelerator", "cp"), 2, "invalid topology"),
        (("train", "accelerator", "pp"), 2, "invalid topology"),
        (("train", "accelerator", "tp"), 4, "invalid topology"),
        (("rollout", "vllm", "visible_devices"), "0", "disjoint from the trainer"),
        (("rollout", "vllm", "host"), "0.0.0.0", "loopback"),
        (("rollout", "vllm", "port"), True, "explicit integer"),
        (("rollout", "vllm", "dtype"), "float32", "requires bfloat16"),
        (("rollout", "vllm", "gpu_memory_utilization"), 1.0, "between 0 and 1"),
        (("rollout", "vllm", "request_concurrency"), 2, "replaced"),
        (("rollout", "vllm", "api_server_count"), 2, "controlled by vLLM upstream"),
        (("rollout", "vllm", "server_hccl_if_base_port"), 62000, "configured together"),
        (("evaluation", "max_samples"), 0, "max_samples must be positive"),
        (("evaluation", "ignore_eos"), 1, "ignore_eos must be a boolean"),
        (("train", "learning_gate", "max_step"), True, "positive integer or null"),
    ],
)
def test_config_rejects_invalid_runtime_contracts(
    validated_recipe: dict, path: tuple[str, ...], value: object, error: str,
) -> None:
    """Malformed production settings fail in validation before loading any model."""
    section = validated_recipe
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        validate_config(validated_recipe, build_algorithm(validated_recipe["algorithm"]))


def test_all_st_recipes_pass_production_validation(
    validated_recipe: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ST topology and resume phase must remain accepted by the real config validator."""
    runtime = importlib.import_module("tests.common.rl_st_cases")
    for case in runtime.CASES:
        devices = list(range(case.cards))
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, devices[:case.world])))
        for phase in range(1, 3 if case.resume else 2):
            config = runtime.prepare_config(case, phase, (8100, 8200), devices)
            config["model"].update(validated_recipe["model"])
            for field in ("train_path", "test_path"):
                config["data"][field] = validated_recipe["data"][field]
            validate_config(resolve_vllm_automatic_limits(config), build_algorithm(config["algorithm"]))


def _complete_config(tmp_path: Path) -> dict:
    """Build a complete configuration with real temporary model metadata."""
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
        "train": _training_config(tmp_path),
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
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("npu"))
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
    assert runtime.model.to_dict()["_target_"] == "rl.roles.qwen3_builder.build_causal_lm"
    assert runtime.model.torch_dtype == "bfloat16"
    assert runtime.optimizer.target.to_dict()["_target_"] == "hyper_parallel.components.optim.AdamW"


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [("qwen3_moe", "Qwen3MoeForCausalLM"), ("deepseek_v3", "DeepseekV3ForCausalLM")],
)
def test_removed_model_families_fail_before_runtime_construction(
    tmp_path: Path, model_type: str, architecture: str,
) -> None:
    """A Qwen3 label cannot bypass checkpoint identity validation."""
    config = _complete_config(tmp_path)
    config_path = Path(config["model"]["weights_path"]) / "config.json"
    config_path.write_text(json.dumps({"model_type": model_type, "architectures": [architecture]}))
    with pytest.raises(ValueError, match="Unsupported RL model identity"):
        build_model_registration(config)


@pytest.mark.parametrize("ep", [0, 2, 4, True, "1"])
def test_dense_runtime_rejects_expert_parallelism(tmp_path: Path, ep: object) -> None:
    """Runtime construction rejects stale EP settings even without full validation."""
    config = _complete_config(tmp_path)
    config["train"]["accelerator"]["ep"] = ep
    with pytest.raises(ValueError, match="train.accelerator.ep=1"):
        build_runtime_config(config)


@pytest.mark.parametrize("option", ["enable_expert_parallel", "enable_eplb"])
def test_dense_rollout_rejects_expert_options(option: str) -> None:
    """Removed expert options fail rather than silently starting dense rollout."""
    with pytest.raises(ValueError, match=option):
        config_module._validate_dense_parallelism({option: True}, {})


@pytest.mark.parametrize("implementation", ["native", "hyper"])
def test_dense_registration_preserves_both_rollout_implementations(
    tmp_path: Path, implementation: str,
) -> None:
    """Checkpoint identity resolves both supported dense rollout paths."""
    registration = build_model_registration(_complete_config(tmp_path))
    rollout = resolve_vllm_model(registration, implementation)
    assert rollout.family == "qwen3"
    assert rollout.architecture == (
        "HyperQwen3ForCausalLM" if implementation == "hyper" else "Qwen3ForCausalLM"
    )
    config_module._validate_dense_parallelism(
        {"enable_expert_parallel": False, "enable_eplb": False}, {"ep": 1},
    )


@pytest.fixture(name="validation_config")
def fixture_validation_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Provide a valid recipe with mocked accelerator and environment registration."""
    config = _complete_config(tmp_path)
    config["rollout"]["vllm"]["max_num_seqs"] = 2
    config["train"]["accelerator"]["dp_shard"] = 2
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("npu"))
    monkeypatch.setitem(config_module.ENVIRONMENTS._items, "ut-config-environment", lambda _context: object())
    return config


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
@pytest.mark.parametrize("trainer_tp", [1, 2])
@pytest.mark.parametrize("rollout_tp", [1, 2])
@pytest.mark.parametrize("strategy", ["full_gather", "direct_reshard"])
def test_supported_parallel_and_weight_sync_combinations(
    validation_config: dict, deployment: str, trainer_tp: int, rollout_tp: int, strategy: str,
) -> None:
    """Both deployments retain supported trainer/rollout TP and weight-sync combinations."""
    config = validation_config
    accelerator = config["train"]["accelerator"]
    accelerator.update(tp=trainer_tp, cpu_offload=True)
    world_size = 2 * trainer_tp
    vllm = config["rollout"]["vllm"]
    vllm.update(deployment=deployment, data_parallel_size=world_size // rollout_tp,
                tensor_parallel_size=rollout_tp)
    vllm["weight_sync"]["strategy"] = strategy
    if deployment == "colocated":
        vllm.pop("visible_devices")
    else:
        vllm["visible_devices"] = ",".join(str(device) for device in range(4, 4 + world_size))
    validate_config(config, build_algorithm(config["algorithm"]))


@pytest.mark.parametrize("field", ["data_parallel_size", "tensor_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, True, "2", 1.5, None])
def test_rollout_parallel_sizes_still_require_positive_integers(
    validation_config: dict, field: str, value: object,
) -> None:
    """Unified integer validation rejects zero, negative, Boolean and noninteger sizes."""
    validation_config["rollout"]["vllm"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(validation_config, build_algorithm(validation_config["algorithm"]))


@pytest.mark.parametrize("field,value", [("tp", 3), ("cp", 2), ("pp", 2), ("dp_replicate", 2)])
@pytest.mark.parametrize("engine", ["vllm", "custom"])
def test_training_topology_is_checked_for_all_engines(
    validation_config: dict, monkeypatch: pytest.MonkeyPatch, field: str, value: int, engine: str,
) -> None:
    """Removing weight-sync duplicate checks must not admit unsupported training axes."""
    validation_config["train"]["accelerator"][field] = value
    validation_config["rollout"]["engine"] = engine
    monkeypatch.setattr(config_module, "ROLLOUT_ENGINES", SimpleNamespace(names=(engine,)))
    with pytest.raises(ValueError, match="invalid topology"):
        validate_config(validation_config, build_algorithm(validation_config["algorithm"]))


@pytest.mark.parametrize("rollout_tp", [3, 4])
def test_colocated_device_product_rejects_invalid_tp(validation_config: dict, rollout_tp: int) -> None:
    """The device-product check also rejects nondivisible or oversized rollout TP."""
    validation_config["train"]["accelerator"]["cpu_offload"] = True
    vllm = validation_config["rollout"]["vllm"]
    vllm.update(deployment="colocated", tensor_parallel_size=rollout_tp)
    vllm.pop("visible_devices")
    with pytest.raises(ValueError, match="devices must match"):
        validate_config(validation_config, build_algorithm(validation_config["algorithm"]))


@pytest.mark.parametrize("field,value", [("strategy", "invalid"), ("bucket_size_mb", 0)])
def test_weight_sync_boundary_still_rejects_invalid_options(
    validation_config: dict, field: str, value: object,
) -> None:
    """Weight-sync's canonical resolver remains authoritative for option validation."""
    validation_config["rollout"]["vllm"]["weight_sync"][field] = value
    with pytest.raises(ValueError, match="strategy|bucket_size_mb"):
        validate_config(validation_config, build_algorithm(validation_config["algorithm"]))


def test_model_resolution_still_rejects_invalid_implementation(validation_config: dict) -> None:
    """Model resolution retains the implementation check removed from capacity validation."""
    validation_config["rollout"]["vllm"]["model_implementation"] = "invalid"
    with pytest.raises(ValueError, match="model_implementation"):
        validate_config(validation_config, build_algorithm(validation_config["algorithm"]))


def _training_config(tmp_path):
    """Return fresh optimizer, precision and checkpoint settings for configuration tests."""
    return {
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
    }
