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
"""CPU contracts for HyperAutoModel RL configuration translation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import rl.consistency.qwen3_dense as consistency_module
import rl.consistency.vllm_ascend as consistency_vllm_ascend
import torch
import yaml
from rl.algorithm import build_algorithm
from rl.config import (
    _validate_colocated_vllm,
    _validate_disjoint_vllm,
    _validate_model_implementation,
    _validate_topology,
    _validate_vllm_basics,
    _validate_vllm_limits,
    build_model_registration,
    build_runtime_config,
    resolve_vllm_automatic_limits,
    validate_config,
)
from rl.consistency import (
    QWEN3_ASCEND_CONSISTENCY_V1,
    configure_consistency_profile,
    consistency_profile,
    install_trainer_consistency_profile,
    trainer_sequence_log_probs,
    validate_consistency_model_identity,
)

from examples.train_rl import _apply_override


def test_cli_override_accepts_known_optional_disjoint_devices() -> None:
    """A colocated recipe can become disjoint without weakening strict overrides."""
    config = {"rollout": {"vllm": {"deployment": "colocated"}}}

    _apply_override(config, "--rollout.vllm.visible_devices=2,3,4,5")

    assert config["rollout"]["vllm"]["visible_devices"] == "2,3,4,5"


def test_cli_override_still_rejects_unknown_optional_fields() -> None:
    """Only declared optional paths may be introduced by a CLI override."""
    config = {"rollout": {"vllm": {"deployment": "colocated"}}}

    with pytest.raises(ValueError, match="Unknown configuration override path"):
        _apply_override(config, "--rollout.vllm.unknown_field=value")


def test_colocated_rollout_accepts_trainer_tp2_and_rollout_tp2() -> None:
    """A two-rank Trainer TP group may share one colocated TP2 rollout server."""
    _validate_colocated_vllm(
        {"port": 8200, "data_parallel_size": 1},
        {
            "dp_shard": 1,
            "tp": 2,
            "cpu_offload": True,
            "reshard_after_forward": True,
        },
        rollout_tp=2,
    )


def test_colocated_rollout_rejects_non_divisible_tp() -> None:
    """Colocated TP groups must partition the trainer ranks exactly."""
    with pytest.raises(ValueError, match="Trainer world size divisible"):
        _validate_colocated_vllm(
            {"port": 8200, "data_parallel_size": 1},
            {"dp_shard": 3, "cpu_offload": True, "reshard_after_forward": True},
            rollout_tp=2,
        )


def test_colocated_shared_deployment_uses_one_valid_port() -> None:
    """A colocated deployment owns one endpoint regardless of its DP degree."""
    accelerator = {
        "dp_shard": 4,
        "cpu_offload": True,
        "reshard_after_forward": True,
    }
    _validate_colocated_vllm(
        {"port": 65535, "data_parallel_size": 2},
        accelerator,
        rollout_tp=2,
    )
    with pytest.raises(ValueError, match="explicit integer between 1 and 65535"):
        _validate_colocated_vllm(
            {"port": 65536, "data_parallel_size": 2},
            accelerator,
            rollout_tp=2,
        )


def test_trainer_topology_accepts_tp2_and_rejects_unvalidated_axes() -> None:
    """Phase A opens TP2 while CP, PP, and replicated DP remain fail closed."""
    _validate_topology(
        {"dp_replicate": 1, "dp_shard": 2, "tp": 2, "cp": 1, "pp": 1}
    )

    for topology in (
        {"dp_replicate": 2, "dp_shard": 1, "tp": 2},
        {"dp_shard": 1, "tp": 4},
        {"dp_shard": 1, "tp": 2, "cp": 2},
        {"dp_shard": 1, "tp": 2, "pp": 2},
    ):
        with pytest.raises(ValueError, match="currently supports"):
            _validate_topology(topology)


def test_disjoint_rollout_requires_dp_times_tp_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout DP2 x TP2 requires four devices regardless of Trainer DP."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    _validate_disjoint_vllm(
        {"data_parallel_size": 2, "visible_devices": "2,3,4,5", "port": 8200},
        {"dp_shard": 2},
        rollout_tp=2,
    )
    with pytest.raises(ValueError, match="expected 4 rollout devices"):
        _validate_disjoint_vllm(
            {"data_parallel_size": 2, "visible_devices": "2,3", "port": 8200},
            {"dp_shard": 8},
            rollout_tp=2,
        )


def _consistency_config(
    enabled: bool,
    *,
    trainer_tp: int = 1,
    rollout_tp: int = 1,
    implementation: str = "hyper",
) -> dict:
    """Build the minimal mutable structure owned by consistency mode."""
    return {
        "consistency": {"enabled": enabled},
        "model": {"name": "qwen3", "attn_implementation": "sdpa"},
        "train": {
            "accelerator": {"tp": trainer_tp},
            "mixed_precision": {
                "enabled": False,
                "param_dtype": "float32",
            }
        },
        "rollout": {
            "engine": "vllm",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 17,
            "vllm": {
                "model_implementation": implementation,
                "batch_invariant": False,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "tensor_parallel_size": rollout_tp,
            },
        },
    }


def _colocated_accelerator(dp_shard: int = 2) -> dict[str, object]:
    """Return the residency settings required by colocated rollout."""
    return {
        "dp_shard": dp_shard,
        "cpu_offload": True,
        "reshard_after_forward": True,
    }


def test_shared_vllm_configuration_uses_explicit_parallel_sizes() -> None:
    """The shared deployment is defined directly by its DP and TP sizes."""
    vllm = {
        "deployment": "colocated",
        "data_parallel_size": 2,
        "tensor_parallel_size": 1,
        "port": 8100,
    }

    assert _validate_vllm_basics(vllm) == ("colocated", 2, 1)
    _validate_colocated_vllm(vllm, _colocated_accelerator(), 1)


def test_internal_dp1_tp2_matches_two_rank_colocated_trainer() -> None:
    """A single shared TP2 engine is valid without rank-local ownership."""
    _validate_colocated_vllm(
        {"port": 8100, "data_parallel_size": 1},
        _colocated_accelerator(dp_shard=2),
        rollout_tp=2,
    )


def test_qwen3_production_recipe_uses_selected_performance_defaults() -> None:
    """The production recipe uses the selected scheduler and parallel sizes."""
    recipe_path = (
        Path(__file__).parents[1]
        / "examples"
        / "configs"
        / "qwen3_4b_gsm8k_vllm_production.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    vllm = recipe["rollout"]["vllm"]

    assert recipe["rollout"]["num_return_sequences"] == 4
    assert vllm["data_parallel_size"] == 8
    assert "api_server_count" not in vllm
    assert vllm["gpu_memory_utilization"] == 0.35
    assert vllm["max_num_batched_tokens"] == 2048
    assert vllm["max_num_seqs"] is None
    assert vllm["tensor_parallel_size"] == 1
    accelerator = recipe["train"]["accelerator"]
    assert accelerator["dp_replicate"] == 1
    assert accelerator["tp"] == 1
    assert accelerator["cp"] == 1
    assert accelerator["pp"] == 1
    assert "request_concurrency" not in vllm


def test_example_config_directory_contains_only_major_recipes() -> None:
    """The public config directory stays limited to distinct supported workflows."""
    config_dir = Path(__file__).parents[1] / "examples" / "configs"
    expected = {
        "qwen3_4b_gsm8k_vllm_production.yaml",
        "qwen3_4b_gsm8k_vllm_tp2_consistency.yaml",
    }

    assert {path.name for path in config_dir.glob("*.yaml")} == expected

    for name in expected:
        recipe = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        vllm = recipe["rollout"]["vllm"]
        assert vllm["deployment"] in {"colocated", "disjoint"}
        assert isinstance(vllm["data_parallel_size"], int)


def test_vllm_request_concurrency_is_replaced_by_automatic_child_admission() -> None:
    """Legacy parent concurrency cannot be silently reinterpreted as child capacity."""
    with pytest.raises(ValueError, match="replaced by automatic child admission"):
        _validate_vllm_limits({"request_concurrency": 2})


def test_vllm_max_num_seqs_is_required_for_automatic_child_admission() -> None:
    """Configuration fails before runtime when automatic admission has no capacity."""
    with pytest.raises(ValueError, match="max_num_seqs is required"):
        _validate_vllm_limits({})


def test_vllm_max_num_seqs_cannot_exceed_token_budget() -> None:
    """Manual scheduler limits fail before an invalid vLLM startup."""
    with pytest.raises(ValueError, match="cannot exceed max_num_batched_tokens"):
        _validate_vllm_limits({"max_num_seqs": 9, "max_num_batched_tokens": 8})


def test_vllm_api_server_count_is_owned_by_upstream() -> None:
    """Frontend count is not a persistent user-facing performance switch."""
    with pytest.raises(ValueError, match="controlled by vLLM upstream"):
        _validate_vllm_limits({"api_server_count": 1, "max_num_seqs": 1})


def test_vllm_server_hccl_ports_must_be_a_complete_valid_range() -> None:
    """Server-only HCCL isolation rejects partial or inconsistent ports."""
    with pytest.raises(ValueError, match="must be configured together"):
        _validate_vllm_limits(
            {"max_num_seqs": 1, "server_hccl_if_base_port": 64400}
        )
    with pytest.raises(ValueError, match="must be in.*contain the base port"):
        _validate_vllm_limits(
            {
                "max_num_seqs": 1,
                "server_hccl_if_base_port": 64600,
                "server_hccl_npu_socket_port_range": "64400-64500",
            }
        )
    _validate_vllm_limits(
        {
            "max_num_seqs": 1,
            "server_hccl_if_base_port": 64400,
            "server_hccl_npu_socket_port_range": "64400-64500",
        }
    )


@pytest.mark.parametrize(
    ("base_port", "socket_range"),
    [
        (1023, "1023-1030"),
        (65521, "65500-65521"),
        (65520, "65520-65521"),
    ],
)
def test_vllm_server_hccl_ports_reject_cann_reserved_range(
    base_port: int,
    socket_range: str,
) -> None:
    """HCCL ports stay within the range accepted by the CANN runtime."""
    with pytest.raises(ValueError, match=r"\[1024, 65520\]"):
        _validate_vllm_limits(
            {
                "max_num_seqs": 1,
                "server_hccl_if_base_port": base_port,
                "server_hccl_npu_socket_port_range": socket_range,
            }
        )


@pytest.mark.parametrize(
    ("deployment", "tensor_parallel_size", "expected_max_num_seqs"),
    [
        ("colocated", 1, 14),
        ("colocated", 2, 28),
        ("disjoint", 1, 14),
        ("disjoint", 2, 28),
    ],
)
def test_automatic_max_num_seqs_is_bounded_by_workload_and_kv(
    tmp_path: Path,
    deployment: str,
    tensor_parallel_size: int,
    expected_max_num_seqs: int,
) -> None:
    """Both deployments account for DP engines and TP-local KV blocks."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 2560,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "model": {"weights_path": str(model_path)},
        "data": {"max_prompt_length": 512},
        "rollout": {
            "engine": "vllm",
            "num_return_sequences": 8,
            "max_new_tokens": 512,
            "vllm": {
                "deployment": deployment,
                "data_parallel_size": 8 // tensor_parallel_size,
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": "bfloat16",
                "kv_cache_memory_bytes": 2147483648,
                "max_model_len": 1024,
                "max_num_batched_tokens": 4096,
                "block_size": 128,
                "max_num_seqs": None,
            },
        },
        "agentic": {"max_turns": 1, "max_observation_tokens": 0},
        "train": {"prompt_batch_size": 2, "accelerator": {"dp_shard": 8}},
    }

    resolved = resolve_vllm_automatic_limits(config)

    assert resolved["rollout"]["vllm"]["max_num_seqs"] == expected_max_num_seqs
    assert config["rollout"]["vllm"]["max_num_seqs"] is None


@pytest.mark.parametrize(
    ("data_parallel_size", "prompt_batch_size", "expected_max_num_seqs"),
    [(4, 1, 4), (4, 2, 8), (8, 2, 8)],
)
def test_shared_dp_automatic_capacity_tracks_scaling_workload(
    tmp_path: Path,
    data_parallel_size: int,
    prompt_batch_size: int,
    expected_max_num_seqs: int,
) -> None:
    """DP4/DP8 strong and weak workloads resolve per-engine child capacity automatically."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 2560,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )
    recipe_path = (
        Path(__file__).parents[1]
        / "examples"
        / "configs"
        / "qwen3_4b_gsm8k_vllm_production.yaml"
    )
    config = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config["model"]["weights_path"] = str(model_path)
    config["rollout"]["num_return_sequences"] = 4
    config["rollout"]["vllm"].pop("topology", None)
    config["rollout"]["vllm"]["data_parallel_size"] = data_parallel_size
    config["train"]["prompt_batch_size"] = prompt_batch_size
    config["train"]["accelerator"]["dp_shard"] = data_parallel_size

    resolved = resolve_vllm_automatic_limits(config)

    assert resolved["rollout"]["vllm"]["max_num_seqs"] == expected_max_num_seqs


def test_automatic_max_num_seqs_preserves_manual_override() -> None:
    """An explicit scheduler capacity remains available for isolated A/B runs."""
    config = {"rollout": {"engine": "vllm", "vllm": {"max_num_seqs": 2}}}

    resolved = resolve_vllm_automatic_limits(config)

    assert resolved["rollout"]["vllm"]["max_num_seqs"] == 2


def test_automatic_max_num_seqs_does_not_infer_a_missing_field() -> None:
    """Only explicit null opts into auto capacity; omission remains a config error."""
    config = {"rollout": {"engine": "vllm", "vllm": {}}}

    resolved = resolve_vllm_automatic_limits(config)

    with pytest.raises(ValueError, match="max_num_seqs is required"):
        _validate_vllm_limits(resolved["rollout"]["vllm"])


def test_automatic_limits_reject_removed_topology_before_capacity_resolution() -> None:
    """The pre-validation sizing pass cannot defer or reinterpret old topology config."""
    config = {
        "rollout": {
            "engine": "vllm",
            "vllm": {"topology": "rank_local", "max_num_seqs": 2},
        }
    }

    with pytest.raises(ValueError, match="rollout.vllm.topology was removed"):
        resolve_vllm_automatic_limits(config)


@pytest.mark.parametrize("invalid_block_size", [0, True, 1.5, "128"])
def test_automatic_max_num_seqs_rejects_invalid_resource_before_division(
    tmp_path: Path,
    invalid_block_size: object,
) -> None:
    """Invalid automatic capacity inputs fail with their field instead of arithmetic errors."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 2560,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "model": {"weights_path": str(model_path)},
        "data": {"max_prompt_length": 512},
        "rollout": {
            "engine": "vllm",
            "num_return_sequences": 4,
            "max_new_tokens": 512,
            "vllm": {
                "deployment": "colocated",
                "dtype": "bfloat16",
                "kv_cache_memory_bytes": 2147483648,
                "max_model_len": 1024,
                "max_num_batched_tokens": 2048,
                "block_size": invalid_block_size,
                "tensor_parallel_size": 1,
                "max_num_seqs": None,
            },
        },
        "agentic": {"max_turns": 1},
        "train": {"prompt_batch_size": 2, "accelerator": {"dp_shard": 2}},
    }

    with pytest.raises(ValueError, match="rollout.vllm.block_size must be a positive integer"):
        resolve_vllm_automatic_limits(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"data_parallel_size": 0}, "positive integer"),
        ({"data_parallel_size": True}, "positive integer"),
        ({"data_parallel_size": 3}, "devices must match"),
        ({"port": 65536}, "explicit integer between 1 and 65535"),
    ],
)
def test_shared_vllm_rejects_invalid_colocated_configuration(
    override: dict[str, int],
    message: str,
) -> None:
    """Internal DP must not silently launch an invalid engine or frontend pool."""
    vllm = {
        "port": 8100,
        "data_parallel_size": 2,
        **override,
    }

    with pytest.raises(ValueError, match=message):
        _validate_colocated_vllm(vllm, _colocated_accelerator(), 1)


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
@pytest.mark.parametrize("topology", ["internal_dp", "rank_local", None])
def test_vllm_rejects_removed_topology_option(deployment: str, topology: object) -> None:
    """The removed topology option fails instead of being silently reinterpreted."""
    with pytest.raises(ValueError, match="rollout.vllm.topology was removed"):
        _validate_vllm_basics(
            {
                "deployment": deployment,
                "topology": topology,
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "port": 8100,
            }
        )


@pytest.mark.parametrize("field", ["data_parallel_size", "tensor_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_vllm_parallel_sizes_must_be_positive_integers(field: str, value: object) -> None:
    """DP and TP cannot rely on coercion or non-positive values."""
    vllm = {
        "deployment": "colocated",
        "data_parallel_size": 1,
        "tensor_parallel_size": 1,
        "port": 8100,
        field: value,
    }

    with pytest.raises(ValueError, match=rf"rollout.vllm.{field} must be a positive integer"):
        _validate_vllm_basics(vllm)


def test_trainer_profile_packs_and_restores_right_padded_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed model inputs preserve the Actor's padded output and gradient contract."""

    class PackedModel:
        """Record model-level remove-padding metadata for one CPU contract."""

        def __init__(self) -> None:
            """Initialize captured inputs and logits."""
            self.inputs = None
            self.logits = None

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            """Return differentiable uniform logits for packed inputs."""
            self.inputs = kwargs
            input_ids = kwargs["input_ids"]
            self.logits = torch.zeros(
                (1, input_ids.shape[1], 8),
                dtype=torch.float32,
                requires_grad=True,
            )
            return SimpleNamespace(logits=self.logits)

    monkeypatch.setattr(
        consistency_module._runtime,  # pylint: disable=protected-access
        "installed_profile",
        QWEN3_ASCEND_CONSISTENCY_V1,
    )
    model = PackedModel()
    sequences = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    result = trainer_sequence_log_probs(model, sequences, attention_mask)

    assert result is not None
    assert tuple(result.shape) == (2, 3)
    assert model.inputs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert model.inputs["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
    assert model.inputs["packed_cu_seqlens"].tolist() == [0, 3, 5]
    assert model.inputs["packed_max_seqlen"] == 3
    assert result[:, -1].tolist() == [0.0, 0.0]
    result.sum().backward()
    assert model.logits.grad is not None
    assert bool(model.logits.grad.isfinite().all())


def test_model_registration_uses_checkpoint_identity(tmp_path: Path) -> None:
    """Logical YAML names cannot override the checkpoint's HF architecture."""
    hf_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    config = {
        "model": {
            "registry_name": "logical_name",
            "name": "qwen3",
            "weights_path": str(tmp_path),
            "tokenizer_path": str(tmp_path),
        }
    }

    registration = build_model_registration(config)

    assert registration.hf_architecture == "Qwen3ForCausalLM"
    assert registration.family == "qwen3"
    assert registration.tie_word_embeddings is True


def test_qwen3_resolves_hyper_rollout_adapter(tmp_path: Path) -> None:
    """Qwen3 resolves the Hyper-vLLM implementation selector."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        ),
        encoding="utf-8",
    )
    registration = build_model_registration(
        {
            "model": {
                "registry_name": "qwen3_4b",
                "name": "qwen3",
                "weights_path": str(tmp_path),
                "tokenizer_path": str(tmp_path),
            }
        }
    )

    rollout_model = _validate_model_implementation(
        {"model_implementation": "hyper"},
        registration,
    )

    assert rollout_model.implementation == "hyper"
    assert rollout_model.architecture == "HyperQwen3ForCausalLM"


def test_runtime_config_uses_atomic_hf_loader(tmp_path: Path) -> None:
    """RL translates optimizer/FSDP settings without creating a legacy trainer config."""
    config = {
        "model": {
            "weights_path": str(tmp_path),
            "config_overrides": None,
        },
        "train": {
            "max_steps": 3,
            "prompt_batch_size": 2,
            "seed": 7,
            "comm_backend": "hccl",
            "accelerator": {
                "dp_shard": 2,
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "cpu_offload": True,
                "activation_checkpoint": False,
            },
            "mixed_precision": {
                "enabled": True,
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "output_dtype": None,
            },
            "optimizer": {
                "lr": 1.0e-6,
                "lr_decay_style": "constant",
            },
            "checkpoint": {
                "output_dir": str(tmp_path / "outputs"),
                "save_final": True,
            },
        },
    }

    runtime = build_runtime_config(config)

    assert runtime.model.to_dict()["_target_"] == (
        "hyper_parallel.auto_models._transformers."
        "HyperAutoModelForCausalLM.from_pretrained"
    )
    assert runtime.model.force_hf is True
    assert runtime.training.train_iters == 3
    assert runtime.training.backend == "cpu:gloo,npu:hccl"
    assert runtime.fsdp_config.dp_shard_size == 2
    assert runtime.activation_checkpoint.mode == "off"


def test_runtime_config_keeps_tp_out_of_logical_global_batch(tmp_path: Path) -> None:
    """TP ranks replay one logical DP batch instead of multiplying its size."""
    config = {
        "model": {"weights_path": str(tmp_path), "config_overrides": None},
        "train": {
            "max_steps": 1,
            "prompt_batch_size": 3,
            "accelerator": {"dp_shard": 2, "tp": 2, "cp": 1, "pp": 1},
            "mixed_precision": {"enabled": False},
            "optimizer": {"lr": 1.0e-6},
            "checkpoint": {
                "output_dir": str(tmp_path / "outputs"),
                "save_final": False,
            },
        },
    }

    runtime = build_runtime_config(config)

    assert runtime.training.global_batch_size == 6
    assert runtime.accelerator.tp_size == 2
    assert runtime.fsdp_config.dp_shard_size == 2


def test_consistency_profile_defaults_to_side_effect_free_off() -> None:
    """An omitted consistency section must not rewrite model or rollout settings."""
    config = _consistency_config(False)
    config.pop("consistency")
    original_model = dict(config["model"])
    original_mixed_precision = dict(config["train"]["mixed_precision"])
    original_vllm = dict(config["rollout"]["vllm"])

    profile = configure_consistency_profile(config)
    install_trainer_consistency_profile(config)

    assert profile == "off"
    assert config["model"] == original_model
    assert config["train"]["mixed_precision"] == original_mixed_precision
    assert config["rollout"]["vllm"] == original_vllm


def test_disabled_consistency_does_not_install_process_global_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled path returns before importing or installing numerical kernels."""
    config = _consistency_config(False)
    runtime = consistency_module._runtime  # pylint: disable=protected-access
    monkeypatch.setattr(runtime, "flash_attn_func", None)
    monkeypatch.setattr(runtime, "flash_attn_varlen_func", None)
    monkeypatch.setattr(runtime, "npu_rms_norm", None)
    monkeypatch.setattr(runtime, "installed_profile", "off")
    monkeypatch.setattr(runtime, "installed_rollout_profile", "off")
    monkeypatch.setattr(runtime, "batch_invariant_sum_compatibility_installed", False)

    install_trainer_consistency_profile(config)

    assert consistency_module.consistency_runtime_state() == {
        "trainer_recipe": "off",
        "rollout_recipe": "off",
        "trainer_attention_installed": False,
        "trainer_varlen_attention_installed": False,
        "qwen3_rms_norm_installed": False,
        "batch_invariant_sum_compatibility_installed": False,
    }


def test_qwen3_consistency_profile_owns_paired_runtime_settings() -> None:
    """One enable switch selects numerical settings without changing the workload."""
    config = _consistency_config(True)

    profile = configure_consistency_profile(config)

    assert profile == QWEN3_ASCEND_CONSISTENCY_V1
    assert config["model"]["attn_implementation"] == "hyper_qwen3_npu_consistent_v1"
    assert config["train"]["mixed_precision"] == {
        "enabled": True,
        "output_dtype": None,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
    }
    assert config["rollout"]["vllm"] == {
        "attention_backend": "FLASH_ATTN",
        "batch_invariant": True,
        "block_size": 128,
        "consistency_profile": QWEN3_ASCEND_CONSISTENCY_V1,
        "dtype": "bfloat16",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "logprobs_mode": "raw_logprobs",
        "model_implementation": "hyper",
        "tensor_parallel_size": 1,
    }
    assert config["rollout"]["temperature"] == 0.7
    assert config["rollout"]["top_p"] == 0.8
    assert config["rollout"]["top_k"] == 17


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_qwen3_consistency_requires_matched_tp_without_fixing_degree(tp_size: int) -> None:
    """The consistency contract is matched TP rather than a degree-specific profile."""
    config = _consistency_config(
        True,
        trainer_tp=tp_size,
        rollout_tp=tp_size,
    )

    profile = configure_consistency_profile(config)

    assert profile == QWEN3_ASCEND_CONSISTENCY_V1
    assert config["train"]["accelerator"]["tp"] == tp_size
    assert config["rollout"]["vllm"]["tensor_parallel_size"] == tp_size
    assert config["rollout"]["vllm"]["consistency_profile"] == profile


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_qwen3_consistency_preserves_selected_deployment(deployment: str) -> None:
    """The numerical profile does not replace rollout residency or transport."""
    config = _consistency_config(True, trainer_tp=2, rollout_tp=2)
    config["rollout"]["vllm"]["deployment"] = deployment

    configure_consistency_profile(config)

    assert config["rollout"]["vllm"]["deployment"] == deployment


@pytest.mark.parametrize("keepdim", [False, True])
def test_batch_invariant_non_last_sum_preserves_torch_dimension_order(
    keepdim: bool,
) -> None:
    """The NPU last-dimension adapter retains PyTorch dim=0 semantics."""
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    actual = consistency_module._reduce_non_last_dimension(  # pylint: disable=protected-access
        tensor,
        0,
        keepdim,
        lambda moved, preserve_dim: moved.sum(dim=-1, keepdim=preserve_dim),
    )

    assert torch.equal(actual, tensor.sum(dim=0, keepdim=keepdim))


@pytest.mark.parametrize(("trainer_tp", "rollout_tp"), [(1, 2), (2, 1)])
def test_qwen3_consistency_rejects_mismatched_tp(
    trainer_tp: int,
    rollout_tp: int,
) -> None:
    """Neither role may silently use another tensor-parallel degree."""
    config = _consistency_config(
        True,
        trainer_tp=trainer_tp,
        rollout_tp=rollout_tp,
    )

    with pytest.raises(ValueError, match="requires matched Trainer and rollout TP"):
        configure_consistency_profile(config)


def test_qwen3_consistency_rejects_native_vllm() -> None:
    """Native-vLLM remains valid normally but cannot opt into Hyper bit-exact mode."""
    config = _consistency_config(True, implementation="native")

    with pytest.raises(ValueError, match="model_implementation='hyper'"):
        configure_consistency_profile(config)


def test_qwen3_consistency_profile_checks_checkpoint_identity() -> None:
    """A YAML model name cannot disguise a checkpoint from another Qwen family."""
    config = _consistency_config(True)
    registration = SimpleNamespace(
        hyper_model_name="qwen3",
        hf_architecture="LlamaForCausalLM",
        model_type="llama",
        text_model_type="llama",
    )

    with pytest.raises(ValueError, match="requires checkpoint identity"):
        validate_consistency_model_identity(config, registration)


@pytest.mark.parametrize(
    ("consistency", "message"),
    [
        ({"profile": "legacy"}, "Unsupported consistency configuration keys"),
        ({"enabled": False, "fallback": True}, "Unsupported consistency configuration keys"),
        ({"enabled": "true"}, "consistency.enabled must be a boolean"),
        ("off", "must be a mapping"),
    ],
)
def test_consistency_profile_rejects_invalid_contract(consistency: object, message: str) -> None:
    """Unknown profile names and ad-hoc fallback switches must fail closed."""
    with pytest.raises(ValueError, match=message):
        consistency_profile({"consistency": consistency})


def test_qwen3_consistency_profile_rejects_other_model_families() -> None:
    """The Qwen3 numerical profile cannot silently patch another model family."""
    config = _consistency_config(True)
    config["model"]["name"] = "llama"

    with pytest.raises(ValueError, match="supports only model.name='qwen3'"):
        configure_consistency_profile(config)


def test_enabled_consistency_profile_rejects_non_npu_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled profiles validate the NPU boundary before importing optional kernels."""
    config = _consistency_config(True)
    configure_consistency_profile(config)
    monkeypatch.setattr(
        consistency_module,
        "platform",
        SimpleNamespace(platform_type=None, device_type=lambda: "cpu"),
    )

    with pytest.raises(ValueError, match="requires the Torch NPU platform"):
        install_trainer_consistency_profile(config)


def test_qwen3_npu_rms_norm_uses_profile_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The patched Qwen3 layer must call the profile-owned fused primitive."""
    calls = []

    def fake_npu_rms_norm(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        *,
        epsilon: float,
    ) -> tuple[torch.Tensor, None]:
        """Record and emulate the fused RMSNorm primitive."""
        calls.append((hidden_states, weight, epsilon))
        return hidden_states + weight, None

    monkeypatch.setattr(
        consistency_module._runtime,  # pylint: disable=protected-access
        "npu_rms_norm",
        fake_npu_rms_norm,
    )
    module = SimpleNamespace(
        weight=torch.ones(2, dtype=torch.bfloat16),
        variance_epsilon=1.0e-6,
    )
    hidden_states = torch.zeros(1, 2, dtype=torch.float32)

    output = consistency_module._qwen3_npu_rms_norm_forward(  # pylint: disable=protected-access
        module,
        hidden_states,
    )

    assert output.dtype == torch.bfloat16
    assert torch.equal(output, torch.ones_like(output))
    assert len(calls) == 1
    captured_hidden_states, captured_weight, captured_epsilon = calls[0]
    assert torch.equal(captured_hidden_states, hidden_states.to(torch.bfloat16))
    assert captured_weight is module.weight
    assert captured_epsilon == 1.0e-6


def test_rollout_profile_cannot_be_disabled_after_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed process-global rollout patch cannot be reported as disabled."""
    installed = []
    monkeypatch.setattr(
        consistency_module,
        "validate_rollout_consistency_profile",
        lambda _profile: None,
    )
    monkeypatch.setattr(
        consistency_module,
        "_install_qwen3_npu_rms_norm",
        lambda: installed.append("rms_norm"),
    )
    monkeypatch.setattr(
        consistency_module,
        "install_partial_prefill_rng_fix",
        lambda: installed.append("rng"),
    )
    monkeypatch.setattr(
        consistency_module._runtime,  # pylint: disable=protected-access
        "installed_rollout_profile",
        "off",
    )

    consistency_module.install_rollout_consistency_profile(
        QWEN3_ASCEND_CONSISTENCY_V1
    )
    consistency_module.install_rollout_consistency_profile(
        QWEN3_ASCEND_CONSISTENCY_V1
    )

    assert installed == ["rng", "rms_norm"]
    with pytest.raises(ValueError, match="Cannot disable process-global rollout"):
        consistency_module.install_rollout_consistency_profile("off")


def test_rollout_profile_restores_discarded_partial_prefill_rng() -> None:
    """Partial-prefill sampling must not advance a seeded request generator."""

    class FakeGenerator:
        """Expose the NPU generator offset interface used by the rollout patch."""

        def __init__(self) -> None:
            """Initialize the fake generator offset."""
            self.offset = 0

        def get_offset(self) -> int:
            """Return the current fake generator offset."""
            return self.offset

        def set_offset(self, offset: int) -> None:
            """Set the current fake generator offset."""
            self.offset = offset

    class FakeModelRunner:
        """Model the vLLM-Ascend sample/discard bookkeeping sequence."""

        def __init__(self) -> None:
            """Initialize two seeded generators and one discarded request."""
            self.generators = {0: FakeGenerator(), 1: FakeGenerator()}
            self.input_batch = SimpleNamespace(generators=self.generators)
            self.discard_request_indices = SimpleNamespace(np=[1])
            self.num_discarded_requests = 1

        def _sample(self) -> str:
            for generator in self.generators.values():
                generator.offset += 12
            return "sampled"

        def _bookkeeping_sync(self) -> str:
            for index in self.discard_request_indices.np[: self.num_discarded_requests]:
                self.generators[index].offset -= 4
            return "bookkept"

    consistency_vllm_ascend.patch_partial_prefill_rng(
        FakeModelRunner
    )
    runner = FakeModelRunner()

    assert runner._sample() == "sampled"
    assert runner._bookkeeping_sync() == "bookkept"
    assert runner.generators[0].offset == 12
    assert runner.generators[1].offset == 0

    runner.discard_request_indices.np = [0]
    assert runner._sample() == "sampled"
    assert runner._bookkeeping_sync() == "bookkept"
    assert runner.generators[0].offset == 12
    assert runner.generators[1].offset == 12

    runner.num_discarded_requests = 0
    assert runner._sample() == "sampled"
    assert runner._bookkeeping_sync() == "bookkept"
    assert runner.generators[0].offset == 24
    assert runner.generators[1].offset == 24


def test_consistency_attention_rejects_padding_that_bypasses_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attention callback must not receive padding that bypassed model-level packing."""
    monkeypatch.setattr(
        consistency_module._runtime,  # pylint: disable=protected-access
        "flash_attn_func",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="must be packed before the model forward"):
        consistency_module._flash_attn_npu_attention_forward(  # pylint: disable=protected-access
            None,
            None,
            None,
            None,
            object(),
        )


def test_critic_algorithm_is_rejected_before_runtime_setup() -> None:
    """PPO cannot allocate models before the initial GRPO-only boundary fails."""
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})

    with pytest.raises(NotImplementedError, match="critic-free algorithms only"):
        validate_config({}, algorithm)
