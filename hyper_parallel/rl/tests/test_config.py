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
"""CPU contracts for HyperModels RL configuration translation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import rl.consistency.qwen3_dense as consistency_module
import torch
from rl.algorithm import build_algorithm
from rl.config import (
    _validate_model_implementation,
    build_model_registration,
    build_runtime_config,
    validate_config,
)
from rl.consistency import (
    QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1,
    configure_consistency_profile,
    consistency_profile,
    install_trainer_consistency_profile,
    trainer_sequence_log_probs,
    validate_consistency_model_identity,
)
from rl.roles.weight_sync import vllm_worker


def _consistency_config(profile: str) -> dict:
    """Build the minimal mutable structure owned by a consistency profile."""
    return {
        "consistency": {"profile": profile},
        "model": {"name": "qwen3", "attn_implementation": "sdpa"},
        "train": {
            "mixed_precision": {
                "enabled": False,
                "param_dtype": "float32",
            }
        },
        "rollout": {
            "engine": "vllm",
            "vllm": {
                "model_implementation": "native",
                "batch_invariant": False,
            },
        },
    }


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
        QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1,
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


@pytest.mark.parametrize(
    ("hf_config", "architecture", "family", "tie_word_embeddings"),
    [
        (
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "tie_word_embeddings": True,
            },
            "Qwen3ForCausalLM",
            "qwen3",
            True,
        ),
        (
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "tie_word_embeddings": True,
                },
            },
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5",
            True,
        ),
        (
            {
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5_text",
                "tie_word_embeddings": True,
            },
            "Qwen3_5ForCausalLM",
            "qwen3_5",
            True,
        ),
    ],
)
def test_model_registration_uses_checkpoint_identity(
    tmp_path: Path,
    hf_config: dict,
    architecture: str,
    family: str,
    tie_word_embeddings: bool,
) -> None:
    """Logical YAML names cannot override the checkpoint's HF architecture."""
    (tmp_path / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    config = {
        "model": {
            "registry_name": "logical_name",
            "name": family,
            "weights_path": str(tmp_path),
            "tokenizer_path": str(tmp_path),
        }
    }

    registration = build_model_registration(config)

    assert registration.hf_architecture == architecture
    assert registration.family == family
    assert registration.tie_word_embeddings is tie_word_embeddings


def test_qwen3_resolves_hyper_rollout_adapter(tmp_path: Path) -> None:
    """Qwen3 and Qwen3.5 share the same model implementation selector."""
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


def test_composite_qwen3_5_uses_text_tied_weight_contract(tmp_path: Path) -> None:
    """Actor and rollout tied-head behavior follows their shared text config."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": False,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "tie_word_embeddings": True,
                },
            }
        ),
        encoding="utf-8",
    )

    registration = build_model_registration(
        {
            "model": {
                "registry_name": "qwen3_5",
                "name": "qwen3_5",
                "weights_path": str(tmp_path),
                "tokenizer_path": str(tmp_path),
            }
        }
    )

    assert registration.tie_word_embeddings is True


def test_composite_qwen3_5_does_not_inherit_outer_tie_default(tmp_path: Path) -> None:
    """A present text config owns its tie default even when the outer wrapper is tied."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {"model_type": "qwen3_5_text"},
            }
        ),
        encoding="utf-8",
    )

    registration = build_model_registration(
        {
            "model": {
                "registry_name": "qwen3_5",
                "name": "qwen3_5",
                "weights_path": str(tmp_path),
                "tokenizer_path": str(tmp_path),
            }
        }
    )

    assert registration.tie_word_embeddings is False


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
                "load_path": None,
            },
        },
    }

    runtime = build_runtime_config(config)

    assert runtime.model.to_dict()["_target_"] == (
        "hyper_models._transformers.HyperAutoModelForCausalLM.from_pretrained"
    )
    assert runtime.model.force_hf is True
    assert runtime.training.train_iters == 3
    assert runtime.training.backend == "cpu:gloo,npu:hccl"
    assert runtime.fsdp_config.dp_shard_size == 2
    assert runtime.activation_checkpoint.mode == "off"


def test_consistency_profile_defaults_to_side_effect_free_off() -> None:
    """An omitted consistency section must not rewrite model or rollout settings."""
    config = _consistency_config("off")
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


def test_qwen3_consistency_profile_owns_paired_runtime_settings() -> None:
    """One profile switch must atomically select Trainer FA v2 and rollout FA3."""
    config = _consistency_config(QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1)

    profile = configure_consistency_profile(config)

    assert profile == QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1
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
        "consistency_profile": QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1,
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "logprobs_mode": "raw_logprobs",
        "model_implementation": "hyper",
        "tensor_parallel_size": 1,
    }
    assert config["rollout"]["temperature"] == 1.0
    assert config["rollout"]["top_p"] == 1.0
    assert config["rollout"]["top_k"] == 0


def test_qwen3_consistency_profile_checks_checkpoint_identity() -> None:
    """A YAML model name cannot disguise a checkpoint from another Qwen family."""
    config = _consistency_config(QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1)
    registration = SimpleNamespace(
        hyper_model_name="qwen3",
        hf_architecture="Qwen3_5ForCausalLM",
        model_type="qwen3_5_text",
        text_model_type="qwen3_5_text",
    )

    with pytest.raises(ValueError, match="requires checkpoint identity"):
        validate_consistency_model_identity(config, registration)


@pytest.mark.parametrize(
    ("consistency", "message"),
    [
        ({"profile": "unknown"}, "Unsupported consistency.profile"),
        ({"profile": "off", "fallback": True}, "Unsupported consistency configuration keys"),
        ("off", "must be a mapping"),
    ],
)
def test_consistency_profile_rejects_invalid_contract(consistency: object, message: str) -> None:
    """Unknown profile names and ad-hoc fallback switches must fail closed."""
    with pytest.raises(ValueError, match=message):
        consistency_profile({"consistency": consistency})


def test_qwen3_consistency_profile_rejects_other_model_families() -> None:
    """The versioned Qwen3 numerical profile cannot silently patch Qwen3.5."""
    config = _consistency_config(QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1)
    config["model"]["name"] = "qwen3_5"

    with pytest.raises(ValueError, match="supports only model.name='qwen3'"):
        configure_consistency_profile(config)


def test_enabled_consistency_profile_rejects_non_npu_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled profiles validate the NPU boundary before importing optional kernels."""
    config = _consistency_config(QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1)
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
        vllm_worker,
        "install_vllm_ascend_partial_prefill_rng_fix",
        lambda: installed.append("rng"),
    )
    monkeypatch.setattr(
        consistency_module._runtime,  # pylint: disable=protected-access
        "installed_rollout_profile",
        "off",
    )

    consistency_module.install_rollout_consistency_profile(
        QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1
    )
    consistency_module.install_rollout_consistency_profile(
        QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1
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

    vllm_worker._patch_vllm_ascend_partial_prefill_rng(  # pylint: disable=protected-access
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
