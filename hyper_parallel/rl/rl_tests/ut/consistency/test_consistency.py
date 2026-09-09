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
"""CPU unit tests for the Hyper-RL numerical consistency contract."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.consistency.gates as gates_module
import rl.consistency.qwen3_dense as profile_module
from rl.consistency import (
    QWEN3_ASCEND_CONSISTENCY_V1,
    measure_post_update_old_policy_mismatch,
    validate_consistency_forward_inputs,
    validate_pre_update_consistency,
)
from rl.consistency.vllm_ascend import (
    install_partial_prefill_rng_fix,
    patch_partial_prefill_rng,
)
from rl.dataset.contracts import ExperienceBatch
from hyper_parallel.platform.platform import PlatformType


def test_consistency_profile_installs_shared_recipe_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer and rollout install one matching process recipe without repetition."""
    calls: list[str] = []

    class FakeRegistry(dict):
        """Record Transformers attention registrations without global mutation."""

        def register(self, name: str, value: Any) -> None:
            self[name] = value

    runtime = profile_module._runtime  # pylint: disable=protected-access
    monkeypatch.setattr(runtime, "flash_attn_func", None)
    monkeypatch.setattr(runtime, "flash_attn_varlen_func", None)
    monkeypatch.setattr(runtime, "npu_rms_norm", None)
    monkeypatch.setattr(runtime, "installed_profile", "off")
    monkeypatch.setattr(runtime, "installed_rollout_profile", "off")
    monkeypatch.setattr(runtime, "batch_invariant_sum_compatibility_installed", False)
    monkeypatch.setattr(
        profile_module,
        "platform",
        SimpleNamespace(platform_type=PlatformType.PYTORCH, device_type=lambda: "npu"),
    )
    monkeypatch.setattr(profile_module, "_require_package_versions", lambda: None)
    monkeypatch.setattr(profile_module, "ALL_ATTENTION_FUNCTIONS", FakeRegistry())
    monkeypatch.setattr(
        profile_module,
        "ALL_MASK_ATTENTION_FUNCTIONS",
        FakeRegistry({"flash_attention_2": "mask"}),
    )
    monkeypatch.setattr(
        profile_module,
        "_install_batch_invariant_sum_compatibility",
        lambda: calls.append("batch-sum"),
    )
    monkeypatch.setattr(
        profile_module,
        "_install_qwen3_npu_rms_norm",
        lambda: calls.append("rms-norm"),
    )
    monkeypatch.setattr(
        profile_module,
        "validate_rollout_consistency_profile",
        lambda profile: calls.append(f"validate:{profile}"),
    )
    monkeypatch.setattr(
        profile_module,
        "install_partial_prefill_rng_fix",
        lambda: calls.append("rng"),
    )
    def fake_attention(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Represent either supported flash-attention callable."""
        return args, kwargs
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_npu",
        SimpleNamespace(
            flash_attn_func=fake_attention,
            flash_attn_varlen_func=fake_attention,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_npu_v3",
        SimpleNamespace(flash_attn_with_kvcache=fake_attention),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.batch_invariant",
        SimpleNamespace(
            HAS_ASCENDC_BATCH_INVARIANT=True,
            enable_batch_invariant_mode=lambda: calls.append("batch-mode"),
        ),
    )
    config = {"consistency": {"enabled": True}}

    profile_module.install_trainer_consistency_profile(config)
    profile_module.install_trainer_consistency_profile(config)
    profile_module.install_rollout_consistency_profile(QWEN3_ASCEND_CONSISTENCY_V1)
    profile_module.install_rollout_consistency_profile(QWEN3_ASCEND_CONSISTENCY_V1)

    state = profile_module.consistency_runtime_state()
    assert state["trainer_recipe"] == QWEN3_ASCEND_CONSISTENCY_V1
    assert state["rollout_recipe"] == QWEN3_ASCEND_CONSISTENCY_V1
    assert state["trainer_attention_installed"]
    assert state["trainer_varlen_attention_installed"]
    assert calls == [
        "batch-mode",
        "batch-sum",
        "rms-norm",
        f"validate:{QWEN3_ASCEND_CONSISTENCY_V1}",
        "rng",
        "rms-norm",
        f"validate:{QWEN3_ASCEND_CONSISTENCY_V1}",
    ]


def test_consistency_gate_accepts_right_padded_bit_exact_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Right-padded FP32 action values pass preflight and aggregate exact metrics."""
    sequences = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    attention_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    action_mask = torch.tensor([[False, True, True, True], [False, True, True, False]])
    rollout_log_probs = torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, 99.0]], dtype=torch.float32)
    experience = ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=torch.tensor([1.0, 0.0]),
        old_log_probs=rollout_log_probs,
        responses=("a", "b"),
        generation_seconds=0.0,
        worker_policy_version=3,
        worker_policy_fingerprint="digest-v3",
    )

    def all_gather(output: list[Any], value: Any, group: Any) -> None:
        assert group == "dp"
        output[0] = value
        output[1] = value.copy() if isinstance(value, dict) else value

    monkeypatch.setattr(gates_module.platform, "all_gather_object", all_gather)
    monkeypatch.setattr(gates_module.platform, "get_rank", lambda: 0)

    validate_consistency_forward_inputs(
        experience,
        group="dp",
        group_size=2,
        operation="pre-update",
    )
    metrics = validate_pre_update_consistency(
        experience,
        rollout_log_probs.clone(),
        expected_policy_version=3,
        expected_policy_fingerprint="digest-v3",
        group="dp",
        group_size=2,
    )

    assert metrics == {
        "training/pre_update_exact_valid": 1.0,
        "training/pre_update_exact_tokens": 10.0,
        "training/pre_update_mismatch_count": 0.0,
        "training/pre_update_max_abs_diff": 0.0,
        "training/pre_update_mean_abs_diff": 0.0,
    }


def test_partial_prefill_rng_restores_discarded_request_offsets() -> None:
    """Discarded partial-prefill samples restore their seeded generator offsets."""

    class FakeGenerator:
        """Expose the generator offset interface patched by vLLM-Ascend."""

        def __init__(self) -> None:
            self.offset = 0

        def get_offset(self) -> int:
            return self.offset

        def set_offset(self, offset: int) -> None:
            self.offset = offset

    class FakeModelRunner:
        """Model one sampling and bookkeeping cycle with one discarded row."""

        def __init__(self) -> None:
            self.generators = {0: FakeGenerator(), 1: FakeGenerator()}
            self.input_batch = SimpleNamespace(generators=self.generators)
            self.discard_request_indices = SimpleNamespace(np=[1])
            self.num_discarded_requests = 1

        def _sample(self) -> str:
            for generator in self.generators.values():
                generator.offset += 12
            return "sampled"

        def _bookkeeping_sync(self) -> str:
            return "bookkept"

    patch_partial_prefill_rng(FakeModelRunner)
    patch_partial_prefill_rng(FakeModelRunner)
    runner = FakeModelRunner()

    assert runner._sample() == "sampled"
    assert runner._bookkeeping_sync() == "bookkept"
    assert runner.generators[0].offset == 12
    assert runner.generators[1].offset == 0
    assert getattr(FakeModelRunner, "_hyper_rl_partial_prefill_rng_patched")


def test_consistency_configures_complete_qwen3_recipe() -> None:
    """Production configuration applies the paired Trainer and Hyper-vLLM recipe."""
    config = {
        "consistency": {"enabled": True},
        "model": {"name": "qwen3"},
        "rollout": {
            "engine": "vllm",
            "temperature": 0.7,
            "vllm": {
                "deployment": "colocated",
                "model_implementation": "hyper",
                "tensor_parallel_size": 2,
            },
        },
        "train": {
            "accelerator": {"tp": 2},
            "mixed_precision": {"enabled": False},
        },
    }

    profile = profile_module.configure_consistency_profile(config)

    assert profile == QWEN3_ASCEND_CONSISTENCY_V1
    assert config["model"]["attn_implementation"] == "hyper_qwen3_npu_consistent_v1"
    assert config["train"]["mixed_precision"] == {
        "enabled": True,
        "output_dtype": None,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
    }
    assert config["rollout"]["temperature"] == 0.7
    assert config["rollout"]["vllm"] == {
        "deployment": "colocated",
        "model_implementation": "hyper",
        "tensor_parallel_size": 2,
        "attention_backend": "FLASH_ATTN",
        "batch_invariant": True,
        "block_size": 128,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "logprobs_mode": "raw_logprobs",
        "consistency_profile": QWEN3_ASCEND_CONSISTENCY_V1,
    }


def test_trainer_sequence_log_probs_executes_packed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production packed forward removes right padding and restores padded logprob rows."""
    runtime = profile_module._runtime  # pylint: disable=protected-access
    monkeypatch.setattr(runtime, "installed_profile", QWEN3_ASCEND_CONSISTENCY_V1)
    captured: dict[str, Any] = {}

    class Model:
        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            captured.update(kwargs)
            tokens = kwargs["input_ids"]
            logits = torch.zeros((1, tokens.shape[1], 8), dtype=torch.float32)
            logits[0, 0, 2] = 2.0
            logits[0, 1, 3] = 2.0
            logits[0, 3, 5] = 2.0
            return {"logits": logits}

    sequences = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )

    output = profile_module.trainer_sequence_log_probs(
        Model(),
        sequences,
        attention_mask,
    )

    assert captured["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert captured["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
    assert captured["packed_cu_seqlens"].tolist() == [0, 3, 5]
    assert captured["packed_max_seqlen"] == 3
    assert captured["use_cache"] is False
    assert output.shape == (2, 3)
    assert output[0, :2].tolist() == pytest.approx(output[1, :1].repeat(2).tolist())
    assert output[0, 2].item() == 0.0
    assert output[1, 1:].tolist() == [0.0, 0.0]


def test_consistency_attention_executes_standard_and_packed_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production attention adapter sends expected layouts to dense and varlen kernels."""
    runtime = profile_module._runtime  # pylint: disable=protected-access
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []

    def dense(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert key.shape == value.shape == query.shape
        calls.append(("dense", tuple(query.shape), kwargs))
        return query

    def varlen(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *_args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert key.shape == value.shape == query.shape
        calls.append(("varlen", tuple(query.shape), kwargs))
        return query

    monkeypatch.setattr(runtime, "flash_attn_func", dense)
    monkeypatch.setattr(runtime, "flash_attn_varlen_func", varlen)
    query = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)

    dense_output, dense_cache = profile_module._flash_attn_npu_attention_forward(  # pylint: disable=protected-access
        None,
        query,
        query,
        query,
        None,
        scaling=0.5,
    )
    packed_output, packed_cache = profile_module._flash_attn_npu_attention_forward(  # pylint: disable=protected-access
        None,
        query,
        query,
        query,
        None,
        scaling=0.25,
        packed_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        packed_max_seqlen=3,
    )

    assert dense_output.shape == (1, 3, 2, 2)
    assert packed_output.shape == (1, 3, 2, 2)
    assert dense_cache is None and packed_cache is None
    assert calls == [
        ("dense", (1, 3, 2, 2), {"dropout_p": 0.0, "softmax_scale": 0.5, "causal": True}),
        ("varlen", (3, 2, 2), {"dropout_p": 0.0, "softmax_scale": 0.25, "causal": True}),
    ]


def test_batch_invariant_reduction_and_rmsnorm_preserve_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production compatibility helpers preserve dimensions, dtype, epsilon, and values."""
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    reduced = profile_module._reduce_non_last_dimension(  # pylint: disable=protected-access
        tensor,
        dim=0,
        keepdim=True,
        reduce_last=lambda moved, preserve: moved.sum(dim=-1, keepdim=preserve),
    )
    rms_calls: list[tuple[torch.dtype, float]] = []

    def rms_norm(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        *,
        epsilon: float,
    ) -> tuple[torch.Tensor, None]:
        rms_calls.append((hidden_states.dtype, epsilon))
        return hidden_states + weight, None

    monkeypatch.setattr(
        profile_module._runtime,  # pylint: disable=protected-access
        "npu_rms_norm",
        rms_norm,
    )
    module = SimpleNamespace(
        weight=torch.ones(4, dtype=torch.bfloat16),
        variance_epsilon=1.0e-6,
    )
    output = profile_module._qwen3_npu_rms_norm_forward(  # pylint: disable=protected-access
        module,
        torch.zeros((2, 4), dtype=torch.float32),
    )

    torch.testing.assert_close(reduced, tensor.sum(dim=0, keepdim=True))
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.ones_like(output))
    assert rms_calls == [(torch.bfloat16, 1.0e-6)]


def test_consistency_installs_sum_rmsnorm_and_validates_rollout_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production compatibility installers wire successful kernel dependencies."""
    runtime = profile_module._runtime  # pylint: disable=protected-access
    monkeypatch.setattr(runtime, "batch_invariant_sum_compatibility_installed", False)
    monkeypatch.setattr(runtime, "npu_rms_norm", None)
    original_calls = []

    class DeviceTensor:
        device = SimpleNamespace(type="npu")

        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def dim(self) -> int:
            return self.value.dim()

        def movedim(self, source: int, destination: int) -> torch.Tensor:
            return self.value.movedim(source, destination)

    operations = SimpleNamespace(
        npu_reduce_sum_batch_invariant=lambda tensor, dim, keepdim: tensor.sum(
            dim=dim, keepdim=keepdim
        )
    )

    def original_reduce_sum(tensor: Any, dim: Any = None, keepdim: bool = False) -> Any:
        original_calls.append((dim, keepdim))
        return tensor.sum(dim=dim, keepdim=keepdim)

    batch_module = SimpleNamespace(
        reduce_sum=original_reduce_sum,
        torch=SimpleNamespace(
            ops=SimpleNamespace(batch_invariant_ops=operations),
            sum=torch.sum,
        ),
        HAS_ASCENDC_BATCH_INVARIANT=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend",
        SimpleNamespace(batch_invariant=batch_module),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.batch_invariant", batch_module)

    class RMSNorm:
        pass

    def npu_rms_norm(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        epsilon: float,
    ) -> Any:
        return (hidden * weight + epsilon,)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=npu_rms_norm),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3.modeling_qwen3",
        SimpleNamespace(Qwen3RMSNorm=RMSNorm),
    )
    monkeypatch.setattr(profile_module, "_require_package_versions", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_npu_v3",
        SimpleNamespace(flash_attn_with_kvcache=lambda *args, **kwargs: None),
    )

    profile_module._install_batch_invariant_sum_compatibility()  # pylint: disable=protected-access
    reduced = batch_module.reduce_sum(
        DeviceTensor(torch.arange(6.0).reshape(2, 3)),
        dim=0,
        keepdim=True,
    )
    cpu_reduced = batch_module.reduce_sum(torch.ones((2, 3)), dim=-1)
    profile_module._install_qwen3_npu_rms_norm()  # pylint: disable=protected-access
    rms_module = SimpleNamespace(
        weight=torch.tensor([2.0, 3.0]),
        variance_epsilon=0.5,
    )
    rms_output = getattr(RMSNorm, "forward")(
        rms_module, torch.tensor([[1.0, 2.0]])
    )
    profile_module.validate_rollout_consistency_profile(
        QWEN3_ASCEND_CONSISTENCY_V1
    )

    torch.testing.assert_close(reduced, torch.tensor([[3.0, 5.0, 7.0]]))
    torch.testing.assert_close(cpu_reduced, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(rms_output, torch.tensor([[2.5, 6.5]]))
    assert original_calls == [(-1, False)]
    assert runtime.batch_invariant_sum_compatibility_installed
    assert runtime.npu_rms_norm is npu_rms_norm


def test_consistency_post_update_diagnostic_counts_changed_action_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The successful negative control aggregates only changed trainable token bits."""
    experience = ExperienceBatch(
        trajectories=(),
        sequences=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[True, True, True], [True, True, False]]),
        action_mask=torch.tensor([[False, True, True], [False, True, False]]),
        rewards=torch.tensor([1.0, 0.0]),
        old_log_probs=torch.tensor([[-0.1, -0.2], [-0.3, 0.0]]),
        responses=("a", "b"),
        generation_seconds=0.0,
    )
    actor_log_probs = experience.old_log_probs.clone()
    actor_log_probs[0, 1] += 0.01

    def gather(output: list[Any], value: Any, group: Any) -> None:
        assert group == "dp"
        output[:] = [value, value]

    monkeypatch.setattr(gates_module.platform, "all_gather_object", gather)

    metrics = measure_post_update_old_policy_mismatch(
        experience,
        actor_log_probs,
        group="dp",
        group_size=2,
    )

    assert metrics == {
        "training/post_update_old_policy_tokens": 6.0,
        "training/post_update_old_policy_mismatch_count": 2.0,
        "training/post_update_negative_control_valid": 1.0,
    }


def test_consistency_validates_model_identity_versions_and_installs_rng_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported checkpoint and pinned dependency set install the rollout RNG hook."""
    config = {"consistency": {"enabled": True}}
    registration = SimpleNamespace(
        hyper_model_name="qwen3",
        hf_architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        text_model_type="qwen3",
    )
    profile_module.validate_consistency_model_identity(config, registration)
    versions = dict(profile_module._EXPECTED_PACKAGE_VERSIONS)  # pylint: disable=protected-access
    monkeypatch.setattr(
        profile_module,
        "package_version",
        lambda distribution: f"{versions[distribution]}+local",
    )
    profile_module._require_package_versions()  # pylint: disable=protected-access

    class Runner:
        def _sample(self) -> None:
            return None

        def _bookkeeping_sync(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.model_runner_v1",
        SimpleNamespace(NPUModelRunner=Runner),
    )
    install_partial_prefill_rng_fix()

    assert getattr(Runner, "_hyper_rl_partial_prefill_rng_patched")
