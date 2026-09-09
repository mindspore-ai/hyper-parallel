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
"""CPU unit tests for the main synchronous Trainer orchestration contracts."""
# Tests intentionally exercise stable internal stage boundaries.
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access,unnecessary-lambda

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch

import rl.trainer as trainer_module
import rl.utils.monitoring.metrics as metrics_module
from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.roles import Actor
from rl.trainer import SyncTrainer
from rl.utils.monitoring.metrics import ActorUpdateMetrics


def _rollout() -> ExperienceBatch:
    """Return one valid rollout row for orchestration tests."""
    return ExperienceBatch(
        trajectories=(),
        sequences=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        action_mask=torch.tensor([[False, True]]),
        rewards=torch.ones(1),
        old_log_probs=torch.zeros((1, 1)),
        responses=("answer",),
        generation_seconds=0.1,
    )


def test_trainer_initializes_batch_invariant_settings_before_distributed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration and deterministic communication precede runtime setup."""
    calls: list[str] = []
    resolved = {"algorithm": {"name": "grpo", "loss_aggregation": "token-mean"}}
    monkeypatch.setattr(trainer_module, "resolve_vllm_automatic_limits", lambda config: config)
    monkeypatch.setattr(
        trainer_module,
        "configure_consistency_profile",
        lambda _config: calls.append("consistency-config") or "off",
    )
    monkeypatch.setattr(
        trainer_module,
        "build_algorithm",
        lambda config: calls.append("algorithm") or build_algorithm(config),
    )
    monkeypatch.setattr(trainer_module, "validate_config", lambda *_args: calls.append("validate"))
    monkeypatch.setattr(
        trainer_module,
        "_configure_batch_invariant_communication",
        lambda _config: calls.append("batch-invariant"),
    )
    monkeypatch.setattr(
        trainer_module,
        "install_trainer_consistency_profile",
        lambda _config: calls.append("consistency-install"),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_model_registration",
        lambda _config: calls.append("model-registration") or SimpleNamespace(name="qwen"),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_runtime_config",
        lambda _config: calls.append("runtime-config")
        or SimpleNamespace(training=SimpleNamespace(train_iters=1)),
    )
    monkeypatch.setattr(SyncTrainer, "_setup_runtime", lambda self: calls.append("distributed"))
    monkeypatch.setattr(SyncTrainer, "_validate_runtime_topology", lambda self: calls.append("topology"))
    monkeypatch.setattr(SyncTrainer, "_build_runtime", lambda self: calls.append("roles"))

    trainer = SyncTrainer(resolved)

    assert trainer.state.max_steps == 1
    assert calls.index("batch-invariant") < calls.index("distributed")
    assert calls == [
        "consistency-config",
        "algorithm",
        "validate",
        "batch-invariant",
        "consistency-install",
        "model-registration",
        "runtime-config",
        "distributed",
        "topology",
        "roles",
    ]


def test_trainer_builds_independent_trainable_and_reference_roles() -> None:
    """GRPO construction exposes one trainable Actor and one frozen Reference."""
    trainer = object.__new__(SyncTrainer)
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.resolved_config = {
        "train": {
            "micro_batch_size": 1,
            "response_mini_batch_size": 2,
            "policy_update_epochs": 1,
            "optimizer": {"max_grad_norm": 1.0},
        }
    }
    trainer.device = torch.device("cpu")
    trainer._dp_group_info = None
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    actor_model = torch.nn.Linear(2, 2)
    reference_model = torch.nn.Linear(2, 2)
    trainer._build_one_parallel_model = lambda frozen: reference_model if frozen else actor_model
    scheduler = SimpleNamespace(step=lambda: None)
    trainer._build_optimizer_for = lambda model: (torch.optim.SGD(model.parameters(), lr=0.1), scheduler)

    trainer._build_models_and_optimizers()

    assert isinstance(trainer.actor, Actor)
    assert isinstance(trainer.reference_actor, Actor)
    assert trainer.actor.actor_model is actor_model
    assert trainer.reference_actor.actor_model is reference_model
    assert trainer.actor.actor_model is not trainer.reference_actor.actor_model
    assert trainer.actor.optimizer is not None
    assert trainer.reference_actor.optimizer is None
    assert all(not parameter.requires_grad for parameter in trainer.reference_actor.parameters())


def test_trainer_step_orchestrates_required_role_outputs_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One step orders consistency, required roles, updates, publication, and completion."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer._consistency_profile = "qwen3"
    trainer.algorithm = SimpleNamespace(
        name="acceptance",
        requirements=SimpleNamespace(
            data=SimpleNamespace(reference_log_probs=True, values=True)
        ),
    )
    trainer.state = SimpleNamespace(global_step=0, max_steps=1)
    trainer.device = torch.device("cpu")
    trainer._log_steps = 2
    trainer.evaluator = None
    trainer.checkpoints = SimpleNamespace(will_save=lambda _step: False)
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer._dp_group_info = SimpleNamespace(group=None)
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer._reshard_model = lambda _model: calls.append("reshard")
    trainer.model_registration = SimpleNamespace(name="qwen")
    rollout = _rollout()
    actor_log_probs = rollout.old_log_probs.clone()
    reference_log_probs = torch.full_like(actor_log_probs, -0.5)
    values = torch.full_like(actor_log_probs, 0.25)
    trainer.rollout_manager = SimpleNamespace(
        generate=lambda **_kwargs: calls.append("rollout") or rollout
    )
    trainer.rollout_engine = SimpleNamespace(
        policy_fingerprint="digest-v0",
        prepare_for_training=lambda: calls.append("prepare-training"),
        update_weights=lambda _snapshot: calls.append("update-weights"),
        prepare_for_rollout=lambda: calls.append("prepare-rollout"),
    )

    class ActorRole:
        actor_model = object()

        @staticmethod
        def compute_log_probs(_experience: ExperienceBatch) -> torch.Tensor:
            calls.append("actor-logprobs")
            return actor_log_probs

        @staticmethod
        def update(experience: ExperienceBatch) -> Any:
            assert experience is not None
            calls.append("actor-update")
            return SimpleNamespace(optimizer_steps=1)

    class ReferenceRole:
        @staticmethod
        def compute_log_probs(_experience: ExperienceBatch) -> torch.Tensor:
            calls.append("reference")
            return reference_log_probs

    class CriticRole:
        @staticmethod
        def compute_values(_experience: ExperienceBatch) -> torch.Tensor:
            calls.append("critic-values")
            return values

        @staticmethod
        def update(_experience: ExperienceBatch) -> Any:
            calls.append("critic-update")
            return SimpleNamespace()

    class Preparer:
        @staticmethod
        def prepare(
            experience: ExperienceBatch,
            *,
            reference_log_probs: torch.Tensor,
            values: Any,
        ) -> ExperienceBatch:
            assert reference_log_probs is not None
            assert values is not None
            calls.append("targets")
            return experience

    def complete_step(self: SyncTrainer, **kwargs: Any) -> None:
        assert kwargs["step"] == 1
        assert kwargs["critic_update"] is not None
        assert kwargs["diagnostic_metrics"] == {
            "training/tokens": 1.0,
            "training/exact": 1.0,
            "training/post_update": 1.0,
        }
        calls.append("complete")
        self.state.global_step = 1

    trainer.actor = ActorRole()
    trainer.reference_actor = ReferenceRole()
    trainer.critic = CriticRole()
    trainer.experience_preparer = Preparer()
    trainer._complete_step = MethodType(complete_step, trainer)
    monkeypatch.setattr(trainer_module, "_write_rollout_artifact", lambda *_args: None)
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(trainer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(
        trainer_module,
        "validate_consistency_forward_inputs",
        lambda *_args, **_kwargs: calls.append("forward-contract"),
    )
    monkeypatch.setattr(
        trainer_module,
        "validate_pre_update_consistency",
        lambda *_args, **_kwargs: calls.append("exact-gate")
        or {"training/exact": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "summarize_training_diagnostics",
        lambda *_args: calls.append("diagnostics") or {"training/tokens": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "measure_post_update_old_policy_mismatch",
        lambda *_args, **_kwargs: calls.append("post-update-check")
        or {"training/post_update": 1.0},
    )
    batch = {
        "sample_indices": [0],
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "prompts": ["prompt"],
        "ground_truths": ["answer"],
    }

    trainer._train_step(batch)

    assert calls == [
        "rollout",
        "prepare-training",
        "forward-contract",
        "actor-logprobs",
        "exact-gate",
        "reference",
        "critic-values",
        "targets",
        "diagnostics",
        "actor-update",
        "actor-logprobs",
        "post-update-check",
        "critic-update",
        "reshard",
        "release",
        "update-weights",
        "prepare-rollout",
        "complete",
    ]
    assert trainer.state.global_step == 1


def test_trainer_publication_releases_training_state_before_rollout_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colocated publication releases training allocations before waking rollout."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.actor = SimpleNamespace(actor_model=object())
    trainer.model_registration = SimpleNamespace(name="qwen")
    trainer._reshard_model = lambda _model: calls.append("reshard")
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer.rollout_engine = SimpleNamespace(
        update_weights=lambda _snapshot: calls.append("update"),
        prepare_for_rollout=lambda: calls.append("wake"),
    )
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("stream"))

    trainer._publish_policy(2)

    assert calls == ["stream", "reshard", "release", "update", "wake"]


def test_trainer_completes_metrics_and_evaluation_after_published_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful publication advances step before metrics, evaluation, and logging."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.state = SimpleNamespace(global_step=0, max_steps=2)
    trainer._log_samples = 1
    trainer._log_steps = 1
    trainer.resolved_config = {"train": {"learning_gate": {"enabled": False}}}
    trainer.rollout_engine = SimpleNamespace()
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.checkpoints = SimpleNamespace(
        will_save=lambda _step: True,
        complete_step=lambda *_args, **_kwargs: None,
    )
    trainer.evaluator = SimpleNamespace(
        run=lambda step: calls.append(f"evaluate:{step}") or ({"validation/accuracy": 1.0}, [])
    )
    trainer._tracker = SimpleNamespace(
        log=lambda _metrics, step, **_kwargs: calls.append(f"log:{step}")
    )
    actor_update = ActorUpdateMetrics(0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, 2, 1)

    def summarize(*_args: Any, **_kwargs: Any) -> tuple[dict[str, float], list[Any]]:
        assert trainer.state.global_step == 1
        calls.append("summarize")
        return {"reward/mean": 1.0}, []

    monkeypatch.setattr(trainer_module, "summarize_rollout", summarize)
    monkeypatch.setattr(
        trainer_module,
        "build_training_metrics",
        lambda **_kwargs: calls.append("metrics") or {"train/gradient_norm": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "enforce_learning_gate",
        lambda *_args, **_kwargs: calls.append("gate"),
    )
    monkeypatch.setattr(trainer_module, "uses_colocated_vllm", lambda _config: False)

    SyncTrainer._complete_step(
        trainer,
        step=1,
        batch={},
        rollout=_rollout(),
        actor_update=actor_update,
        critic_update=None,
        diagnostic_metrics={},
    )

    assert trainer.state.global_step == 1
    assert calls == ["summarize", "metrics", "gate", "evaluate:1", "log:1"]


def test_trainer_runs_one_complete_synchronous_training_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production train and step methods complete one strongly synchronous publication."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer._consistency_profile = "off"
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.state = SimpleNamespace(global_step=0, max_steps=1, epoch=0)
    trainer.resolved_config = {
        "train": {
            "checkpoint": {"save_final": False},
            "learning_gate": {"enabled": False},
        }
    }
    trainer.device = torch.device("cpu")
    trainer._log_steps = 1
    trainer._log_samples = 0
    trainer.evaluator = None
    trainer._runtime_started = False
    trainer._dp_group_info = SimpleNamespace(group=None)
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer.model_registration = SimpleNamespace(name="qwen")
    batch = {
        "sample_indices": [0],
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "prompts": ["prompt"],
        "ground_truths": ["answer"],
    }
    trainer.train_dataloader = [batch]
    rollout = _rollout()
    trainer.rollout_manager = SimpleNamespace(
        generate=lambda **_kwargs: calls.append("rollout") or rollout
    )

    class Engine:
        policy_version = 0
        policy_fingerprint = None
        policy_fingerprint_changed = False
        weight_sync_configured_strategy = "full_gather"
        weight_sync_last_strategy = "full_gather"
        weight_sync_fallback_count = 0
        weight_sync_direct_success_count = 0

        @staticmethod
        def prepare_for_training() -> None:
            calls.append("prepare-training")

        @staticmethod
        def update_weights(snapshot: Any) -> None:
            calls.append(f"publish:{snapshot.version}")

        @staticmethod
        def prepare_for_rollout() -> None:
            calls.append("prepare-rollout")

        @staticmethod
        def close() -> None:
            calls.append("engine-close")

    class ActorRole:
        actor_model = torch.nn.Identity()

        @staticmethod
        def compute_log_probs(experience: ExperienceBatch) -> torch.Tensor:
            calls.append("actor-logprobs")
            return experience.old_log_probs.clone()

        @staticmethod
        def update(experience: ExperienceBatch) -> ActorUpdateMetrics:
            assert experience is rollout
            calls.append("actor-update")
            return ActorUpdateMetrics(0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, 1, 1)

    class ReferenceRole:
        @staticmethod
        def compute_log_probs(experience: ExperienceBatch) -> torch.Tensor:
            calls.append("reference")
            return torch.zeros_like(experience.old_log_probs)

    class Preparer:
        @staticmethod
        def prepare(experience: ExperienceBatch, **kwargs: Any) -> ExperienceBatch:
            assert kwargs["reference_log_probs"] is not None
            calls.append("targets")
            return experience

    class Checkpoints:
        @staticmethod
        def validate_resume() -> None:
            calls.append("validate-resume")

        @staticmethod
        def begin(_state: Any) -> None:
            calls.append("checkpoint-begin")

        @staticmethod
        def will_save(_step: int) -> bool:
            return False

        @staticmethod
        def complete_step(_state: Any, **_kwargs: Any) -> None:
            calls.append("checkpoint-step")

        @staticmethod
        def finalize(_state: Any) -> None:
            calls.append("checkpoint-finalize")

    class Tracker:
        @staticmethod
        def log(_metrics: Any, *, step: int, **_kwargs: Any) -> None:
            calls.append(f"log:{step}")

        @staticmethod
        def finish() -> None:
            calls.append("tracker-finish")

    trainer.rollout_engine = Engine()
    trainer.actor = ActorRole()
    trainer.reference_actor = ReferenceRole()
    trainer.critic = None
    trainer.experience_preparer = Preparer()
    trainer.checkpoints = Checkpoints()
    trainer._tracker = Tracker()
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    monkeypatch.setattr(trainer_module, "_write_rollout_artifact", lambda *_args: calls.append("artifact"))
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("stream"))
    monkeypatch.setattr(trainer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_module.platform, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(metrics_module, "_system_memory_metrics", lambda: {})
    monkeypatch.setattr(metrics_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(metrics_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        metrics_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    trainer.train()

    assert trainer.state.global_step == 1
    assert calls.index("rollout") < calls.index("prepare-training")
    assert calls.index("prepare-training") < calls.index("reference")
    assert calls.index("reference") < calls.index("targets")
    assert calls.index("targets") < calls.index("actor-update")
    assert calls.index("actor-update") < calls.index("publish:1")
    assert calls.index("publish:1") < calls.index("prepare-rollout")
    assert calls.index("prepare-rollout") < calls.index("checkpoint-step")
    assert calls[-4:] == ["checkpoint-finalize", "barrier", "tracker-finish", "engine-close"]


def test_trainer_sets_up_runtime_with_mocked_distributed_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production runtime setup builds device, mesh, DP group, and deterministic seeds."""
    calls: list[Any] = []
    dp_mesh = SimpleNamespace(get_group=lambda: "dp-group")
    mesh_context = SimpleNamespace(
        device_mesh="device-mesh",
        dp_cp_mesh=dp_mesh,
        dp_size=2,
    )
    setup = SimpleNamespace(mesh_context=mesh_context)
    handle = SimpleNamespace(set_device=lambda rank: calls.append(("set-device", rank)))
    trainer = object.__new__(SyncTrainer)
    trainer.runtime_config = SimpleNamespace(
        training=SimpleNamespace(backend="hccl", seed=1234)
    )
    trainer._runtime_started = False
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        trainer_module,
        "initialize_distributed",
        lambda backend: calls.append(("initialize", backend)),
    )
    monkeypatch.setattr(
        trainer_module,
        "create_distributed_setup_from_config",
        lambda config: calls.append(("mesh", config)) or setup,
    )
    monkeypatch.setattr(trainer_module.platform, "device", lambda rank: torch.device("cpu", rank))
    monkeypatch.setattr(trainer_module.platform, "device_type", lambda: "cpu")
    monkeypatch.setattr(trainer_module.platform, "get_device_handle", lambda _kind: handle)
    monkeypatch.setattr(trainer_module.platform, "manual_seed", lambda seed: calls.append(("seed", seed)))

    trainer._setup_runtime()

    assert trainer._runtime_started
    assert trainer.distributed_setup is setup
    assert trainer.parallel_dims is mesh_context
    assert trainer.mesh == "device-mesh"
    assert trainer.device == torch.device("cpu", 1)
    assert trainer._dp_group_info.group == "dp-group"
    assert trainer._dp_group_info.rank_size == 2
    assert calls == [
        ("initialize", "hccl"),
        ("mesh", trainer.runtime_config),
        ("set-device", 1),
        ("seed", 1234),
    ]


def test_trainer_builds_tokenizer_data_and_dataloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production data construction wires tokenizer, splits, sampler, collate, and loader."""
    calls: list[Any] = []

    class Tokenizer:
        eos_token_id = 2
        eos_token = "<eos>"
        pad_token_id = None
        padding_side = "right"

        @property
        def pad_token(self) -> str:
            return self.eos_token

        @pad_token.setter
        def pad_token(self, value: str) -> None:
            assert value == self.eos_token
            self.pad_token_id = self.eos_token_id

    class Dataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("dataset", kwargs))

        @staticmethod
        def __len__() -> int:
            return 8

    class Loader:
        def __init__(self, dataset: Any, **kwargs: Any) -> None:
            calls.append(("loader", dataset, kwargs))
            self.kwargs = kwargs

    tokenizer = Tokenizer()
    monkeypatch.setattr(
        trainer_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: calls.append(("tokenizer", args, kwargs)) or tokenizer,
    )
    monkeypatch.setattr(trainer_module, "PromptDataset", Dataset)
    monkeypatch.setattr(trainer_module, "StatefulDataLoader", Loader)
    trainer = object.__new__(SyncTrainer)
    trainer.resolved_config = {
        "model": {"tokenizer_path": "/model"},
        "data": {
            "train_path": "/train.parquet",
            "test_path": "/test.parquet",
            "max_prompt_length": 32,
            "prompt_column": "prompt",
            "answer_column": "answer",
            "prompt_instruction": "instruction",
            "max_train_samples": 6,
            "shuffle": True,
            "num_workers": 2,
            "prefetch_factor": 3,
            "pin_memory": False,
        },
        "evaluation": {"enabled": True},
        "train": {"prompt_batch_size": 2},
    }
    trainer.parallel_dims = SimpleNamespace(dp_size=2, dp_rank=1)
    trainer.runtime_config = SimpleNamespace(training=SimpleNamespace(seed=1234))

    trainer._build_tokenizer_and_data()

    assert tokenizer.pad_token_id == 2
    assert tokenizer.padding_side == "left"
    dataset_calls = [call for call in calls if call[0] == "dataset"]
    assert len(dataset_calls) == 2
    assert dataset_calls[0][1]["parquet_path"] == "/train.parquet"
    assert dataset_calls[0][1]["max_samples"] == 6
    assert dataset_calls[1][1]["parquet_path"] == "/test.parquet"
    loader_call = next(call for call in calls if call[0] == "loader")
    assert loader_call[2]["batch_size"] == 2
    assert loader_call[2]["num_workers"] == 2
    assert loader_call[2]["prefetch_factor"] == 3
    assert loader_call[2]["pin_memory"] is False
    assert loader_call[2]["drop_last"] is True


def test_rollout_artifact_writes_complete_synchronized_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Production artifact writer persists exact tensors and committed worker identity."""
    rollout = ExperienceBatch(
        **{
            **_rollout().__dict__,
            "worker_policy_version": 2,
            "worker_policy_fingerprint": "digest-v2",
        }
    )
    monkeypatch.setenv("HYPER_RL_ROLLOUT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("HYPER_RL_ROLLOUT_ARTIFACT_STRATEGY", "direct_reshard")
    monkeypatch.setenv("HYPER_RL_WEIGHT_ORACLE_RUN_ID", "run-2")
    monkeypatch.setattr(trainer_module.platform, "get_rank", lambda: 0)

    trainer_module._write_rollout_artifact(rollout, policy_version=2)

    path = tmp_path / "direct_reshard-policy2-rank0.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["strategy"] == "direct_reshard"
    assert artifact["oracle_run_id"] == "run-2"
    assert artifact["policy_version"] == 2
    assert artifact["worker_policy_version"] == 2
    assert artifact["worker_policy_fingerprint"] == "digest-v2"
    assert artifact["sequences"]["values"] == [[1, 2]]
    assert artifact["attention_mask"]["values"] == [[1, 1]]
    assert artifact["action_mask"]["values"] == [[0, 1]]
    assert artifact["raw_logprobs"]["values"] == [[0.0]]


def test_trainer_builds_rollout_evaluator_and_tp_request_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production rollout construction wires TP ownership and separate evaluation settings."""
    calls: list[Any] = []

    class Engine:
        def configure_trainer_tensor_parallel(self, **kwargs: Any) -> None:
            calls.append(("tp", kwargs))

    engine = Engine()
    managers: list[Any] = []

    def manager(**kwargs: Any) -> Any:
        result = SimpleNamespace(kwargs=kwargs)
        managers.append(result)
        return result

    monkeypatch.setattr(
        trainer_module,
        "build_rollout_engine",
        lambda rollout, model: calls.append(("engine", rollout, model)) or engine,
    )
    monkeypatch.setattr(trainer_module, "RolloutManager", manager)
    monkeypatch.setattr(
        trainer_module,
        "Evaluator",
        lambda **kwargs: calls.append(("evaluator", kwargs)) or SimpleNamespace(**kwargs),
    )
    trainer = object.__new__(SyncTrainer)
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.model_registration = SimpleNamespace(name="qwen")
    trainer.model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[2, 3])
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    trainer.parallel_dims = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_size=1,
        device_mesh={"tp": SimpleNamespace(get_group=lambda: "tp-group")},
    )
    trainer.resolved_config = {
        "rollout": {
            "engine": "vllm",
            "num_return_sequences": 4,
            "max_new_tokens": 16,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 5,
            "seed": 10,
        },
        "agentic": {
            "environment": "env",
            "max_turns": 2,
            "max_observation_tokens": 8,
            "max_episode_tokens": 32,
            "interaction_mode": "multi_turn",
        },
        "evaluation": {
            "enabled": True,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "do_sample": False,
            "batch_size": 2,
            "max_samples": 4,
            "log_samples": 1,
            "progress_steps": 1,
        },
    }
    trainer._evaluation_enabled = True
    trainer.test_dataset = [object()]
    trainer.collate_fn = object()
    trainer.device = torch.device("cpu")

    trainer._build_rollout_runtime()

    tp_call = next(call for call in calls if call[0] == "tp")
    assert tp_call[1] == {
        "group": "tp-group",
        "tp_rank": 1,
        "tp_size": 2,
        "request_rank": 0,
        "request_size": 1,
    }
    assert len(managers) == 2
    assert managers[0].kwargs["engine"] is engine
    assert managers[0].kwargs["num_return_sequences"] == 4
    assert managers[0].kwargs["do_sample"] is True
    assert managers[0].kwargs["collect_old_log_probs"] is True
    assert managers[0].kwargs["eos_token_ids"] == (2, 3)
    assert managers[1].kwargs["engine"] is engine
    assert managers[1].kwargs["num_return_sequences"] == 1
    assert managers[1].kwargs["do_sample"] is False
    assert trainer.evaluator.rollout_manager is managers[1]


def test_trainer_builds_runtime_and_tracker_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production runtime and tracker methods retain component order and rank-zero config."""
    calls: list[Any] = []
    trainer = object.__new__(SyncTrainer)
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.resolved_config = {
        "train": {
            "checkpoint": {"output_dir": "/tmp/checkpoints"},
        },
        "logging": {
            "backends": ["console", "wandb"],
            "project_name": "project",
            "experiment_name": "experiment",
            "log_steps": 2,
            "log_samples": 3,
            "wandb": {
                "mode": "offline",
                "entity": "team",
                "directory": "/tmp/wandb",
            },
        },
    }
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer._build_tokenizer_and_data = lambda: calls.append("data")
    trainer._build_models_and_optimizers = lambda: calls.append("models")
    trainer._build_rollout_runtime = lambda: calls.append("rollout")
    tracker_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        trainer_module,
        "RLCheckpointManager",
        lambda *args: calls.append(("checkpoint", args)) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        trainer_module,
        "TrainingTracker",
        lambda **kwargs: tracker_kwargs.update(kwargs) or calls.append("tracker") or SimpleNamespace(),
    )
    monkeypatch.setattr(trainer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.platform, "get_world_size", lambda: 4)

    trainer._build_runtime()

    assert calls[:3] == ["data", "models", "rollout"]
    assert calls[3][0] == "checkpoint"
    assert calls[4] == "tracker"
    assert trainer.experience_preparer.algorithm is trainer.algorithm
    assert trainer._log_steps == 2
    assert trainer._log_samples == 3
    assert tracker_kwargs["rank"] == 0
    assert tracker_kwargs["world_size"] == 4
    assert tracker_kwargs["backends"] == ("console", "wandb")
    assert tracker_kwargs["wandb_mode"] == "offline"


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_trainer_validates_supported_shared_runtime_topology(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """Production topology validation accepts one consistent shared DP2xTP1 deployment."""
    trainer = object.__new__(SyncTrainer)
    trainer.parallel_dims = SimpleNamespace(
        dp_size=2,
        cp_size=1,
        tp_size=1,
        pp_size=1,
    )
    vllm: dict[str, Any] = {
        "deployment": deployment,
        "data_parallel_size": 2,
        "tensor_parallel_size": 1,
        "port": 8100,
    }
    if deployment == "disjoint":
        vllm["visible_devices"] = "4,5"
    trainer.resolved_config = {"rollout": {"engine": "vllm", "vllm": vllm}}
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(trainer_module.platform, "get_world_size", lambda: 2)

    def gather(output: list[Any], value: Any) -> None:
        output[:] = [value, value]

    monkeypatch.setattr(trainer_module.platform, "all_gather_object", gather)

    trainer._validate_runtime_topology()


def test_trainer_releases_colocated_training_state_for_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production residency release reshards roles, validates CPU state, and clears cache."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.resolved_config = {
        "rollout": {"engine": "vllm", "vllm": {"deployment": "colocated"}}
    }
    trainer.actor = SimpleNamespace(actor_model="actor")
    trainer.reference_actor = SimpleNamespace(actor_model="reference")
    trainer.critic = SimpleNamespace(
        critic_model="critic",
        optimizer=SimpleNamespace(state={}),
    )
    trainer.optimizer = SimpleNamespace(state={})
    trainer._reshard_model = lambda model: calls.append(f"reshard:{model}")
    trainer._run_rank_synchronized = lambda name, callback: calls.append(name) or callback()
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))
    handle = SimpleNamespace(empty_cache=lambda: calls.append("empty-cache"))
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("hsdp-sync"))
    monkeypatch.setattr(trainer_module.platform, "get_current_stream", lambda: stream)
    monkeypatch.setattr(trainer_module.platform, "device_type", lambda: "cpu")
    monkeypatch.setattr(trainer_module.platform, "get_device_handle", lambda _kind: handle)

    trainer._release_training_state_for_rollout()

    assert calls == [
        "training-state release",
        "hsdp-sync",
        "reshard:actor",
        "reshard:reference",
        "reshard:critic",
        "stream",
        "allocator-cache release",
        "empty-cache",
        "stream",
    ]


def test_trainer_builds_required_models_optimizers_and_role_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GRPO construction creates independent trainable/reference models and Actor roles."""
    trainer = object.__new__(SyncTrainer)
    trainer.algorithm = build_algorithm(
        {"name": "grpo", "loss_aggregation": "token-mean"}
    )
    trainer.resolved_config = {
        "train": {
            "optimizer": {"max_grad_norm": 2.0},
            "micro_batch_size": 1,
            "response_mini_batch_size": 2,
            "policy_update_epochs": 3,
        }
    }
    trainer.device = torch.device("cpu")
    trainer._dp_group_info = "dp-group"
    trainer.parallel_dims = SimpleNamespace(dp_size=2)
    trainable_model = torch.nn.Linear(2, 2)
    reference_model = torch.nn.Linear(2, 2)
    models = iter((trainable_model, reference_model))
    frozen_flags = []

    def build_model(frozen: bool) -> Any:
        frozen_flags.append(frozen)
        return next(models)

    trainer._build_one_parallel_model = build_model
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}])
    scheduler = SimpleNamespace()
    trainer._build_optimizer_for = lambda model: (optimizer, scheduler)
    actor_calls = []

    def actor(**kwargs: Any) -> Any:
        actor_calls.append(kwargs)
        return SimpleNamespace(
            actor_model=kwargs["actor_model"],
            optimizer=kwargs.get("optimizer"),
            lr_scheduler=kwargs.get("lr_scheduler"),
        )

    monkeypatch.setattr(trainer_module, "Actor", actor)

    trainer._build_models_and_optimizers()

    assert frozen_flags == [False, True]
    assert actor_calls[0] == {
        "actor_model": trainable_model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "device": torch.device("cpu"),
        "dp_group_info": "dp-group",
        "dp_size": 2,
        "response_mini_batch_size": 2,
        "update_epochs": 3,
        "max_grad_norm": 2.0,
        "algorithm": trainer.algorithm,
        "micro_batch_size": 1,
    }
    assert actor_calls[1] == {
        "actor_model": reference_model,
        "algorithm": trainer.algorithm,
        "micro_batch_size": 1,
    }
    assert trainer.actor.actor_model is trainable_model
    assert trainer.reference_actor.actor_model is reference_model
    assert trainer.critic is None
    assert trainer.model is trainable_model
    assert trainer.optimizer is optimizer
    assert trainer.lr_scheduler is scheduler


def test_trainer_runtime_helpers_cover_epoch_cleanup_and_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampler rollover, recursive CPU state, reshard, cleanup, and rank sync compose."""
    trainer = object.__new__(SyncTrainer)
    trainer.state = SimpleNamespace(epoch=0)
    sampler = trainer_module._DistributedPromptSampler(  # pylint: disable=protected-access
        7,
        rank=1,
        world_size=2,
        seed=3,
        shuffle=True,
    )
    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert len(sampler) == 3
    assert len(first_epoch) == len(second_epoch) == 3
    assert first_epoch != second_epoch
    epochs = []
    trainer.sampler = SimpleNamespace(set_epoch=lambda epoch: epochs.append(epoch))
    trainer.train_dataloader = [{"sample": 2}]

    batch, iterator = trainer._next_batch(iter(()))

    assert batch == {"sample": 2}
    assert next(iter([iterator])) is iterator
    assert trainer.state.epoch == 1
    assert epochs == [1]

    roots = [SimpleNamespace(reshard=lambda: epochs.append("reshard")) for _ in range(2)]
    monkeypatch.setattr(trainer_module, "iter_hsdp_roots", lambda _model: roots)
    trainer._reshard_model(object())
    trainer._validate_optimizer_cpu_residency(
        SimpleNamespace(
            state={
                "parameter": {
                    "step": torch.tensor(1),
                    "moments": [torch.tensor([1.0]), (torch.tensor([2.0]),)],
                }
            }
        ),
        "actor",
    )
    assert epochs[-2:] == ["reshard", "reshard"]

    monkeypatch.setattr(trainer_module.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, value]),
    )
    assert trainer._run_rank_synchronized("success", lambda: "result") == "result"

    cleanup = []
    trainer._tracker = SimpleNamespace(finish=lambda: cleanup.append("tracker"))
    trainer.rollout_engine = SimpleNamespace(close=lambda: cleanup.append("rollout"))
    trainer._runtime_started = True
    monkeypatch.setattr(
        trainer_module,
        "destroy_process_group",
        lambda: cleanup.append("distributed"),
    )

    trainer._cleanup_distributed()

    assert cleanup == ["tracker", "rollout", "distributed"]
    assert trainer._tracker is None
    assert not trainer._runtime_started


def test_trainer_complete_step_coordinates_colocated_checkpoint_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saving step evaluates, enters training residency, commits, and restores rollout."""
    calls = []
    trainer = object.__new__(SyncTrainer)
    trainer.state = SimpleNamespace(global_step=0, max_steps=2)
    trainer._log_samples = 0
    trainer._log_steps = 10
    trainer.resolved_config = {"train": {"learning_gate": {"enabled": False}}}
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.checkpoints = SimpleNamespace(
        will_save=lambda _step: True,
        complete_step=lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    trainer.evaluator = SimpleNamespace(
        run=lambda step: calls.append(f"evaluate:{step}")
        or ({"validation/accuracy": 1.0}, [{"sample": step}])
    )
    trainer._tracker = SimpleNamespace(
        log=lambda *_args, **_kwargs: calls.append("log")
    )
    trainer.rollout_engine = SimpleNamespace(
        prepare_for_training=lambda: calls.append("training"),
        prepare_for_rollout=lambda: calls.append("rollout"),
    )
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    monkeypatch.setattr(trainer_module, "uses_colocated_vllm", lambda _config: True)
    monkeypatch.setattr(trainer_module, "summarize_rollout", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(trainer_module, "build_training_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(trainer_module, "enforce_learning_gate", lambda *_args, **_kwargs: None)
    update = SimpleNamespace(total_loss=0.1, gradient_norm=1.0)

    SyncTrainer._complete_step(
        trainer,
        step=1,
        batch={},
        rollout=_rollout(),
        actor_update=update,
        critic_update=None,
        diagnostic_metrics={},
    )

    assert trainer.state.global_step == 1
    assert calls == [
        "evaluate:1",
        "log",
        "training",
        "release",
        "checkpoint",
        "release",
        "rollout",
    ]
