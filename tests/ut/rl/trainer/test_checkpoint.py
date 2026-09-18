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
"""CPU contracts for the top-level RL distributed checkpoint lifecycle."""
# White-box regression tests intentionally exercise internal state and lifecycle hooks.
# pylint: disable=protected-access
# pylint: disable=missing-public-docstring

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

import rl.checkpoint as checkpoint_backend
from rl.checkpoint import (
    RLCheckpointManager,
    _clone_shared_checkpoint_tensors,
)


class _Stateful:
    """Minimal state-dict owner used for model, optimizer, and scheduler tests."""

    def __init__(self, value: str) -> None:
        """Store a visible state value."""
        self.value = value

    def state_dict(self) -> dict[str, Any]:
        """Return mutable state consumed by distributed checkpoint IO."""
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Apply a restored state value."""
        self.value = str(state["value"])

    def named_modules(self) -> Any:
        """Expose a module-like iterator for checkpoint topology helpers."""
        return iter((("", self),))


def test_checkpoint_state_clones_tied_storage_without_dropping_alias() -> None:
    """Safetensors receives both tied names backed by independent storage."""
    tied = torch.arange(4, dtype=torch.float32)
    state = {
        "model.embed_tokens.weight": tied,
        "lm_head.weight": tied,
        "model.norm.weight": tied.clone(),
    }

    checkpoint_state, aliases = _clone_shared_checkpoint_tensors(state)

    assert aliases == (("lm_head.weight", "model.embed_tokens.weight"),)
    assert set(checkpoint_state) == set(state)
    assert torch.equal(
        checkpoint_state["model.embed_tokens.weight"],
        checkpoint_state["lm_head.weight"],
    )
    assert (
        checkpoint_state["model.embed_tokens.weight"].data_ptr()
        != checkpoint_state["lm_head.weight"].data_ptr()
    )


def _manager(tmp_path: Path) -> tuple[RLCheckpointManager, Any]:
    """Build a manager around independently stateful role components."""
    trainer = SimpleNamespace(
        state=SimpleNamespace(max_steps=2),
        model=_Stateful("model-live"),
        model_registration=SimpleNamespace(hf_architecture="Qwen3ForCausalLM"),
        optimizer=_Stateful("optimizer-live"),
        lr_scheduler=_Stateful("scheduler-live"),
        train_dataloader=_Stateful("dataloader-live"),
        device="npu:0",
        device_handle=SimpleNamespace(
            get_rng_state=lambda _device: "device-rng",
            set_rng_state=lambda _state, _device: None,
        ),
    )
    manager = RLCheckpointManager(
        trainer,
        {
            "output_dir": str(tmp_path),
            "save_steps": 1,
            "save_final": True,
            "load_path": str(tmp_path / "step_1"),
        },
        {},
        lambda _operation, callback: callback(),
    )
    return manager, trainer


def test_save_persists_distributed_and_device_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor state is collective while RNG and dataloader state remain rank-local."""
    manager, _ = _manager(tmp_path)
    saves = []
    dispatch = MagicMock()
    monkeypatch.setattr(checkpoint_backend, "SkipDTensorDispatch", lambda: dispatch)
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        checkpoint_backend.torch,
        "get_rng_state",
        lambda: "cpu-rng",
    )
    monkeypatch.setattr(
        checkpoint_backend,
        "dcp_save",
        lambda state, **kwargs: saves.append((state, kwargs)),
    )
    state = SimpleNamespace(
        global_step=1,
        epoch=2,
        consumed_samples=3,
        consumed_tokens=4,
    )

    manager.complete_step(state, loss=0.5, grad_norm=1.0)

    assert saves[0][1]["use_collectives"] is True
    assert set(saves[0][0]) == {"model"}
    assert saves[1][1]["use_collectives"] is False
    rank_state = pickle.loads(saves[1][0]["runtime"])
    assert rank_state["cpu_rng"] == "cpu-rng"
    assert rank_state["device_rng"] == "device-rng"
    assert rank_state["dataloader"] == {"value": "dataloader-live"}
    assert rank_state["optimizer"] == {"value": "optimizer-live"}
    dispatch.__enter__.assert_called_once_with()
    dispatch.__exit__.assert_called_once()
    assert manager.directory(1).joinpath("checkpoint_complete.json").is_file()


def test_resume_restores_role_state_and_both_rng_domains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume restores policy progress, optimizer state, and CPU/device generators."""
    manager, trainer = _manager(tmp_path)
    checkpoint_dir = manager.directory(1)
    checkpoint_dir.mkdir()
    checkpoint_dir.joinpath("extra_state.json").write_text(
        '{"global_step": 1, "epoch": 2, "consumed_samples": 3, "consumed_tokens": 4}',
        encoding="utf-8",
    )
    restored_rng = []
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.torch, "get_rng_state", lambda *_args: "live-rng")
    monkeypatch.setattr(
        checkpoint_backend.torch,
        "set_rng_state",
        lambda state, *args: restored_rng.append((state, args)),
    )

    monkeypatch.setattr(
        trainer.device_handle,
        "set_rng_state",
        lambda state, device: restored_rng.append((state, (device,))),
    )

    def load(state: dict[str, Any], **kwargs: Any) -> None:
        """Inject deterministic distributed or rank-local checkpoint values."""
        if kwargs["use_collectives"]:
            state["model"] = {"value": "model-restored"}
        else:
            state["runtime"] = {
                "cpu_rng": "cpu-restored",
                "device_rng": "device-restored",
                "dataloader": {"value": "dataloader-restored"},
                "optimizer": {"value": "optimizer-restored"},
                "scheduler": {"value": "scheduler-restored"},
            }

    monkeypatch.setattr(checkpoint_backend, "dcp_load", load)
    state = SimpleNamespace(
        global_step=0,
        epoch=0,
        consumed_samples=0,
        consumed_tokens=0,
    )

    manager.begin(state)

    assert trainer.model.value == "model-restored"
    assert trainer.optimizer.value == "optimizer-restored"
    assert trainer.lr_scheduler.value == "scheduler-restored"
    assert trainer.train_dataloader.value == "dataloader-restored"
    assert restored_rng == [
        ("cpu-restored", ()),
        ("device-restored", (trainer.device,)),
    ]
    assert vars(state) == {
        "global_step": 1,
        "epoch": 2,
        "consumed_samples": 3,
        "consumed_tokens": 4,
    }


@pytest.mark.parametrize("step", [0, 1, 2])
def test_finalize_saves_even_when_no_periodic_save_ran(tmp_path: Path, step: int) -> None:
    """Fresh and resumed runs produce a final checkpoint in this output directory."""
    manager, _ = _manager(tmp_path)
    manager.config["verify_reload"] = True
    manager._save = MagicMock()
    manager._verify_reload = MagicMock()
    manager._export_hf = MagicMock()
    state = SimpleNamespace(global_step=step)

    manager.finalize(state)

    manager._save.assert_called_once_with(state)
    manager._verify_reload.assert_called_once_with(manager.directory(step))
    manager._export_hf.assert_called_once_with(manager.directory(step))


def test_finalize_reuses_successful_periodic_save(tmp_path: Path) -> None:
    """Finalization exports HF weights without rewriting the same resumable state."""
    manager, _ = _manager(tmp_path)
    manager._last_saved_step = 1
    manager._save = MagicMock()
    manager._export_hf = MagicMock()

    manager.finalize(SimpleNamespace(global_step=1))

    manager._save.assert_not_called()
    manager._export_hf.assert_called_once_with(manager.directory(1))


def test_disabled_final_save_does_not_export(tmp_path: Path) -> None:
    """save_final=false disables both final resumable state and HF export."""
    manager, _ = _manager(tmp_path)
    manager.config["save_final"] = False
    manager._save = MagicMock()
    manager._export_hf = MagicMock()

    manager.finalize(SimpleNamespace(global_step=1))

    manager._save.assert_not_called()
    manager._export_hf.assert_not_called()


@pytest.mark.parametrize("writing_rank", [True, False])
def test_hf_export_gathers_on_every_rank_and_saves_tokenizer_on_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writing_rank: bool,
) -> None:
    """Non-writer ranks must participate in the distributed full-weight gather."""
    manager, trainer = _manager(tmp_path)
    trainer.tokenizer = MagicMock()
    trainer.model.config = MagicMock(architectures=["HSDPQwen3ForCausalLM"])
    exporter = MagicMock()
    exporter.return_value.save_pretrained.return_value = writing_rank
    monkeypatch.setattr(checkpoint_backend, "CheckpointManager", exporter)
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0 if writing_rank else 1)

    manager._export_hf(manager.directory(1))

    exporter.assert_called_once_with(trainer.model)
    exporter.return_value.save_pretrained.assert_called_once_with(
        manager.directory(1) / "hf", save_original_format=True,
    )
    if writing_rank:
        trainer.model.config.save_pretrained.assert_called_once_with(manager.directory(1) / "hf")
        trainer.tokenizer.save_pretrained.assert_called_once_with(str(manager.directory(1) / "hf"))
    else:
        trainer.model.config.save_pretrained.assert_not_called()
        trainer.tokenizer.save_pretrained.assert_not_called()
    assert trainer.model.config.architectures == ["Qwen3ForCausalLM"]


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
@pytest.mark.parametrize("hsdp_class", [True, False])
def test_final_hf_weights_reload_with_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tie_word_embeddings: bool, hsdp_class: bool,
) -> None:
    """Exported Qwen3 safetensors round-trip with native HF names and tied weights."""
    manager, trainer = _manager(tmp_path)
    trainer.model = Qwen3ForCausalLM(Qwen3Config.from_dict({
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "tie_word_embeddings": tie_word_embeddings,
    })).to(dtype=torch.bfloat16)
    if hsdp_class:
        trainer.model.__class__ = type("HSDPQwen3ForCausalLM", (type(trainer.model),), {})
    trainer.tokenizer = MagicMock()
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0)

    manager._export_hf(manager.directory(1))

    export_dir = manager.directory(1) / "hf"
    restored = Qwen3ForCausalLM.from_pretrained(export_dir, local_files_only=True, dtype="auto")
    assert (export_dir / "model.safetensors").is_file()
    assert json.loads((export_dir / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
    assert trainer.model.config.architectures == ["Qwen3ForCausalLM"]
    assert restored.dtype == torch.bfloat16
    for name, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    if tie_word_embeddings:
        assert restored.lm_head.weight is restored.model.embed_tokens.weight


@pytest.mark.parametrize("failure_stage", ["preparation", "model", "runtime"])
def test_failed_overwrite_removes_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    """An interrupted overwrite cannot leave a checkpoint advertised as complete."""
    manager, trainer = _manager(tmp_path)
    checkpoint_dir = manager.directory(1)
    checkpoint_dir.mkdir()
    manifest = checkpoint_dir / "checkpoint_complete.json"
    manifest.write_text('{"step": 1, "world_size": 1}', encoding="utf-8")
    manager._last_saved_step = 1
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.torch, "get_rng_state", lambda *_args: b"rng")
    if failure_stage == "preparation":
        trainer.model.state_dict = MagicMock(side_effect=OSError("prepare failed"))

    def save(unused_state: Any, **kwargs: Any) -> None:
        """Fail before either collective model or rank-local runtime completes."""
        del unused_state
        if kwargs["use_collectives"] == (failure_stage == "model"):
            raise OSError("save failed")

    monkeypatch.setattr(checkpoint_backend, "dcp_save", save)
    with pytest.raises(OSError):
        manager.complete_step(SimpleNamespace(global_step=1), loss=0.5, grad_norm=1.0)

    assert not manifest.exists()
    assert manager._last_saved_step is None


@pytest.mark.parametrize("missing", ["manifest", "runtime", "progress", "world_size"])
def test_resume_preflight_rejects_incomplete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    """Missing progress or rank state and incompatible world sizes fail before load."""
    manager, _ = _manager(tmp_path)
    checkpoint_dir = manager.directory(1)
    checkpoint_dir.mkdir()
    if missing != "manifest":
        (checkpoint_dir / "checkpoint_complete.json").write_text(
            json.dumps({"step": 1, "world_size": 2 if missing == "world_size" else 1}),
            encoding="utf-8",
        )
    if missing != "runtime":
        (checkpoint_dir / "rank_0").mkdir()
    if missing != "progress":
        (checkpoint_dir / "extra_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.dist, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError):
        manager.validate_resume()


def test_dcp_roundtrip_restores_weights_and_adam_moments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real DCP files restore updated model tensors, Adam moments, and RL progress."""
    manager, trainer = _manager(tmp_path)
    trainer.model = torch.nn.Linear(3, 2)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01, foreach=False)
    trainer.model(torch.ones(2, 3)).sum().backward()
    trainer.optimizer.step()
    expected_weights = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    expected_optimizer = trainer.optimizer.state_dict()
    state = SimpleNamespace(global_step=1, epoch=2, consumed_samples=3, consumed_tokens=4)
    # Both DCP and the RL manager share the native Torch collectives.
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    restored_rng = []
    monkeypatch.setattr(checkpoint_backend.torch, "set_rng_state", restored_rng.append)
    monkeypatch.setattr(trainer.device_handle, "set_rng_state", lambda rng, _device: restored_rng.append(rng))

    manager.complete_step(state, loss=0.5, grad_norm=1.0)
    trainer.model = torch.nn.Linear(3, 2)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.1, foreach=False)
    trainer.train_dataloader.value = "changed"
    restored_state = SimpleNamespace(global_step=0, epoch=0, consumed_samples=0, consumed_tokens=0)
    manager.validate_resume()
    manager.begin(restored_state)

    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    actual_optimizer = trainer.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for param_id, values in expected_optimizer["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(actual_optimizer["state"][param_id][name], value, rtol=0, atol=0)
    assert trainer.train_dataloader.value == "dataloader-live"
    assert vars(restored_state) == vars(state)
    assert len(restored_rng) == 2
