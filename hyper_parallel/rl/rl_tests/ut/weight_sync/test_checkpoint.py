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
"""CPU contracts for the composed RL distributed checkpoint lifecycle."""
# pylint: disable=missing-public-docstring

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import rl.roles.weight_sync.checkpoint as checkpoint_backend
from rl.roles.weight_sync.checkpoint import (
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
        optimizer=_Stateful("optimizer-live"),
        lr_scheduler=_Stateful("scheduler-live"),
        train_dataloader=_Stateful("dataloader-live"),
        device="npu:0",
        device_handle=object(),
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
    monkeypatch.setattr(checkpoint_backend.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        checkpoint_backend.platform,
        "get_rng_state",
        lambda _device=None, device_handle=None: (
            "device-rng" if device_handle is not None else "cpu-rng"
        ),
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
    monkeypatch.setattr(checkpoint_backend.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_backend.platform, "get_rng_state", lambda *_args: "live-rng")
    monkeypatch.setattr(
        checkpoint_backend.platform,
        "set_rng_state",
        lambda state, *args: restored_rng.append((state, args)),
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
        ("device-restored", (trainer.device, trainer.device_handle)),
    ]
    assert vars(state) == {
        "global_step": 1,
        "epoch": 2,
        "consumed_samples": 3,
        "consumed_tokens": 4,
    }
