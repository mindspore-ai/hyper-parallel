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
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
)

import rl.checkpoint as checkpoint_backend
from rl.checkpoint import (
    RLCheckpointManager,
    _clone_shared_checkpoint_tensors,
)
from rl.roles.model_setup import resolve_model

import hyper_parallel.models._transformers.auto_model as auto_model_backend
import hyper_parallel.models._transformers.checkpoint_loader as checkpoint_loader_backend
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard


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


def test_rank_local_master_state_pickle_preserves_hidden_precision(tmp_path) -> None:
    """Distributed FP32 master shards serialize locally without BF16 rounding."""
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        master = torch.tensor([0.9984, 1.0016], dtype=torch.float32, requires_grad=True)
        distributed = DTensor.from_local(master, mesh, [Shard(0)])
        state = {"_mixed_precision_optimizer": {"fp32_from_fp16_params": {"weight": distributed}},
                 "nested": [(distributed,)], "step": 16}
        restored = pickle.loads(pickle.dumps(checkpoint_backend._rank_local_optimizer_state(state)))
        tensor = restored["_mixed_precision_optimizer"]["fp32_from_fp16_params"]["weight"]
        assert tensor.__class__ is torch.Tensor
        assert not tensor.requires_grad
        assert torch.equal(tensor, master)
        assert tensor.dtype is torch.float32
        assert not torch.equal(tensor, tensor.bfloat16().float())
        assert torch.equal(restored["nested"][0][0], master)
        assert restored["step"] == 16
    finally:
        if created_group:
            dist.destroy_process_group()


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


@pytest.mark.parametrize("checkpoint_policy", [False, None, "true"])
def test_resume_rejects_optimizer_precision_change(tmp_path, monkeypatch, checkpoint_policy) -> None:
    """A BF16-only checkpoint cannot silently recreate lost FP32 master precision."""
    manager, _ = _manager(tmp_path)
    manager.resolved_config = {"train": {"optimizer": {"fp32_main_params": True}}}
    directory = tmp_path / "step_1"
    directory.mkdir()
    manifest = {"world_size": 1, "critic": False, "distillation_fingerprint": None}
    if checkpoint_policy is not None:
        manifest["optimizer_fp32_main_params"] = checkpoint_policy
    (directory / "checkpoint_complete.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(checkpoint_backend.dist, "get_world_size", lambda: 1)
    with pytest.raises(RuntimeError, match="precision policy differs"):
        manager.validate_resume()


@pytest.mark.parametrize("topology", [None, {"dp_shard": 1, "tp": 2}])
def test_resume_rejects_missing_or_changed_master_partition(tmp_path, monkeypatch, topology) -> None:
    """Equal world sizes cannot make different rank-local optimizer partitions compatible."""
    manager, _ = _manager(tmp_path)
    manager.resolved_config = {"train": {"optimizer": {"fp32_main_params": True},
                                         "accelerator": {"dp_shard": 2, "tp": 1}}}
    directory = tmp_path / "step_1"
    directory.mkdir()
    manifest = {"world_size": 2, "critic": False, "distillation_fingerprint": None,
                "optimizer_fp32_main_params": True, "optimizer_parallel_topology": topology}
    (directory / "checkpoint_complete.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(checkpoint_backend.dist, "get_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="parallel topology differs"):
        manager.validate_resume()


@pytest.mark.parametrize("recipe", [None, "off", "qwen3_ascend_consistency_v1"])
def test_resume_rejects_changed_numerical_recipe(tmp_path, monkeypatch, recipe) -> None:
    """A fused/off optimizer partition cannot be restored into the native 0.23 recipe."""
    manager, _ = _manager(tmp_path)
    directory = tmp_path / "step_1"
    directory.mkdir()
    manifest = {"world_size": 1, "critic": False, "distillation_fingerprint": None}
    if recipe is not None:
        manifest["training_consistency_profile"] = recipe
    (directory / "checkpoint_complete.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(checkpoint_backend.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpoint_backend, "consistency_profile", lambda _config: "qwen3_ascend_consistency_v023")
    with pytest.raises(RuntimeError, match="numerical consistency recipe differs"):
        manager.validate_resume()


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


def _qwen35_export_actor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Build actual conditional checkpoint weights through the public CPU text loader."""
    text = Qwen3_5TextConfig.from_dict({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 2, "linear_num_value_heads": 6,
        "layer_types": ["linear_attention", "full_attention"], "pad_token_id": 0,
        "max_position_embeddings": 128, "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    })
    outer = Qwen3_5Config.from_dict({
        "text_config": text.to_dict(),
        "vision_config": {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                          "out_hidden_size": 32, "num_position_embeddings": 16, "deepstack_visual_indexes": []},
    })
    conditional = Qwen3_5ForConditionalGeneration(outer).cpu()
    source = tmp_path / "conditional-source"
    conditional.save_pretrained(source)
    options = {"registry_name": "tiny_qwen35", "name": "qwen3_5", "weights_path": str(source),
               "tokenizer_path": str(source), "trust_remote_code": False, "tokenizer_trust_remote_code": False}
    registration = resolve_model(options)
    # Explicit placement keeps the real public loader hardware-independent.
    monkeypatch.setattr(auto_model_backend, "_current_device", lambda: torch.device("cpu"))
    from_config = HyperAutoModelForCausalLM._from_config_parent_class

    def from_text_config(config: Any, **kwargs: Any) -> Any:
        """Keep export fixtures independent of HF composite-config normalization."""
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
        text_config_class = config.sub_configs.get("text_config")
        if text_config_class is not None and model_class.config_class is text_config_class:
            config = config.get_text_config()
        return from_config(config, **kwargs)

    with monkeypatch.context() as loader_patch:
        loader_patch.setattr(HyperAutoModelForCausalLM, "_from_config_parent_class", staticmethod(from_text_config))
        actor = HyperAutoModelForCausalLM.from_pretrained(
            str(source), torch_dtype=torch.float32, attn_implementation="eager", force_hf=True,
            local_files_only=True, trust_remote_code=False,
        )
    actor.config.rms_norm_eps = 2e-6
    actor.__class__ = type("HSDPQwen3_5ForCausalLM", (type(actor),), {})
    with torch.no_grad():
        actor.model.layers[0].linear_attn.in_proj_qkv.weight.add_(0.01)
    return actor, registration


@pytest.mark.parametrize("writing_rank", [True, False])
def test_qwen35_hf_export_retains_conditional_identity_and_actual_text_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writing_rank: bool,
) -> None:
    """Text policy export preserves native conditional names and public reload identity."""
    actor, registration = _qwen35_export_actor(tmp_path, monkeypatch)
    manager, trainer = _manager(tmp_path / "checkpoints")
    trainer.model = actor
    trainer.model_registration = registration
    trainer.tokenizer = MagicMock()
    monkeypatch.setattr(checkpoint_backend.dist, "get_rank", lambda: 0 if writing_rank else 1)
    monkeypatch.setattr(checkpoint_loader_backend, "is_initialized", lambda: True)

    manager._export_hf(manager.directory(1))

    export_dir = manager.directory(1) / "hf"
    assert actor.config.model_type == "qwen3_5_text", f"Expected text Actor, got {actor.config.model_type}"
    assert actor.config.architectures == ["Qwen3_5ForCausalLM"], (
        f"Expected actual text architecture, got {actor.config.architectures}"
    )
    assert actor.config.rms_norm_eps == 2e-6, f"Expected live Actor epsilon 2e-6, got {actor.config.rms_norm_eps}"
    if not writing_rank:
        assert not export_dir.exists(), f"Non-writer unexpectedly created export {export_dir}"
        trainer.tokenizer.save_pretrained.assert_not_called()
        return
    trainer.tokenizer.save_pretrained.assert_called_once_with(str(export_dir))
    saved_config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["model_type"] == "qwen3_5", f"Expected conditional model_type, got {saved_config}"
    assert saved_config["architectures"] == ["Qwen3_5ForConditionalGeneration"], (
        f"Expected native conditional architecture, got {saved_config['architectures']}"
    )
    assert saved_config["text_config"]["model_type"] == "qwen3_5_text", (
        f"Expected nested text model_type, got {saved_config['text_config']['model_type']}"
    )
    assert saved_config["text_config"]["rms_norm_eps"] == actor.config.rms_norm_eps, (
        f"Expected actual Actor epsilon {actor.config.rms_norm_eps}, got {saved_config['text_config']['rms_norm_eps']}"
    )
    source_config = json.loads((Path(registration.weights_path) / "config.json").read_text(encoding="utf-8"))
    assert saved_config["vision_config"] == source_config["vision_config"], (
        f"Expected original vision schema {source_config['vision_config']}, got {saved_config['vision_config']}"
    )
    with safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
        expected_keys = {
            "model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name
            for name in actor.state_dict()
        }
        assert set(checkpoint.keys()) == expected_keys, (
            f"Expected conditional text keys {expected_keys}, got {set(checkpoint.keys())}"
        )
    restored, report = AutoModelForCausalLM.from_pretrained(
        export_dir, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    assert not report["missing_keys"] and not report["unexpected_keys"] and not report["mismatched_keys"], (
        f"Expected complete native text reload, got {report}"
    )
    assert set(restored.state_dict()) == set(actor.state_dict()), (
        f"Expected Actor keys {set(actor.state_dict())}, got restored keys {set(restored.state_dict())}"
    )
    for name, tensor in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    exported_registration = resolve_model({
        "registry_name": "tiny_qwen35_export", "name": "qwen3_5", "weights_path": str(export_dir),
        "tokenizer_path": str(export_dir), "trust_remote_code": False, "tokenizer_trust_remote_code": False,
    })
    assert exported_registration.family == registration.family, (
        f"Expected original family {registration.family}, got {exported_registration.family}"
    )


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
