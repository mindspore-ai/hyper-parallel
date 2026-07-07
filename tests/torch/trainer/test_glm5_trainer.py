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
"""GLM5 Trainer integration tests."""
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from hyper_parallel.models.glm5 import GLM5Config, GLM5ForCausalLM, prepare_glm5_batch
from hyper_parallel.models.glm5.attention import GLM5AttentionCore
from hyper_parallel.models.glm5.parallelize import parallelize_glm5
from hyper_parallel.models.spec import get_spec
from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, TrainerState
from hyper_parallel.trainer.callbacks import base as callback_base
from hyper_parallel.trainer.callbacks.base import CheckpointCallback
from hyper_parallel.trainer.utils.discovery import discover_model_spec


class _RecordingDataloader:
    """Minimal stateful dataloader stand-in."""

    def __init__(self, position: int = 0) -> None:
        """Initialize the in-memory dataloader position."""
        self.position = position

    def state_dict(self) -> dict:
        """Return a resumable dataloader state."""
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        """Restore the dataloader position."""
        self.position = state["position"]


def _tiny_config_kwargs() -> dict:
    """Return the small GLM5 config used by Trainer tests."""
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_dense_layers": 2,
        "max_position_embeddings": 64,
    }


def _build_tiny_model() -> GLM5ForCausalLM:
    """Build a tiny dense GLM5 model for trainer-path checks."""
    return GLM5ForCausalLM(GLM5Config(**_tiny_config_kwargs()))


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the shifted CausalLM loss convention used by Trainer baselines."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def test_glm5_discovery_and_build():
    """
    Feature: GLM5 Trainer model discovery
    Description: Discover GLM5 ModelSpec and build a tiny model from config overrides.
    Expectation: The registered builder creates a GLM5 model with requested values.
    """
    discover_model_spec("glm5")
    spec = get_spec("glm5")
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            vocab_size=None,
            hidden_size=None,
            intermediate_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            max_position_embeddings=None,
            config_overrides=_tiny_config_kwargs(),
        )
    )

    model = spec.build_model_fn(cfg)

    assert isinstance(model, GLM5ForCausalLM)
    assert model.config.vocab_size == 32
    assert model.config.num_hidden_layers == 2


def test_glm5_trainer_step_applies_prepare_batch_fn(monkeypatch):
    """
    Feature: GLM5 Trainer train_step integration
    Description: Run one BaseTrainer.train_step through the GLM5 prepare-batch hook.
    Expectation: CP-prepared labels/positions are used and one optimizer step runs.
    """
    torch.manual_seed(0)
    prepared = {}

    def _record_prepare_batch(batch: dict, model: GLM5ForCausalLM) -> dict:
        prepared_batch = prepare_glm5_batch(batch, model)
        prepared["position_ids"] = prepared_batch["position_ids"]
        prepared["labels"] = prepared_batch["labels"]
        return prepared_batch

    spec = SimpleNamespace(prepare_batch_fn=_record_prepare_batch, clip_grad_fn=None)
    monkeypatch.setattr(trainer_base, "get_spec", lambda _: spec)
    monkeypatch.setattr(trainer_base.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_base, "hsdp_sync_stream", lambda: None)
    args = SimpleNamespace(
        model=SimpleNamespace(name="glm5"),
        train=SimpleNamespace(
            max_steps=2,
            optimizer=SimpleNamespace(
                loss_aggregation="token_weighted",
                max_grad_norm=1.0,
            ),
        ),
    )
    trainer = BaseTrainer(args)
    model = _build_tiny_model()
    setattr(model, "_cp_size", 2)
    setattr(model, "_cp_rank", 0)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer.lr_scheduler = None
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    setattr(trainer, "_dp_group_info", SimpleNamespace(rank_size=1))
    trainer.on_substep_end = lambda: None
    trainer.on_pre_optimizer_step = lambda grad_norm=None: None
    with torch.no_grad():
        first_weight = model.model.embed_tokens.weight.clone()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    metrics = trainer.train_step(iter([[batch]]))

    assert trainer.state.global_step == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert prepared["position_ids"].tolist() == [0, 1, 2]
    assert prepared["labels"].tolist() == [[2, 3, 4]]
    assert getattr(trainer, "_last_global_tokens") == 3
    assert not torch.equal(model.model.embed_tokens.weight, first_weight)


def test_glm5_loss_matches_causal_lm_shifted_ce():
    """
    Feature: GLM5 Trainer loss semantics
    Description: Compare GLM5 loss with shifted CausalLM loss.
    Expectation: GLM5 loss matches the Transformers/LLaMAFactory label shift.
    """
    torch.manual_seed(0)
    model = _build_tiny_model()
    input_ids = torch.tensor([
        [0, 0, 3, 4, 5, 6],
        [0, 7, 8, 9, 10, 11],
    ])
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
    ])
    position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
    labels = input_ids.masked_fill(attention_mask == 0, -100)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
    )

    assert torch.allclose(
        output["loss"],
        _causal_lm_loss(output["logits"], labels),
        atol=1e-6,
        rtol=0,
    )


def test_glm5_cp_batch_shards_inputs_and_shifted_labels():
    """
    Feature: GLM5 CP batch preparation
    Description: Shard input tokens and labels across two CP ranks.
    Expectation: Shifted labels and position ids remain globally aligned.
    """
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
    }
    rank0_model = SimpleNamespace(**{"_cp_size": 2, "_cp_rank": 0})
    rank1_model = SimpleNamespace(**{"_cp_size": 2, "_cp_rank": 1})

    rank0 = prepare_glm5_batch(batch, rank0_model)
    rank1 = prepare_glm5_batch(batch, rank1_model)

    assert rank0["input_ids"].tolist() == [[10, 11, 12]]
    assert rank0["labels"].tolist() == [[11, 12, 13]]
    assert rank0["position_ids"].tolist() == [0, 1, 2]
    assert rank1["input_ids"].tolist() == [[13, 14, 0]]
    assert rank1["labels"].tolist() == [[14, -100, -100]]
    assert rank1["position_ids"].tolist() == [3, 4, 5]


def test_glm5_cp_batch_shards_4d_attention_mask():
    """
    Feature: GLM5 CP batch preparation
    Description: Shard a 4D additive attention mask across two CP ranks.
    Expectation: Local mask query/key dimensions match the local token shard.
    """
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "attention_mask": torch.zeros(1, 1, 4, 4),
    }
    rank1_model = SimpleNamespace(**{"_cp_size": 2, "_cp_rank": 1})

    rank1 = prepare_glm5_batch(batch, rank1_model)

    assert rank1["input_ids"].tolist() == [[12, 13]]
    assert rank1["attention_mask"].shape == (1, 1, 2, 2)


def test_glm5_attention_core_slices_cp_4d_attention_mask():
    """
    Feature: GLM5 CP attention core
    Description: Run a local core shard with a full 4D additive mask.
    Expectation: The mask is sliced to the core-local query/key dimensions.
    """
    core = GLM5AttentionCore(scale=1.0)
    setattr(core, "_cp_size", 2)
    setattr(core, "_cp_rank", 1)
    query = torch.randn(1, 2, 1, 4)
    key = torch.randn(1, 2, 1, 4)
    value = torch.randn(1, 2, 1, 4)
    attention_mask = torch.zeros(1, 1, 4, 4)

    output = core(query, key, value, attention_mask=attention_mask)

    assert output.shape == query.shape


def test_glm5_parallelize_rejects_tp_until_supported():
    """
    Feature: GLM5 TP guard
    Description: Request tensor parallel before GLM5 TP apply is implemented.
    Expectation: Parallelization raises NotImplementedError.
    """
    model = _build_tiny_model()
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            accelerator=SimpleNamespace(tp=2, cp=1, ep=1),
            gradient_checkpointing=SimpleNamespace(activation_checkpoint="off"),
        ),
    )

    with pytest.raises(NotImplementedError, match="GLM5 TP is not supported"):
        parallelize_glm5(model, mesh={}, cfg=cfg)


def test_glm5_checkpoint_callback_round_trip(tmp_path, monkeypatch):
    """
    Feature: GLM5 checkpoint save and resume
    Description: Save a tiny GLM5 training state and restore it.
    Expectation: Model, optimizer, scheduler, RNG, dataloader, and step restore.
    """

    def _save_state_dict(state_dict, checkpoint_id, use_collectives=False):
        del use_collectives
        torch.save(
            {
                key: value.detach().cpu().clone()
                for key, value in state_dict.items()
            },
            os.path.join(checkpoint_id, "model_state.pt"),
        )

    def _load_state_dict(state_dict, checkpoint_id, use_collectives=False):
        del use_collectives
        payload = torch.load(
            os.path.join(checkpoint_id, "model_state.pt"),
            map_location="cpu",
            weights_only=True,
        )
        for key, value in state_dict.items():
            value.copy_(payload[key])

    set_rng_calls = []
    monkeypatch.setattr(callback_base, "dcp_save", _save_state_dict)
    monkeypatch.setattr(callback_base, "dcp_load", _load_state_dict)
    monkeypatch.setattr(callback_base.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(
        callback_base.platform,
        "get_rng_state",
        lambda: torch.tensor([1, 2, 3]),
    )
    monkeypatch.setattr(callback_base.platform, "set_rng_state", set_rng_calls.append)

    torch.manual_seed(1)
    model = _build_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    dataloader = _RecordingDataloader(position=12)
    checkpoint_cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        save_steps=1,
        save_async=False,
        load_path=None,
        save_hf_weights=False,
    )
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint_cfg)),
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=dataloader,
        dispatch_save_event=lambda *_args, **_kwargs: None,
        dispatch_load_event=lambda *_args, **_kwargs: None,
    )
    callback = CheckpointCallback(trainer)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(input_ids=input_ids, labels=input_ids)["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    expected_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    state = TrainerState(max_steps=10)
    state.global_step = 3
    state.epoch = 1
    callback.on_step_end(state, loss=0.0, grad_norm=0.0)
    save_dir = tmp_path / "step_3"

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    dataloader.position = 0
    state.global_step = 0
    state.epoch = 0
    checkpoint_cfg.load_path = str(save_dir)
    callback.load_path = str(save_dir)

    callback.on_train_begin(state)

    for key, value in model.state_dict().items():
        assert torch.allclose(value, expected_state[key])
    assert state.global_step == 3
    assert state.epoch == 1
    assert dataloader.position == 12
    assert set_rng_calls and torch.equal(set_rng_calls[-1], torch.tensor([1, 2, 3]))
