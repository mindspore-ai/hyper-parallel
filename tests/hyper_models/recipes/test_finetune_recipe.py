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
"""Integration tests for FinetuneRecipe — distributed deps mocked (03 §5-§6)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import hyper_models.recipes.llm.train_ft as train_ft
from hyper_models.components.checkpoint.config import CheckpointingConfig
from hyper_models.components.checkpoint.checkpointing import Checkpointer
from hyper_models.components.distributed.infrastructure import (
    DistributedSetup,
    MeshContext,
)
from hyper_models.components.loss import LossConfig, MaskedCrossEntropy
from hyper_models.components.training.callback import (
    CallbackManager,
    TrainingCallback,
)
from hyper_models.components.training.step_scheduler import (
    StepScheduler,
    StepSchedulerConfig,
)
from hyper_models.recipes.llm.train_ft import FinetuneRecipe


@pytest.fixture
def env(monkeypatch):
    """Monkeypatch all distributed/external deps of FinetuneRecipe.setup().

    Returns (calls, cfg, mocks) — `calls` records mocked build order.
    """
    calls = []

    def recorder(name, ret=None):
        def fn(*args, **kwargs):
            calls.append(name)
            return ret
        return fn

    # ── module-level patches ──
    monkeypatch.setattr(train_ft, "initialize_distributed", recorder("initialize_distributed", MagicMock()))
    monkeypatch.setattr(train_ft, "setup_logging", recorder("setup_logging"))
    monkeypatch.setattr(train_ft, "apply_cache_compatibility_patches", recorder("apply_cache_compatibility_patches"))
    monkeypatch.setattr(train_ft, "destroy_process_group", recorder("destroy_process_group"))

    rng_instance = MagicMock()
    monkeypatch.setattr(train_ft, "StatefulRNG", recorder("StatefulRNG", rng_instance))

    distributed_setup = DistributedSetup(mesh_context=MeshContext())
    monkeypatch.setattr(
        train_ft, "create_distributed_setup_from_config",
        recorder("create_distributed_setup_from_config", distributed_setup),
    )

    callback_manager = MagicMock()
    monkeypatch.setattr(
        train_ft, "build_callback_manager",
        recorder("build_callback_manager", callback_manager),
    )

    model = nn.Linear(4, 4)
    monkeypatch.setattr(train_ft, "build_model", recorder("build_model", (model, None)))

    batches = [
        {"input_ids": torch.randint(0, 16, (1, 8)), "labels": torch.randint(0, 16, (1, 8))}
        for _ in range(4)
    ]
    monkeypatch.setattr(train_ft, "build_dataloader", recorder("build_dataloader", (batches, None)))
    monkeypatch.setattr(train_ft, "build_validation_dataloader", recorder("build_validation_dataloader", {}))

    mfu = SimpleNamespace(flops_per_token=1e8, peak_tflops=100.0)
    auto_mfu = MagicMock()
    auto_mfu.from_config.side_effect = lambda *a, **k: (calls.append("AutoMFU.from_config"), mfu)[1]
    monkeypatch.setattr(train_ft, "AutoMFU", auto_mfu)

    # ── cfg ──
    loss_cfg = LossConfig()
    orig_loss_build = loss_cfg.build
    loss_cfg.build = lambda: (calls.append("loss.build"), orig_loss_build())[1]

    ss_cfg = StepSchedulerConfig(
        max_steps=2, local_batch_size=1, global_batch_size=1,
        ckpt_every_steps=500, num_train_epochs=1,
    )
    orig_ss_build = ss_cfg.build
    def ss_build(*args, **kwargs):
        calls.append("step_scheduler.build")
        return orig_ss_build(*args, **kwargs)
    ss_cfg.build = ss_build

    opt = MagicMock()
    opt.param_groups = [{"lr": 1e-3}]
    optimizer_cfg = MagicMock()
    optimizer_cfg.build.side_effect = lambda *a, **k: (calls.append("optimizer.build"), [opt])[1]
    optimizer_cfg.max_grad_norm = 1.0

    lr_scheduler_cfg = MagicMock()
    lr_scheduler_cfg.build.side_effect = lambda *a, **k: (calls.append("lr_scheduler.build"), [MagicMock()])[1]

    cfg = SimpleNamespace(
        model=MagicMock(),
        training=SimpleNamespace(seed=42),
        accelerator=SimpleNamespace(dp_shard_size=1, tp_size=1),
        step_scheduler=ss_cfg,
        loss=loss_cfg,
        checkpoint=CheckpointingConfig(checkpoint_dir="/tmp/ut_ckpt", restore_from=None),
        optimizer=optimizer_cfg,
        lr_scheduler=lr_scheduler_cfg,
        dataset=None, dataloader=None, packed_sequence=None,
        magi=None, peft=None, wandb=None,
    )

    mocks = SimpleNamespace(
        rng=rng_instance,
        callback_manager=callback_manager,
        model=model,
        batches=batches,
        optimizer=opt,
        mfu=mfu,
    )
    return calls, cfg, mocks


# ── setup() 组件构建 ──

def test_setup_component_order(env):
    calls, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert calls == [
        "initialize_distributed",          # ①
        "setup_logging",                   # ②
        "apply_cache_compatibility_patches",
        "StatefulRNG",                     # ③
        "create_distributed_setup_from_config",  # ④
        "build_callback_manager",          # ⑥
        "loss.build",                      # ⑦
        "build_model",                     # ⑪
        "optimizer.build",                 # ⑫
        "build_dataloader",                # ⑬
        "build_validation_dataloader",     # ⑭
        "step_scheduler.build",            # ⑮
        "lr_scheduler.build",              # ⑯
        "AutoMFU.from_config",             # ⑲
    ]


def test_setup_creates_rng(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.rng is mocks.rng


def test_setup_creates_distributed_setup(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert isinstance(recipe.distributed_setup, DistributedSetup)
    assert recipe.mesh is recipe.distributed_setup.mesh_context
    assert recipe.dp_cp_mesh is None  # stub MeshContext 无 device_mesh → 兜底


def test_setup_creates_callback_manager(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.callback_manager is mocks.callback_manager


def test_setup_builds_loss(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert isinstance(recipe.loss, MaskedCrossEntropy)


def test_setup_builds_checkpointer(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert isinstance(recipe.checkpointer, Checkpointer)
    assert recipe.checkpoint_config is cfg.checkpoint


def test_setup_builds_model(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.model is mocks.model
    assert recipe.model_parts == [mocks.model]  # 无 .parts → [model]


def test_setup_builds_optimizer(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.optimizer == [mocks.optimizer]
    cfg.optimizer.build.assert_called_once()


def test_setup_builds_dataloader(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.dataloader is mocks.batches
    assert recipe.tokenizer is None
    assert recipe.val_dataloaders == {}


def test_setup_builds_step_scheduler(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert isinstance(recipe.step_scheduler, StepScheduler)
    assert recipe.step_scheduler.max_steps == 2
    assert recipe.step_scheduler.grad_acc_steps == 1


def test_setup_builds_lr_scheduler(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert isinstance(recipe.lr_scheduler, list)
    cfg.lr_scheduler.build.assert_called_once()


def test_setup_lr_scheduler_optional(env):
    _, cfg, _ = env
    cfg.lr_scheduler = None
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.lr_scheduler is None


def test_setup_registers_state(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe._state_tracked == [
        ("model", "model"),
        ("optimizer", "optimizer"),
        ("lr_scheduler", "lr_scheduler"),
        ("rng", "rng"),
        ("dataloader", "dataloader"),
        ("step_scheduler", "train_state"),
    ]


def test_setup_loads_checkpoint(env, monkeypatch):
    _, cfg, _ = env
    load_calls = []
    monkeypatch.setattr(
        FinetuneRecipe, "load_checkpoint",
        lambda self, restore_from: load_calls.append(restore_from),
    )
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert load_calls == [None]  # cfg.checkpoint.restore_from


def test_setup_creates_mfu(env):
    _, cfg, mocks = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    assert recipe.mfu_calc is mocks.mfu


# ── 训练主循环 ──

class _Recorder(TrainingCallback):
    def __init__(self, events):
        self.events = events

    def on_train_begin(self):
        self.events.append("begin")

    def on_step_end(self, state):
        self.events.append(("step", state.step, state.epoch))

    def on_train_end(self):
        self.events.append("end")


def _start_train_loop(recipe):
    events = []
    recipe.callback_manager = CallbackManager()
    recipe.callback_manager.register(_Recorder(events))
    recipe._run_train_optim_step = MagicMock(return_value={
        "loss": 0.5, "grad_norm": 1.0, "lr": 1e-3,
        "step_time": 0.1, "tps": 100.0, "mfu": 0.3, "num_tokens": 8,
    })
    recipe.save_checkpoint = MagicMock()
    recipe.checkpointer = MagicMock()
    recipe.run_train_validation_loop()
    return events


def test_train_loop_epochs(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    events = _start_train_loop(recipe)
    # max_steps=2, grad_acc=1 → 2 步；epoch=0
    assert events == ["begin", ("step", 1, 0), ("step", 2, 0), "end"]
    assert recipe._run_train_optim_step.call_count == 2


def test_train_loop_callback_driver(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    events = _start_train_loop(recipe)
    step_events = [e for e in events if isinstance(e, tuple)]
    assert len(step_events) == 2  # on_step_end 被驱动 2 次


def test_train_loop_final_save(env):
    _, cfg, _ = env
    recipe = FinetuneRecipe()
    recipe.setup(cfg)
    _start_train_loop(recipe)
    recipe.save_checkpoint.assert_called_once()
    kwargs = recipe.save_checkpoint.call_args.kwargs
    assert kwargs["is_final_checkpoint"] is True
    # final save 先于 checkpointer.close（04 顺序约束）
    recipe.checkpointer.close.assert_called_once()
