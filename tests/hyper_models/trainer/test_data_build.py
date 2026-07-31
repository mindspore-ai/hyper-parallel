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
"""Tests for the staged Trainer dummy-data construction path."""

from types import SimpleNamespace

from hyper_models.components.data import build_micro_batch_collator
from hyper_models.components.datasets import build_dummy_dataset
from hyper_models.components.distributed.infrastructure import MeshContext
from hyper_models.data import build_train_dataloader
from hyper_models.trainer.base import BaseTrainer
from hyper_models.trainer.config import Target, TrainerConfig, TrainingConfig


def _unused_target() -> Target:
    """Return a target required by ``TrainerConfig`` but unused in this test."""
    return Target(lambda: None, target_path="tests.unused")


def _model_target(
    *,
    distributed_setup: object,
    peft_config: object,
) -> SimpleNamespace:
    """Return a complete model-build result for delegation testing."""
    model = object()
    return SimpleNamespace(
        model=model,
        optimizer_init="optimizer-init",
        model_config="model-config",
        model_parts=[model],
        hsdp_model_parts=[],
        distributed_setup=distributed_setup,
        peft_config=peft_config,
    )


def test_trainer_model_stage_delegates_to_config_target() -> None:
    """Inject runtime arguments and accept all state from the model target."""
    distributed_setup = object()
    peft_config = object()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=Target(_model_target, target_path="tests.model_target"),
        optimizer=_unused_target(),
        peft=peft_config,
    )
    trainer.distributed_setup = distributed_setup

    trainer._build_model()

    assert trainer.peft_config is peft_config
    assert trainer.model_parts == [trainer.model]
    assert trainer.model_config == "model-config"
    assert trainer.optimizer_init == "optimizer-init"
    assert trainer.hsdp_model_parts == []


def test_trainer_data_stages_return_micro_batches() -> None:
    """Build assets, dataset, collator, and dataloader in Trainer order."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        training=TrainingConfig(
            max_steps=3,
            global_batch_size=8,
            micro_batch_size=2,
            seed=17,
        ),
        dataset=Target(
            build_dummy_dataset,
            target_path="hyper_models.components.datasets.build_dummy_dataset",
            num_samples=16,
            seq_len=6,
        ),
        collate_fn=Target(
            build_micro_batch_collator,
            target_path=(
                "hyper_models.components.data.build_micro_batch_collator"
            ),
        ),
        dataloader=Target(
            build_train_dataloader,
            target_path="hyper_models.data.build_train_dataloader",
            shuffle=True,
            drop_last=True,
            use_background_prefetcher=False,
        ),
    )
    trainer.mesh = MeshContext(dp_size=2, dp_rank=1)
    trainer.model_config = SimpleNamespace(vocab_size=19)

    trainer._build_model_assets()
    trainer._build_dataset()
    trainer._build_collate_fn()
    trainer._build_dataloader()

    micro_batches = next(iter(trainer.train_dataloader))

    assert trainer.tokenizer is None
    assert trainer.chat_template is None
    assert trainer.model_assets == [trainer.model_config]
    assert trainer.train_steps == 3
    assert len(micro_batches) == 2
    assert all(micro_batch["input_ids"].shape == (2, 6) for micro_batch in micro_batches)
    assert all(micro_batch["labels"].shape == (2, 6) for micro_batch in micro_batches)
