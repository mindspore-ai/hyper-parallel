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

import pytest

from hyper_models.components.data import (
    DataLoader,
    DummyDataset,
    IdentityDataTransform,
    MakeMicroBatchCollator,
)
from hyper_models.components.distributed.infrastructure import MeshContext
from hyper_models.trainer.base import BaseTrainer
from hyper_models.trainer.config import (
    ActivationCheckpointConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)


def _unused_target() -> Target:
    """Return a target required by ``TrainerConfig`` but unused in this test."""
    return Target(lambda: None, target_path="tests.unused")


def _value_target(*, value: object) -> object:
    """Return a configured value for dependency-order tests."""
    return value


def _model_target(
    *,
    distributed_setup: object,
    peft_config: object,
    activation_checkpoint: object,
) -> SimpleNamespace:
    """Return a model object for delegation testing."""
    return SimpleNamespace(
        config=SimpleNamespace(model_type="fake"),
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        activation_checkpoint=activation_checkpoint,
    )


def _optimizer_target(
    *,
    model: object,
    device_mesh: object,
    is_peft: bool,
) -> SimpleNamespace:
    """Return the runtime context received by the optimizer target."""
    return SimpleNamespace(
        model=model,
        device_mesh=device_mesh,
        is_peft=is_peft,
    )


def test_trainer_model_stage_delegates_to_config_target() -> None:
    """Build the model directly and derive Trainer-owned state."""
    distributed_setup = object()
    peft_config = object()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=Target(_model_target, target_path="tests.model_target"),
        optimizer=_unused_target(),
        peft=peft_config,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )
    trainer.distributed_setup = distributed_setup

    trainer._build_model()

    assert trainer.peft_config is peft_config
    assert trainer.model.distributed_setup is distributed_setup
    assert trainer.model.peft_config is peft_config
    assert trainer.model.activation_checkpoint == "full"
    assert trainer.model_parts == [trainer.model]
    assert trainer.model_config is trainer.model.config
    assert trainer.hsdp_model_parts == []


def test_trainer_model_stage_passes_off_activation_checkpointing_mode() -> None:
    """Pass the disabled activation-checkpoint mode through unchanged."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=Target(_model_target, target_path="tests.model_target"),
        optimizer=_unused_target(),
        activation_checkpoint=ActivationCheckpointConfig(mode="off"),
    )
    trainer.distributed_setup = object()

    trainer._build_model()

    assert trainer.model.activation_checkpoint == "off"


def test_trainer_optimizer_stage_passes_runtime_context() -> None:
    """Pass model, mesh, and PEFT state to the configured optimizer target."""
    model = object()
    device_mesh = object()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=Target(
            _optimizer_target,
            target_path="tests.optimizer_target",
        ),
        peft=object(),
    )
    trainer.model = model
    trainer.device_mesh = device_mesh
    trainer.peft_config = trainer.config.peft

    trainer._build_optimizer()

    assert trainer.optimizer.model is model
    assert trainer.optimizer.device_mesh is device_mesh
    assert trainer.optimizer.is_peft is True


def test_trainer_data_stages_return_micro_batches() -> None:
    """Build assets, dataset, collator, and dataloader in Trainer order."""
    tokenizer = object()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        tokenizer=Target(
            _value_target,
            target_path="tests.value_target",
            value=tokenizer,
        ),
        training=TrainingConfig(
            max_steps=3,
            global_batch_size=8,
            micro_batch_size=2,
            seed=17,
        ),
        data_transform=Target(
            IdentityDataTransform,
            target_path=(
                "hyper_models.components.data.identity_transform."
                "IdentityDataTransform"
            ),
        ),
        dataset=Target(
            DummyDataset,
            target_path="hyper_models.components.data.datasets.DummyDataset",
            num_samples=16,
            seq_len=6,
            vocab_size=19,
            seed=23,
        ),
        collate_fn=Target(
            MakeMicroBatchCollator,
            target_path="hyper_models.components.data.data_collator.MakeMicroBatchCollator",
        ),
        dataloader=Target(
            DataLoader,
            target_path="hyper_models.components.data.dataloader.DataLoader",
            shuffle=True,
            drop_last=True,
            use_background_prefetcher=False,
        ),
    )
    trainer.mesh = MeshContext(dp_size=2, dp_rank=1)
    trainer.model_config = SimpleNamespace(vocab_size=19)

    trainer._build_model_assets()
    trainer._build_data_transform()
    trainer._build_dataset()
    trainer._build_collate_fn()
    trainer._build_dataloader()
    trainer._compute_train_steps()

    micro_batches = next(iter(trainer.train_dataloader))

    assert trainer.chat_template is None
    assert trainer.tokenizer is tokenizer
    assert trainer.model_assets == [trainer.model_config, tokenizer]
    assert isinstance(trainer.data_transform, IdentityDataTransform)
    assert trainer.data_transform.tokenizer is tokenizer
    assert trainer.train_dataset.transform is trainer.data_transform
    assert trainer.train_steps == 3
    assert len(micro_batches) == 2
    assert all(micro_batch["input_ids"].shape == (2, 6) for micro_batch in micro_batches)
    assert all(micro_batch["labels"].shape == (2, 6) for micro_batch in micro_batches)


def test_trainer_computes_steps_from_dataloader_length() -> None:
    """Derive total steps from finite dataloader length and epoch count."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        training=TrainingConfig(max_steps=None, num_train_epochs=3),
    )
    trainer.train_dataloader = [object(), object()]

    trainer._compute_train_steps()

    assert trainer.train_steps == 6


def test_trainer_requires_max_steps_for_unsized_dataloader() -> None:
    """Require an explicit step count when dataloader length is unavailable."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        training=TrainingConfig(max_steps=None),
    )
    trainer.train_dataloader = object()

    with pytest.raises(ValueError, match="does not have a finite length"):
        trainer._compute_train_steps()
