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

from hyper_models.components.datasets import DatasetConfig
from hyper_models.components.distributed.infrastructure import MeshContext
from hyper_models.trainer.base import BaseTrainer
from hyper_models.trainer.config import ModelConfig, TrainerConfig, TrainingConfig


def test_trainer_data_stages_return_micro_batches() -> None:
    """Build assets, dataset, collator, and dataloader in Trainer order."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=ModelConfig(name="dummy"),
        training=TrainingConfig(
            max_steps=3,
            global_batch_size=8,
            micro_batch_size=2,
            seed=17,
        ),
        dataset=DatasetConfig(
            num_samples=16,
            seq_len=6,
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
