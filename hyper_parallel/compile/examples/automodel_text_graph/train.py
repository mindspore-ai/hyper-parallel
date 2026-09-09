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
"""Minimal graph-mode text trainer prototype reusing AutoModel + TextTrainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from transformers import LlamaConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from hyper_parallel.compile import GraphTrainer  # pylint: disable=C0413
from hyper_parallel.components.checkpoint.config import (  # pylint: disable=C0413
    CheckpointingConfig,
)
from hyper_parallel.components.optim.builders import AdamW  # pylint: disable=C0413
from hyper_parallel.components.optim.lr_scheduler import (  # pylint: disable=C0413
    MultiLRScheduler,
)
from hyper_parallel.models._transformers import (  # pylint: disable=C0413
    HyperAutoModelForCausalLM,
)
from hyper_parallel.trainer.config import (  # pylint: disable=C0413
    AcceleratorConfig,
    DataLoaderConfig,
    DatasetConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)


class TinyTextDataset(Dataset):
    """Tiny fixed-shape dataset for graph-mode smoke tests."""

    def __init__(self, *, vocab_size: int, seq_len: int, size: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=gen,
        )
        labels = input_ids.roll(shifts=-1)
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


def build_tiny_automodel(
    *,
    distributed_setup=None,
    compile_config=None,
    activation_checkpoint=None,
    swap_inputs: bool = False,
    activation_swap: str = "none",
    model_init_dtype=None,
) -> torch.nn.Module:
    """Build a tiny AutoModel through the shared HyperAutoModel facade."""
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        swap_inputs=swap_inputs,
        activation_swap=activation_swap,
        model_init_dtype=model_init_dtype,
    )


def build_dummy_dataset(
    *,
    transform=None,
    tokenizer=None,
    mesh_context=None,
    training_config=None,
) -> Dataset:
    """Build the tiny text dataset expected by the Trainer."""
    del transform, tokenizer, mesh_context, training_config
    return TinyTextDataset(vocab_size=128, seq_len=16, size=4)


def build_simple_collate_fn(*, mesh_context=None):
    """Stack per-sample tensors into one local micro-batch."""
    del mesh_context

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in keys
        }

    return _collate


class SimpleGetBatch:
    """Minimal DataLoader-to-model batch adapter for the graph prototype."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, data_iterator: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = next(data_iterator)
        batch = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }
        input_ids = batch["input_ids"]
        batch["attention_mask"] = torch.ones_like(input_ids)
        batch["position_ids"] = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0).expand_as(input_ids)
        loss_inputs = {"labels": batch["labels"]}
        return batch, loss_inputs


def build_simple_get_batch(
    *,
    mesh_context=None,
    device=None,
    tokenizer=None,
    data_config=None,
    pp_shared_data: bool = False,
) -> SimpleGetBatch:
    """Build the graph prototype batch adapter."""
    del mesh_context, tokenizer, data_config
    if pp_shared_data:
        raise ValueError("The minimal graph-mode prototype does not support pp_shared_data")
    return SimpleGetBatch(device=device)


def build_simple_dataloader(
    *,
    dataset: Dataset,
    collate_fn,
    batch_sampler=None,
    batch_size: int = 1,
    dp_world_size: int = 1,
    max_seq_len=None,
    seed: int = 0,
    drop_last: bool = False,
    use_background_prefetcher: bool = False,
) -> TorchDataLoader:
    """Build a plain PyTorch DataLoader for the graph prototype."""
    del dp_world_size, max_seq_len, seed, use_background_prefetcher
    if batch_sampler is not None:
        return TorchDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=drop_last,
    )


def build_config() -> TrainerConfig:
    """Create the minimal TrainerConfig used by the graph prototype."""
    return TrainerConfig(
        model=Target(
            _target_=build_tiny_automodel,
            target_path="__main__.build_tiny_automodel",
        ),
        optimizer=OptimizerConfig(
            target=Target(
                _target_=AdamW,
                target_path="hyper_parallel.components.optim.builders.AdamW",
                adamw_config={"lr": 1.0e-4, "adamw_weight_decay": 0.0},
            ),
        ),
        lr_scheduler=Target(
            _target_=MultiLRScheduler,
            target_path="hyper_parallel.components.optim.lr_scheduler.MultiLRScheduler",
            lr_decay_style="constant",
            lr_config={"lr": 1.0e-4, "lr_warmup_steps": 0},
        ),
        training=TrainingConfig(
            train_iters=2,
            global_batch_size=1,
            micro_batch_size=1,
            logging_steps=1,
        ),
        accelerator=AcceleratorConfig(),
        dataset=DatasetConfig(
            target=Target(
                _target_=build_dummy_dataset,
                target_path="__main__.build_dummy_dataset",
            ),
        ),
        dataloader=DataLoaderConfig(
            target=Target(
                _target_=build_simple_dataloader,
                target_path="__main__.build_simple_dataloader",
                drop_last=False,
                use_background_prefetcher=False,
            ),
            collate_fn=Target(
                _target_=build_simple_collate_fn,
                target_path="__main__.build_simple_collate_fn",
            ),
            get_batch=Target(
                _target_=build_simple_get_batch,
                target_path="__main__.build_simple_get_batch",
            ),
        ),
        checkpoint=CheckpointingConfig(save_ckpt=False),
    )


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29621")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    trainer = GraphTrainer.from_text_config(build_config())
    print("=" * 70)
    print("GraphTextTrainer minimal AutoModel prototype")
    print("=" * 70)
    trainer.train()
    print("=" * 70)
    print("Prototype run finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
