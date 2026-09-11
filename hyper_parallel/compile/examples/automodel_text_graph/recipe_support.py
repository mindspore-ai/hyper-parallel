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
"""Importable helpers for GraphTrainer YAML smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from transformers import LlamaConfig

from hyper_parallel.models import HyperAutoModelForCausalLM


DATA_DIR = Path(__file__).with_name("data")
ONLINE_TEXT_PATH = DATA_DIR / "tiny_text.jsonl"


class ToyTokenizer:
    """Minimal tokenizer compatible with the plaintext data transform."""

    def __init__(
        self,
        *,
        vocab_size: int = 128,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eod = eos_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Map UTF-8 bytes into a small bounded vocabulary."""
        del add_special_tokens
        base = self.vocab_size - 3
        return [3 + (byte % base) for byte in text.encode("utf-8")]


class TinyTextDataset(Dataset):
    """Tiny fixed-shape dataset for graph-mode smoke tests."""

    def __init__(self, *, vocab_size: int, seq_len: int, size: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=generator,
        )
        labels = input_ids.roll(shifts=-1)
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


class RebuildableDataLoader:
    """Recreate a fresh DataLoader iterator for every epoch."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        collate_fn,
        batch_sampler=None,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes to the batch sampler when supported."""
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def _build_loader(self) -> TorchDataLoader:
        if self.batch_sampler is not None:
            return TorchDataLoader(
                self.dataset,
                batch_sampler=self.batch_sampler,
                collate_fn=self.collate_fn,
            )
        return TorchDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=self.drop_last,
        )

    def __iter__(self):
        return iter(self._build_loader())

    def __len__(self) -> int:
        return len(self._build_loader())


class SimpleGetBatch:
    """Minimal DataLoader-to-model batch adapter for graph-mode text tests."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(
        self,
        data_iterator: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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


def build_toy_tokenizer(
    pretrained_model_name_or_path: str | None = None,
    **_: Any,
) -> ToyTokenizer:
    """Build a tiny tokenizer for the online plaintext smoke test."""
    del pretrained_model_name_or_path
    return ToyTokenizer()


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


def build_fixed_dummy_dataset(
    *,
    transform=None,
    tokenizer=None,
    mesh_context=None,
    training_config=None,
    vocab_size: int = 128,
    seq_len: int = 16,
    size: int = 40,
) -> Dataset:
    """Build a fixed-shape dataset that keeps graph tracing stable."""
    del transform, tokenizer, mesh_context, training_config
    return TinyTextDataset(vocab_size=vocab_size, seq_len=seq_len, size=size)


def build_simple_collate_fn(*, mesh_context=None):
    """Stack per-sample tensors into one local micro-batch."""
    del mesh_context

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in batch[0].keys()
        }

    return _collate


def build_simple_get_batch(
    *,
    mesh_context=None,
    device=None,
    tokenizer=None,
    data_config=None,
    pp_shared_data: bool = False,
) -> SimpleGetBatch:
    """Build the graph-mode batch adapter for fixed-shape smoke tests."""
    del mesh_context, tokenizer, data_config
    if pp_shared_data:
        raise ValueError("The graph smoke test does not support pp_shared_data")
    return SimpleGetBatch(device=device)


def build_rebuildable_dataloader(
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
) -> RebuildableDataLoader:
    """Build a DataLoader wrapper that refreshes iterators per epoch."""
    del dp_world_size, max_seq_len, seed, use_background_prefetcher
    return RebuildableDataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
    )
