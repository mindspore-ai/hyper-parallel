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
"""Self-contained data targets for the config-driven graph-mode example.

The standard ``TextTrainer`` data stack (indexed/online datasets, tokenizer,
``TextParallelBatch``) exists to feed real text corpora. This example only
demonstrates the graph-mode Trainer wiring, so it plugs three tiny targets
into the same ``TrainerConfig`` extension points instead:

- ``build_dataset`` builds a deterministic synthetic mapping Dataset;
- ``build_collate_fn`` returns the default collator;
- ``build_get_batch`` returns a callable producing ``(model_inputs,
  loss_inputs)`` for one batch, which is the one contract
  ``TextTrainer.forward_backward_step`` requires.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset, default_collate

__all__ = [
    "SyntheticTextDataset",
    "build_dataset",
    "build_collate_fn",
    "build_get_batch",
]


class SyntheticTextDataset(Dataset):
    """A single constant next-token-prediction batch without any on-disk data.

    Every sample is identical, so regardless of batch-sampler shuffling each
    optimizer step trains on the same batch. The model should then overfit it
    and the loss should fall monotonically — the convergence signal that tells
    a correct graph-mode run from a silently broken one.
    """

    def __init__(
        self, vocab_size: int, seq_length: int, num_samples: int, seed: int = 42
    ) -> None:
        """Record the synthetic sample layout."""
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            0, vocab_size, (seq_length,), generator=generator
        )
        self.labels = torch.roll(self.input_ids, shifts=-1, dims=0)

    def __len__(self) -> int:
        """Return the number of synthetic samples."""
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        """Return the same constant ``input_ids``/``labels`` next-token pair."""
        del index
        return {"input_ids": self.input_ids.clone(), "labels": self.labels.clone()}


def build_dataset(
    *,
    vocab_size: int,
    data_config: dict[str, Any],
    num_samples: int,
    seed: int = 42,
    mesh_context: Any = None,
    training_config: Any = None,
    transform: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> SyntheticTextDataset:
    """YAML ``dataset`` target: build the synthetic mapping Dataset.

    Args:
        vocab_size: Vocabulary size for the synthetic tokens.
        data_config: Required max_seq_len for every synthetic sample.
        num_samples: Number of synthetic samples.
        seed: Base seed for deterministic sampling.
        mesh_context: Trainer mesh (unused; injection point).
        training_config: Trainer training config (unused; injection point).
        transform: Sample transform (unused; no text tokenization).
        tokenizer: Tokenizer (unused; no text tokenization).
        **kwargs: Ignored extra build kwargs.

    Returns:
        The synthetic Dataset.
    """
    del mesh_context, training_config, transform, tokenizer, kwargs
    return SyntheticTextDataset(
        vocab_size=vocab_size,
        seq_length=data_config["max_seq_len"],
        num_samples=num_samples,
        seed=seed,
    )


def build_collate_fn(*, mesh_context: Any = None, **kwargs: Any) -> Callable:
    """YAML ``dataloader.collate_fn`` target: return the default collator."""
    del mesh_context, kwargs
    return default_collate


class SyntheticGetBatch:
    """Build one step's ``(model_inputs, loss_inputs)`` from a collated batch.

    Tensors are placed on the training ``device`` here (not only later in the
    Trainer's ``preforward``): ``TextTrainer`` computes ``current_token_counts``
    from ``loss_inputs`` *before* ``preforward``, and those counts become
    explicit graph inputs. Leaving them on CPU breaks graph execution on
    accelerators (``No backend type associated with device type cpu``).
    """

    def __init__(self, device: Any = None) -> None:
        """Record the device that every batch tensor is placed on."""
        self.device = device

    def __call__(self, data_iterator: Any) -> tuple[dict, dict]:
        """Fetch one batch and split model-facing from loss-only inputs.

        Args:
            data_iterator: Iterator yielding one collated batch mapping.

        Returns:
            ``(model_inputs, loss_inputs)`` for one forward-backward step.
        """
        batch = next(data_iterator)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
        model_inputs = {"input_ids": input_ids, "labels": labels}
        loss_inputs = {"labels": labels}
        return model_inputs, loss_inputs


def build_get_batch(
    *,
    mesh_context: Any = None,
    device: Any = None,
    tokenizer: Any = None,
    data_config: Any = None,
    pp_shared_data: bool = False,
    **kwargs: Any,
) -> SyntheticGetBatch:
    """YAML ``dataloader.get_batch`` target: build the get-batch runtime."""
    del mesh_context, tokenizer, data_config, pp_shared_data, kwargs
    return SyntheticGetBatch(device=device)
