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
"""Deterministic dummy LM dataset.

Each sample's tokens are derived from ``base_seed + idx`` so the same global
sample index always yields the same content, independent of DP topology or
sampler shuffling.
"""
import logging
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from hyper_parallel.data.registry import DATASET_REGISTRY


logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Deterministic random-token dataset.

    Args:
        num_samples: Total dataset length.
        seq_length: Tokens per sample.
        vocab: Token range upper bound (exclusive).
        base_seed: Seed offset; each sample uses ``base_seed + idx``.
    """

    def __init__(self, num_samples: int, seq_length: int, vocab: int, base_seed: int = 42) -> None:
        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.vocab = int(vocab)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(self.base_seed + idx)
        input_ids = torch.randint(0, self.vocab, (self.seq_length,), generator=g)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


@DATASET_REGISTRY.register("dummy")
def build_dummy(*, base: Any, args: Any, **_: Any) -> DummyDataset:
    """Build the dummy LM dataset from trainer config.

    Picks ``vocab`` from ``model.config.vocab_size`` when available, else
    defaults to 32000 (matches the pre-refactor behaviour in BaseTrainer).
    """
    seq_len = args.data.max_seq_len
    vocab_size = base.model.config.vocab_size
    base_seed = args.train.seed
    total_samples = base.state.max_steps * args.train.global_batch_size
    ds = DummyDataset(total_samples, seq_len, vocab_size, base_seed=base_seed)
    logger.info("Dummy dataset created: %d samples, seq_len=%d", total_samples, seq_len)
    return ds
