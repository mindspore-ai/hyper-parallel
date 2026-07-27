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
"""Data pipeline stubs — following design doc 02_data_pipeline.md.

Stub implementations for build_dataloader and build_validation_dataloader.
"""

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class _DummyDictDataset(Dataset):
    """Simple dict-yielding dataset for skeleton testing.

    Each sample is a dict with ``input_ids`` and ``labels`` tensors, which is
    the contract expected by the training loop.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            0, vocab_size, (num_samples, seq_len), generator=g,
        )
        self.labels = self.input_ids.clone()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def build_dataloader(
    cfg_dataset=None,
    cfg_dataloader=None,
    cfg_model=None,
    cfg_packed_sequence=None,
    seed: int = 42,
    local_batch_size: int = 1,
    global_batch_size: Optional[int] = None,
    max_steps: int = -1,
    val_check_interval: Optional[int] = None,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    pp_enabled: bool = False,
    cp_size: int = 1,
    model=None,
    **kwargs,
) -> tuple[DataLoader, Any]:
    """Build training dataloader and tokenizer.

    Stub — returns a simple DataLoader wrapping a dict-yielding dummy dataset.
    Full implementation follows 02_data_pipeline.md.

    Returns:
        (dataloader, tokenizer) — tokenizer is None in stub.
    """
    # Rank-aware seed so different DP ranks see different samples.
    dataset = _DummyDictDataset(
        num_samples=100,
        seq_len=32,
        vocab_size=1000,
        seed=seed + dp_rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger.warning("build_dataloader: using stub implementation with dummy data")

    return dataloader, None  # tokenizer=None


def build_validation_dataloader(
    cfg_dataset=None,
    cfg_dataloader=None,
    cfg_model=None,
    cfg_packed_sequence=None,
    seed: int = 42,
    local_batch_size: int = 1,
    global_batch_size: Optional[int] = None,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    pp_enabled: bool = False,
    cp_size: int = 1,
    model=None,
    **kwargs,
) -> dict[str, DataLoader]:
    """Build validation dataloader(s).

    Stub — returns a dict with a single dummy validation dataloader.

    Returns:
        dict[str, DataLoader] — validation dataloaders keyed by name.
    """
    dataset = _DummyDictDataset(
        num_samples=20,
        seq_len=32,
        vocab_size=1000,
        seed=seed + 1000 + dp_rank,
    )

    val_dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        drop_last=False,
    )

    logger.warning("build_validation_dataloader: using stub implementation with dummy data")

    return {"default": val_dataloader}
