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
"""Dataset components used by Trainer targets."""

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Generate deterministic token IDs and labels for Trainer smoke tests."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        """Initialize the generated dataset.

        Args:
            num_samples: Number of generated samples.
            seq_len: Number of tokens per sample.
            vocab_size: Exclusive upper bound for generated token IDs.
            seed: Random seed used by the local generator.
        """
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            0,
            vocab_size,
            (num_samples, seq_len),
            generator=generator,
        )
        self.labels = self.input_ids.clone()

    def __len__(self) -> int:
        """Return the generated sample count."""
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one token-and-label sample."""
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


__all__ = ["DummyDataset"]
