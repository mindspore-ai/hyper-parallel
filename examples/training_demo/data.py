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
"""Example-only deterministic data for the tiny Qwen3-MoE demo."""

from collections.abc import Callable, Mapping
from typing import Any, Optional

# This example intentionally uses the PyTorch Dataset contract because the
# Trainer demo selects the PyTorch backend. The repository lint checker treats
# ``examples/`` as backend-neutral, so keep the explicit backend dependency
# local to this example module rather than changing framework code.
# pylint: disable=forbidden-backend-import
import torch
from torch.utils.data import Dataset


class TinyCausalDataset(Dataset):
    """Generate deterministic inputs with next-token labels before CP sharding."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 42,
        transform: Optional[Callable[[Any], Any]] = None,
        data_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Build fixed-length samples for this offline training example."""
        self.transform = transform
        self.data_config = dict(data_config or {})
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(
            0,
            vocab_size,
            (num_samples, seq_len + 1),
            generator=generator,
        )
        self.input_ids = tokens[:, :-1]
        self.labels = tokens[:, 1:]

    def __len__(self) -> int:
        """Return the number of generated samples."""
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one model-ready causal language-model sample."""
        return {
            "tokens": self.input_ids[index],
            "labels": self.labels[index],
        }


__all__ = ["TinyCausalDataset"]
