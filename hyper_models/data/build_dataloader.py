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

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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

    Stub — returns a simple DataLoader wrapping a list of dummy batches.
    Full implementation follows 02_data_pipeline.md.

    Returns:
        (dataloader, tokenizer) — tokenizer is None in stub.
    """
    import torch
    from torch.utils.data import TensorDataset

    # Create a dummy dataset of 100 samples
    dummy_input_ids = torch.randint(0, 1000, (100, 32))
    dummy_labels = dummy_input_ids.clone()
    dummy_dataset = TensorDataset(dummy_input_ids, dummy_labels)

    dataloader = DataLoader(
        dummy_dataset,
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
    import torch
    from torch.utils.data import TensorDataset

    dummy_input_ids = torch.randint(0, 1000, (20, 32))
    dummy_labels = dummy_input_ids.clone()
    dummy_dataset = TensorDataset(dummy_input_ids, dummy_labels)

    val_dataloader = DataLoader(
        dummy_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        drop_last=False,
    )

    logger.warning("build_validation_dataloader: using stub implementation with dummy data")

    return {"default": val_dataloader}