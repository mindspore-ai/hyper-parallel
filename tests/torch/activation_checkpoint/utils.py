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
"""Utility functions for activation checkpointing tests."""
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


@contextmanager
def seed_memory_time_context(seed=42):
    """Context manager to set seed, track peak memory and execution time."""
    set_seed(seed)
    torch.npu.reset_peak_memory_stats()
    torch.npu.empty_cache()
    start_time = time.time()

    stats = {}

    try:
        yield stats
    finally:
        exec_time = time.time() - start_time
        peak_mem_gb = torch.npu.max_memory_allocated() / (1024 ** 3)
        torch.npu.empty_cache()
        stats["peak_mem"] = peak_mem_gb
        stats["exec_time"] = exec_time


def prepare_data(batch_size=8, seq_len=512, num_samples=64):
    dataset = TensorDataset(
        torch.randint(0, 10000, (num_samples, seq_len)),
        torch.randint(0, 10000, (num_samples, seq_len))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def train_one_mode(model, dataloader, train_steps=5):
    """Run training for one mode and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for step, (x, y) in enumerate(dataloader):
        if step >= train_steps:
            break
        x, y = x.npu(), y.npu()
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
