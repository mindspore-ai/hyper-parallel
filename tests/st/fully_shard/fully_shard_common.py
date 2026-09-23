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
"""fully_shard worker helpers (not a pytest module — do not name test_*.py)."""
import time

import torch
from torch import optim

from hyper_parallel import hsdp_sync_stream


def train(model, data, comm_async=True, train_steps=10):
    """Train a model for a few steps and optionally wait async fully_shard comm."""
    train_steps = max(train_steps, 1) + 1

    model = model.npu()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    cost_time = 0
    for i in range(train_steps):
        if i != 0:
            start_time = time.time()
        loss = model(data)
        loss.backward(torch.ones_like(loss))
        if comm_async:
            hsdp_sync_stream()
        if i != 0:
            end_time = time.time()
            cost_time += end_time - start_time
        optimizer.step()
        optimizer.zero_grad()
    return cost_time
