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
"""Trainer-facing fakes for ``tests/ut/trainer``.

``FakeTrainerComponents`` records the build/lifecycle/callback order that a
``BaseTrainer`` drives, so trainer tests can assert stage ordering and
callback fan-out without a real model, optimizer, or device.
"""

import pytest


class FakeTrainerComponents:
    """Recording stand-ins for the objects a trainer build produces.

    ``log`` is the single ordered event stream; every fake appends
    ``(component, event)`` pairs as the trainer touches it.
    """

    def __init__(self):
        self.log = []

    def _record(self, component, event):
        self.log.append((component, event))

    def build_model(self):
        self._record("model", "build")
        return _FakeModel(self.log)

    def build_optimizer(self, model):
        self._record("optimizer", "build")
        return _FakeOptimizer(self.log)

    def build_lr_scheduler(self, optimizer):
        self._record("lr_scheduler", "build")
        return _FakeLRScheduler(self.log)

    def build_dataloader(self):
        self._record("dataloader", "build")
        return _FakeDataLoader(self.log)

    def assert_order(self, expected):
        """Assert a subsequence of (component, event) pairs in order."""
        remaining = list(expected)
        for entry in self.log:
            if remaining and entry == remaining[0]:
                remaining.pop(0)
        assert not remaining, f"events {remaining} missing from log {self.log}"


class _FakeModel:
    def __init__(self, log):
        self.log = log
        self.log.append(("model", "init"))

    def forward(self, batch):
        self.log.append(("model", "forward"))
        return batch

    def train(self, mode=True):
        self.log.append(("model", f"train({mode})"))

    def eval(self):
        self.train(False)


class _FakeOptimizer:
    def __init__(self, log):
        self.log = log
        self.log.append(("optimizer", "init"))

    def zero_grad(self):
        self.log.append(("optimizer", "zero_grad"))

    def step(self):
        self.log.append(("optimizer", "step"))


class _FakeLRScheduler:
    def __init__(self, log):
        self.log = log
        self.log.append(("lr_scheduler", "init"))

    def step(self):
        self.log.append(("lr_scheduler", "step"))


class _FakeDataLoader:
    def __init__(self, log, batches=2):
        self.log = log
        self.log.append(("dataloader", "init"))
        self.batches = list(range(batches))

    def __iter__(self):
        self.log.append(("dataloader", "iter"))
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@pytest.fixture
def fake_trainer_components():
    """A fresh recording set of trainer components."""
    return FakeTrainerComponents()
