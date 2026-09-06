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
"""In-memory checkpoint backend for ``tests/ut/auto_models/checkpoint``.

``FakeCheckpointBackend`` never touches the filesystem or DCP; it records the
save/load call order and round-trips state dicts in memory so checkpoint
orchestration can be asserted on CPU.
"""

import pytest


class FakeCheckpointBackend:
    """In-memory save/load with a recorded call sequence."""

    def __init__(self):
        self.calls = []
        self.storage = {}

    def save(self, state_dict, path):
        """Record the save and keep a shallow copy of the state dict."""
        self.calls.append(("save", path, sorted(state_dict)))
        self.storage[path] = dict(state_dict)

    def load(self, path):
        """Return the recorded state dict; KeyError when never saved."""
        self.calls.append(("load", path, None))
        return self.storage[path]

    def exists(self, path):
        return path in self.storage

    def assert_call_order(self, expected):
        """Assert the exact (op, path) sequence."""
        actual = [(op, path) for op, path, _ in self.calls]
        assert actual == list(expected), f"checkpoint calls {actual} != {list(expected)}"


@pytest.fixture
def fake_checkpoint_backend():
    """A fresh in-memory checkpoint backend."""
    return FakeCheckpointBackend()
