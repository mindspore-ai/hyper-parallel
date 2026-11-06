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
"""Exercise public mode selection without constructing device resources."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl.trainer import build
from rl.trainer import sync, fully_async, colocate_async


def test_sync_entry_constructs_resources_once(monkeypatch):
    """The synchronous public entry must terminate and pass assembled resources."""
    config = {"train": {}}
    resources = SimpleNamespace()
    factory = Mock(return_value=resources)
    trainer = Mock()
    monkeypatch.setattr(build, "build_resources", factory)
    monkeypatch.setattr(sync, "SyncTrainer", trainer)
    assert build.build_trainer(config) is trainer.return_value
    factory.assert_called_once_with(config)
    trainer.assert_called_once_with(resources)


@pytest.mark.parametrize("key,module,name", [
    ("async_training", fully_async, "FullyAsyncTrainer"),
    ("colocate_async", colocate_async, "ColocateAsyncTrainer"),
])
def test_async_entry_delegates_construction(monkeypatch, key, module, name):
    """Each asynchronous mode owns its construction and receives the recipe."""
    config = {key: {}}
    trainer = Mock()
    monkeypatch.setattr(module, name, trainer)
    assert build.build_trainer(config) is trainer.return_value
    trainer.assert_called_once_with(config)


def test_entry_rejects_ambiguous_mode():
    """Ambiguous mode selection fails before any distributed initialization."""
    with pytest.raises(ValueError, match="exactly one"):
        build.build_trainer({"async_training": {}, "colocate_async": {}})
