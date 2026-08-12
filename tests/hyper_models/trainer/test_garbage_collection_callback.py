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
"""Tests for periodic Trainer garbage collection."""

from types import SimpleNamespace

import hyper_models.trainer.callbacks.garbage_collection_callback as callback_module
from hyper_models.trainer.callbacks import GarbageCollectionCallback, TrainerState


def _build_trainer(gc_steps: int, empty_cache_steps: int) -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    training = SimpleNamespace(
        gc_steps=gc_steps,
        empty_cache_steps=empty_cache_steps,
    )
    return SimpleNamespace(mesh=None, config=SimpleNamespace(training=training))


def test_garbage_collection_callback_uses_independent_cadences(monkeypatch) -> None:
    """Collect Python garbage and clear device cache independently."""
    callback = GarbageCollectionCallback(_build_trainer(gc_steps=2, empty_cache_steps=3))
    calls = []
    monkeypatch.setattr(callback_module.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(callback_module, "get_device_type", lambda: "npu")
    monkeypatch.setattr(callback_module, "empty_cache", lambda: calls.append("empty_cache"))

    for step in range(1, 7):
        callback.on_step_end(TrainerState(global_step=step))

    assert calls == [
        "collect",
        "empty_cache",
        "collect",
        "collect",
        "empty_cache",
    ]


def test_garbage_collection_callback_skips_allocator_cleanup_on_cpu(monkeypatch) -> None:
    """Collect Python garbage but skip allocator cleanup on CPU."""
    callback = GarbageCollectionCallback(_build_trainer(gc_steps=1, empty_cache_steps=1))
    calls = []
    monkeypatch.setattr(callback_module.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(callback_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(callback_module, "empty_cache", lambda: calls.append("empty_cache"))

    callback.on_step_end(TrainerState(global_step=1))

    assert calls == ["collect"]
