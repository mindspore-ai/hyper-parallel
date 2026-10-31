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
"""Verify independent early backward without changing another graph's root state."""
# Backend hook tests intentionally traverse the existing core/backend import cycle.
# pylint: disable=cyclic-import
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

# Import only the selected backend so the test also runs in framework-only environments.
if get_platform().platform_type == PlatformType.MINDSPORE:
    from hyper_parallel.platform.mindspore.fully_shard import scheduler as backend
    Scheduler = backend.MindSporeHSDPSchedulerV2
else:
    from hyper_parallel.platform.torch.fully_shard import scheduler as backend
    Scheduler = backend.TorchHSDPSchedulerV2


def _make_scheduler(monkeypatch):
    """Build the hook control-flow seam without communication or device kernels."""
    scheduler = object.__new__(Scheduler)
    scheduler.cell = object()
    scheduler.hsdp_state = SimpleNamespace(module_name="isolated-test")
    scheduler.scheduler_state = FSDPSchedulerState.FORWARD
    scheduler.isolated_backward = True
    scheduler._is_root = True
    scheduler._backward_hook = Mock()
    scheduler._hsdp_backward_pre_hook = Mock()
    scheduler._register_backward_pre_hook = Mock()
    scheduler._restore_forward_prefetch_after_recompute = Mock()
    scheduler._hsdp_forward_hook = Mock(return_value="forward-output")
    scheduler.wait_for_pending_reductions = Mock()
    if get_platform().platform_type == PlatformType.MINDSPORE:
        monkeypatch.setattr(backend, "_pynative_executor", SimpleNamespace(queue_backward_final_callback=Mock()))
    else:
        engine = SimpleNamespace(queue_callback=Mock())
        monkeypatch.setattr(backend, "Variable", SimpleNamespace(_execution_engine=engine))
    return scheduler


def test_public_isolated_backward_setting():
    """The API is per unit and rejects non-boolean values."""
    module = HSDPModule()
    module.hsdp_scheduler = SimpleNamespace(isolated_backward=False)
    module.set_isolated_backward(True)
    assert module.hsdp_scheduler.isolated_backward is True
    module.set_isolated_backward(False)
    assert module.hsdp_scheduler.isolated_backward is False
    with pytest.raises(ValueError, match="bool"):
        module.set_isolated_backward(1)


@pytest.mark.parametrize("previous", [False, True])
def test_isolated_pre_hook_preserves_other_root_state(monkeypatch, previous):
    """An early unit's pre-hook must not publish a top-level root backward."""
    scheduler = _make_scheduler(monkeypatch)
    monkeypatch.setattr(HSDPSchedulerV2, "root_bp_state", previous)
    marker = object()
    assert scheduler._backward_pre_hook(marker) is marker
    assert HSDPSchedulerV2.root_bp_state is previous
    scheduler._hsdp_backward_pre_hook.assert_called_once()


def test_isolated_callback_preserves_another_root_backward(monkeypatch):
    """Finalizing an early unit must not clear a concurrently running root graph."""
    scheduler = _make_scheduler(monkeypatch)
    monkeypatch.setattr(HSDPSchedulerV2, "root_bp_state", True)
    scheduler._root_backward_hook()
    assert HSDPSchedulerV2.root_bp_state is True
    scheduler._backward_hook.assert_called_once()
    scheduler.wait_for_pending_reductions.assert_called_once()


def test_isolated_forward_is_not_another_graphs_recompute(monkeypatch):
    """An independent forward still reshards while a different root is in backward."""
    scheduler = _make_scheduler(monkeypatch)
    monkeypatch.setattr(HSDPSchedulerV2, "root_bp_state", True)
    assert scheduler._forward_hook(scheduler.cell, (), "outputs") == "forward-output"
    scheduler._hsdp_forward_hook.assert_called_once()
    scheduler._restore_forward_prefetch_after_recompute.assert_not_called()


def test_regular_backward_keeps_the_existing_root_lifecycle(monkeypatch):
    """The isolation behavior is opt-in and does not change normal graphs."""
    scheduler = _make_scheduler(monkeypatch)
    scheduler.isolated_backward = False
    monkeypatch.setattr(HSDPSchedulerV2, "root_bp_state", False)
    scheduler._backward_pre_hook(object())
    assert HSDPSchedulerV2.root_bp_state is True
    scheduler._root_backward_hook()
    assert HSDPSchedulerV2.root_bp_state is False
