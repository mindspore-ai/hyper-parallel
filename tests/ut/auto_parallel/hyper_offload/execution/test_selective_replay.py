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
"""End-to-end contracts for plan-scoped bindings with real CPU byte copies."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from hyper_parallel.auto_parallel.hyper_offload import (
    OffloadConfig,
    OffloadSession,
    skip_offload,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import (
    ActivationTracker,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType as Action,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
)


class BytePool:
    """Use real CPU byte copies with mocked accelerator queues."""

    def acquire(self, size: int) -> torch.Tensor:
        """Allocate a host byte buffer."""
        return torch.empty(size, dtype=torch.uint8)

    def release(self, buffer: torch.Tensor, event: object = None) -> None:
        """CPU copies complete synchronously; no reuse is needed."""

    def clear(self) -> None:
        """No persistent allocations."""


class CopyFirstActivation:
    """Offload the first activation and prefetch before its view is consumed."""

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Build a small schedule containing real byte transfers."""
        assert trace.ops[0].name == "sin.default"
        assert trace.ops[1].name == "cos.default"
        assert trace.ops[2].name == "view.default"
        sid = next(
            a.storage_id for a in trace.ops[0].accesses if a.kind == AccessKind.WRITE
        )
        self.sid = sid
        schedule = ResidencySchedule()
        schedule.add_post(0, Action.COPY_D2H, sid)
        schedule.add_post(0, Action.RELEASE_DEVICE, sid)
        schedule.add_pre(2, Action.COPY_H2D, sid)
        schedule.add_post(4, Action.RELEASE_HOST, sid)
        return schedule


class CPUActivationTracker(ActivationTracker):
    """Keep production identity rules; permit CPU activations in this test."""

    def get_activation_sid(self, tensor: torch.Tensor) -> int | None:
        """Resolve only tracked tensor or storage identities."""
        if isinstance(tensor, ShadowTensor):
            return tensor.storage_id
        sid = self._storage_tracker.get(tensor.untyped_storage())
        return sid if sid in self._activation_sids else None


class CheckFirstActivation(CopyFirstActivation):
    """Check completion after the producer; enforce release one op later."""

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Exercise optional retirement before the next native operation."""
        schedule = super().build(trace)
        schedule.post[0] = [action for action in schedule.post[0] if action.kind == Action.COPY_D2H]
        schedule.add_post(0, Action.TRY_RELEASE_DEVICE, self.sid)
        schedule.add_post(1, Action.RELEASE_DEVICE, self.sid)
        return schedule


def compute(x: torch.Tensor, replace_left: torch.Tensor | None = None) -> tuple:
    """Compute and differentiate through one moved and one native activation."""
    a = x.sin()
    b = x.cos()
    left = a.view(-1)
    right = b.view(-1)
    loss = ((left if replace_left is None else replace_left) * right).sum()
    loss.backward()
    return a, b, left, loss


@pytest.fixture(autouse=True)
def cpu_queues() -> Iterator[None]:
    """Keep queue semantics mocked without replacing tensor computations."""
    with (patch("torch.accelerator.current_stream"), patch("torch.Event"),
          patch("torch.Tensor.record_stream")):
        yield


@pytest.mark.parametrize("opaque", [False, True])
def test_native_view_created_outside_tracking_keeps_original_input_contract(opaque):
    """FSDP may make an untracked native alias of a tracked layer output."""
    planner = CopyFirstActivation()
    session = OffloadSession(OffloadConfig(planner=planner))
    manager = session.residency_manager
    manager._host_pool = BytePool()
    manager._copy_stream = MagicMock()
    manager._copy_stream.device = torch.device("cpu")
    manager.wait_for_transfers = lambda: None
    manager.sync_all_transfers = lambda: None
    session.executor._timer = Mock()
    session.executor._timer.stop.return_value = 0.0
    session.executor._tracker = CPUActivationTracker()

    def run(value: torch.Tensor) -> torch.Tensor:
        """Exercise a native alias produced outside activation tracking."""
        a, b = value.sin(), value.cos()
        left = a.view(-1)
        # Model the dispatch-disabled FSDP boundary: warmup's ShadowTensor
        # resolves to an ordinary view which intentionally has no binding.
        if opaque:
            escaped = skip_offload(lambda z: z * 2)(b).view(-1)
        else:
            with torch.utils._python_dispatch._disable_current_modes():
                escaped = b.view(-1)
        loss = (left * escaped).sum()
        loss.backward()
        return loss

    data = torch.linspace(-2, 2, 16).reshape(4, 4)
    reference = data.clone().requires_grad_()
    expected = run(reference)
    with patch(
        "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
        return_value=(10.0, 10.0),
    ):
        warmup = data.clone().requires_grad_()
        with session:
            run(warmup)
    value = data.clone().requires_grad_()
    with session:
        actual = run(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_managed_input", [False, True])
@pytest.mark.parametrize("copy_complete", [None, False, True])
def test_partial_replay_preserves_gradients_aliases_and_retained_native_outputs(
    invalid_managed_input, copy_complete,
):
    planner = CopyFirstActivation() if copy_complete is None else CheckFirstActivation()
    session = OffloadSession(OffloadConfig(planner=planner))
    manager = session.residency_manager
    manager._host_pool = BytePool()
    manager._copy_stream = MagicMock()
    manager._copy_stream.device = torch.device("cpu")
    manager.wait_for_transfers = lambda: None
    manager.sync_all_transfers = lambda: None
    session.executor._timer = Mock()
    session.executor._timer.stop.return_value = 0.0
    session.executor._tracker = CPUActivationTracker()
    data = torch.linspace(-2, 2, 16).reshape(4, 4)
    reference = data.clone().requires_grad_()
    expected = compute(reference)
    with patch(
        "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
        return_value=(10.0, 10.0),
    ):
        warmup = data.clone().requires_grad_()
        with session:
            compute(warmup)
    detach_sids = {
        sid
        for op in session.executor._guide
        if op.name == "detach.default"
        for sid in op.input_bindings.values()
    }
    assert session.executor._managed_sids == {planner.sid}
    # Unrelated gradient storage stays native even with an active copy plan.
    assert detach_sids - {planner.sid}
    if invalid_managed_input:
        value = data.clone().requires_grad_()
        replacement = value.flatten()
        with (
            pytest.raises(RuntimeError, match="input bindings differ"),
            session,
        ):
            compute(value, replace_left=replacement)
        return
    for _ in range(2):
        value = data.clone().requires_grad_()
        with (
            patch.object(manager, "copy_d2h", wraps=manager.copy_d2h) as d2h,
            patch("torch.Event") as events,
        ):
            events.return_value.query.return_value = copy_complete
            with session:
                actual = compute(value)
            assert d2h.call_count == 1
            if copy_complete is not None:
                events.return_value.query.assert_called()
        assert isinstance(actual[0], ShadowTensor)
        assert isinstance(actual[2], ShadowTensor)
        assert type(actual[1]) is torch.Tensor  # pylint: disable=unidiomatic-typecheck  # Require a native output.
        assert actual[0]._buffer is actual[2]._buffer
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        # The native output must survive later session resets and keep values.
        retained_native = actual[1]
        session.executor.reset()
        torch.testing.assert_close(retained_native, expected[1], rtol=0, atol=0)
