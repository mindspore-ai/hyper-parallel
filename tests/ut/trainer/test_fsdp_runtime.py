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
"""Unit tests for Trainer-side FSDP runtime diagnostics."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.trainer.runtime.fsdp import FSDPRuntimeDiagnostics


def _make_state() -> SimpleNamespace:
    """Build an empty duck-typed Torch FSDP state."""
    return SimpleNamespace(param_group=None, hsdp_params=[])


class _FSDPUnit(torch.nn.Module):
    """CPU module carrying only the scheduler fields consumed by diagnostics."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(features, features)
        self.hsdp_scheduler = SimpleNamespace(
            scheduler_ctx=SimpleNamespace(root_bp_state=False),
            hsdp_state=_make_state(),
            forward_prefetch_cells=[],
            backward_prefetch_cells=[],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the test projection."""
        return self.projection(inputs)


class _TwoUnitModel(_FSDPUnit):
    """Root plus two nested FSDP units with an optional second branch."""

    def __init__(self, run_second: bool = True) -> None:
        super().__init__()
        del self.projection
        self.first = _FSDPUnit()
        self.second = _FSDPUnit()
        self.run_second = run_second

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run child units in the fixed first-to-second execution order."""
        outputs = self.first(inputs)
        if self.run_second:
            outputs = self.second(outputs)
        return outputs


class TestFSDPRuntimeDiagnostics(unittest.TestCase):
    """Verify order and optimizer-boundary checks without distributed setup."""

    def setUp(self) -> None:
        """Create the CPU test model and attach diagnostics."""
        self.model = _TwoUnitModel()
        self.model.first.hsdp_scheduler.forward_prefetch_cells = [self.model.second]
        self.model.second.hsdp_scheduler.backward_prefetch_cells = [self.model.first]
        self.diagnostics = FSDPRuntimeDiagnostics([self.model])

    def tearDown(self) -> None:
        """Remove hooks so each test owns an isolated observer."""
        self.diagnostics.close()

    def _run_micro_step(self) -> None:
        """Run one forward/backward micro-step on CPU."""
        self.diagnostics.begin_micro_step(global_step=3, micro_step=0)
        inputs = torch.randn(2, 4, requires_grad=True)
        self.model(inputs).sum().backward()

    def test_matching_forward_backward_order_passes(self) -> None:
        """Accept prefetch edges that follow both observed execution phases."""
        self._run_micro_step()

        self.assertIsNone(self.diagnostics.validate_before_optimizer_step())

    def test_forward_prefetch_to_earlier_unit_fails(self) -> None:
        """Reject a construction-order edge that points backward at runtime."""
        self.model.first.hsdp_scheduler.forward_prefetch_cells = []
        self.model.second.hsdp_scheduler.forward_prefetch_cells = [self.model.first]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-008"):
            self.diagnostics.validate_before_optimizer_step()

    def test_prefetched_conditional_target_that_did_not_execute_fails(self) -> None:
        """Reject a prefetch target skipped by a conditional execution branch."""
        self.model.run_second = False
        self.model.first.hsdp_scheduler.forward_prefetch_cells = [self.model.second]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "target did not execute"):
            self.diagnostics.validate_before_optimizer_step()

    def test_non_fused_pending_all_gather_fails(self) -> None:
        """Reject an unconsumed per-parameter all-gather before optimizer.step."""
        hsdp_param = SimpleNamespace(
            _param_fqn="first.projection.weight",
            allgather_comm_ctx=SimpleNamespace(
                allgather_handle=object(),
                allgather_output=None,
            ),
        )
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [hsdp_param]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-009"):
            self.diagnostics.validate_before_optimizer_step()

    def test_non_fused_completed_but_unconsumed_output_fails(self) -> None:
        """Reject a non-fused output retained after its async handle completed."""
        hsdp_param = SimpleNamespace(
            _param_fqn="first.projection.weight",
            allgather_comm_ctx=SimpleNamespace(
                allgather_handle=None,
                allgather_output=object(),
            ),
        )
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [hsdp_param]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "output=not-consumed"):
            self.diagnostics.validate_before_optimizer_step()

    def test_fused_completed_but_unconsumed_all_gather_fails(self) -> None:
        """Reject a fused result even when its async handle is already complete."""
        result = SimpleNamespace(handle=None)
        bucket = SimpleNamespace(all_gather_result=result)
        self.model.first.hsdp_scheduler.hsdp_state.param_group = SimpleNamespace(
            all_gather_buckets=[bucket]
        )
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "completed-not-consumed"):
            self.diagnostics.validate_before_optimizer_step()

    def test_child_parameter_gradient_without_child_forward_fails(self) -> None:
        """Detect direct child-weight access that bypasses the FSDP unit call."""
        parameter = self.model.first.projection.weight
        parameter.main_grad = torch.ones_like(parameter)
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [
            SimpleNamespace(
                _param_fqn="first.projection.weight",
                sharded_param=parameter,
            )
        ]
        self.diagnostics.begin_micro_step(global_step=3, micro_step=0)
        inputs = torch.randn(2, 4, requires_grad=True)
        self.model.second(inputs).sum().backward()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-007"):
            self.diagnostics.validate_before_optimizer_step()


if __name__ == "__main__":
    unittest.main()
