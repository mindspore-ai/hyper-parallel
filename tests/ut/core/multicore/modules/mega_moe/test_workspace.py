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
"""Unit tests for MegaMoe workspace sizing and stream ordering."""

import unittest
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.modules.mega_moe import workspace as workspace_module
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _resolve_receive_capacity,
)
from hyper_parallel.core.multicore.modules.mega_moe.workspace import (
    _EVENT_COUNTER_BYTES,
    _WORKSPACE_ALIGNMENT,
    MegaMoeWorkspace,
    _spec_workspace_bytes,
)


class TestMegaMoeWorkspaceSizing(unittest.TestCase):
    """Validate SHMEM planning without allocating accelerator memory."""

    @staticmethod
    def _specification(expert_capacity_factor):
        """Build a fixed capacity-planning specification."""
        return {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": expert_capacity_factor,
            "ep_size": 8,
        }

    def test_capacity_planning_covers_lossless_and_bounded_modes(self) -> None:
        """Reserve the EP maximum by default and align explicit factors."""
        specification = self._specification(None)
        routed_slots = 256
        expected_capacity = 2048
        element_size = 2
        expected_bytes = (
            (expected_capacity + routed_slots)
            * specification["hidden_size"]
            * element_size
            + 2 * _EVENT_COUNTER_BYTES
            + 4 * (_WORKSPACE_ALIGNMENT - 1)
        )

        actual_capacity = _resolve_receive_capacity(
            specification["expert_capacity_factor"],
            routed_slots,
            specification["ep_size"],
        )
        actual_bytes = _spec_workspace_bytes(specification, element_size)

        self.assertEqual(actual_capacity, expected_capacity)
        self.assertEqual(actual_bytes, expected_bytes)
        self.assertEqual(
            _resolve_receive_capacity(
                1.5,
                routed_slots,
                specification["ep_size"],
            ),
            384,
        )

    def test_reuse_waits_with_event_without_host_synchronize(self) -> None:
        """Order serial reuse across streams without blocking the CPU."""
        symmetric_memory = Mock()
        workspace = MegaMoeWorkspace(symmetric_memory=symmetric_memory, shared=True)
        completion_event = Mock()
        current_stream = Mock()
        workspace.completion_event = completion_event
        workspace.used = True

        with (
            patch.object(
                workspace_module.torch.npu,
                "current_stream",
                return_value=current_stream,
            ),
            patch.object(workspace_module.torch.npu, "synchronize") as mock_synchronize,
        ):
            workspace.claim()
            workspace.release()

        current_stream.wait_event.assert_called_once_with(completion_event)
        completion_event.record.assert_called_once_with(current_stream)
        mock_synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
