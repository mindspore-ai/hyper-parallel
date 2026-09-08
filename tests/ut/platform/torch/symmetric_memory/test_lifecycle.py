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
"""Unit tests for the minimal process-wide Torch SHMEM lifecycle."""

import unittest
from typing import Any
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist

from hyper_parallel.platform.torch.symmetric_memory import (
    lifecycle as lifecycle_module,
)
from hyper_parallel.platform.torch.symmetric_memory.lifecycle import (
    _resolve_world,
    acquire_symmetric_memory,
)
from hyper_parallel.platform.torch.symmetric_memory.symmetric_memory import (
    TorchSymmetricMemoryHandler,
)


class TestTorchSymmetricMemoryLifecycle(unittest.TestCase):
    """Validate the whole-world owner contract without accelerator hardware."""

    def setUp(self) -> None:
        """Reset shared state and deterministic environment before each test."""
        self._environment = patch.dict(
            lifecycle_module.os.environ,
            {
                "SYMMETRIC_MEMORY_HEAP_SIZE": str(1024**3),
                "SHMEM_IP_PORT": "tcp://127.0.0.1:8662",
            },
            clear=False,
        )
        self._environment.start()
        self.addCleanup(self._environment.stop)
        lifecycle_module._PROCESS_STATE.reset()
        TorchSymmetricMemoryHandler._owner = None
        TorchSymmetricMemoryHandler.comm_streams = []
        TorchSymmetricMemoryHandler.compute_streams = []

    def tearDown(self) -> None:
        """Discard mocked process and handler state after each test."""
        lifecycle_module._PROCESS_STATE.reset()
        TorchSymmetricMemoryHandler._owner = None
        TorchSymmetricMemoryHandler.comm_streams = []
        TorchSymmetricMemoryHandler.compute_streams = []

    def test_handler_and_mega_owner_share_one_native_runtime(self) -> None:
        """Finalize only after both the ordinary and MegaMoe owners close."""
        manager = Mock()
        manager.attr_init.return_value = 0
        manager.finalize.return_value = 0
        handler_tensor = torch.empty(4)
        manager.malloc.return_value = handler_tensor
        with patch.object(
            lifecycle_module,
            "_load_manager",
            return_value=manager,
        ):
            self.assertIs(
                TorchSymmetricMemoryHandler.empty((4,), torch.float32),
                handler_tensor,
            )
            handler_owner = TorchSymmetricMemoryHandler._owner
            mega_owner = acquire_symmetric_memory()
            mega_owner.close()
            manager.finalize.assert_not_called()
            handler_owner.free(handler_tensor)
            TorchSymmetricMemoryHandler.close()

        self.assertIsNotNone(handler_owner)
        self.assertTrue(handler_owner.closed)
        manager.attr_init.assert_called_once_with(
            0,
            1,
            1024**3,
            "tcp://127.0.0.1:8662",
        )
        manager.free.assert_called_once_with(handler_tensor)
        manager.finalize.assert_called_once_with()
        self.assertFalse(lifecycle_module._PROCESS_STATE.initialized)

    def test_subgroup_is_rejected_before_native_initialization(self) -> None:
        """Keep this increment scoped to one communicator over the whole world."""
        subgroup = object()

        def _world_size(group: Any = None) -> int:
            return 4 if group is None else 2

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_rank", return_value=0),
            patch.object(dist, "get_world_size", side_effect=_world_size),
            self.assertRaisesRegex(ValueError, "whole distributed world"),
        ):
            _resolve_world(subgroup)

    def test_active_heap_rejects_a_larger_later_request(self) -> None:
        """Reject a heap increase after the process-wide runtime is active."""
        manager = Mock()
        manager.attr_init.return_value = 0
        manager.finalize.return_value = 0
        with patch.object(
            lifecycle_module,
            "_load_manager",
            return_value=manager,
        ):
            first = acquire_symmetric_memory()
            try:
                with (
                    patch.dict(
                        lifecycle_module.os.environ,
                        {"SYMMETRIC_MEMORY_HEAP_SIZE": "2147483648"},
                        clear=False,
                    ),
                    self.assertRaisesRegex(RuntimeError, "smaller than the requested"),
                ):
                    acquire_symmetric_memory()
            finally:
                first.close()

        manager.attr_init.assert_called_once()
        manager.finalize.assert_called_once_with()

    def test_owner_tracks_regular_and_aligned_allocations(self) -> None:
        """Delegate both allocation forms and invalidate tensors after free."""
        manager = Mock()
        manager.attr_init.return_value = 0
        manager.finalize.return_value = 0
        regular = torch.empty(4)
        aligned = torch.empty(8)
        manager.malloc.return_value = regular
        manager.aligned_malloc.return_value = aligned
        with patch.object(
            lifecycle_module,
            "_load_manager",
            return_value=manager,
        ):
            owner = acquire_symmetric_memory()
            self.assertIs(owner.empty((4,), torch.float32), regular)
            self.assertIs(owner.aligned_empty((8,), torch.float32, 512), aligned)
            owner.free(regular)
            owner.free(aligned)
            owner.close()

        manager.malloc.assert_called_once_with([4], torch.float32)
        manager.aligned_malloc.assert_called_once_with([8], torch.float32, 512)
        self.assertEqual(manager.free.call_args_list, [call(regular), call(aligned)])
        self.assertEqual(regular.untyped_storage().nbytes(), 0)
        self.assertEqual(aligned.untyped_storage().nbytes(), 0)
        manager.finalize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
