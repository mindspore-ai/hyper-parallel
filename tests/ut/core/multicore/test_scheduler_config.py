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
"""CPU-only tests for multicore scheduler ctypes configuration."""

import ctypes
import importlib.util
import unittest
from pathlib import Path


_CONFIG_PATH = (
    Path(__file__).resolve().parents[4]
    / "hyper_parallel"
    / "core"
    / "multicore"
    / "scheduler"
    / "config.py"
)
_CONFIG_SPEC = importlib.util.spec_from_file_location("multicore_scheduler_config", _CONFIG_PATH)
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)


class TestSchedulerConfig(unittest.TestCase):
    """Validate defaults that keep TaskDesc profiling metadata optional."""

    def test_task_desc_defaults_profile_metadata_to_invalid(self):
        """Keep a new task from being assigned a stage or owner before profile resolution."""
        task_desc = _CONFIG_MODULE.TaskDescC()

        self.assertEqual(
            task_desc.profile_desc_id,
            _CONFIG_MODULE.INVALID_PROFILE_DESC_ID,
            msg=f"Unexpected default profile desc ID: got={task_desc.profile_desc_id}",
        )
        self.assertEqual(
            task_desc.profile_owner_id,
            _CONFIG_MODULE.INVALID_PROFILE_OWNER_ID,
            msg=f"Unexpected default profile owner ID: got={task_desc.profile_owner_id}",
        )

    def test_task_desc_profile_fields_use_reserved_tail_slots(self):
        """Give the two reserved tail slots explicit profiling semantics."""
        self.assertEqual(
            ctypes.sizeof(_CONFIG_MODULE.TaskDescC),
            576,
            msg=f"Unexpected TaskDesc size: got={ctypes.sizeof(_CONFIG_MODULE.TaskDescC)}",
        )
        self.assertEqual(
            _CONFIG_MODULE.TaskDescC.profile_desc_id.offset,
            _CONFIG_MODULE.TaskDescC.extra_value_2.offset + ctypes.sizeof(ctypes.c_uint32),
            msg="profile_desc_id does not reuse the first reserved tail slot",
        )
        self.assertEqual(
            _CONFIG_MODULE.TaskDescC.profile_owner_id.offset,
            _CONFIG_MODULE.TaskDescC.profile_desc_id.offset + ctypes.sizeof(ctypes.c_uint32),
            msg="profile_owner_id is not the final TaskDesc field",
        )


if __name__ == "__main__":
    unittest.main()
