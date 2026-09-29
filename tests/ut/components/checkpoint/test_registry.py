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
"""Unit tests for the checkpointer backend registry."""

import unittest

from hyper_parallel.components.checkpoint.registry import Registry


def _global_backend() -> None:
    """Stand in for a globally registered backend."""


def _local_backend() -> None:
    """Stand in for a local backend override."""


class TestCheckpointRegistry(unittest.TestCase):
    """Verify Registry follows the MutableMapping removal contract."""

    def test_registry_names_are_instance_local(self) -> None:
        """
        Feature: Checkpoint registry ownership.
        Description: Construct two registry instances with different names.
        Expectation: Each instance owns an independent list of registry names.
        """
        first = Registry("first")
        second = Registry("second")

        self.assertEqual(first.registry, ["first"])
        self.assertEqual(second.registry, ["second"])
        self.assertIsNot(first.registry, second.registry)

    def test_pop_removes_global_registration(self) -> None:
        """
        Feature: Checkpoint registry removal.
        Description: Pop a backend registered in the global mapping.
        Expectation: The backend is returned and no longer registered.
        """
        registry = Registry("unit")
        registry.register("backend", _global_backend)

        self.assertIs(registry.pop("backend"), _global_backend)
        self.assertNotIn("backend", registry)

    def test_clear_removes_global_and_local_registrations(self) -> None:
        """
        Feature: Checkpoint registry cleanup.
        Description: Clear a registry containing local and global backends.
        Expectation: No registered keys remain after cleanup.
        """
        registry = Registry("unit")
        registry.register("overridden", _global_backend)
        registry["overridden"] = _local_backend
        registry.register("global", _global_backend)

        registry.clear()

        self.assertEqual(len(registry), 0)
        self.assertEqual(registry.valid_keys(), [])


if __name__ == "__main__":
    unittest.main()
