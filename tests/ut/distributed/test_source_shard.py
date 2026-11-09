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
"""Unit tests for managed source-shard metadata selection."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.distributed._builder.source_shard import _build_managed_source_shard_info


def _manager_without_meshes() -> SimpleNamespace:
    """A manager whose meshes are absent, so no default metadata exists."""
    return SimpleNamespace(
        mesh_context=SimpleNamespace(fsdp_non_moe_mesh=None, device_mesh=None))


class TestBuildManagedSourceShardInfo(unittest.TestCase):
    """Parameter selection for one FSDP wrap call."""

    def test_ignored_parameters_are_excluded_from_the_mapping(self):
        """Ignored params neither appear in the result nor need metadata."""
        owner = nn.Module()
        owner.dense = nn.Parameter(torch.zeros(2))
        owner.external = nn.Parameter(torch.zeros(2))
        metadata = {owner.dense: SimpleNamespace(name="dense-meta")}
        result = _build_managed_source_shard_info(
            _manager_without_meshes(), owner, metadata, {owner.external})
        self.assertEqual(set(result), {owner.dense})
        self.assertIs(result[owner.dense], metadata[owner.dense])

    def test_uncovered_managed_parameter_is_rejected(self):
        """A managed parameter without metadata and no default raises."""
        owner = nn.Module()
        owner.dense = nn.Parameter(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "does not cover an FSDP-managed parameter"):
            _build_managed_source_shard_info(_manager_without_meshes(), owner, {})

    def test_none_metadata_stays_none(self):
        """No metadata mapping means no managed selection at all."""
        owner = nn.Module()
        owner.dense = nn.Parameter(torch.zeros(2))
        self.assertIsNone(
            _build_managed_source_shard_info(_manager_without_meshes(), owner, None))


if __name__ == "__main__":
    unittest.main()
