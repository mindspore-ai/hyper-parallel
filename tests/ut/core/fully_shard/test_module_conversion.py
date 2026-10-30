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
"""Model conversion must release obsolete fused parameter storage."""

from types import SimpleNamespace
import unittest

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.param_group import AllGatherBucket
from tests.common.mark_utils import arg_mark


class ConvertedLinear(HSDPModule, torch.nn.Linear):
    """Exercise the real conversion hook with a CPU-only fused bucket."""

    def __init__(self) -> None:
        """Attach a fused bucket whose storage must follow parameter conversion."""
        torch.nn.Linear.__init__(self, 3, 2, bias=False)
        HSDPModule.__init__(self)
        self.bucket = AllGatherBucket(
            hsdp_params=[SimpleNamespace(_sharded_param_data=self.weight)],
            shard_group=None, shard_rank=0, shard_world_size=1,
            dtype=self.weight.dtype, metadata=None, flat_param_buffer=self.weight.detach(),
        )
        self.hsdp_scheduler = SimpleNamespace(hsdp_state=SimpleNamespace(
            hsdp_params=[], param_group=SimpleNamespace(all_gather_buckets=[self.bucket]),
        ))

    def reshard(self) -> None:
        """The test model starts sharded and has no distributed collectives."""


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestModuleConversion(unittest.TestCase):
    """Keep zero-copy reuse valid while dropping obsolete allocations."""

    def test_conversion_releases_invalid_fused_storage(self) -> None:
        """A real dtype conversion must let the old storage expire immediately."""
        model = ConvertedLinear()
        parameter = model.weight
        storage = StorageWeakRef(model.bucket.flat_param_buffer.untyped_storage())
        model.to(dtype=torch.float64)
        self.assertIs(model.weight, parameter)
        self.assertIsNone(model.bucket.flat_param_buffer)
        self.assertTrue(storage.expired())

    def test_noop_conversion_keeps_valid_fused_storage(self) -> None:
        """A no-op conversion must preserve the existing zero-copy allocation."""
        model = ConvertedLinear()
        buffer = model.bucket.flat_param_buffer
        model.to(device="cpu")
        self.assertIs(model.bucket.flat_param_buffer, buffer)
        self.assertTrue(model.bucket.is_flat_buffer_valid())
