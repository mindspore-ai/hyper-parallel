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
"""Engram layout compatibility and borrowed communication ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.engram import Engram
from hyper_parallel.components.modules.engram.engram import engram_sharding, validate_engram_layout
from hyper_parallel.components.modules.module_family import FamilyShardingContext
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding
from tests.common.engram_family import small_config


class TestEngramFamilyBinding(unittest.TestCase):
    """Validate effective placements and resource injection without process groups."""

    def test_omitted_replicate_axes_keep_the_same_effective_layout(self) -> None:
        """Sparse YAML axis maps are equivalent to complete module-owned placements."""
        module = Engram(config=small_config())
        context = FamilyShardingContext(("tp", "cp", "ep"), {"tp": 2, "cp": 2, "ep": 2}, {})
        spec = engram_sharding(module=module, module_fqn="", member=None, context=context)
        spec.params = {"q_weight": {"tp": Replicate()}, "k_weight": {"tp": Replicate()},
                       "wkv.weight": {"tp": Replicate()}, "embed.weight": {"ep": Shard(0)}}
        validate_engram_layout(module=module, sharding=spec, context=context)
        spec.params["wkv.weight"]["tp"] = Shard(0)
        with self.assertRaisesRegex(ValueError, "wkv.weight"):
            validate_engram_layout(module=module, sharding=spec, context=context)

    def test_layout_requires_every_parameter_and_balanced_table_rows(self) -> None:
        """Compatibility does not admit missing parameters or unsupported EP partitions."""
        module = Engram(config=small_config())
        context = FamilyShardingContext(("ep",), {"tp": 1, "cp": 1, "ep": 2}, {})
        spec = engram_sharding(module=module, module_fqn="", member=None, context=context)
        del spec.params["q_weight"]
        with self.assertRaisesRegex(ValueError, "replicated fusion weights"):
            validate_engram_layout(module=module, sharding=spec, context=context)
        spec = engram_sharding(module=module, module_fqn="", member=None, context=context)
        module.padded_num_embeddings = 15
        with self.assertRaisesRegex(ValueError, "EP-size multiple"):
            validate_engram_layout(module=module, sharding=spec, context=context)

    def test_ep_binding_borrows_existing_groups(self) -> None:
        """The family supplies job-owned EP resources to its sparse lookup stage."""
        module = Engram(config=small_config())
        group = object()
        axis = SimpleNamespace(get_local_rank=lambda: 1, get_group=lambda: group)
        mesh = SimpleNamespace(ep_size=2, cp_size=1, tp_size=1, tp_rank=0,
                               sequence_parallel=False, fsdp_moe_mesh={"ep": axis})
        module.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        hidden = torch.randn(1, 3, 2, 4)
        ids = torch.tensor([[1, 2, 3]])
        with patch.object(module, "parallel_forward", return_value=hidden) as stage:
            self.assertIs(module(hidden, ids), hidden)
        self.assertIs(stage.call_args.kwargs["ep_group"], group)
        self.assertEqual(stage.call_args.kwargs["ep_rank"], 1)
        self.assertEqual(stage.call_args.kwargs["ep_size"], 2)

    def test_local_binding_preserves_outputs_and_gradients(self) -> None:
        """DP-only execution uses local embedding math without world collectives."""
        module = Engram(config=small_config())
        hidden = torch.randn(1, 3, 2, 4, requires_grad=True)
        ids = torch.tensor([[1, 2, 3]])
        expected = module(hidden, ids)
        mesh = SimpleNamespace(ep_size=1, cp_size=1, tp_size=1, sequence_parallel=False)
        module.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        with patch("torch.distributed.all_gather", side_effect=AssertionError("Unexpected world collective")):
            actual = module(hidden, ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))
