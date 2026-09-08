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
"""Characterization: ``apply_sharding_plan`` forward install and identity.

Locks the current master's applier behavior before the migration splits the
builder (plan §14.6 item 7), using only fake meshes and empty placement
contracts (Gate-1: no process group):

* forward wrappers are installed exactly once per boundary per apply call,
  with the production/validate branch selected by ``validate_mode``;
* the entry point fails fast on a missing device mesh and on injection /
  ``region_dispatch`` contract violations, before any forward is touched;
* tied-weight detection and storage sharing keep parameter identity.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel.distributed.apply import (
    apply_sharding_plan,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _replicate_tied_weights,
    detect_tied_weights,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh


class _TinyChild(nn.Module):
    """Boundary module with a deterministic forward."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return x * 2 + self.weight.sum()


class _TinyModel(nn.Module):
    """One-boundary model; deliberately has no ``config`` attribute."""

    def __init__(self):
        super().__init__()
        self.child = _TinyChild()


class _TiedModel(nn.Module):
    """HF-style tied embedding/lm_head sharing one parameter."""

    def __init__(self, tie_word_embeddings=True, share=True):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if share:
            self.lm_head.weight = self.embed_tokens.weight


def _boundary_plan():
    """A plan with one boundary module and empty placement contracts."""
    return ShardingPlan(
        modules={"child": ModuleShardingSpec(params={})},
        mesh_dim_names=(),
    )


class TestApplyEntryValidation(unittest.TestCase):
    """Entry-point argument validation and return identity."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_mesh_raises(self):
        """A None device mesh fails fast with a dedicated message."""
        with self.assertRaisesRegex(ValueError, "requires a DeviceMesh"):
            apply_sharding_plan(_TinyModel(), ShardingPlan(), None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_plan_returns_model_identity(self):
        """An empty plan mutates nothing and returns the same model object."""
        model = _TinyModel()
        original_forward = model.child.forward
        result, source_shard_info = apply_sharding_plan(
            model, ShardingPlan(mesh_dim_names=()), FakeDeviceMesh()
        )
        self.assertIs(result, model)
        self.assertIsNone(source_shard_info)
        self.assertEqual(model.child.forward.__func__, original_forward.__func__)


class TestForwardInstall(unittest.TestCase):
    """Production/validate wrapper installation characterization."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_validate_mode_installs_validate_forward_once(self):
        """validate_mode=True installs one validate wrapper around the original."""
        model = _TinyModel()
        result, source_shard_info = apply_sharding_plan(
            model, _boundary_plan(), FakeDeviceMesh(), validate_mode=True
        )
        self.assertIs(result, model)
        self.assertIsNone(source_shard_info)  # validate keeps DTensors; no info
        # functools.wraps copies __name__; the code object keeps the wrapper name
        self.assertEqual(model.child.forward.__code__.co_name, "validate_forward")
        # exactly one wrapper: __wrapped__ is the original forward
        self.assertIs(model.child.forward.__wrapped__.__func__, _TinyChild.forward)
        # empty contracts: the boundary is a pass-through for plain tensors
        output = model.child(torch.ones(2, 2))
        self.assertTrue(torch.equal(output, torch.ones(2, 2) * 2 + 4.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_production_mode_installs_production_forward_once(self):
        """validate_mode=False installs one production wrapper around the original."""
        model = _TinyModel()
        result, source_shard_info = apply_sharding_plan(
            model, _boundary_plan(), FakeDeviceMesh(), validate_mode=False
        )
        self.assertIs(result, model)
        # no DTensor parameters were produced, so no FSDP source metadata
        self.assertIsNone(source_shard_info)
        self.assertEqual(model.child.forward.__code__.co_name, "production_forward")
        self.assertIs(model.child.forward.__wrapped__.__func__, _TinyChild.forward)
        output = model.child(torch.ones(2, 2))
        self.assertTrue(torch.equal(output, torch.ones(2, 2) * 2 + 4.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_repeated_apply_nests_wrappers(self):
        """Current behavior: a second apply wraps again (no idempotency guard).

        Recorded deliberately: the split ``_builder/forward_rewriter`` targets
        idempotent re-installation; this snapshot makes the behavior change
        explicit when the new component replaces the applier.
        """
        model = _TinyModel()
        apply_sharding_plan(model, _boundary_plan(), FakeDeviceMesh(), validate_mode=True)
        apply_sharding_plan(model, _boundary_plan(), FakeDeviceMesh(), validate_mode=True)
        self.assertEqual(model.child.forward.__code__.co_name, "validate_forward")
        self.assertEqual(
            model.child.forward.__wrapped__.__code__.co_name, "validate_forward"
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_injection_without_region_dispatch_fails_before_wrapping(self):
        """local_compute_fn without region_dispatch fails fast; forward untouched."""
        model = _TinyModel()
        spec = ModuleShardingSpec(params={}, local_compute_fn=lambda *a, **k: None)
        plan = ShardingPlan(modules={"child": spec}, mesh_dim_names=())
        with self.assertRaisesRegex(ValueError, "region_dispatch"):
            apply_sharding_plan(model, plan, FakeDeviceMesh(), validate_mode=True)
        # the original bound method is untouched (no wrapper installed)
        self.assertIs(getattr(model.child.forward, "__func__", None), _TinyChild.forward)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_region_dispatch_true_without_injection_fails(self):
        """region_dispatch=True with no injection is rejected as redundant."""
        model = _TinyModel()
        spec = ModuleShardingSpec(params={}, region_dispatch=True)
        plan = ShardingPlan(modules={"child": spec}, mesh_dim_names=())
        with self.assertRaisesRegex(ValueError, "redundant"):
            apply_sharding_plan(model, plan, FakeDeviceMesh(), validate_mode=True)
        self.assertIs(getattr(model.child.forward, "__func__", None), _TinyChild.forward)


class TestTiedWeights(unittest.TestCase):
    """Tied-weight detection and within-rank storage sharing."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_detect_tied_weights(self):
        """tie_word_embeddings + shared parameter yields the (embed, lm_head) pair."""
        model = _TiedModel(tie_word_embeddings=True, share=True)
        self.assertEqual(
            detect_tied_weights(model),
            [("embed_tokens.weight", "lm_head.weight")],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_detect_tied_weights_disabled(self):
        """No tied pairs when the config does not tie word embeddings."""
        model = _TiedModel(tie_word_embeddings=False, share=True)
        self.assertEqual(detect_tied_weights(model), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_replicate_tied_weights_shares_storage(self):
        """End B's data is rebound to end A's storage; parameter objects stay."""
        model = _TiedModel(tie_word_embeddings=False, share=False)
        embed_param = model.embed_tokens.weight
        head_param = model.lm_head.weight
        self.assertNotEqual(embed_param.data_ptr(), head_param.data_ptr())
        _replicate_tied_weights(
            model, [("embed_tokens.weight", "lm_head.weight")]
        )
        self.assertIs(model.embed_tokens.weight, embed_param)
        self.assertIs(model.lm_head.weight, head_param)
        self.assertEqual(
            model.lm_head.weight.data_ptr(), model.embed_tokens.weight.data_ptr()
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_replicate_tied_weights_tolerates_missing_fqn(self):
        """A pair whose FQN does not resolve is skipped without error."""
        model = _TiedModel(tie_word_embeddings=False, share=False)
        head_data_ptr = model.lm_head.weight.data_ptr()
        _replicate_tied_weights(model, [("nonexistent.weight", "lm_head.weight")])
        self.assertEqual(model.lm_head.weight.data_ptr(), head_data_ptr)


if __name__ == "__main__":
    unittest.main()
