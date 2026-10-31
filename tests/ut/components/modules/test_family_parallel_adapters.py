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
"""Family-registered parallel adapters, without model-specific dependencies."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.family import (
    ModuleFamily, get_family_parallel_adapter, get_module_family, register_module_family,
)
from hyper_parallel.components.modules.gqa_attention import GQAAttention, GatedGQAAttention
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.distributed.context_parallel import gqa_attention as cp_adapter
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)
from tests.common.module_parallel_utils import (
    PackedExperts, make_attention, make_attention_source,
)


class _MoeOwner(nn.Module):
    """Model-owned router and forward contract before Experts replacement."""

    def __init__(self) -> None:
        """Build parameters following the native model ownership layout."""
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = PackedExperts()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep the model forward as the signature anchor for the EP factory."""
        return hidden_states


def _replaced_model():
    model = nn.Module()
    model.layer = nn.Module()
    model.config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
    model.layer.self_attn = make_attention_source()
    model.layer.mlp = _MoeOwner()
    rules = (
        ModuleReplacementSpec(("layer.self_attn",), GQAAttention),
        ModuleReplacementSpec(("layer.mlp.experts",), GroupedExperts),
    )
    return apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])[0]


class TestFamilyAutomaticBinding(unittest.TestCase):
    """The builder consumes Family defaults after real module replacements."""

    def test_builder_binds_cp_and_ep_without_manual_factories(self):
        model = _replaced_model()
        device_mesh = SimpleNamespace(mesh_dim_names=("cp", "ep"), mesh_shape=(2, 2))
        mesh = SimpleNamespace(device_mesh=device_mesh, tp_size=1, cp_size=2, ep_size=2,
                               sequence_parallel=False, loss_parallel=False)
        with patch.object(model_builder, "apply_sharding_plan", return_value=(model, {})) as apply:
            model_builder._plan_and_apply_sharding(model, mesh, ShardingPlanner(), True, False)
        plan = apply.call_args.args[1]
        cp = plan.modules["layer.self_attn"]
        self.assertIs(cp.inner_wrapper, cp_adapter.gqa_attention_cp_wrapper)
        self.assertEqual(cp.inner_target, "self")
        self.assertFalse(cp.region_dispatch)
        ep = plan.modules["layer.mlp"]
        self.assertIs(ep.local_compute_fn.callable, get_family_parallel_adapter(GroupedExperts, "ep"))
        self.assertTrue(callable(ep.local_compute_fn.router_fn))
        self.assertTrue(callable(ep.local_compute_fn.combine))
        self.assertFalse(ep.region_dispatch)
        self.assertNotIn("layer.mlp.experts", plan.modules)

    def test_inactive_axes_do_not_bind(self):
        plan = ShardingPlanner().plan(_replaced_model(), SimpleNamespace(mesh_dim_names=()))
        self.assertIsNone(plan.modules["layer.self_attn"].inner_wrapper)
        self.assertIsNone(plan.modules["layer.mlp"].local_compute_fn)

    def test_explicit_injections_win_even_without_model_semantics(self):
        model = _replaced_model()
        model.config.architectures = ["UnregisteredForCausalLM"]
        cp, ep = Mock(), Mock()
        overrides = {
            "layer.self_attn": ModuleShardingSpec(inner_target="self", inner_wrapper=cp, region_dispatch=False),
            "layer.mlp": ModuleShardingSpec(local_compute_fn=ep, region_dispatch=False),
        }
        plan = ShardingPlanner(overrides).plan(
            model, SimpleNamespace(mesh_dim_names=("cp", "ep")), cp_size=2, ep_size=2,
        )
        self.assertIs(plan.modules["layer.self_attn"].inner_wrapper, cp)
        self.assertIs(plan.modules["layer.mlp"].local_compute_fn, ep)

    def test_unknown_model_requires_ep_semantics(self):
        model = _replaced_model()
        model.config.architectures = ["UnregisteredForCausalLM"]
        with self.assertRaisesRegex(ValueError, "expert_parallel_options"):
            ShardingPlanner().plan(model, SimpleNamespace(mesh_dim_names=("ep",)), ep_size=2)

    def test_instance_forward_is_not_wrapped_twice(self):
        model = _replaced_model()
        model.layer.self_attn.forward = Mock()
        model.layer.mlp.forward = Mock()
        plan = ShardingPlanner().plan(
            model, SimpleNamespace(mesh_dim_names=("cp", "ep")), cp_size=2, ep_size=2,
        )
        self.assertIsNone(plan.modules["layer.self_attn"].inner_wrapper)
        self.assertIsNone(plan.modules["layer.mlp"].local_compute_fn)

    def test_conflicting_dispatch_setting_is_rejected(self):
        overrides = {"layer.self_attn": ModuleShardingSpec(region_dispatch=True)}
        with self.assertRaisesRegex(ValueError, "Family CP requires"):
            ShardingPlanner(overrides).plan(
                _replaced_model(), SimpleNamespace(mesh_dim_names=("cp",)), cp_size=2,
            )


class TestFamilyParallelDeclarations(unittest.TestCase):
    """A binding belongs to the existing concrete Family, with lazy resolution."""

    def test_registered_types_and_numerical_variants_are_distinct(self):
        for module_type in (GQAAttention, GatedGQAAttention):
            self.assertIs(get_family_parallel_adapter(module_type, "cp"), cp_adapter.gqa_attention_cp_wrapper)
        ep = get_family_parallel_adapter(GroupedExperts, "ep")
        self.assertEqual(ep.__name__, "grouped_experts_ep_compute_fn")
        self.assertIsNone(get_family_parallel_adapter(GQAAttention, "ep"))
        class UnregisteredVariant(GroupedExperts):
            pass
        self.assertIsNone(get_family_parallel_adapter(UnregisteredVariant, "ep"))

    def test_lazy_binding_is_an_immutable_snapshot(self):
        class Baseline:
            pass
        adapters = {"cp": "missing_optional_dependency.adapter"}
        with patch.dict(registry._FAMILIES):
            with patch.object(registry, "import_module") as loader:
                register_module_family(ModuleFamily("lazy", Baseline, {}, parallel_adapters=adapters))
                loader.assert_not_called()
            adapters["cp"] = "another_module.adapter"
            self.assertEqual(get_module_family(Baseline).parallel_adapters["cp"],
                             "missing_optional_dependency.adapter")
            with self.assertRaises(TypeError):
                get_module_family(Baseline).parallel_adapters["cp"] = "replacement.adapter"
            with self.assertRaises(ModuleNotFoundError):
                get_family_parallel_adapter(Baseline, "cp")

    def test_invalid_binding_is_rejected(self):
        class Baseline:
            pass
        with patch.dict(registry._FAMILIES):
            for adapters in ({"tp": "some_module.adapter"}, {"cp": "not_a_path"}):
                with self.assertRaisesRegex(ValueError, "parallel adapter"):
                    register_module_family(ModuleFamily("invalid", Baseline, {}, parallel_adapters=adapters))
            with self.assertRaisesRegex(TypeError, "mapping"):
                register_module_family(ModuleFamily("invalid", Baseline, {}, parallel_adapters=None))


class TestGqaFamilyCpAdapter(unittest.TestCase):
    """Actual GQA uses its selected kernel through the registered adapter."""

    def test_implicit_and_external_masks_match_unsharded_forward(self):
        for rank in (0, 1):
            for mask_kind in ("implicit", "boolean", "additive"):
                with self.subTest(rank=rank, mask=mask_kind):
                    torch.manual_seed(23)
                    module = make_attention()
                    reference = deepcopy(module)
                    selected = Mock(side_effect=module.attention_interface)
                    module.attention_interface = selected
                    mesh = Mock()
                    mesh.size.return_value = 2
                    mesh.get_local_rank.return_value = rank
                    adapter = get_family_parallel_adapter(type(module), "cp")
                    _commit_forward_rewrite(adapter(module, None, None, mesh, None))
                    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
                    mask = None
                    if mask_kind != "implicit":
                        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
                        mask[..., 1] = False
                        if mask_kind == "additive":
                            mask = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill(~mask, float("-inf"))
                    expected, _ = reference(hidden, attention_mask=mask)
                    qkv = reference.linear_qkv(hidden).view(1, 4, 2, 8)
                    _, key, value = qkv.split((4, 2, 2), -1)
                    with patch("hyper_parallel.distributed.context_parallel.wrappers.flex_cp_allgather",
                               return_value=(key.transpose(1, 2), value.transpose(1, 2))):
                        actual, _ = module(hidden[:, rank * 2:(rank + 1) * 2], attention_mask=mask)
                    torch.testing.assert_close(actual, expected[:, rank * 2:(rank + 1) * 2])
                    self.assertEqual(selected.call_count, 1)

    def test_unsupported_inputs_fail_before_communication(self):
        module = make_attention()
        mesh = Mock()
        mesh.size.return_value = 2
        adapter = get_family_parallel_adapter(type(module), "cp")
        _commit_forward_rewrite(adapter(module, None, None, mesh, None))
        hidden = torch.zeros(1, 2, 8, dtype=torch.float64)
        with patch("hyper_parallel.distributed.context_parallel.wrappers.flex_cp_allgather") as gather:
            with self.assertRaisesRegex(ValueError, "cache"):
                module(hidden, past_key_values=object())
            with self.assertRaisesRegex(ValueError, "packed"):
                module(hidden, actual_seq_len=[2])
            gather.assert_not_called()
