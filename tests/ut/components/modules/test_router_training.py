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
"""Training and initialization contracts for the unified Gate family."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from examples.torch.module_family.config import GateConfig
from hyper_parallel.components.losses.aux_loss import (
    aux_loss_scale_context, bind_aux_loss_scale, sequence_load_balancing_loss,
)
from hyper_parallel.components.modules import Gate
from hyper_parallel.components.modules.moe.router import Gate as CanonicalGate
from hyper_parallel.components.modules.moe.router import balanced_expert_indices
from hyper_parallel.components.modules.moe.moe_utils import (
    accumulate_router_load, router_sharding, sync_and_update_expert_bias,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding, validate_parallel_module
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.trainer.callbacks.moe_monitor_callback import MoEMonitorCallback
from tests.common.mark_utils import arg_mark
from tests.common.op_precision import assert_precision


def _router(**settings):
    """Create the public Gate with model-level training flags."""
    config = GateConfig(4, 4, 1, scoring_func="softmax", normalize_topk=False)
    return Gate(config=config, context={"module_configs": (SimpleNamespace(**settings),)})


def _outputs(member, hidden, **metadata):
    """Return learned weights and actual expert IDs from the Gate contract."""
    return member(hidden, **metadata)[1:]


class TestRouterTraining(unittest.TestCase):
    """Verify actual-route accounting, independent composition and borrowed meshes."""

    def setUp(self) -> None:
        """Keep RNG and checkpoint device state local to CPU cases."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(38)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_training_features_default_to_disabled(self) -> None:
        """Feature: Explicit router training opt-in.
        Description: HF's enclosing-model loss default must not enable a second family loss.
        Expectation: Gate defaults to zero coefficients and no correction parameter.
        """
        member = _router(router_aux_loss_coef=.001)
        self.assertEqual(member.aux_loss_coeff, 0.)
        self.assertEqual(member.expert_bias_update_rate, 0.)
        self.assertFalse(member.enable_expert_bias)
        self.assertIsNone(getattr(member, "bias", None))
        _outputs(member, torch.randn(2, 3, 4))
        self.assertIsNone(member.last_aux_loss)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_canonical_options_override_legacy_fields_including_disabled_values(self) -> None:
        """Feature: Canonical router option names.
        Description: Read old configs while allowing explicit new zero/False values to disable features.
        Expectation: Gate prefers canonical fields and preserves old-config compatibility.
        """
        legacy_options = (
            {"router_fix_router": True, "router_aux_loss_coeff": .1, "router_bias_update_rate": .01},
            {"fix_router": True, "aux_loss_coeff": .1, "expert_bias_update_rate": .01},
        )
        canonical = {
            "moe_router_force_load_balancing": False, "moe_aux_loss_coeff": 0., "moe_router_bias_update_rate": 0.,
        }
        for legacy in legacy_options:
            with self.subTest(legacy=legacy):
                original = _router(**legacy)
                self.assertTrue(original.fix_router)
                self.assertEqual(original.aux_loss_coeff, .1)
                self.assertEqual(original.expert_bias_update_rate, .01)
                member = _router(**dict(legacy, **canonical))
                self.assertFalse(member.fix_router)
                self.assertEqual(member.aux_loss_coeff, 0.)
                self.assertEqual(member.expert_bias_update_rate, 0.)
                self.assertFalse(member.enable_expert_bias)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_auxiliary_loss_and_gradient_use_fixed_dispatch_ids(self) -> None:
        """Feature: Fixed routing and auxiliary loss.
        Description: Force learned expert zero, then compare post-fix loss/gradient with an explicit formula.
        Expectation: Learned weights stay unchanged while both auxiliary counts and gradients use fixed IDs.
        """
        member = _router(moe_router_force_load_balancing=True, moe_aux_loss_coeff=.07)
        reference = _router()
        with torch.no_grad():
            member.weight.copy_(torch.eye(4) * 3)
            reference.weight.copy_(member.weight)
        hidden = torch.tensor([[2., .3, .1, .1], [1., .2, .1, .1], [3., .1, .2, .1]], requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        expected_weights, learned_ids = _outputs(reference, other)
        weights, fixed_ids = _outputs(member, hidden)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        self.assertTrue(torch.equal(learned_ids, torch.zeros_like(learned_ids)))
        self.assertTrue(torch.equal(fixed_ids, torch.tensor([[0], [1], [2]])))
        scores = (other @ reference.weight.T).float().softmax(-1)
        fraction = torch.tensor([1/3, 1/3, 1/3, 0.])
        explicit_loss = 4 * (fraction * (scores / scores.sum(-1, keepdim=True)).mean(0)).sum()
        learned_loss = 4 * scores[:, 0].mean()
        torch.testing.assert_close(member.last_aux_loss, explicit_loss, rtol=1e-6, atol=1e-7)
        self.assertFalse(torch.isclose(member.last_aux_loss, learned_loss))
        (weights.sum() * 0).backward()
        (.07 * explicit_loss).backward()
        torch.testing.assert_close(hidden.grad, other.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(member.weight.grad, reference.weight.grad, rtol=1e-6, atol=1e-7)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_aux_free_counts_once_per_backward_with_masks_and_checkpoint(self) -> None:
        """Feature: Aux-loss-free step state.
        Description: Accumulate two microbatches with padding masks and both checkpoint modes.
        Expectation: Actual post-fix IDs are counted once; a step updates standard bias then clears counts.
        """
        for replay in (None, False, True):
            with self.subTest(replay=replay):
                member = _router(moe_router_force_load_balancing=True, moe_router_bias_update_rate=.01)
                valid = torch.tensor([[True, True, False]])

                def forward(states: torch.Tensor, router: torch.nn.Module = member,
                            token_mask: torch.Tensor = valid) -> torch.Tensor:
                    """Route one microbatch with padding metadata.

                    Args:
                        states: Local token activations.
                    """
                    return _outputs(router, states, token_mask=token_mask)[0]

                for _ in range(2):
                    hidden = torch.randn(1, 3, 4, requires_grad=True)
                    weights = forward(hidden) if replay is None else checkpoint(forward, hidden,
                                                                              use_reentrant=replay)
                    weights.sum().backward()
                torch.testing.assert_close(member.tokens_per_expert, torch.tensor([2, 2, 0, 0]))
                monitor = MoEMonitorCallback(member)
                monitor.on_step_end()
                torch.testing.assert_close(member.bias, torch.tensor([-.01, -.01, .01, .01]), rtol=0, atol=0)
                self.assertFalse(hasattr(member, "bias_vl"))
                self.assertEqual(member.tokens_per_expert.count_nonzero(), 0)
                self.assertFalse(member.bias.requires_grad)
                self.assertNotIn("tokens_per_expert", member.state_dict())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_loss_scaling_and_packed_sequences_follow_explicit_objective(self) -> None:
        """Feature: Packed sequence auxiliary gradients and gradient scaling.
        Description: Bind a scaled task objective with explicit sequence metadata and masked padding.
        Expectation: Injected gradients equal the same scaled sum of task and sequence auxiliary losses.
        """
        member, reference = _router(moe_aux_loss_coeff=.03), _router()
        reference.weight.data.copy_(member.weight)
        hidden = torch.randn(4, 4, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        sequence_ids = torch.tensor([0, 0, 1, 1])
        mask = torch.tensor([True, False, True, True])
        with aux_loss_scale_context():
            weights, _ = _outputs(member, hidden, token_mask=mask, sequence_ids=sequence_ids, num_sequences=2)
            loss = weights.square().sum()
            bind_aux_loss_scale(loss)
            (loss * .125).backward()
        expected_weights, ids = _outputs(reference, other)
        scores = (other @ reference.weight.T).float().softmax(-1)
        auxiliary = sequence_load_balancing_loss(scores, ids, sequence_ids, 2, mask)
        ((expected_weights.square().sum() + .03 * auxiliary) * .125).backward()
        torch.testing.assert_close(hidden.grad, other.grad, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(member.weight.grad, reference.weight.grad, atol=1e-7, rtol=1e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_statistics_binding_and_metadata_slice(self) -> None:
        """Feature: Model-independent statistics mesh binding.
        Description: Bind distinct DP/CP/TP axes and TP-extended EP owners, then pass gathered metadata.
        Expectation: No duplicate EP reduction, complete replicated state, and the matching local TP mask.
        """
        axes = {name: SimpleNamespace(size=lambda: 2, get_group=lambda name=name: name,
                                      get_local_rank=lambda: 1) for name in ("dp", "cp", "tp")}

        class DeviceMesh(dict):
            """Existing named axes; no process group construction in this test."""
            mesh_dim_names = ("dp", "cp", "tp")

        mesh = SimpleNamespace(device_mesh=DeviceMesh(axes), sequence_parallel=False, ep_size=2,
                               fsdp_moe_mesh={"ep": SimpleNamespace(size=lambda: 4)})
        member = _router(moe_router_force_load_balancing=True, moe_router_bias_update_rate=.01)
        member.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        self.assertEqual(member.router_ep_size, 4)
        self.assertEqual(member.sequence_partition_groups, ("cp", "tp"))
        self.assertEqual(member.expert_bias_update_groups, ("dp", "cp", "tp"))
        sharding = router_sharding(module=member, module_fqn="gate", member=None, context=None)
        validate_parallel_module(member, member.parallel_module_spec(), sharding, None)
        sharding.params[next(iter(sharding.params))] = {"tp": Shard(0)}
        with self.assertRaisesRegex(ValueError, "replicated"):
            validate_parallel_module(member, member.parallel_module_spec(), sharding, None)
        weights, ids = _outputs(member, torch.randn(1, 2, 4),
                                token_mask=torch.tensor([[True, True, False, True]]))
        weights.sum().backward()
        torch.testing.assert_close(ids, torch.tensor([[0], [1]]))
        torch.testing.assert_close(member.tokens_per_expert, torch.tensor([0, 1, 0, 0]))
        with patch("hyper_parallel.components.modules.moe.moe_utils.dist.all_reduce") as reduce:
            sync_and_update_expert_bias(member)
        self.assertEqual([call.kwargs["group"] for call in reduce.call_args_list], ["dp", "cp", "tp"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_public_compatibility_adoption_and_fp32_state(self) -> None:
        """Feature: Public and compatibility imports plus source state adoption.
        Description: Adopt new family members with enabled balancing and convert model dtype.
        Expectation: Imports identify one factory; weights/counters/bias are retained and bias stays FP32.
        """
        self.assertIs(CanonicalGate, Gate)
        source = _router(moe_router_bias_update_rate=.01)
        target = Gate(module=source)
        self.assertIs(target.weight, source.weight)
        self.assertIs(target.tokens_per_expert, source.tokens_per_expert)
        self.assertIs(target.bias, source.bias)
        value = torch.tensor([.012345, -.345678, .456789, -.567891])
        target.bias.data.copy_(value)
        target.to(dtype=torch.bfloat16)
        self.assertEqual(target.bias.dtype, torch.float32)
        torch.testing.assert_close(target.bias, value, rtol=0, atol=0)
        transforms = target.make_transforms()
        names = {name for name, _ in target.named_parameters()}
        self.assertEqual({item.source_patterns[0] for item in transforms}, names)
        for item in transforms:
            self.assertEqual(item.source_patterns, item.target_patterns)
        target.reset_parameters()
        self.assertEqual(target.bias.count_nonzero(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_checkpoint_import_derives_missing_correction_bias(self) -> None:
        """Feature: Router checkpoint conversion.
        Description: Import a projection-only checkpoint without depending on a model router adapter.
        Expectation: Projection storage is preserved, new bias is zero, and resume keeps saved bias.
        """
        member = _router(moe_router_bias_update_rate=.001)
        projection = member.weight
        source = torch.randn_like(projection)
        operation = member.make_transforms()[0].operations[0]
        state = operation.convert({"weight": [source]}, ["weight"], ["weight", "bias"])
        self.assertIs(state["weight"], source)
        self.assertEqual(state["bias"].dtype, torch.float32)
        torch.testing.assert_close(state["bias"], torch.zeros(4), rtol=0, atol=0)
        member.load_state_dict(state, strict=True)
        self.assertIs(member.weight, projection)
        torch.testing.assert_close(member.weight, source, rtol=0, atol=0)
        state["bias"] = torch.full_like(member.bias, .25)
        member.load_state_dict(state, strict=True)
        torch.testing.assert_close(member.bias, state["bias"], rtol=0, atol=0)
        exported = operation.reverse_op.convert(state, ["weight", "bias"], ["weight"])
        self.assertEqual(set(exported), {"weight"})
        self.assertIs(exported["weight"], source)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_checkpoint_adoption_preserves_existing_correction_bias(self) -> None:
        """Feature: Router checkpoint conversion.
        Description: Adopt a family member whose checkpoint already owns correction state.
        Expectation: Identity conversions preserve the existing nonzero correction bias.
        """
        source = _router(moe_router_bias_update_rate=.001)
        source.bias.data.fill_(.25)
        member = Gate(module=source)
        self.assertTrue(member.source_has_correction_bias)
        for rule in member.make_transforms():
            self.assertEqual(rule.source_patterns, rule.target_patterns)
        member.load_state_dict(source.state_dict(), strict=True)
        torch.testing.assert_close(member.bias, source.bias, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_disabled_eval_and_invalid_configuration(self) -> None:
        """Feature: Opt-in router features and boundary validation.
        Description: Evaluate disabled/enabled gates and malformed masks, flags and fixed-route dimensions.
        Expectation: Defaults keep original math; evaluation has no training side effects and invalid inputs fail.
        """
        member = Gate(config=GateConfig(4, 4, 1, normalize_topk=False))
        hidden = torch.randn(3, 4)
        logits, weights, ids = member(hidden)
        scores = torch.sigmoid((hidden @ member.weight.T))
        expected_ids = scores.topk(1, sorted=False).indices
        torch.testing.assert_close(logits, (hidden @ member.weight.T), rtol=0, atol=0)
        torch.testing.assert_close(weights, scores.gather(1, expected_ids), rtol=0, atol=0)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        for settings in ({"moe_router_force_load_balancing": "false"}, {"moe_aux_loss_coeff": -1},
                         {"moe_router_bias_update_rate": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                _router(**settings)
        with self.assertRaisesRegex(ValueError, "divisible"):
            balanced_expert_indices(torch.zeros(3, 1, dtype=torch.int64), 4, 3)
        with self.assertRaisesRegex(ValueError, "token_mask"):
            member(hidden, token_mask=torch.ones(3))
        member = _router(moe_aux_loss_coeff=.1, moe_router_bias_update_rate=.01).eval()
        member(hidden)[1].sum().backward()
        self.assertIsNone(member.last_aux_loss)
        self.assertEqual(member.tokens_per_expert.count_nonzero(), 0)
        with self.assertRaisesRegex(TypeError, "model routers"):
            Gate(module=torch.nn.Linear(4, 4))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_materialization_restores_state_and_empty_auxiliary_is_zero(self) -> None:
        """Feature: Router training state materialization.
        Description: Materialize meta members and rebuild counts without altering checkpoint parameters.
        Expectation: Own biases reset to zero, buffers rebuild to zero, and empty auxiliary inputs are finite.
        """
        with torch.device("meta"):
            member = _router(moe_aux_loss_coeff=.1, moe_router_bias_update_rate=.01)
        self.assertTrue(member.bias.is_meta)
        member.to_empty(device="cpu")
        member.reset_parameters()
        self.assertEqual(member.bias.count_nonzero(), 0)
        member.bias.data.fill_(.123)
        member.tokens_per_expert.fill_(87)
        rebuild_materialized_state(member, MaterializationContext("checkpoint_load", torch.device("cpu")))
        self.assertEqual(member.tokens_per_expert.count_nonzero(), 0)
        torch.testing.assert_close(member.bias, torch.full((4,), .123), rtol=0, atol=0)
        weights, ids = _outputs(member, torch.empty(0, 4))
        self.assertEqual(weights.shape, (0, 1))
        self.assertEqual(ids.shape, (0, 1))
        self.assertEqual(member.last_aux_loss, 0)
        weights.sum().backward()
        self.assertEqual(member.tokens_per_expert.count_nonzero(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_int32_native_ids_accumulate_int64_counts(self) -> None:
        """Feature: Native fused gate ID compatibility.
        Description: Feed int32 IDs to the shared backward load accountant without modality metadata.
        Expectation: Scatter indices and updates use int64 while original routing weight gradients remain exact.
        """
        ids = torch.tensor([[0, 2], [0, 1]], dtype=torch.int32)
        weights = torch.randn(2, 2, requires_grad=True)
        counts = torch.zeros(4, dtype=torch.int64)
        accumulate_router_load(weights, ids, None, counts).sum().backward()
        torch.testing.assert_close(counts, torch.tensor([2, 1, 1, 0]))
        torch.testing.assert_close(weights.grad, torch.ones_like(weights), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mega_torch_variant_shares_training_semantics(self) -> None:
        """Feature: Gate execution variants share optional training features.
        Description: Compare ordinary and MegaGate CPU paths with fixed IDs and both balancing features.
        Expectation: Outputs, gradients and backward counts agree without importing the optional native backend.
        """
        config = GateConfig(4, 4, 2, scoring_func="sqrtsoftplus")
        context = {"module_configs": (SimpleNamespace(moe_router_force_load_balancing=True,
                    moe_aux_loss_coeff=.01, moe_router_bias_update_rate=.001),)}
        left = Gate(config=config, context=context)
        right = Gate(config=config, context=context, implementation="mega")
        right.load_state_dict(left.state_dict())
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        actual, expected = left(hidden), right(other)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        actual[1].square().sum().backward()
        expected[1].square().sum().backward()
        assert_precision(hidden.grad, other.grad, dtype=torch.float32, backward=True, label="router/input")
        assert_precision(left.weight.grad, right.weight.grad, dtype=torch.float32,
                         backward=True, label="router/weight")
        torch.testing.assert_close(left.tokens_per_expert, right.tokens_per_expert)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mega_vision_selection_keeps_standard_training_state(self) -> None:
        """Feature: MegaGate vision selection compatibility.
        Description: Compare visual selection with an explicit correction bias on the ordinary Gate.
        Expectation: Outputs and gradients match exactly; common load state stays [E] and bias_vl stays FP32.
        """
        config = GateConfig(4, 4, 2, scoring_func="sqrtsoftplus", vision_enabled=True)
        context = {"module_configs": (SimpleNamespace(moe_aux_loss_coeff=.01),)}
        reference = Gate(config=config, context=context)
        context = {"module_configs": (SimpleNamespace(moe_aux_loss_coeff=.01, moe_router_bias_update_rate=.001),)}
        member = Gate(config=config, context=context, implementation="mega")
        with torch.no_grad():
            member.weight.copy_(reference.weight)
            member.bias.copy_(torch.tensor([.1, -.3, .2, -.1]))
            member.bias_vl.copy_(torch.tensor([-.2, .3, -.1, .1]))
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        mask = torch.tensor([[True, False, True], [False, True, False]])
        correction = torch.where(mask.reshape(-1, 1), member.bias_vl, member.bias).detach()
        actual, expected = member(hidden, mask), reference(other, correction_bias=correction)
        for result, value in zip(actual, expected):
            torch.testing.assert_close(result, value, rtol=0, atol=0)
        actual[1].square().sum().backward()
        expected[1].square().sum().backward()
        assert_precision(hidden.grad, other.grad, dtype=torch.float32, backward=True, label="router/input")
        assert_precision(member.weight.grad, reference.weight.grad, dtype=torch.float32,
                         backward=True, label="router/weight")
        torch.testing.assert_close(member.tokens_per_expert, torch.bincount(actual[2].flatten(), minlength=4))
        biases = {name: getattr(member, name).detach().clone() for name in member.fp32_parameter_names}
        member.to(dtype=torch.bfloat16)
        for name, bias in biases.items():
            self.assertEqual(getattr(member, name).dtype, torch.float32)
            torch.testing.assert_close(getattr(member, name), bias, rtol=0, atol=0)
