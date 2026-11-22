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
"""Training features on new router families without migrating native model routers."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from examples.torch.module_family.config import GateConfig, TokenChoiceTopKRouterConfig
from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.losses.moe_aux_loss import sequence_load_balancing_loss
from hyper_parallel.components.modules import Gate, TokenChoiceTopKRouter
from hyper_parallel.components.modules.moe.router import Gate as CanonicalGate
from hyper_parallel.components.modules.moe.router import balanced_expert_indices
from hyper_parallel.components.modules.moe.moe_utils import (
    router_sharding, sync_and_update_expert_bias,
)
from hyper_parallel.components.modules.moe.router import TokenChoiceTopKRouterModule
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding, validate_parallel_module
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.trainer.callbacks.moe_monitor_callback import MoEMonitorCallback
from tests.common.mark_utils import arg_mark


def _router(kind: str, **settings):
    """Create either public protocol with common model-level training flags."""
    config = GateConfig(4, 4, 1, scoring_func="softmax", normalize_topk=False)
    factory = Gate
    if kind == "router":
        factory = TokenChoiceTopKRouter
        config = TokenChoiceTopKRouterConfig(4, 4, score_func="softmax")
    return factory(config=config, context={"module_configs": (SimpleNamespace(**settings),)})


def _outputs(member, hidden, **metadata):
    """Normalize the two public protocols solely for assertions."""
    result = member(hidden, **metadata)
    return result[:2] if isinstance(member, TokenChoiceTopKRouterModule) else result[1:]


def _weight(member):
    return member.gate.weight if isinstance(member, TokenChoiceTopKRouterModule) else member.weight


class TestRouterTraining(unittest.TestCase):
    """Verify actual-route accounting, independent composition and borrowed meshes."""

    def setUp(self) -> None:
        """Keep RNG and checkpoint device state local to CPU cases."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(38)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_auxiliary_loss_and_gradient_use_fixed_dispatch_ids(self) -> None:
        """Feature: Fixed routing and auxiliary loss on both new families.
        Description: Force learned expert zero, then compare post-fix loss/gradient with an explicit formula.
        Expectation: Learned weights stay unchanged while both auxiliary counts and gradients use fixed IDs.
        """
        for kind in ("gate", "router"):
            with self.subTest(kind=kind):
                member = _router(kind, router_fix_router=True, router_aux_loss_coeff=.07)
                reference = _router(kind)
                with torch.no_grad():
                    _weight(member).copy_(torch.eye(4) * 3)
                    _weight(reference).copy_(_weight(member))
                hidden = torch.tensor([[2., .3, .1, .1], [1., .2, .1, .1], [3., .1, .2, .1]], requires_grad=True)
                other = hidden.detach().clone().requires_grad_()
                expected_weights, learned_ids = _outputs(reference, other)
                weights, fixed_ids = _outputs(member, hidden)
                torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                self.assertTrue(torch.equal(learned_ids, torch.zeros_like(learned_ids)))
                self.assertTrue(torch.equal(fixed_ids, torch.tensor([[0], [1], [2]])))
                scores = (other @ _weight(reference).T).float().softmax(-1)
                fraction = torch.tensor([1/3, 1/3, 1/3, 0.])
                explicit_loss = 4 * (fraction * (scores / scores.sum(-1, keepdim=True)).mean(0)).sum()
                learned_loss = 4 * scores[:, 0].mean()
                torch.testing.assert_close(member.last_aux_loss, explicit_loss, rtol=1e-6, atol=1e-7)
                self.assertFalse(torch.isclose(member.last_aux_loss, learned_loss))
                (weights.sum() * 0).backward()
                (.07 * explicit_loss).backward()
                torch.testing.assert_close(hidden.grad, other.grad, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(_weight(member).grad, _weight(reference).grad, rtol=1e-6, atol=1e-7)
                if kind == "router":
                    torch.testing.assert_close(member(hidden)[2], torch.tensor([1, 1, 1, 0]))


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_loss_scaling_and_packed_sequences_follow_explicit_objective(self) -> None:
        """Feature: Packed sequence auxiliary gradients and gradient scaling.
        Description: Bind a scaled task objective with explicit sequence metadata and masked padding.
        Expectation: Injected gradients equal the same scaled sum of task and sequence auxiliary losses.
        """
        for kind in ("gate", "router"):
            with self.subTest(kind=kind):
                member, reference = _router(kind, router_aux_loss_coeff=.03), _router(kind)
                _weight(reference).data.copy_(_weight(member))
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
                scores = (other @ _weight(reference).T).float().softmax(-1)
                auxiliary = sequence_load_balancing_loss(scores, ids, sequence_ids, 2, mask)
                ((expected_weights.square().sum() + .03 * auxiliary) * .125).backward()
                torch.testing.assert_close(hidden.grad, other.grad, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(_weight(member).grad, _weight(reference).grad, atol=1e-7, rtol=1e-6)
