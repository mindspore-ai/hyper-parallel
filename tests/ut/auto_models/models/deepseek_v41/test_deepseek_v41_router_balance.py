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
"""V4.1 modality balancing across backward, recomputation and optimizer steps."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.utils.moe_utils import MoEMonitorCallback, sync_and_update_expert_bias
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_router_aux_loss_wrapper,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.trainer.base import BaseTrainer
from tests.common.mark_utils import arg_mark


def make_router(rate: float = 0.001) -> DeepseekV41TopKRouter:
    """Make deterministic opposite routing preferences for text and images."""
    router = DeepseekV41TopKRouter(SimpleNamespace(
        hidden_size=4, num_local_experts=3, num_experts_per_tok=1,
        scoring_func="sigmoid", routed_scaling_factor=1.0, router_aux_loss_coef=0.0001,
        router_bias_update_rate=rate, v41_vision_enabled=True,
    ))
    with torch.no_grad():
        router.weight.fill_(0.1)
        router.bias.copy_(torch.tensor([1., 0., 0.]))
        router.bias_vl.copy_(torch.tensor([0., 0., 1.]))
    return router


class TestModalityRouterBalance(unittest.TestCase):
    """Use the real router and updater, mocking only distributed communication."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_checkpoint_and_accumulation_count_each_microbatch_once(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Both checkpoint modes preserve exact text/image counts across two backwards.
        Expectation: Both checkpoint modes accumulate exactly the same text/image counts as eager execution.
        """
        for mode in (None, False, True):
            with self.subTest(reentrant=mode):
                router = make_router()
                images = torch.tensor([[False, True, True, False]])
                valid = torch.tensor([[True, True, False, True]])

                def forward(hidden: torch.Tensor) -> torch.Tensor:
                    """Return the routing-weight carrier to the checkpoint wrapper."""
                    return router(hidden, images, valid)[1]

                for _ in range(2):
                    hidden = torch.ones(1, 4, 4, requires_grad=True)
                    before = router.tokens_per_expert.clone()
                    weights = forward(hidden) if mode is None else checkpoint(forward, hidden, use_reentrant=mode)
                    torch.testing.assert_close(router.tokens_per_expert, before)
                    weights.sum().backward()
                torch.testing.assert_close(router.tokens_per_expert, torch.tensor([[4, 0, 0], [0, 0, 2]]))
                self.assertFalse(router.tokens_per_expert.requires_grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_frozen_router_still_records_load_when_experts_train(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Load accounting is independent of whether the router parameters receive gradients.
        Expectation: Expert backward records loads without creating a router parameter gradient.
        """
        router = make_router()
        router.aux_loss_coeff = 0
        router.weight.requires_grad_(False)
        weights = router(torch.ones(1, 2, 4))[1]
        expert_output = torch.ones_like(weights, requires_grad=True)
        (expert_output * weights).sum().backward()
        torch.testing.assert_close(router.tokens_per_expert, torch.tensor([[2, 0, 0], [0, 0, 0]]))
        self.assertIsNone(router.weight.grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_independent_updates_empty_modality_and_reset(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Text/image overloads change different biases; a missing modality stays unchanged.
        Expectation: Text/image updates differ as expected; absent modalities stay unchanged and counts reset.
        """
        router = make_router(0.02)
        router.tokens_per_expert.copy_(torch.tensor([[8, 0, 0], [0, 0, 4]]))
        text_before, image_before = router.bias.clone(), router.bias_vl.clone()
        sync_and_update_expert_bias(router)
        torch.testing.assert_close(router.bias - text_before, torch.tensor([-0.02, 0.02, 0.02]))
        torch.testing.assert_close(router.bias_vl - image_before, torch.tensor([0.02, 0.02, -0.02]))
        torch.testing.assert_close(router.tokens_per_expert, torch.zeros(2, 3, dtype=torch.long))
        image_before = router.bias_vl.clone()
        router.tokens_per_expert[0].copy_(torch.tensor([2, 0, 0]))
        sync_and_update_expert_bias(router)
        torch.testing.assert_close(router.bias_vl, image_before)
        self.assertFalse(router.bias.requires_grad)
        self.assertFalse(router.bias_vl.requires_grad)
        self.assertNotIn("tokens_per_expert", router.state_dict())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_optimizer_step_updates_once_after_all_optimizers(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Trainer invokes the existing shared monitor once per logical update.
        Expectation: Two optimizers cause one bias update; a subsequent empty step causes no change.
        """
        router = make_router()
        router(torch.ones(1, 3, 4))[1].sum().backward()
        before = router.bias.clone()
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.optimizer = [Mock(), Mock()]
        trainer.lr_scheduler = None
        trainer.model_integration = Mock()
        trainer.moe_monitor = MoEMonitorCallback(router)
        trainer.step_optimizers_and_schedulers()
        torch.testing.assert_close(router.bias - before, torch.tensor([-0.001, 0.001, 0.001]))
        torch.testing.assert_close(router.tokens_per_expert, torch.zeros(2, 3, dtype=torch.long))
        trainer.step_optimizers_and_schedulers()
        torch.testing.assert_close(router.bias - before, torch.tensor([-0.001, 0.001, 0.001]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_checkpoint_materialization_resets_nonpersistent_counts(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Loading a checkpoint restores biases but starts a fresh step's counters.
        Expectation: Checkpoint biases survive and counters reset to integer zeros.
        """
        router = make_router()
        state = router.state_dict()
        router.to_empty(device="cpu")
        router.tokens_per_expert.fill_(123)
        router.load_state_dict(state)
        rebuild_materialized_state(router, MaterializationContext(reason="checkpoint_load", device=torch.device("cpu")))
        torch.testing.assert_close(router.tokens_per_expert, torch.zeros(2, 3, dtype=torch.long))
        torch.testing.assert_close(router.bias, state["bias"])
        router.bfloat16()
        self.assertEqual(router.tokens_per_expert.dtype, torch.long)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_model_owned_groups_override_generic_defaults(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: The adapter's DP and token-partition groups control both modalities.
        Expectation: Each model-provided group is reduced once and generic defaults are not used.
        """
        router = make_router()
        groups = (object(), object())
        router.expert_bias_update_groups = groups
        router.tokens_per_expert[0, 0] = 1
        with patch("hyper_parallel.core.utils.moe_utils.dist.all_reduce") as reduce_counts:
            sync_and_update_expert_bias(router, dp_group=object())
        self.assertEqual([call.kwargs["group"] for call in reduce_counts.call_args_list], list(groups))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_eval_disabled_and_invalid_update_rate(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Evaluation and disabled bias balancing never add loads.
        Expectation: Disabled/eval counts stay zero and invalid rates raise ValueError.
        """
        for training, rate in ((False, 0.001), (True, 0.0)):
            router = make_router(rate).train(training)
            router(torch.ones(1, 2, 4))[1].sum().backward()
            torch.testing.assert_close(router.tokens_per_expert, torch.zeros(2, 3, dtype=torch.long))
        for rate in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "router_bias_update_rate"):
                make_router(rate)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_tp_ep_metadata_alignment_and_bias_statistics_domain(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Slice sequence IDs and modality masks before the TP-local EP router.
        Expectation: Masks and IDs follow the TP shard; bias groups contain DP, CP and TP.
        """
        tp, cp, dp = Mock(), Mock(), Mock()
        for axis in (tp, cp, dp):
            axis.size.return_value = 2
        tp.get_local_rank.return_value = 1
        mesh = Mock()
        mesh.mesh_dim_names = ("dp", "cp", "tp")
        mesh.__getitem__ = Mock(side_effect={"dp": dp, "cp": cp, "tp": tp}.__getitem__)
        gate = Mock()
        module = SimpleNamespace(gate=gate, forward=Mock())
        requests = deepseek_v41_router_aux_loss_wrapper(module, mesh, tp, cp, Mock())
        ids = torch.tensor([[0, 0, 1, 2]])
        images = torch.tensor([[False, False, True, False]])
        requests[1].forward(torch.ones(1, 2, 4), image_mask=images, sequence_ids=ids, num_sequences=3)
        actual = gate.forward.call_args.kwargs
        torch.testing.assert_close(actual["sequence_ids"], ids[:, 2:])
        torch.testing.assert_close(actual["image_mask"], images[:, 2:])
        self.assertEqual(actual["sequence_partition_groups"], (cp.get_group(), tp.get_group()))
        self.assertEqual(requests[1].companion_attrs["expert_bias_update_groups"],
                         (dp.get_group(), cp.get_group(), tp.get_group()))
