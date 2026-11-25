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
"""Unit tests for the pipeline-schedule name resolver + shared-parameter info."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.core.pipeline_parallel.scheduler import ScheduleDualPipeV
from hyper_parallel.core.pipeline_parallel.stage import PipelineStage, SharedParameterInfo
from hyper_parallel.core.pipeline_parallel.utils import MetaStepType
from tests.common.mark_utils import arg_mark


class TestSharedParameterInfoReFetch(unittest.TestCase):
    """``SharedParameterInfo`` re-fetches the live parameter from its owner.

    Needed for the meta-init pipeline path, where the stage is materialized
    (``to_empty`` replaces the ``nn.Parameter`` object) *after* the shared info
    is created — a stale capture would have ``grad=None`` at sync time.
    """

    def test_refetches_current_param_from_owner(self):
        """With ``owner_module``/``param_name``, ``parameter`` tracks replacement."""
        module = nn.Linear(4, 4)
        original = module.weight
        info = SharedParameterInfo(module.weight, [0, 1], owner_module=module, param_name="weight")
        assert info.parameter is original, "should start at the captured parameter"
        # Simulate ``to_empty`` swapping the Parameter object in place.
        module.weight = nn.Parameter(torch.zeros(4, 4))
        assert info.parameter is module.weight, "must re-fetch the live parameter"
        assert info.parameter is not original, "must not return the stale parameter"

    def test_static_capture_without_owner(self):
        """Without ``owner_module`` the captured parameter object is returned as-is."""
        module = nn.Linear(4, 4)
        original = module.weight
        info = SharedParameterInfo(module.weight, [0, 1])
        module.weight = nn.Parameter(torch.zeros(4, 4))
        assert info.parameter is original, "no owner -> keep the captured parameter"


class TestSyncSharedParametersGradGuards(unittest.TestCase):
    """``sync_shared_parameters_grad`` guards and zero-fill semantics.

    A shared info whose materialization handshake has not completed has
    ``group=None`` and must skip the collective. A tied end whose backward has
    not run has ``grad=None`` but must still ENTER the collective with a zero
    contribution — skipping would desync the peers' matching ``all_reduce``.
    """

    @staticmethod
    def _make_info(grad, group, requires_grad=True):
        """Build a ``SharedParameterInfo`` with a given grad and group."""
        param = nn.Parameter(torch.zeros(2, 2), requires_grad=requires_grad)
        param.grad = grad
        info = SharedParameterInfo(param, [0, 1])
        info.group = group
        return info

    def _call(self, info):
        """Invoke the unbound method on a duck-typed stage that owns ``info``."""
        fake_stage = SimpleNamespace(_shared_parameters=[info], _has_backward=True)
        with patch("hyper_parallel.core.pipeline_parallel.stage.dist") as mock_dist:
            mock_dist.all_reduce.return_value = None
            PipelineStage.sync_shared_parameters_grad(fake_stage)
        return mock_dist

    def test_zero_fills_when_grad_is_none(self):
        """A ``None`` grad still enters the collective with a zero contribution."""
        info = self._make_info(grad=None, group="pp_group")
        mock_dist = self._call(info)
        torch.testing.assert_close(info.parameter.grad, torch.zeros_like(info.parameter))
        assert mock_dist.all_reduce.call_count == 1, \
            (f"a grad-less shared end must still all-reduce (zeros), "
             f"got {mock_dist.all_reduce.call_count} calls")

    def test_skips_when_group_is_none(self):
        """A ``None`` group (handshake not yet run) must not reach ``all_reduce``."""
        info = self._make_info(grad=torch.ones(2, 2), group=None)
        mock_dist = self._call(info)
        mock_dist.all_reduce.assert_not_called()

    def test_skips_when_frozen(self):
        """``requires_grad=False`` peers all skip (tied/frozen is group-wide)."""
        info = self._make_info(grad=None, group="pp_group", requires_grad=False)
        mock_dist = self._call(info)
        mock_dist.all_reduce.assert_not_called()

    def test_reduces_when_grad_and_group_ready(self):
        """A ready entry reduces its grad exactly once over its group."""
        grad = torch.ones(2, 2)
        info = self._make_info(grad=grad, group="pp_group")
        mock_dist = self._call(info)
        assert mock_dist.all_reduce.call_count == 1, \
            (f"ready shared grad should all-reduce once, "
             f"got {mock_dist.all_reduce.call_count} calls")
        reduced = mock_dist.all_reduce.call_args.args[0]
        assert reduced is grad, \
            (f"all_reduce should receive the live grad tensor, "
             f"got {type(reduced)}")

    def test_noncontiguous_gradient_copies_back_after_async_wait(self):
        """A native reduction preserves the original grad and waits before copying."""
        grad = torch.ones(2, 2).t()
        info = self._make_info(grad=grad, group=object())
        fake_stage = SimpleNamespace(_shared_parameters=[info], _has_backward=True)
        completed = []

        def all_reduce(tensor: torch.Tensor, group: object, async_op: bool) -> SimpleNamespace:
            """Delay the mock reduction until the returned work is waited."""
            self.assertIs(group, info.group)
            self.assertTrue(async_op)
            self.assertTrue(tensor.is_contiguous())
            self.assertIsNot(tensor, grad)

            def wait() -> None:
                """Finish the reduction while the original gradient is still untouched."""
                torch.testing.assert_close(grad, torch.ones_like(grad))
                tensor.mul_(2)
                completed.append(True)

            return SimpleNamespace(wait=wait)

        with patch("torch.distributed.all_reduce", side_effect=all_reduce):
            PipelineStage.sync_shared_parameters_grad(fake_stage)

        self.assertEqual(completed, [True])
        self.assertIs(info.parameter.grad, grad)
        torch.testing.assert_close(grad, torch.full_like(grad, 2))


class TestDualPipeVSchedule(unittest.TestCase):
    """Validate the V-shaped execution order and its local autograd boundary."""

    @staticmethod
    def _make_schedule(micro_batches: int = 4) -> ScheduleDualPipeV:
        schedule = object.__new__(ScheduleDualPipeV)
        schedule.real_stage_num = 2
        schedule._stage_num = 4  # pylint: disable=protected-access
        schedule.micro_batch_num = micro_batches
        schedule.exec_order = {}
        return schedule

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_each_microbatch_has_ordered_forward_and_backward(self) -> None:
        """Feature: DualPipeV execution order.
        Description: Construct two ranks with four virtual stages and four microbatches.
        Expectation: Each computation occurs once with receives before it and sends after it.
        """
        schedule = self._make_schedule()
        schedule.construct_exec_order()
        for rank, order in schedule.exec_order.items():
            owned_stages = (rank, 3 - rank)
            for stage in owned_stages:
                for micro in range(schedule.micro_batch_num):
                    indices = {
                        step.type: index
                        for index, step in enumerate(order)
                        if step.stage_index == stage and step.micro_index == micro
                    }
                    self.assertLess(indices[MetaStepType.FWD], indices[MetaStepType.BWD])
                    self.assertEqual(
                        sum(step.type == MetaStepType.FWD and step.stage_index == stage
                            and step.micro_index == micro for step in order),
                        1,
                    )
                    if stage not in (0, 3):
                        self.assertLess(indices[MetaStepType.FWD_RECV], indices[MetaStepType.FWD])
                    if stage not in (0, 3):
                        self.assertLess(indices[MetaStepType.BWD], indices[MetaStepType.BWD_SEND])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_local_fold_uses_direct_recv_and_send_steps(self) -> None:
        """Feature: DualPipeV local fold.
        Description: Inspect the last rank's middle edge and its adjacent operations.
        Expectation: Local handoff steps remain while distributed P2P is omitted on that edge.
        """
        schedule = self._make_schedule()
        order = schedule._construct_rank_exec_order(1)  # pylint: disable=protected-access
        fold_recv = [step for step in order if step.type == MetaStepType.FWD_RECV and step.stage_index == 2]
        fold_send = [step for step in order if step.type == MetaStepType.BWD_SEND and step.stage_index == 2]
        self.assertEqual(len(fold_recv), schedule.micro_batch_num)
        self.assertEqual(len(fold_send), schedule.micro_batch_num)
        self.assertFalse(any(step.type == MetaStepType.FWD_SEND and step.stage_index == 1 for step in order))
        self.assertFalse(any(step.type == MetaStepType.BWD_RECV and step.stage_index == 1 for step in order))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rejects_too_few_microbatches(self) -> None:
        """Feature: DualPipeV configuration validation.
        Description: Construct two ranks with fewer than two microbatches per rank.
        Expectation: Schedule construction raises a descriptive ValueError.
        """
        schedule = self._make_schedule(micro_batches=3)
        with self.assertRaisesRegex(ValueError, "at least 4 micro-batches"):
            schedule.construct_exec_order()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_local_fold_detaches_inputs_and_returns_gradients(self) -> None:
        """Feature: DualPipeV local autograd boundary.
        Description: Transfer one activation and one gradient between the two local halves.
        Expectation: The input is a detached leaf and the gradient reaches the source recv slot.
        """
        schedule = self._make_schedule()
        source = SimpleNamespace(stage_index=1, fwd_outputs_cache={}, grad_recv_info={})
        destination = SimpleNamespace(stage_index=2, args_recv_info={}, bwd_cache={})
        output = torch.randn(2, 3, requires_grad=True)
        source.fwd_outputs_cache[0] = (output,)
        destination.bwd_cache[0] = [torch.ones_like(output)]
        schedule.stages = [SimpleNamespace(stage_index=1), destination]
        schedule._stage_dict = {1: source, 2: destination}  # pylint: disable=protected-access

        with patch("hyper_parallel.core.pipeline_parallel.scheduler.dist.get_rank", return_value=1):
            schedule.recv_fwd(destination, 0)
            schedule.send_bwd(destination, 0)

        received = destination.args_recv_info[0][0].buffer
        self.assertIsNot(received, output)
        self.assertIsNone(received.grad_fn)
        self.assertTrue(received.requires_grad)
        torch.testing.assert_close(received, output)
        torch.testing.assert_close(source.grad_recv_info[0][0].buffer, torch.ones_like(output))


if __name__ == "__main__":
    unittest.main()
