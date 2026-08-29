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
"""Real-constructor and ``run_with_dataiterator`` tests for ScheduleMPipeTranspose.

The ordering tests build schedules via ``object.__new__`` and the executor
tests build the executor directly, so neither exercises the real
``ScheduleMPipeTranspose.__init__`` (stage init, executor wiring, handler
registration) nor the ``run_with_dataiterator`` driver entry point.  These
tests cover both without a distributed runtime:

* Construction patches only the stage module's ``platform`` (the UT
  convention, see ``test_style.py``: no real process group in ``tests/ut``)
  so ``PipelineStage.init`` resolves its PP group single-process; everything
  else in the constructor chain is real.
* Execution runs at PP=1, where the schedule emits no P2P at all, so the
  full DATA_LOAD -> MPIPE_TRANSPOSE_FWD -> body forward/backward path runs
  on the real platform against a same-weights single-process reference.
"""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # noqa: E402  pylint: disable=wrong-import-position
from torch import nn  # noqa: E402  pylint: disable=wrong-import-position

import hyper_parallel.core.pipeline_parallel.stage as stage_module  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel import PipelineStage  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.core.pipeline_parallel.mpipe import (  # noqa: E402  pylint: disable=wrong-import-position
    MpipeStepType,
    ScheduleMPipeTranspose,
)
from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose import (  # noqa: E402  pylint: disable=wrong-import-position
    MPipeTransposeExecutor,
)

DIM = 8
VOCAB = 16
BATCH = 2
SEQ = 5
SEED = 1234

# Every handler ``_setup_mpipe_execution`` must register (one per MPIPE_* step).
_MPIPE_HANDLER_TYPES = frozenset({
    MpipeStepType.MPIPE_PARAM_BROADCAST,
    MpipeStepType.MPIPE_TRANSPOSE_FWD,
    MpipeStepType.MPIPE_FWD_SEND,
    MpipeStepType.MPIPE_FWD_RECV,
    MpipeStepType.MPIPE_GRAPH_SEND,
    MpipeStepType.MPIPE_GRAPH_RECV,
    MpipeStepType.MPIPE_TRANSPOSE_BWD,
    MpipeStepType.MPIPE_GRAD_SEND,
    MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD,
    MpipeStepType.MPIPE_GRAD_REDUCE,
})


def _stage_platform_patch(world_size):
    """Patch the stage module's ``platform`` for single-process construction.

    ``PipelineStage.init`` resolves rank / world size / the PP group through
    the stage module's ``platform`` binding; a real resolution would need
    ``init_process_group``, which ``tests/ut`` never does.  Scope the patch
    to construction only: the run path must see the real platform (tensor
    ops, sens building).
    """
    mock_plat = MagicMock()
    mock_plat.get_rank.return_value = 0
    mock_plat.get_world_size.return_value = world_size
    mock_plat.create_group.return_value = MagicMock(name="pp_group")
    return patch.object(stage_module, "platform", mock_plat)


class _FrozenTower(nn.Module):
    """Frozen embedding preprocess (the transposed visual-tower stand-in)."""

    def __init__(self) -> None:
        """Build the frozen embedding."""
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.embed.weight.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the integer input ids.

        Args:
            input_ids (Tensor): The integer token ids.
        """
        return self.embed(input_ids)


class _LossBody(nn.Module):
    """Single-stage body producing a scalar loss (first == last stage at PP=1)."""

    def __init__(self) -> None:
        """Build the body linear layer."""
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project the hidden state and reduce to a scalar loss.

        Args:
            hidden (Tensor): The preprocess (tower) output features.
        """
        return self.linear(hidden).pow(2).sum()


def _build_real_schedule(micro_batch_num, preprocess, submodule, stage_num=1,
                         **mpipe_kwargs):
    """Build a ScheduleMPipeTranspose through the REAL constructor chain.

    Returns ``(schedule, stage)``.  Only stage-side platform resolution is
    patched (see ``_stage_platform_patch``); the schedule, executor, and
    handler registration all run for real.
    """
    with _stage_platform_patch(world_size=stage_num):
        stage = PipelineStage(submodule, stage_index=0, stage_num=stage_num,
                              device=torch.device("cpu"))
        schedule = ScheduleMPipeTranspose(
            [stage],
            micro_batch_num=micro_batch_num,
            preprocess_module=preprocess,
            num_transpose_layers=2,
            **mpipe_kwargs)
    return schedule, stage


def _micro_batches(count, seed=SEED):
    """``count`` deterministic {input_ids, labels} micro-batch dicts."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator),
            "labels": torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator),
        }
        for _ in range(count)
    ]


def _count(order, step_type):
    """Count the steps of ``step_type`` in one rank's order."""
    return sum(1 for step in order if step is not None and step.type == step_type)


class TestMPipeTransposeRealConstructor(unittest.TestCase):
    """The real ``__init__``: stage init, executor wiring, handler registration."""

    def test_frozen_constructor_binds_executor_and_handlers(self):
        """
        Feature: ScheduleMPipeTranspose real construction (frozen tower).
        Description: Build the schedule at PP=4 / M=8 through the real
            constructor with a frozen preprocess.
        Expectation: the platform executor is bound, every MPIPE_* handler is
            registered, the frozen-tower flags and derived counts are set, and
            the built order carries the transpose prefix but no broadcast /
            stage-0-backward transport and no retain flag.
        """
        schedule, stage = _build_real_schedule(
            8, _FrozenTower(), _LossBody(), stage_num=4)
        # pylint: disable=protected-access
        assert isinstance(schedule._executor, MPipeTransposeExecutor)
        assert _MPIPE_HANDLER_TYPES.issubset(schedule._custom_fn_map.keys()), \
            f"missing handlers: {_MPIPE_HANDLER_TYPES - schedule._custom_fn_map.keys()}"
        assert schedule.requires_all_rank_input is True
        assert not schedule.has_trainable_preprocess and not schedule.owner_backward
        assert schedule.num_transpose_layers == 2
        assert schedule.num_transpose_micro_batches == 4  # min(PP=4, M=8)
        assert schedule._overlap_b_f is False, \
            "MPipe must force overlap_b_f off on the base schedule"
        assert schedule._DATA_KEYS == ("input_ids",)
        assert sorted(schedule.exec_order.keys()) == [0, 1, 2, 3]
        # Frozen: no broadcast, no stage-0-backward input transport, no retain.
        for order in schedule.exec_order.values():
            assert _count(order, MpipeStepType.MPIPE_PARAM_BROADCAST) == 0
            assert _count(order, MpipeStepType.MPIPE_GRAPH_SEND) == 0
            assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == 0
        assert not getattr(stage, "retain_backward_graph", False)
        # Round-robin ("full" default): rank 0 owns {0, 4} and receives the rest.
        rank0 = schedule.exec_order[0]
        assert _count(rank0, MetaStepType.DATA_LOAD) == 2
        assert _count(rank0, MpipeStepType.MPIPE_TRANSPOSE_FWD) == 2
        assert _count(rank0, MpipeStepType.MPIPE_FWD_RECV) == 6
        rank1 = schedule.exec_order[1]
        assert _count(rank1, MpipeStepType.MPIPE_FWD_SEND) == 2

    def test_trainable_constructor_emits_broadcast_and_retain_flag(self):
        """
        Feature: ScheduleMPipeTranspose real construction (trainable tower).
        Description: Build the schedule at PP=4 / M=4 with a trainable
            preprocess (stage-0 backward default).
        Expectation: the trainable flag is derived from the module, every rank
            broadcasts the tower params once, the stage-0-backward transport
            and backward steps are emitted, and the stage is flagged to retain
            its body-backward graph.
        """
        schedule, stage = _build_real_schedule(
            4, nn.Linear(DIM, DIM), _LossBody(), stage_num=4)
        assert schedule.has_trainable_preprocess
        assert not schedule.owner_backward  # opt-in only
        assert stage.retain_backward_graph is True
        for order in schedule.exec_order.values():
            assert _count(order, MpipeStepType.MPIPE_PARAM_BROADCAST) == 1
        assert _count(schedule.exec_order[1], MpipeStepType.MPIPE_GRAPH_SEND) == 1
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_GRAPH_RECV) == 3
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_TRANSPOSE_BWD) == 4

    def test_owner_backward_gating_through_real_constructor(self):
        """
        Feature: ScheduleMPipeTranspose owner-backward constructor gating.
        Description: Request ``owner_backward=True`` once with a trainable and
            once with a frozen preprocess (torch platform).
        Expectation: effective for the trainable tower (GRAD_REDUCE emitted on
            every rank, no stage-0-backward transport); silently ineffective
            for the frozen tower (falls back, no owner-backward steps).
        """
        trainable, _ = _build_real_schedule(
            4, nn.Linear(DIM, DIM), _LossBody(), stage_num=4, owner_backward=True)
        assert trainable.owner_backward is True
        for order in trainable.exec_order.values():
            assert _count(order, MpipeStepType.MPIPE_GRAD_REDUCE) == 1
            assert _count(order, MpipeStepType.MPIPE_GRAPH_SEND) == 0
        frozen, _ = _build_real_schedule(
            4, _FrozenTower(), _LossBody(), stage_num=4, owner_backward=True)
        assert frozen.owner_backward is False
        for order in frozen.exec_order.values():
            assert _count(order, MpipeStepType.MPIPE_GRAD_REDUCE) == 0

    def test_constructor_forwards_optional_knobs(self):
        """
        Feature: ScheduleMPipeTranspose constructor knob forwarding.
        Description: Pass ``less_memory``, ``output_consumer``,
            ``overflow_mode`` and a kwarg batch-dim spec through the real
            constructor.
        Expectation: ``less_memory`` reaches the interleaved base, the
            consumer is exposed via the property, the overflow mode drives
            ``owned_micros``, and the kwarg spec extends ``_DATA_KEYS``.
        """
        consumer = lambda ctx, micro, out: None  # noqa: E731  pylint: disable=unnecessary-lambda-assignment
        schedule, _ = _build_real_schedule(
            8, _FrozenTower(), _LossBody(), stage_num=4,
            less_memory=True, output_consumer=consumer, overflow_mode="min",
            kwargs_batch_dim={"attention_mask": 0})
        # pylint: disable=protected-access
        assert schedule._less_memory is True
        assert schedule.output_consumer is consumer
        # "min": rank 0 absorbs the overflow micros, other owners keep one.
        assert schedule.owned_micros(0) == frozenset({0, 4, 5, 6, 7})
        assert schedule.owned_micros(1) == frozenset({1})
        assert schedule._DATA_KEYS == ("input_ids", "attention_mask")


class TestMPipeTransposeRunWithDataIterator(unittest.TestCase):
    """``run_with_dataiterator`` at PP=1: the full local schedule run."""

    @staticmethod
    def _build_run_setup(micro_batch_num):
        """Seeded (schedule, tower, body, reference-copies) for a PP=1 run."""
        torch.manual_seed(SEED)
        tower = _FrozenTower()
        body = _LossBody()
        ref_tower = copy.deepcopy(tower)
        ref_body = copy.deepcopy(body)
        schedule, _ = _build_real_schedule(micro_batch_num, tower, body)
        return schedule, tower, body, ref_tower, ref_body

    def test_run_matches_reference_and_tracks_tokens(self):
        """
        Feature: run_with_dataiterator end-to-end (PP=1, frozen tower).
        Description: Run M=3 micro-batches pulled from a data iterator through
            the real schedule (DATA_LOAD -> MPIPE_TRANSPOSE_FWD -> body
            forward/backward), then compare against a same-weights
            single-process reference.
        Expectation: the returned per-micro losses and the body gradients
            match the reference, the frozen tower accumulates no gradient,
            ``last_local_tokens`` counts the valid (non-pad) shifted targets,
            and the iterator/dp-group state is bound on the schedule.
        """
        micro_batch_num = 3
        schedule, tower, body, ref_tower, ref_body = self._build_run_setup(micro_batch_num)
        data = _micro_batches(micro_batch_num)
        iterator = iter(data)
        dp_sentinel = object()

        losses = schedule.run_with_dataiterator(
            iterator, pp_fsdp_composed=False, dp_group_info=dp_sentinel)

        ref_losses = [ref_body(ref_tower(mb["input_ids"])) for mb in data]
        torch.stack(ref_losses).sum().backward()
        assert len(losses) == micro_batch_num
        for micro, (loss, ref_loss) in enumerate(zip(losses, ref_losses)):
            assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-6), \
                f"micro {micro} loss mismatch: {loss.item()} vs {ref_loss.item()}"
        for (name, param), (_, ref_param) in zip(body.named_parameters(),
                                                 ref_body.named_parameters()):
            assert param.grad is not None, f"body.{name} has no grad after the run"
            assert torch.allclose(param.grad, ref_param.grad, atol=1e-6), \
                f"body.{name} grad mismatch"
        assert all(p.grad is None for p in tower.parameters()), \
            "frozen tower must accumulate no gradient"
        # Shifted targets pad the last column with -100: BATCH*(SEQ-1) per micro.
        assert schedule.last_local_tokens == micro_batch_num * BATCH * (SEQ - 1)
        assert schedule.data_iterator is iterator
        # pylint: disable=protected-access
        assert schedule._pp_fsdp_composed is False
        assert schedule._dp_group_info is dp_sentinel
        with self.assertRaises(StopIteration):  # every micro was consumed
            next(iterator)

    def test_second_run_accumulates_grads_and_resets_token_count(self):
        """
        Feature: run_with_dataiterator repeated runs (grad accumulation).
        Description: Run the same M=2 data twice through one schedule (the
            trainer's accumulation pattern; the executor per-run reset must
            find its caches empty).
        Expectation: body gradients accumulate to twice the single-pass
            reference while ``last_local_tokens`` is per-run (reset to the
            single-pass count, not doubled).
        """
        micro_batch_num = 2
        schedule, _, body, ref_tower, ref_body = self._build_run_setup(micro_batch_num)
        data = _micro_batches(micro_batch_num)
        for _ in range(2):
            schedule.run_with_dataiterator(iter(data), pp_fsdp_composed=False)

        ref_losses = [ref_body(ref_tower(mb["input_ids"])) for mb in data]
        torch.stack(ref_losses).sum().backward()
        for (name, param), (_, ref_param) in zip(body.named_parameters(),
                                                 ref_body.named_parameters()):
            assert torch.allclose(param.grad, 2 * ref_param.grad, atol=1e-6), \
                f"body.{name} grad must accumulate over runs"
        assert schedule.last_local_tokens == micro_batch_num * BATCH * (SEQ - 1), \
            "last_local_tokens is per-run and must reset, not accumulate"

    def test_fsdp_composed_allreduces_token_count(self):
        """
        Feature: run_with_dataiterator PP+FSDP token accounting.
        Description: Run with ``pp_fsdp_composed=True``; the DATA_LOAD step
            must all-reduce each micro's valid-token count over the given DP
            group (patched ``torch.distributed.all_reduce`` doubles it,
            standing in for a 2-replica SUM).
        Expectation: ``last_local_tokens`` is the reduced (doubled) count and
            the all-reduce ran once per micro over the given group.
        """
        micro_batch_num = 2
        schedule, _, _, _, _ = self._build_run_setup(micro_batch_num)
        dp_group = MagicMock(name="dp_group")

        def _double(tensor, group=None, async_op=False):  # pylint: disable=unused-argument
            """SUM over 2 equal replicas: double in place."""
            tensor.mul_(2)

        with patch("torch.distributed.all_reduce", side_effect=_double) as mock_reduce:
            schedule.run_with_dataiterator(
                iter(_micro_batches(micro_batch_num)), pp_fsdp_composed=True,
                dp_group_info=SimpleNamespace(group=dp_group))
        assert schedule.last_local_tokens == 2 * micro_batch_num * BATCH * (SEQ - 1)
        assert mock_reduce.call_count == micro_batch_num
        assert all(call.kwargs.get("group") is dp_group
                   for call in mock_reduce.call_args_list)

    def test_error_path_drains_inflight_p2p(self):
        """
        Feature: run_with_dataiterator error-path P2P drain.
        Description: Feed an iterator that exhausts before every DATA_LOAD
            (mid-run error) with stale recv handles cached from a previous
            run's overlap.
        Expectation: the error propagates AND every cached handle is waited
            and popped (the ``finally`` drain contract), so no CommHandle is
            destroyed without ``wait()``.
        """
        schedule, _, _, _, _ = self._build_run_setup(3)
        fwd_handle = MagicMock(name="fwd_handle")
        bwd_handle = MagicMock(name="bwd_handle")
        schedule.fwd_handle_cache[(0, 0)] = [fwd_handle]
        schedule.bwd_handle_cache[(0, 0)] = [bwd_handle]

        with self.assertRaises(StopIteration):
            schedule.run_with_dataiterator(iter(_micro_batches(1)),
                                           pp_fsdp_composed=False)
        fwd_handle.wait.assert_called_once()
        bwd_handle.wait.assert_called_once()
        assert not schedule.fwd_handle_cache and not schedule.bwd_handle_cache, \
            "the finally drain must pop every cached handle"


if __name__ == "__main__":
    unittest.main()
