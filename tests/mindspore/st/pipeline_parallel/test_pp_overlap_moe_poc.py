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
"""Pytest entry: MindSpore PP+EP+overlap end-to-end PoC.

4-card test — full integration of the comm/compute overlap stack
on MindSpore PyNative.  Smoke-validates that the schedule, callback,
CommComputeOverlap, OverlapExpertParallel, and AsyncCollectiveTensor
chain runs end-to-end without deadlock and produces non-zero grads.
"""
from tests.common.mark_utils import arg_mark  # pylint: disable=W0611
from tests.common.distributed_launcher import msrun_case

PP_OVERLAP_MOE_POC = "pp_overlap_moe_poc.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_end_to_end():
    """End-to-end PP+EP+overlap on MindSpore PyNative.

    Feature: Full MindSpore comm/compute overlap stack integration.
    Description:
        8 ranks, PP=4 × EP=2.  Interleaved 1F1B with B/F overlap +
        P2P overlap drives a 2-chunk × 2-layer Attention+MoE pipeline.
        OverlapExpertParallel wires MoE a2a through
        AsyncCollectiveTensor; CommComputeOverlap drives paired
        BWD+FWD threads via A/B/C/D sync hooks.
    Expectation:
        Iteration completes, grads non-zero on at least some params.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_end_to_end",
               12351, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute():
    """PP+EP+overlap + activation checkpoint via ``checkpoint_wrapper``.

    Feature: chunk-level activation checkpoint composing with overlap_b_f.
    Description:
        8 ranks, PP=4 × EP=2.  Same topology as
        :func:`test_pp_overlap_moe_end_to_end` but each chunk is wrapped
        with
        :func:`hyper_parallel.core.activation_checkpoint.checkpoint_wrapper`.
        The chunk's forward re-run is fired serially before the paired
        backward by :meth:`PipelineStage.recompute_one_chunk` and reused
        during backward, so the re-run never races the FWD thread's forward
        record (MS PyNative does not support concurrent FWD-record +
        BWD-replay) and only the grad phase overlaps with the FWD thread.
    Expectation:
        Iteration completes without deadlock and yields non-zero grads.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute",
               12353, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute_per_layer():
    """Per-layer (multi-segment) activation checkpoint under overlap_b_f.

    Feature: per-layer (multi-segment) recompute composing with overlap_b_f.
    Description:
        8 ranks, PP=4 × EP=2.  Each chunk wraps EACH layer in its own
        ``checkpoint`` segment (``_MoEChunk.enable_per_layer_recompute``), so
        backward reuses several pre-fired re-runs.  The re-runs are fired
        serially before the paired backward by
        ``PipelineStage.recompute_one_chunk``.  Compares overlap+recompute
        grads against a sync baseline.
    Expectation:
        No deadlock; grads and last-rank losses match the sync baseline
        within ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute_per_layer",
               12355, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute_mixed():
    """Mixed per-layer recompute (some layers recompute, some don't) under overlap_b_f.

    Feature: mixed per-layer recompute — the case op-granularity SAC cannot
        express.
    Description:
        8 ranks, PP=4 × EP=2, 2 layers per chunk.  Only layer 0 of each chunk
        is recomputed; layer 1 runs directly (its activations are kept).  The
        recomputed layer's re-run is fired serially before the paired backward
        by ``PipelineStage.recompute_one_chunk`` and reused during backward.
        Compares grads against a sync baseline.
    Expectation:
        No deadlock; grads and last-rank losses match the sync baseline
        within ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute_mixed",
               12356, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_accuracy():
    """Numerical equivalence of overlap stack vs sync baseline.

    Feature: Accuracy check for the MindSpore PP+EP+overlap stack.
    Description:
        4 ranks, PP=2 × EP=2.  Builds the same Attention+MoE pipeline
        twice with identical seed and identical input — once with the
        overlap stack (``OverlapExpertParallel(overlap)`` +
        ``overlap_p2p=True`` + ``overlap_b_f=True``) and once with
        the sync baseline (``OverlapExpertParallel(overlap=None)``,
        both overlap flags off).  Compares per-micro-batch losses on
        the last PP rank and per-parameter gradients on every rank.
    Expectation:
        Losses and grads match within ``rtol=1e-3, atol=1e-3``.  A
        mismatch typically means the lazy ``AsyncCollectiveTensor``
        wait is dropping a comm, the dual-thread callback corrupted
        autograd state, or the reordered permute/unpermute changed
        numerics.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_accuracy",
               12352, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_variable_layers():
    """Heterogeneous per-chunk layer counts under overlap_b_f.

    Feature: ``HookCoordinator.depart`` graceful one-party-left drain.
    Description:
        8 ranks, PP=4 × EP=2.  Each PP rank holds 2 interleaved chunks
        with ``[3, 2]`` MoE layers (NOT uniform), so every OVERLAP_B_F
        step in the 1F1B steady state pairs a BWD and FWD chunk of
        different layer counts.  The dual-thread rendezvous counts then
        differ by a full layer; without ``depart`` the shorter chunk's
        thread returns and the longer chunk's thread blocks forever on
        the 2-party barrier.  Compares overlap vs sync-baseline grads.
    Expectation:
        Iteration completes without deadlock; grads and per-micro-batch
        losses match the sync baseline within ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_variable_layers",
               12354, worker_num=8, local_worker_num=8)


#@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
#          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_dxdw_accuracy():
    """Numerical equivalence of the dx/dw split vs the sync baseline.

    Feature: Accuracy check for the ``OVERLAP_B_F`` dx/dw split.
    Description:
        8 ranks, PP=4 × EP=2.  Builds the same Attention+MoE pipeline
        twice with identical seed and identical input — a sync baseline
        with the overlap stack OFF (the same ground truth as
        ``test_pp_overlap_moe_accuracy``) and the dx/dw split path
        (``enable_dxdw_split=True``: each ``OVERLAP_B_F`` pairs
        ``(BWD_INPUT, FWD)`` and the matching ``BWD_WEIGHT`` runs after the
        pair's P2P gap, so the grad send issues before dw).  Each of
        ``NUM_STEPS`` steps feeds the same input to both and asserts
        equivalence — the comparison is mandatory.  Compares per-micro-batch
        losses on the last PP rank and per-parameter gradients on every rank.
    Expectation:
        Losses and grads match within ``rtol=1e-3, atol=1e-3``.  A
        mismatch typically means: ``backward_input_one_chunk`` did not
        write ``bwd_cache`` (so the scheduler's ``BWD_SEND`` sends a stale
        grad), dw's ``grad_fn`` lost intermediates after dx, or the overlap
        path diverged from the sync baseline.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_dxdw_accuracy",
               12357, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute_save_a2a():
    """Chunk recompute that keeps (does not recompute) the EP all-to-all.

    Feature: op-granularity selective recompute (SAC ``MUST_SAVE`` for the a2a)
        composing with overlap_b_f.
    Description:
        8 ranks, PP=4 x EP=2.  Each chunk's forward saves every EP all-to-all
        output; the serial re-forward fired by ``recompute_one_chunk`` restores
        them instead of re-issuing the HCCL all-to-all, while every other op is
        recomputed.  This is the only path that recomputes a layer's compute
        while keeping its a2a (layer-granularity recompute re-runs the a2a).
        Compares overlap+save-a2a grads against a sync baseline and checks the
        a2a went through SAC's save+restore path (not re-communicated).
    Expectation:
        No deadlock; grads and last-rank losses match the sync baseline within
        ``rtol=1e-3, atol=1e-3``; the a2a is classified in both the forward and
        the recompute phase.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute_save_a2a",
               12358, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute_save_a2a_dxdw():
    """Save-a2a chunk recompute combined with the dx/dw split.

    Feature: ``enable_dxdw_split`` x activation checkpoint.
    Description:
        8 ranks, PP=4 x EP=2, same SAC save-a2a policy as
        ``test_pp_overlap_moe_recompute_save_a2a`` but with the schedule-level
        dx/dw split enabled.  The split backward halves must enter the chunk's
        pre-fired recompute session on their own thread; otherwise dx's unpack
        misses the cache and lazily re-runs the chunk forward on the BWD
        thread, deadlocking against the paired forward's overlap hooks.
    Expectation:
        No hang; the a2a is SAC-saved and restored exactly once (the pre-fired
        recompute); grads and last-rank losses match the sync baseline within
        ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute_save_a2a_dxdw",
               12366, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_accuracy_batch_p2p():
    """Same-peer duplex P2P batching — numerical equivalence vs sync baseline.

    Feature: ``p2p_transport="batch"`` — ``coalesce_p2p`` merges each same-peer
        send+recv into one ``batch_isend_irecv`` (TX||RX duplex); both endpoints
        batched, matched per-peer FIFO.
    Description:
        8 ranks, PP=4 x EP=2.  Overlap stack built with
        ``p2p_transport="batch"`` compared against the plain sync baseline.
        Coalescing only regroups the launch, so numerics must be unchanged.
    Expectation:
        No ``HcclBatchISendIRecv`` EI0005, no deadlock; losses and grads match
        the sync baseline within ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_accuracy_batch_p2p",
               12359, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_accuracy_multi_stream_p2p() -> None:
    """Multi-stream PP groups — numerical equivalence and deadlock smoke test.

    Feature: ``p2p_transport="multi_stream"`` keeps same-peer duplex batching while
        routing each adjacent PP rank pair through its own two-rank group.
    Description:
        Launch the distributed overlap PoC using peer-specific P2P groups.
    Expectation:
        No group-init or P2P matching deadlock; losses and grads match the sync
        baseline within ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_accuracy_multi_stream_p2p",
               12367, worker_num=8, local_worker_num=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_accuracy_boundary():
    """fwd-boundary batching (EXPERIMENTAL opt-in) — equivalence vs baseline.

    Feature: ``p2p_transport="boundary"``: the overlap's ``F_SEND`` (payload
        ready when the forward finishes; the backward is the long pole) plus
        the next slot's recvs are issued mid-overlap by the stage's
        after-forward hook via ``exec_boundary_p2p``, so the activation send
        leaves ~half a slot early and no send rides a compute-gating recv
        handle (a2a-friendly).  The auto default under overlap_b_f is the
        measured-beneficial duplex "batch"; passing here is the prerequisite
        for ever promoting boundary.
    Description:
        8 ranks, PP=4 x EP=2.  Overlap stack built with
        ``p2p_transport="boundary"`` compared against the plain sync baseline.
    Expectation:
        No EI0005, no hang; losses and grads match the sync baseline within
        ``rtol=1e-3, atol=1e-3``.
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_accuracy_boundary",
               12365, worker_num=8, local_worker_num=8)
