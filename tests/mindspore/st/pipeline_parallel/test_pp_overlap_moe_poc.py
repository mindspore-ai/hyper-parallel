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
from tests.mindspore.st.utils import msrun_case

PP_OVERLAP_MOE_POC = "pp_overlap_moe_poc.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
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
