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


#@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
#          card_mark="allcards", essential_mark="essential")
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


#@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
#          card_mark="allcards", essential_mark="essential")
def test_pp_overlap_moe_recompute():
    """End-to-end smoke check: PP+EP+overlap + ``ms.recompute(use_reentrant=False)``.

    Feature: HookCoordinator recompute bypass via context_fn.
    Description:
        8 ranks, PP=4 × EP=2.  Same topology as
        :func:`test_pp_overlap_moe_end_to_end` but every chunk is
        wrapped in ``ms.recompute(use_reentrant=False, context_fn=
        coord.recompute_context_fn)``.  The ``context_fn`` brackets
        the backward-time forward re-run with
        ``set_recomputing(True/False)`` so the re-run's A/B/C/D and
        CHUNK_START/CHUNK_END hooks become no-ops and the BWD-paired
        rendezvous count stays balanced.
    Expectation:
        Iteration completes without deadlock, grads non-zero on at
        least some params.  A deadlock here usually means the
        recompute bypass is not bracketing the re-run (e.g. someone
        passed ``use_reentrant=True``, which is the MindSpore default
        and is *not* hookable from Python).
    """
    msrun_case(3, PP_OVERLAP_MOE_POC, "test_pp_overlap_moe_recompute_smoke",
               12353, worker_num=8, local_worker_num=8)


#@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
#          card_mark="allcards", essential_mark="essential")
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
