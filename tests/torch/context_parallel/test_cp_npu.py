# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Pytest entry-points for redesigned ContextParallel NPU tests.

Each test spawns worker processes via torchrun and delegates to the
corresponding test function in cp_npu_test.py (NPU/hccl).

Port allocation:
  13000-13099  2-card tests  (Groups 1, 2, 4, 5, 6 where cp=2)
  13100-13199  4-card tests  (Group 3, and 4-card tests in Groups 5, 6)
"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

_FILE = "cp_npu_test.py"


# ---------------------------------------------------------------------------
# Group 1: Pure Ulysses (2-card)
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bnsd_fa_noncausal():
    """U1: Pure Ulysses CP=2, BNSD, npu_fusion_attention non-causal.

    Feature: ContextParallel Ulysses mode
    Description: Two ranks each hold a local Q/K/V slice in BNSD layout;
        ATA scatters sequence and gathers heads so full-head attention is computed
        locally, then the reverse ATA restores the sequence-sharded output.
    Expectation: CP output on each rank matches the corresponding slice of the
        single-card reference output.
    """
    torchrun_case(_FILE, "test_ulysses_bnsd_fa_noncausal", master_port=13000, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bnsd_fa_causal_explicit():
    """U2: Pure Ulysses CP=2, BNSD, npu_fusion_attention causal (sparse_mode=0 explicit mask).

    Feature: ContextParallel Ulysses causal attention with explicit mask
    Description: Two ranks run Ulysses ATA on a BNSD tensor; the causal mask is
        passed as a full S×S boolean matrix with sparse_mode=0 so the operator
        applies the exact user-specified mask pattern.
    Expectation: CP output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_ulysses_bnsd_fa_causal_explicit", master_port=13001, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bnsd_fa_causal_leftup():
    """U2b: Pure Ulysses CP=2, BNSD, npu_fusion_attention causal (sparse_mode=2 leftUpCausal).

    Feature: ContextParallel Ulysses causal attention with compressed causal mask
    Description: Two ranks run Ulysses ATA on a BNSD tensor using sparse_mode=2
        (leftUpCausal compressed 2048×2048 mask) to express the causal constraint.
    Expectation: CP output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_ulysses_bnsd_fa_causal_leftup", master_port=13002, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bsnd_fa_noncausal():
    """U3: Pure Ulysses CP=2, BSND, npu_fusion_attention non-causal.

    Feature: ContextParallel Ulysses mode with BSND layout
    Description: Two ranks each hold a local Q/K/V slice in BSND layout;
        the Ulysses ATA operates on the sequence dimension to redistribute heads.
    Expectation: CP output on each rank matches the corresponding slice of the
        single-card reference output.
    """
    torchrun_case(_FILE, "test_ulysses_bsnd_fa_noncausal", master_port=13003, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bnsd_sdpa_noncausal():
    """U4: Pure Ulysses CP=2, BNSD, F.scaled_dot_product_attention non-causal.

    Feature: ContextParallel Ulysses mode with PyTorch SDPA operator
    Description: Two ranks run Ulysses ATA using F.scaled_dot_product_attention
        instead of npu_fusion_attention, verifying that the dispatcher correctly
        intercepts the SDPA call for DTensor inputs.
    Expectation: CP SDPA output matches single-card reference.
    """
    torchrun_case(_FILE, "test_ulysses_bnsd_sdpa_noncausal", master_port=13004, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_bnsd_sdpa_causal():
    """U5: Pure Ulysses CP=2, BNSD, F.scaled_dot_product_attention causal (is_causal=True).

    Feature: ContextParallel Ulysses causal attention with SDPA
    Description: Two ranks run Ulysses ATA on BNSD tensors with is_causal=True;
        the dispatcher adjusts the per-rank causal mask offset automatically.
    Expectation: CP causal SDPA output matches single-card causal reference.
    """
    torchrun_case(_FILE, "test_ulysses_bnsd_sdpa_causal", master_port=13005, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_tnd_fa_noncausal():
    """U6: Pure Ulysses CP=2, TND format, npu_fusion_attention non-causal.

    Feature: ContextParallel Ulysses mode with TND (token-num-dim) layout
    Description: Two ranks run Ulysses ATA on TND-layout Q/K/V tensors,
        exercising the non-standard token-first sequence format.
    Expectation: CP output on each rank matches the corresponding TND slice of
        the single-card reference output.
    """
    torchrun_case(_FILE, "test_ulysses_tnd_fa_noncausal", master_port=13006, num_proc=2)


# ---------------------------------------------------------------------------
# Group 2: Pure Colossal AI (2-card)
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bsh_fa_noncausal():
    """C1: Pure Colossal CP=2, BSH, npu_fusion_attention non-causal.

    Feature: ContextParallel Colossal AI mode with BSH layout
    Description: Two ranks each hold a local Q sequence shard in BSH layout;
        K/V are all-gathered to give each rank full context before attention.
    Expectation: CP output on each rank matches the corresponding sequence slice
        of the single-card reference output.
    """
    torchrun_case(_FILE, "test_colossal_bsh_fa_noncausal", master_port=13010, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_sbh_fa_noncausal():
    """C2: Pure Colossal CP=2, SBH, npu_fusion_attention non-causal.

    Feature: ContextParallel Colossal AI mode with SBH layout
    Description: Two ranks each hold a local Q sequence shard in SBH layout
        (sequence-first); K/V are all-gathered before each rank computes attention.
    Expectation: CP output on each rank matches the corresponding SBH sequence
        slice of the single-card reference output.
    """
    torchrun_case(_FILE, "test_colossal_sbh_fa_noncausal", master_port=13011, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bsnd_fa_noncausal():
    """C3: Pure Colossal CP=2, BSND, npu_fusion_attention non-causal.

    Feature: ContextParallel Colossal AI mode with BSND layout
    Description: Two ranks each hold a local Q slice in BSND layout; K/V are
        all-gathered so each rank attends over the full sequence.
    Expectation: CP output on each rank matches the corresponding BSND sequence
        slice of the single-card reference output.
    """
    torchrun_case(_FILE, "test_colossal_bsnd_fa_noncausal", master_port=13012, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bsnd_fa_causal_leftup():
    """C3b: Pure Colossal CP=2, BSND, npu_fusion_attention causal (sparse_mode=2).

    Feature: ContextParallel Colossal AI causal attention with sparse_mode=2
    Description: Two ranks run Colossal AI CP on BSND tensors using sparse_mode=2
        (leftUpCausal). The dispatcher adjusts pre/next_tokens and converts
        sparse_mode=2→4 per rank to compute the correct causal window.
    Expectation: CP causal output matches the single-card causal reference for
        any sequence length, not just the compressed-mask fixed size of 2048.
    """
    torchrun_case(_FILE, "test_colossal_bsnd_fa_causal_leftup", master_port=13013, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bnsd_fa_noncausal():
    """C4: Pure Colossal CP=2, BNSD, npu_fusion_attention non-causal.

    Feature: ContextParallel Colossal AI mode with BNSD layout
    Description: Two ranks each hold a local Q slice in BNSD layout; K/V are
        all-gathered so each rank attends over the full sequence length.
    Expectation: CP output on each rank matches the corresponding BNSD sequence
        slice of the single-card reference output.
    """
    torchrun_case(_FILE, "test_colossal_bnsd_fa_noncausal", master_port=13014, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bnsd_fa_causal_explicit():
    """C5: Pure Colossal CP=2, BNSD, npu_fusion_attention causal (sparse_mode=0, offset mask).

    Feature: ContextParallel Colossal AI causal attention with explicit offset mask
    Description: Two ranks run Colossal AI CP on BNSD tensors; the causal mask is
        constructed as a full S×S boolean matrix with per-rank sequence offset
        applied, using sparse_mode=0 for exact mask matching.
    Expectation: CP causal output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_colossal_bnsd_fa_causal_explicit", master_port=13015, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bnsd_fa_causal_leftup():
    """C5b: Pure Colossal CP=2, BNSD, npu_fusion_attention causal (sparse_mode=2).

    Feature: ContextParallel Colossal AI causal attention with sparse_mode=2 in BNSD
    Description: Two ranks run Colossal AI CP on BNSD tensors using sparse_mode=2
        (leftUpCausal). Verifies the dispatcher adjusts pre/next_tokens correctly
        for the BNSD layout variant.
    Expectation: CP causal output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_colossal_bnsd_fa_causal_leftup", master_port=13016, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bnsd_sdpa_noncausal():
    """C6: Pure Colossal CP=2, BNSD, F.scaled_dot_product_attention non-causal.

    Feature: ContextParallel Colossal AI mode with PyTorch SDPA operator
    Description: Two ranks run Colossal AI CP using F.scaled_dot_product_attention;
        K/V are all-gathered before the SDPA call so each rank attends over the
        full sequence.
    Expectation: CP SDPA output matches single-card reference.
    """
    torchrun_case(_FILE, "test_colossal_bnsd_sdpa_noncausal", master_port=13017, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_bnsd_sdpa_causal():
    """C7: Pure Colossal CP=2, BNSD, F.sdpa causal with global offset mask.

    Feature: ContextParallel Colossal AI causal attention with SDPA and offset mask
    Description: Two ranks run Colossal AI CP using F.scaled_dot_product_attention
        with a causal mask that incorporates per-rank global sequence offset, so
        each rank's local Q attends correctly over the gathered K/V.
    Expectation: CP causal SDPA output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_colossal_bnsd_sdpa_causal", master_port=13018, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_colossal_tnd_fa_causal():
    """C8: Pure Colossal CP=2, TND format, npu_fusion_attention causal (sparse_mode=3).

    Feature: ContextParallel Colossal AI causal attention in TND format
    Description: Two ranks run Colossal AI CP on TND tensors with sparse_mode=3
        (rightDownCausal). The dispatcher adjusts pre/next_tokens per rank for
        the correct causal window in TND format.
    Expectation: CP causal TND output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_colossal_tnd_fa_causal", master_port=13020, num_proc=2)


# ---------------------------------------------------------------------------
# Group 3: Hybrid CP (4-card)
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_hybrid_bnsd_fa_noncausal():
    """H1: Hybrid CP=4 (ds=2, co=2), BNSD, npu_fusion_attention non-causal.

    Feature: ContextParallel Hybrid mode combining Ulysses and Colossal AI
    Description: Four ranks form a 2-D mesh (co=2, ds=2); Ulysses ATA scatters
        heads on ds-submesh while Colossal AI all-gathers K/V on co-submesh,
        combining both strategies in a BNSD non-causal setting.
    Expectation: Hybrid CP output matches the single-card reference.
    """
    torchrun_case(_FILE, "test_hybrid_bnsd_fa_noncausal", master_port=13100, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_hybrid_bnsd_fa_causal_leftup():
    """H2: Hybrid CP=4 (ds=2, co=2), BNSD, npu_fusion_attention causal (sparse_mode=2).

    Feature: ContextParallel Hybrid causal attention with sparse_mode=2
    Description: Four ranks use hybrid CP (ds=2, co=2) in BNSD layout with
        sparse_mode=2 (leftUpCausal). Both Ulysses head-scatter and Colossal AI
        K/V all-gather are combined, with per-rank causal mask adjustment.
    Expectation: Hybrid CP causal output matches the single-card causal reference.
    """
    torchrun_case(_FILE, "test_hybrid_bnsd_fa_causal_leftup", master_port=13101, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_hybrid_bnsd_sdpa_noncausal():
    """H3: Hybrid CP=4 (ds=2, co=2), BNSD, F.sdpa non-causal with single-card baseline.

    Feature: ContextParallel Hybrid mode with PyTorch SDPA operator
    Description: Four ranks use hybrid CP (ds=2, co=2) in BNSD layout with
        F.scaled_dot_product_attention. Verifies that both the DTensor-based
        SDPA dispatcher and the hybrid mesh routing produce correct results.
    Expectation: Hybrid CP SDPA output matches the single-card reference.
    """
    torchrun_case(_FILE, "test_hybrid_bnsd_sdpa_noncausal", master_port=13102, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_hybrid_tnd_fa_causal():
    """H4: Hybrid CP=4 (ds=2, co=2), TND format, npu_fusion_attention causal (sparse_mode=3).

    Feature: ContextParallel Hybrid causal attention in TND format
    Description: Four ranks use hybrid CP (ds=2, co=2) in TND layout with
        sparse_mode=3 (rightDownCausal). Ulysses ATA and Colossal AI K/V
        all-gather are combined for token-format sequences.
    Expectation: Hybrid CP causal TND output matches the single-card reference.
    """
    torchrun_case(_FILE, "test_hybrid_tnd_fa_causal", master_port=13103, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_hybrid_2d_mesh_equiv():
    """H5: Hybrid CP=4 with 2D mesh — numerical result matches 1D mesh and single-card ref.

    Feature: ContextParallel 2D mesh equivalence with 1D mesh
    Description: Verifies that using an explicit 2-D DeviceMesh for hybrid CP
        produces numerically identical results to a 1-D mesh and to the single-card
        reference, confirming mesh topology does not affect correctness.
    Expectation: All three outputs (2D mesh, 1D mesh, single-card) are equal.
    """
    torchrun_case(_FILE, "test_hybrid_2d_mesh_equiv", master_port=13104, num_proc=4)


# ---------------------------------------------------------------------------
# Group 4: Load Balancing (2-card)
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_lb_bsnd_fa_causal():
    """L1: Q-exchange LB (BSND, sparse_mode=2) == plain Colossal causal.

    Feature: ContextParallel load balancing for causal attention in BSND
    Description: Two ranks use the Q-exchange head-tail load balancing strategy
        in BSND layout with sparse_mode=2 causal mask. Verifies that balanced
        token distribution produces the same result as unbalanced Colossal AI CP.
    Expectation: Load-balanced CP output matches plain Colossal AI causal output.
    """
    torchrun_case(_FILE, "test_lb_bsnd_fa_causal", master_port=13030, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_lb_bnsd_fa_causal():
    """L2: Q-exchange LB (BNSD, sparse_mode=0 offset mask) == plain Colossal causal.

    Feature: ContextParallel load balancing for causal attention in BNSD
    Description: Two ranks use the Q-exchange head-tail load balancing strategy
        in BNSD layout with sparse_mode=0 explicit offset causal mask. Verifies
        the balanced distribution equals the plain Colossal AI causal result.
    Expectation: Load-balanced CP output matches plain Colossal AI causal output.
    """
    torchrun_case(_FILE, "test_lb_bnsd_fa_causal", master_port=13031, num_proc=2)


# ---------------------------------------------------------------------------
# Group 5: AsyncContextParallel NPU
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_async_ulysses_forward_npu():
    """A1: AsyncContextParallel Ulysses CP=2 forward output matches single-card on NPU.

    Feature: AsyncContextParallel overlap of projection GEMM with all-to-all
    Description: Two ranks use AsyncContextParallel in Pure Ulysses mode (cp=2).
        Projection post-hooks launch async A2A; the attention pre-hook waits and
        completes the head scatter before the attention operator runs.
    Expectation: Async CP forward output matches the synchronous single-card reference.
    """
    torchrun_case(_FILE, "test_async_ulysses_forward_npu", master_port=13200, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_async_ulysses_backward_npu():
    """A2: AsyncContextParallel Ulysses CP=2 backward gradients match single-card ref on NPU.

    Feature: AsyncContextParallel backward gradient correctness
    Description: Two ranks use AsyncContextParallel in Pure Ulysses mode; the
        backward pass launches async head→seq A2A inside the autograd function and
        waits it in projection backward pre-hooks to overlap with proj GEMM.
    Expectation: Gradients for Q/K/V projection weights match the single-card
        reference gradients.
    """
    torchrun_case(_FILE, "test_async_ulysses_backward_npu", master_port=13201, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_async_hybrid_forward_npu():
    """A3: AsyncContextParallel Hybrid (cp=4, ds=2, co=2) forward matches single-card on NPU.

    Feature: AsyncContextParallel in Hybrid mode forward correctness
    Description: Four ranks use AsyncContextParallel in Hybrid mode (ds=2, co=2).
        Async Ulysses A2A overlaps with projection GEMMs; synchronous Colossal AI
        K/V all-gather follows after waiting for the Ulysses A2A.
    Expectation: Hybrid async CP forward output matches the single-card reference.
    """
    torchrun_case(_FILE, "test_async_hybrid_forward_npu", master_port=13202, num_proc=4)


# ---------------------------------------------------------------------------
# Group 6: API & Integration
# ---------------------------------------------------------------------------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_api_npu():
    """I1: parallelize_module API with Colossal CP=2 BSHD float16.

    Feature: parallelize_module high-level API for ContextParallel
    Description: Uses the parallelize_module API to apply Colossal AI CP=2 to a
        simple attention module on 2 NPU ranks in BSHD float16 format, verifying
        that the declarative API correctly wires up all forward hooks.
    Expectation: Module output after parallelize_module matches the reference and
        contains no NaN values.
    """
    torchrun_case(_FILE, "test_parallelize_module_api_npu", master_port=13040, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_tp_cp_combination_npu():
    """I2: TP=2 × CP=2 Colossal BSHD float16 — shape and no-NaN verification.

    Feature: Combined Tensor Parallelism and Context Parallelism
    Description: Four ranks form a 2-D mesh with TP=2 and CP=2; weights are
        column/row sharded via TP while the sequence is sharded via Colossal AI CP,
        verifying that both parallelism strategies compose correctly.
    Expectation: Output shape is correct and contains no NaN values.
    """
    torchrun_case(_FILE, "test_tp_cp_combination_npu", master_port=13110, num_proc=4)
