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
"""Unit and seam tests for document-aware DSA KV compaction."""

# pylint: disable=wrong-import-position

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn

from tests.common.mark_utils import arg_mark

# The DSA replacement imports ``aggregate_hidden`` whose custom-op package only
# exists in the NPU training image; stub it when unavailable.
if "omni_training_custom_ops" not in sys.modules:
    try:
        import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
    except ImportError:
        sys.modules["omni_training_custom_ops"] = ModuleType("omni_training_custom_ops")

from hyper_parallel.models.deepseek_v32.adapter.distributed import (  # noqa: E402
    context_parallel,
    kv_compact,
)
from hyper_parallel.models.deepseek_v32.adapter.distributed.kv_compact import (  # noqa: E402
    CompactPlan,
    build_compact_plan,
    compact_pair_grad,
    get_compact_plan,
    interval_select,
)
from hyper_parallel.components.modules.dsa_attention import (  # noqa: E402
    DeepseekV32DSAAttention,
)

HIDDEN = 32
Q_RANK = 16
KV_RANK = 16
ROPE_DIM = 8
NOPE_DIM = 16
V_DIM = 16
NUM_HEADS = 2
NUM_INDEX_HEADS = 2
INDEX_DIM = 12


def _restore_cache() -> None:
    """Start and end every test with a clean plan cache."""
    kv_compact._plan_cache().clear()  # pylint: disable=protected-access


class TestCompactPlan(unittest.TestCase):
    """Feature: compaction plan construction from document boundaries."""

    def setUp(self) -> None:
        """Isolate the plan cache per test."""
        _restore_cache()
        self.addCleanup(_restore_cache)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_plan_matches_design_example(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build plans for the design example ends [12, 28, 40, 64] over three query blocks.
        Expectation: Each block keeps only its earliest active document window and the rebased kv_ends.
        """
        # Query block [48, 56): the earliest active document starts at 40,
        # shrinking the 56-token causal prefix to 16 tokens (71.4%).
        plan = build_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        )
        self.assertEqual(plan.windows, ((40, 56),))
        self.assertEqual(plan.total_tokens, 16)
        self.assertEqual(plan.kv_ends.tolist(), [16])

        # Query block [32, 40) inside document [28, 40): the document's
        # earlier tokens stay, later documents drop out entirely.
        plan = build_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=32, local_q_end=40,
        )
        self.assertEqual(plan.windows, ((28, 40),))
        self.assertEqual(plan.kv_ends.tolist(), [12])

        # Query block [56, 64): the last document is clamped at the block end.
        plan = build_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=56, local_q_end=64,
        )
        self.assertEqual(plan.windows, ((40, 64),))
        self.assertEqual(plan.kv_ends.tolist(), [24])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_trivial_prefix_returns_none(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Request plans for blocks whose causal prefix has nothing to drop.
        Expectation: Trivial prefixes return None so the plain prefix path stays.
        """
        self.assertIsNone(build_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=0, local_q_end=8,
        ))
        self.assertIsNone(build_compact_plan(
            [64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_mid_document_history_is_kept(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build a plan for a block starting inside the middle of a document.
        Expectation: The document head stays and the window stops at the block end.
        """
        plan = build_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=30, local_q_end=38,
        )
        self.assertEqual(plan.windows, ((28, 38),))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_per_row_windows_and_metadata(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build a plan for a batch of two rows with independent documents.
        Expectation: Each row compacts against its own documents and the kv_ends rebase per row.
        """
        # Row 0 documents [0,3), [3,8); row 1 documents [8,11), [11,14), [14,16).
        plan = build_compact_plan(
            [3, 8, 11, 14, 16], batch_size=2, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        self.assertEqual(plan.windows, ((3, 8), (8 + 3, 16)))
        # Row 0 keeps document [3, 8) = 5 tokens; row 1 keeps its two active
        # documents [11, 14) and [14, 16) = 3 + 2 tokens.
        self.assertEqual(plan.kv_ends.tolist(), [5, 8, 10])
        self.assertEqual(plan.total_tokens, 10)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_row_crossing_document_clamps_to_row(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build a plan with a document spanning a CP row edge.
        Expectation: The row segment clamps to the row edge and the windows cover only active tokens.
        """
        # Document [3, 10) spans the row-0/row-1 edge; row 1's segment of it
        # is [8, 10), which row 1's queries see as [0, 2) of their own row.
        plan = build_compact_plan(
            [3, 10, 16], batch_size=2, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        self.assertEqual(plan.windows, ((3, 8), (8 + 2, 16)))
        # Row 0 keeps [3, 8) = 5; row 1 keeps [10, 16) = 6.
        self.assertEqual(plan.kv_ends.tolist(), [5, 11])
        self.assertEqual(plan.total_tokens, 11)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_mixed_rows_keep_full_prefix_where_trivial(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build a plan with one trivial row and one compactable row.
        Expectation: The trivial row keeps its full prefix window while the other row compacts.
        """
        # Row 0's query starts in the first document (d0 = 0); row 1's query
        # starts inside its second document, which begins at 11.
        plan = build_compact_plan(
            [8, 11, 16], batch_size=2, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        self.assertEqual(plan.windows, ((0, 8), (8 + 3, 16)))
        # Row 0 keeps its full prefix (8 tokens); row 1 keeps [11, 16) = 5.
        self.assertEqual(plan.kv_ends.tolist(), [8, 13])
        self.assertEqual(plan.total_tokens, 13)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_empty_documents_are_deduplicated(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build a plan with duplicated boundary ends (empty documents).
        Expectation: Empty documents are filtered before planning and the windows stay correct.
        """
        plan = build_compact_plan(
            [12, 12, 12, 28, 40, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        )
        self.assertEqual(plan.windows, ((40, 56),))
        self.assertEqual(plan.kv_ends.tolist(), [16])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_malformed_boundaries_are_rejected(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Build plans with under-covering, decreasing ends, and an inverted query block.
        Expectation: ValueError names the violated rule for every malformed input.
        """
        with self.assertRaisesRegex(ValueError, "cover"):
            build_compact_plan(
                [12, 28, 40], batch_size=1, global_seq_len=64,
                local_q_start=48, local_q_end=56,
            )
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            build_compact_plan(
                [28, 12, 40, 64], batch_size=1, global_seq_len=64,
                local_q_start=48, local_q_end=56,
            )
        with self.assertRaisesRegex(ValueError, "query block"):
            build_compact_plan(
                [12, 28, 40, 64], batch_size=1, global_seq_len=64,
                local_q_start=56, local_q_end=48,
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_plan_cache_shares_and_evicts(self):
        """
        Feature: compaction plan construction from document boundaries.
        Description: Request plans with identical and distinct metadata and overflow the cache bound.
        Expectation: Identical metadata shares one plan object, distinct content rebuilds, and eviction stays bounded.
        """
        first = get_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        )
        again = get_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        )
        self.assertIs(first, again)
        # Different content never collides with another micro-batch's plan.
        other = get_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=56, local_q_end=64,
        )
        self.assertIsNot(first, other)
        # Push the cache past its bound; the oldest key is evicted and the
        # next request rebuilds an equal but distinct plan object.
        for extra in range(kv_compact._COMPACT_PLAN_CACHE_LIMIT):  # pylint: disable=protected-access
            get_compact_plan(
                [12, 28, 40, 64 + extra + 1], batch_size=1, global_seq_len=64 + extra + 1,
                local_q_start=48, local_q_end=56,
            )
        rebuilt = get_compact_plan(
            [12, 28, 40, 64], batch_size=1, global_seq_len=64,
            local_q_start=48, local_q_end=56,
        )
        self.assertIsNot(first, rebuilt)
        self.assertEqual(rebuilt.windows, first.windows)


class TestIntervalSelect(unittest.TestCase):
    """Feature: window selection, value slicing, and gradient transposition."""

    def setUp(self) -> None:
        """Isolate the plan cache per test."""
        _restore_cache()
        self.addCleanup(_restore_cache)
        torch.manual_seed(2026)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_select_returns_packed_window_values(self):
        """
        Feature: window selection, value slicing, and gradient transposition.
        Description: Select the plan windows from a gathered KV tensor.
        Expectation: The compacted tensor is the TND-packed concatenation of the per-row window slices.
        """
        plan = build_compact_plan(
            [3, 8, 11, 14, 16], batch_size=2, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        kv = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1, 1)
        compact = interval_select(kv, plan)
        self.assertEqual(tuple(compact.shape), (1, 10, 1, 1))
        # Row 1's window is the flat range (11, 16), i.e. within-row [3, 8).
        expected = torch.cat([kv[0, 3:8], kv[1, 3:8]]).reshape(1, 10, 1, 1)
        self.assertTrue(torch.equal(compact, expected))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_select_rejects_shape_mismatch(self):
        """
        Feature: window selection, value slicing, and gradient transposition.
        Description: Select with a tensor whose shape disagrees with the plan.
        Expectation: ValueError reports the expected shape.
        """
        plan = build_compact_plan(
            [3, 8], batch_size=1, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        with self.assertRaisesRegex(ValueError, "interval_select expects"):
            interval_select(torch.zeros(1, 16, 1), plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_grad_scatters_back_with_zero_gaps(self):
        """
        Feature: window selection, value slicing, and gradient transposition.
        Description: Backpropagate through interval_select with a weighted output.
        Expectation: Gradients fill the selected windows and the dropped head gets zeros.
        """
        plan = build_compact_plan(
            [3, 8, 11, 14, 16], batch_size=2, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        kv = torch.randn(2, 8, 2, 3, requires_grad=True)
        compact = interval_select(kv, plan)
        weights = torch.randn_like(compact)
        (compact * weights).sum().backward()
        expected = torch.zeros_like(kv)
        expected[0, 3:8] = weights.reshape(2, 5, 2, 3)[0]
        expected[1, 3:8] = weights.reshape(2, 5, 2, 3)[1]
        self.assertTrue(torch.allclose(kv.grad, expected))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_overlapping_windows_accumulate_gradients(self):
        """
        Feature: window selection, value slicing, and gradient transposition.
        Description: Select through a manually built plan whose windows overlap.
        Expectation: Overlapping positions sum their gradients and unselected slots get zero.
        """
        # A manually built plan whose windows overlap on [5, 8), as one
        # buffer serving two query blocks under dynamic pairing would.
        plan = CompactPlan(
            batch_size=1, row_stride=8,
            windows=((3, 8), (5, 8)),
            kv_ends=torch.tensor([5, 8], dtype=torch.int32),
            q_ends=torch.tensor([5, 8], dtype=torch.int32),
            total_tokens=8,
        )
        kv = torch.randn(1, 8, 1, 2, requires_grad=True)
        compact = interval_select(kv, plan)
        compact.sum().backward()
        expected = torch.tensor([0.0, 0, 0, 1, 1, 2, 2, 2]).reshape(1, 8, 1, 1)
        self.assertTrue(torch.equal(kv.grad, expected.expand(1, 8, 1, 2)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_compact_pair_grad_matches_autograd(self):
        """
        Feature: window selection, value slicing, and gradient transposition.
        Description: Compare compact_pair_grad against the autograd backward of interval_select.
        Expectation: The helper applies the same transpose and both agree elementwise.
        """
        plan = build_compact_plan(
            [3, 8], batch_size=1, global_seq_len=8,
            local_q_start=4, local_q_end=8,
        )
        kv = torch.randn(1, 8, 1, 4, requires_grad=True)
        compact = interval_select(kv, plan)
        compact.sum().backward()
        manual = compact_pair_grad(
            torch.ones(5, 1, 4), plan, (8, 1, 4)
        )
        self.assertTrue(torch.equal(kv.grad.reshape(8, 1, 4), manual))


class _FakeConfig:
    """Attribute set the V3.2 DSA replacement reads from an HF config."""

    def __init__(self, compact: bool = False) -> None:
        """Populate the attributes the V3.2 replacement reads."""
        self.num_attention_heads = NUM_HEADS
        self.q_lora_rank = Q_RANK
        self.kv_lora_rank = KV_RANK
        self.qk_rope_head_dim = ROPE_DIM
        self.qk_nope_head_dim = NOPE_DIM
        self.v_head_dim = V_DIM
        self.index_head_dim = INDEX_DIM
        self.index_topk = 2
        self.index_n_heads = NUM_INDEX_HEADS
        self.attention_dropout = 0.0
        self.dsa_loss_coeff = 0.0
        self.dsa_tnd_compact_kv = compact


class _FakeHFIndexer(nn.Module):
    """Minimal V3.2 indexer surface used by the replacement."""

    def __init__(self) -> None:
        """Build the minimal V3.2 indexer surface."""
        super().__init__()
        self.head_dim = INDEX_DIM
        self.n_heads = NUM_INDEX_HEADS
        self.index_topk = 2
        self.wq_b = nn.Linear(Q_RANK, NUM_INDEX_HEADS * INDEX_DIM, bias=False)
        self.wk = nn.Linear(HIDDEN, NUM_INDEX_HEADS * INDEX_DIM, bias=False)
        self.k_norm = nn.LayerNorm(NUM_INDEX_HEADS * INDEX_DIM)
        self.weights_proj = nn.Linear(HIDDEN, NUM_INDEX_HEADS, bias=False)


class _FakeHFAttention(nn.Module):
    """Minimal V3.2 attention surface used by the replacement."""

    def __init__(self, config: _FakeConfig) -> None:
        """Build the minimal V3.2 attention surface from ``config``."""
        super().__init__()
        self.config = config
        self.layer_idx = 0
        self.scaling = 1.0
        self.num_heads = NUM_HEADS
        self.q_lora_rank = Q_RANK
        self.kv_lora_rank = KV_RANK
        self.qk_rope_head_dim = ROPE_DIM
        self.qk_nope_head_dim = NOPE_DIM
        self.v_head_dim = V_DIM
        self.q_a_proj = nn.Linear(HIDDEN, Q_RANK, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, KV_RANK + ROPE_DIM, bias=False)
        self.q_a_layernorm = nn.LayerNorm(Q_RANK)
        self.kv_a_layernorm = nn.LayerNorm(KV_RANK)
        self.q_b_proj = nn.Linear(Q_RANK, NUM_HEADS * (NOPE_DIM + ROPE_DIM), bias=False)
        self.kv_b_proj = nn.Linear(KV_RANK, NUM_HEADS * (NOPE_DIM + V_DIM), bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * V_DIM, HIDDEN, bias=False)
        self.indexer = _FakeHFIndexer()


class _FakeCPMesh:
    """Rank-1 view of a size-2 CP mesh."""

    def size(self) -> int:
        """Return the CP world size."""
        return 2

    def get_local_rank(self) -> int:
        """Return this rank's position in the CP group."""
        return 1


class _FakeGatherHandle:
    """Stand-in for an async allgather: rank-0's chunk ahead of the local one."""

    def __init__(self, peer: torch.Tensor, local: torch.Tensor, dim: int) -> None:
        """Concatenate ``peer`` and ``local`` on ``wait``."""
        self._output = torch.cat([peer, local], dim=dim)

    def wait(self) -> torch.Tensor:
        """Return the autograd-visible gathered tensor."""
        return self._output


def _last_document_kv(attention_states, sequence_lengths):
    """Sum the last document's KV tokens, per the KV-side metadata.

    Document isolation means both the plain-prefix and the compact layout
    contain the same last-document tokens, so attention restricted to that
    document must produce identical outputs and gradients on both paths.
    """
    ends = sequence_lengths[1].tolist()
    start = ends[-2] if len(ends) >= 2 else 0
    flat_kv = attention_states[1].reshape(-1, attention_states[1].shape[-1])
    return flat_kv[start:ends[-1]].sum()


class TestCPForwardCompact(unittest.TestCase):
    """Feature: compaction inside the DSA CP forward seam."""

    LOCAL_SEQ_LEN = 8
    GLOBAL_SEQ_LEN = 16
    DOC_ENDS = [6, 16]

    def setUp(self) -> None:
        """Patch the CP communication and the dense kernel with CPU fakes."""
        _restore_cache()
        self.addCleanup(_restore_cache)
        torch.manual_seed(2026)

        def _fake_gather_launch(tensor, dim, mesh):  # pylint: disable=unused-argument
            """Concatenate rank-0's fixed chunk ahead of the local chunk."""
            del mesh
            peer = torch.randn(
                tensor.shape, dtype=tensor.dtype, device=tensor.device,
            ).requires_grad_(True)
            handle = _FakeGatherHandle(peer, tensor, dim)
            return handle

        for item in (
            patch.object(context_parallel, "async_cp_allgather_launch", _fake_gather_launch),
        ):
            item.start()
            self.addCleanup(item.stop)

    def _build_attention(self, compact: bool) -> DeepseekV32DSAAttention:
        """Build one replacement with the compaction flag set from config."""
        attention = DeepseekV32DSAAttention(
            module=_FakeHFAttention(_FakeConfig(compact=compact)),
        )
        object.__setattr__(attention, "_hp_deepseek_v32_dsa_cp_mesh", _FakeCPMesh())
        return attention

    def _run_forward(self, attention: DeepseekV32DSAAttention):
        """Run the CP forward for rank 1 and return (output, input, captured)."""
        hidden = torch.randn(1, self.LOCAL_SEQ_LEN, HIDDEN, requires_grad=True)
        captured = {}

        def _capture_run_kernels(module, index_states, attention_states, lengths, *, use_dense):
            """Record the kernel inputs and keep them in the output graph."""
            del module, index_states, use_dense
            captured["attention_states"] = attention_states
            captured["lengths"] = lengths
            scalar = _last_document_kv(attention_states, lengths)
            return scalar * torch.ones(
                1, self.LOCAL_SEQ_LEN, NUM_HEADS, KV_RANK,
                dtype=scalar.dtype, device=scalar.device,
            )

        with patch.object(context_parallel, "_run_cp_dsa_kernels", _capture_run_kernels):
            output, _ = context_parallel._deepseek_v32_dsa_cp_forward(  # pylint: disable=protected-access
                attention,
                hidden,
                None,
                None,
                None,
                None,
                self.DOC_ENDS,
                use_dense=True,
            )
        return output, hidden, captured

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_compact_forward_matches_prefix_forward(self):
        """
        Feature: compaction inside the DSA CP forward seam.
        Description: Run the CP forward seam with compaction on and off under identical inputs.
        Expectation: Outputs and gradients match the prefix path while the KV layout shrinks to the compact window.
        """
        torch.manual_seed(11)
        compact_attention = self._build_attention(compact=True)
        torch.manual_seed(11)
        prefix_attention = self._build_attention(compact=False)

        # The fake gather draws a fresh random peer per launch, so each run
        # must start from the same RNG state to see identical inputs.
        torch.manual_seed(1234)
        compact_out, compact_hidden, compact_cap = self._run_forward(compact_attention)
        torch.manual_seed(1234)
        prefix_out, prefix_hidden, prefix_cap = self._run_forward(prefix_attention)

        # Same inputs, same result: compaction is layout-only.
        self.assertTrue(torch.allclose(compact_out, prefix_out, atol=1e-5))
        self.assertTrue(torch.allclose(compact_hidden, prefix_hidden))
        compact_out.sum().backward()
        prefix_out.sum().backward()
        for name, parameter in compact_attention.named_parameters():
            other = dict(prefix_attention.named_parameters())[name]
            if parameter.grad is None or other.grad is None:
                # The fake kernel only flows gradients through the KV path
                # and o_proj; parameters outside it stay grad-free on both
                # sides.
                self.assertIsNone(parameter.grad, f"unexpected grad for {name}")
                self.assertIsNone(other.grad, f"unexpected grad for {name}")
                continue
            self.assertTrue(
                torch.allclose(parameter.grad, other.grad, atol=1e-5),
                f"gradient mismatch for {name}",
            )
        # The hidden-state gradient flows back through the trunk unchanged.
        self.assertTrue(torch.allclose(compact_hidden.grad, prefix_hidden.grad, atol=1e-5))

        # The compact layout really is smaller and metadata-rebased.
        compact_kv = compact_cap["attention_states"][1]
        prefix_kv = prefix_cap["attention_states"][1]
        self.assertEqual(compact_kv.shape[1], 10)
        self.assertEqual(prefix_kv.shape[1], self.GLOBAL_SEQ_LEN)
        self.assertEqual(compact_cap["lengths"][1].tolist(), [10])
        self.assertEqual(prefix_cap["lengths"][1].tolist(), self.DOC_ENDS)
        # Query-side metadata is rebased to the same active documents: the
        # dead-head document [0, 6) has no queries in this rank's window, so
        # its empty TND sub-sequence drops from the query lengths too.
        self.assertEqual(compact_cap["lengths"][0].tolist(), [8])
        self.assertEqual(prefix_cap["lengths"][0].tolist(), [0, 8])
        # The compact KV values are the prefix KV window values.
        self.assertTrue(torch.equal(
            compact_kv.reshape(-1, compact_kv.shape[-1]),
            prefix_kv[:, 6:self.GLOBAL_SEQ_LEN].reshape(-1, compact_kv.shape[-1]),
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_compact_rebases_query_metadata(self):
        """
        Feature: compaction inside the DSA CP forward seam.
        Description: Build plans whose dead head contains complete documents.
        Expectation: The query-side ends rebase together with the KV ends so the TND pairs stay aligned.
        """
        # Document [6, 16) holds every rank-1 query; the dead head [0, 6)
        # belongs to a document without queries in this rank's window.
        plan = build_compact_plan(
            self.DOC_ENDS, batch_size=1, global_seq_len=self.GLOBAL_SEQ_LEN,
            local_q_start=self.LOCAL_SEQ_LEN, local_q_end=self.GLOBAL_SEQ_LEN,
        )
        self.assertEqual(plan.kv_ends.tolist(), [10])
        self.assertEqual(plan.q_ends.tolist(), [8])

        # A second document inside the dead head keeps both sides aligned.
        plan = build_compact_plan(
            [3, 6, 16], batch_size=1, global_seq_len=16,
            local_q_start=8, local_q_end=16,
        )
        self.assertEqual(plan.windows, ((6, 16),))
        self.assertEqual(plan.kv_ends.tolist(), [10])
        self.assertEqual(plan.q_ends.tolist(), [8])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_flag_follows_module_configuration(self):
        """
        Feature: compaction inside the DSA CP forward seam.
        Description: Run the seam for modules with dsa_tnd_compact_kv set to true and false.
        Expectation: Only the flagged run compacts the KV input and its metadata.
        """
        attention = self._build_attention(compact=True)
        output, _, captured = self._run_forward(attention)
        self.assertEqual(captured["lengths"][1].tolist(), [10])
        self.assertEqual(output.shape, (1, self.LOCAL_SEQ_LEN, HIDDEN))

        attention = self._build_attention(compact=False)
        _, _, captured = self._run_forward(attention)
        self.assertEqual(captured["lengths"][1].tolist(), self.DOC_ENDS)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_compact_requires_document_boundaries(self):
        """
        Feature: compaction inside the DSA CP forward seam.
        Description: Run the compact path without packed document boundaries.
        Expectation: ValueError names the missing packed-sequence boundaries.
        """
        attention = self._build_attention(compact=True)
        with patch.object(context_parallel, "_run_cp_dsa_kernels", lambda *a, **k: None):
            with self.assertRaisesRegex(ValueError, "requires the packed-sequence"):
                context_parallel._deepseek_v32_dsa_cp_forward(  # pylint: disable=protected-access
                    attention,
                    torch.randn(1, self.LOCAL_SEQ_LEN, HIDDEN),
                    None, None, None, None,
                    None,
                    use_dense=True,
                )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_trivial_documents_keep_prefix_path(self):
        """
        Feature: compaction inside the DSA CP forward seam.
        Description: Run the compact path when one document covers the whole row.
        Expectation: The plain rectangular prefix layout and metadata pass through unchanged.
        """
        attention = self._build_attention(compact=True)
        # One document covering the row: rank 1's query block [8, 16) starts
        # inside the first document, so compaction degenerates to the prefix.
        ends = [self.GLOBAL_SEQ_LEN]
        hidden = torch.randn(1, self.LOCAL_SEQ_LEN, HIDDEN, requires_grad=True)
        captured = {}

        def _capture(module, index_states, attention_states, lengths, *, use_dense):
            """Record the kernel inputs and return a dummy output."""
            del module, index_states, use_dense
            captured["lengths"] = lengths
            captured["attention_states"] = attention_states
            return torch.zeros(
                1, self.LOCAL_SEQ_LEN, NUM_HEADS, KV_RANK,
                dtype=hidden.dtype, device=hidden.device,
            )

        with patch.object(context_parallel, "_run_cp_dsa_kernels", _capture):
            context_parallel._deepseek_v32_dsa_cp_forward(  # pylint: disable=protected-access
                attention, hidden, None, None, None, None, ends, use_dense=True,
            )
        self.assertEqual(captured["lengths"][1].tolist(), ends)
        # The KV tensor keeps the plain rectangular prefix layout.
        self.assertEqual(captured["attention_states"][1].shape[1], self.GLOBAL_SEQ_LEN)


if __name__ == "__main__":
    unittest.main()
