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
"""Unit tests for the fused Lightning-Indexer gate and adapter helpers."""
import os
import unittest
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import hyper_parallel.components.modules.shared_compressed_dsa_attention as scda  # noqa: E402  # pylint: disable=wrong-import-position
from hyper_parallel.models.deepseek_v41.adapter.ops import (  # noqa: E402  # pylint: disable=wrong-import-position
    fused_lightning_indexer as fused,
)

fused.register_fused_indexer()


class TestFusedIndexerGate(unittest.TestCase):
    """Availability gate: env escape hatch, lazy probe, failure latch."""

    def setUp(self) -> None:
        """Snapshot the module-level gate state."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)

    def tearDown(self) -> None:
        """Restore the gate state and the escape-hatch env var."""
        fused._FUSED_INDEXER_STATE.update(self._saved)
        os.environ.pop("V41_DISABLE_FUSED_INDEXER", None)

    def test_env_escape_hatch_forces_torch_path(self):
        fused._FUSED_INDEXER_STATE.update(checked=False, available=False)
        with patch.dict(os.environ, {"V41_DISABLE_FUSED_INDEXER": "1"}):
            self.assertFalse(fused._fused_indexer_available())

    def test_probe_result_is_cached(self):
        fused._FUSED_INDEXER_STATE.update(checked=True, available=False)
        self.assertFalse(fused._fused_indexer_available())
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True)
        self.assertTrue(fused._fused_indexer_available())

    def test_fullscore_gate_rejects_cpu_tensors(self):
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True)
        query = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        key = torch.randn(1, 4, 128, dtype=torch.bfloat16)
        self.assertFalse(
            fused._fused_fullscore_usable(query, key, None, 2))


class TestConfigurationSwitch(unittest.TestCase):
    """The fused path is selected by configuration, not the environment."""

    def setUp(self) -> None:
        """Pretend the operator is present so only the flag decides."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True)

    def tearDown(self) -> None:
        """Restore the module-level gate state."""
        fused._FUSED_INDEXER_STATE.update(self._saved)

    def test_use_fused_false_takes_the_torch_path(self):
        """use_fused=False must not reach the operator, even on CPU inputs."""
        torch.manual_seed(5)
        seq, ratio, top_k = 8, 2, 2
        query = torch.randn(1, seq, 4, 16)
        key = torch.randn(1, seq // ratio, 16)
        merge_w = torch.randn(1, seq, 4)
        out = scda.compressed_causal_topk(
            query, key, merge_w, compress_ratio=ratio, sparse_count=top_k,
            query_chunk_size=4, use_fused=False)
        self.assertEqual(tuple(out.shape), (1, seq, top_k))

    def test_module_copies_the_switch_off_the_attention_module(self):
        """The indexer takes fused_indexer from the source attention module."""

        class _Attn(torch.nn.Module):
            """Minimal stand-in carrying the attributes the indexer reads."""

            def __init__(self, enabled: bool) -> None:
                """Build the stand-in with the switch under test."""
                super().__init__()
                self.q_b_proj = torch.nn.Linear(8, 8, bias=False)
                self.weights_proj = torch.nn.Linear(8, 4, bias=False)
                self.compress_ratio = 2
                self.num_heads = 4
                self.head_dim = 16
                self.index_topk = 2
                self.owns_key = True
                self.is_candidate_source = False
                self.uses_candidates = False
                self.candidate_topk_blocks = 4
                self.candidate_block_size = 8
                self.loss_coeff = 1.0
                self.fused_indexer = enabled

        self.assertFalse(scda.SharedCompressedDSAIndexer(_Attn(False)).fused_indexer)
        self.assertTrue(scda.SharedCompressedDSAIndexer(_Attn(True)).fused_indexer)

    def test_helpers_honour_use_fused_false(self):
        """use_fused=False keeps the torch path even when the op is present."""
        torch.manual_seed(5)
        seq, ratio, top_k = 8, 2, 2
        out = scda.compressed_causal_topk(
            torch.randn(1, seq, 4, 16), torch.randn(1, seq // ratio, 16),
            torch.randn(1, seq, 4), compress_ratio=ratio, sparse_count=top_k,
            query_chunk_size=4, use_fused=False)
        self.assertEqual(tuple(out.shape), (1, seq, top_k))


class TestProviderRegistry(unittest.TestCase):
    """The shared chains run the reference path when no provider is installed."""

    def tearDown(self) -> None:
        """Re-install the V4.1 provider for the rest of the suite."""
        fused.register_fused_indexer()

    def test_selection_runs_without_a_provider(self):
        """An unregistered build keeps working: the hook is optional."""
        scda.register_fused_selection_provider(None)
        out = scda.compressed_causal_topk(
            torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16), torch.randn(1, 8, 4),
            compress_ratio=2, sparse_count=2, query_chunk_size=4)
        self.assertEqual(tuple(out.shape), (1, 8, 2))

    def test_registration_installs_the_v41_provider(self):
        """Importing the V4.1 adapter is what puts the operator in the path."""
        scda.register_fused_selection_provider(None)
        fused.register_fused_indexer()
        self.assertIsInstance(scda._FUSED_SELECTION_PROVIDER,
                              fused.FusedLightningIndexerProvider)


class TestEngagementSignal(unittest.TestCase):
    """A run states positively that the operator ran; absence of a warning does not."""

    def setUp(self) -> None:
        """Pretend the operator is present and clear the engagement latch."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True, engaged=False)

    def tearDown(self) -> None:
        """Restore the module-level gate state."""
        fused._FUSED_INDEXER_STATE.clear()
        fused._FUSED_INDEXER_STATE.update(self._saved)

    def test_torch_path_never_claims_engagement(self):
        """Taking the torch path must leave the latch untouched."""
        scda.compressed_causal_topk(
            torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16), torch.randn(1, 8, 4),
            compress_ratio=2, sparse_count=2, query_chunk_size=4, use_fused=False)
        self.assertFalse(fused._FUSED_INDEXER_STATE["engaged"])

    def test_fused_call_records_engagement_once(self):
        """A successful fused call logs once and latches the fact."""
        result = torch.zeros(1, 8, 2, dtype=torch.int32)
        with (patch.object(fused, "_fused_causal_usable", return_value=True),
              patch.object(fused, "_fused_compressed_causal_topk", return_value=result),
              self.assertLogs(fused._FUSED_LOGGER, level="INFO") as logs):
            scda.compressed_causal_topk(
                torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16), torch.randn(1, 8, 4),
                compress_ratio=2, sparse_count=2, query_chunk_size=4)
        self.assertTrue(fused._FUSED_INDEXER_STATE["engaged"])
        self.assertIn("engaged", logs.output[0])


class TestGateConsistency(unittest.TestCase):
    """The two gates must agree on the constraints they share."""

    def setUp(self) -> None:
        """Pretend the operator is present so only the shapes decide."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True)

    def tearDown(self) -> None:
        """Restore the module-level gate state."""
        fused._FUSED_INDEXER_STATE.update(self._saved)

    def test_head_and_dtype_constraints_match(self):
        """Head count, head width and dtype are one contract, not two copies."""
        key = torch.randn(1, 128, 128, dtype=torch.bfloat16)
        for heads in (4, 8, 16, 24, 32, 48, 64):
            for head_dim in (64, 128):
                for dtype in (torch.bfloat16, torch.float32):
                    query = torch.randn(1, 8, heads, head_dim, dtype=dtype)
                    supported = (heads in fused._FUSED_INDEXER_HEADS
                                 and head_dim == 128
                                 and dtype in (torch.bfloat16, torch.float16))
                    with self.subTest(heads=heads, head_dim=head_dim, dtype=dtype):
                        # Both gates also require an NPU device, so on CPU they
                        # must both be False; the point is that they agree.
                        self.assertEqual(
                            fused._fused_causal_usable(query, None, None, 512, 2),
                            fused._fused_fullscore_usable(query, key, None, 2))
                        if not supported:
                            self.assertFalse(fused._fused_causal_usable(query, None, None, 512, 2))


class TestOutOfMemoryContract(unittest.TestCase):
    """Operator OOM propagates instead of latching the fused path off."""

    def setUp(self) -> None:
        """Pretend the operator is present so the fused branch is entered."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)
        fused._FUSED_INDEXER_STATE.update(checked=True, available=True)

    def tearDown(self) -> None:
        """Restore the module-level gate state."""
        fused._FUSED_INDEXER_STATE.update(self._saved)

    @staticmethod
    def _oom(*_args, **_kwargs):
        """Stand in for an operator call that runs out of device memory."""
        raise torch.OutOfMemoryError("fused indexer probe")

    def test_causal_topk_propagates_oom(self):
        """Causal top-k must raise OOM and keep the gate open."""
        with (patch.object(fused, "_fused_causal_usable", return_value=True),
              patch.object(fused, "_fused_compressed_causal_topk", self._oom)):
            with self.assertRaises(torch.OutOfMemoryError):
                scda.compressed_causal_topk(
                    torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16),
                    torch.randn(1, 8, 4), compress_ratio=2, sparse_count=2,
                    query_chunk_size=4)
        self.assertTrue(fused._FUSED_INDEXER_STATE["available"])

    def test_entry_validation_runs_before_the_fused_branch(self):
        """Both paths must reject the same arguments, gate open or not."""
        with patch.object(fused, "_fused_causal_usable", return_value=True):
            with self.assertRaises(ValueError):
                scda.compressed_causal_topk(
                    torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16),
                    torch.randn(1, 8, 4), compress_ratio=2, sparse_count=2,
                    query_chunk_size=0)

    def test_topk_and_candidates_propagates_oom(self):
        """Top-k with candidate blocks must raise OOM and keep the gate open."""
        with (patch.object(fused, "_fused_fullscore_usable", return_value=True),
              patch.object(fused, "_fused_source_segment", self._oom)):
            with self.assertRaises(torch.OutOfMemoryError):
                scda.compressed_causal_topk_and_candidates(
                    torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16),
                    torch.randn(1, 8, 4), compress_ratio=2, sparse_count=2,
                    topk_blocks=1, block_size=2, query_chunk_size=4)
        self.assertTrue(fused._FUSED_INDEXER_STATE["available"])

    def test_candidate_topk_propagates_oom(self):
        """Candidate-restricted re-selection must raise OOM and keep the gate open."""
        with (patch.object(fused, "_fused_fullscore_usable", return_value=True),
              patch.object(fused, "_fused_reindex_segment", self._oom)):
            with self.assertRaises(torch.OutOfMemoryError):
                scda.compressed_candidate_topk(
                    torch.randn(1, 8, 4, 16), torch.randn(1, 4, 16),
                    torch.randn(1, 8, 4), torch.zeros(1, 8, 1, dtype=torch.int32),
                    compress_ratio=2, sparse_count=2, block_size=2,
                    query_chunk_size=4)
        self.assertTrue(fused._FUSED_INDEXER_STATE["available"])


class TestAscendingWithInvalidTail(unittest.TestCase):
    """Output-format helper: ascending order with trailing -1 padding."""

    def test_sorts_and_masks_invalid(self):
        indices = torch.tensor([[5, 2, 7, 9], [9, 9, 1, 0]], dtype=torch.int64)
        out = fused._ascending_with_invalid_tail(indices, invalid_value=9)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out[0].tolist(), [2, 5, 7, -1])
        self.assertEqual(out[1].tolist(), [0, 1, -1, -1])


class TestTorchReferencePath(unittest.TestCase):
    """The torch fallback keeps its output contract when the gate is off."""

    def setUp(self) -> None:
        """Force the torch fallback path for the whole test case."""
        self._saved = dict(fused._FUSED_INDEXER_STATE)
        fused._FUSED_INDEXER_STATE.update(checked=True, available=False)

    def tearDown(self) -> None:
        """Restore the module-level gate state."""
        fused._FUSED_INDEXER_STATE.update(self._saved)

    def test_causal_topk_contract_cpu(self):
        torch.manual_seed(7)
        seq, ratio, top_k = 8, 2, 2
        comp_len = seq // ratio
        query = torch.randn(1, seq, 4, 16)
        key = torch.randn(1, comp_len, 16)
        merge_w = torch.randn(1, seq, 4)
        out = scda.compressed_causal_topk(
            query, key, merge_w, compress_ratio=ratio, sparse_count=top_k,
            query_chunk_size=4)
        self.assertEqual(tuple(out.shape), (1, seq, top_k))
        for i in range(seq):
            visible = (i + 1) // ratio
            row = [x for x in out[0, i].tolist() if x >= 0]
            self.assertEqual(len(row), min(top_k, visible))
            self.assertEqual(row, sorted(row))
            for pos in row:
                self.assertLess(pos, visible)


if __name__ == "__main__":
    unittest.main()
