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
"""CPU tests for sparse-attention construction, metadata and indexer scaling."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.components.modules.attention.deepseek_sparse_attention import (
    CPDeepseekSparseAttentionModule,
    DeepseekSparseAttention,
    DeepseekSparseAttentionModule,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.cp_deepseek_sparse_attention import (
    _gather_key_banks,
    build_cp_sequence_metadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    SparseAttentionSequenceMetadata,
)
from hyper_parallel.components.modules.module_family import FamilyRequest, build_family
from hyper_parallel.distributed.module_binding import ParallelBinding
from hyper_parallel.models.module_build import prepare_model_components
from tests.common.mark_utils import arg_mark


def _config() -> SimpleNamespace:
    """Return small native dimensions without a Transformers model dependency."""
    return SimpleNamespace(
        hidden_size=16, num_attention_heads=2, q_lora_rank=8, kv_lora_rank=8,
        qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        index_head_dim=8, index_n_heads=2, index_topk=1024,
        device="cpu", dtype=torch.float32,
    )


def _gather_cp_tokens(bank, *, cp_size, rank, ends, overlap):
    """Use global token IDs as all three key banks with mocked CP transport."""
    batch, length = bank.shape[:2]
    width = length // cp_size
    local = bank[:, rank * width:(rank + 1) * width]
    metadata = build_cp_sequence_metadata(
        batch_size=batch, local_seq_len=width, device=bank.device,
        cp_size=cp_size, cp_rank=rank, actual_seq_len=ends,
    )
    context = Mock()
    context.gather_sequence.return_value = (bank, None)
    receipt = Mock()
    receipt.wait.return_value = (bank, None)
    context.launch_gather_sequence.return_value = receipt
    index, attention = _gather_key_banks(
        context, overlap, (local, local, local.squeeze(-1)), (local, local, local, local), metadata,
    )
    banks = (index[1].flatten(), attention[1].flatten(), attention[3].flatten())
    return metadata, local.flatten(), banks


def _causal_keys(metadata, key_bank, query_index):
    """Apply the kernel's right-aligned causal rule to one packed query."""
    query_start = key_start = 0
    for query_end, key_end in zip(metadata.query_ends.tolist(), metadata.key_ends.tolist()):
        if query_index < query_end:
            offset = (key_end - key_start) - (query_end - query_start)
            count = offset + query_index - query_start + 1
            return key_bank[key_start:key_start + count]
        query_start, key_start = query_end, key_end
    raise ValueError(f"Query index {query_index} exceeds the packed query endpoints")


class TestDeepseekSparseAttention(unittest.TestCase):
    """Cover family wiring and local numerical behavior without device kernels."""

    def setUp(self) -> None:
        """Keep parameter initialization deterministic and preserve the caller RNG."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(7)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_config_selects_local_and_cp_members(self) -> None:
        """Feature: DSA native family construction.

        Description: Construct local and CP members directly from native config.
        Expectation: The selected member type and parameter dimensions match the requested topology.
        """
        for cp_size, expected_type in ((1, DeepseekSparseAttentionModule), (2, CPDeepseekSparseAttentionModule)):
            with self.subTest(cp_size=cp_size):
                module = DeepseekSparseAttention(
                    config=_config(), context={"cp_size": cp_size}, kv_cp_size=cp_size,
                )

                self.assertIs(type(module), expected_type)
                self.assertEqual(module.linear_qkv.weight.shape, (20, 16))
                self.assertEqual(module.index_topk, 1024)
                self.assertEqual({parameter.device.type for parameter in module.parameters()}, {"cpu"})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_defaults_match_factory_and_deferred_requests(self) -> None:
        """Feature: DSA family default options.

        Description: Construct immediate and deferred requests without optional CP arguments.
        Expectation: Both paths retain the documented KV transport and overlap defaults.
        """
        expected = {
            "kv_cp_size": 1, "kv_cp_strategy": "allgather", "cp_comm_overlap": False,
        }
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                if deferred:
                    module = build_family(FamilyRequest(DeepseekSparseAttention, {}), config=_config())
                else:
                    module = DeepseekSparseAttention(config=_config())
                self.assertIs(type(module), DeepseekSparseAttentionModule)
                self.assertEqual({name: getattr(module, name) for name in expected}, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_promotion_preserves_parameters_and_eval_mode(self) -> None:
        """Feature: DSA deferred CP promotion.

        Description: Promote a loaded local member to its CP implementation.
        Expectation: Parameters, values and evaluation mode are preserved.
        """
        source = DeepseekSparseAttention(config=_config()).eval()
        parameters = dict(source.named_parameters())
        values = {name: parameter.detach().clone() for name, parameter in parameters.items()}

        promoted = DeepseekSparseAttention(module=source, context={"cp_size": 2}, kv_cp_size=2)

        self.assertIs(type(promoted), CPDeepseekSparseAttentionModule)
        self.assertFalse(promoted.training)
        self.assertEqual(set(dict(promoted.named_parameters())), set(parameters))
        for name, parameter in promoted.named_parameters():
            self.assertIs(parameter, parameters[name])
            torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deferred_cp_preparation_preserves_parameters_and_options(self) -> None:
        """Feature: Deferred DSA topology preparation.

        Description: Record CP options during CPU or meta construction and bind them later.
        Expectation: Promotion preserves parameters, options and module mode.
        """
        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                config = _config()
                config.device = device
                source = DeepseekSparseAttention(config=config, kv_cp_size=2, cp_comm_overlap=True).eval()
                parameters = dict(source.named_parameters())
                self.assertIs(type(source), DeepseekSparseAttentionModule)
                with self.assertRaises(ValueError):
                    prepare_model_components(source, {"cp_size": 1})

                promoted, _ = prepare_model_components(source, {"cp_size": 2})

                self.assertIs(type(promoted), CPDeepseekSparseAttentionModule)
                self.assertEqual(promoted.kv_cp_size, 2)
                self.assertTrue(promoted.cp_comm_overlap)
                self.assertFalse(promoted.training)
                for name, parameter in promoted.named_parameters():
                    self.assertIs(parameter, parameters[name])
                    self.assertEqual(parameter.device.type, device)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_rejects_inconsistent_topology_and_unsupported_options(self) -> None:
        """Feature: DSA CP topology validation.

        Description: Pass mismatched CP degrees, invalid KV sizes and removed options.
        Expectation: Unsupported requests fail during construction or deferred preparation.
        """
        for cp_size, options in ((2, {}), (1, {"kv_cp_size": 2}), (4, {"kv_cp_size": 2})):
            for deferred in (False, True):
                with self.subTest(cp_size=cp_size, options=options, deferred=deferred), self.assertRaises(ValueError):
                    if deferred:
                        build_family(
                            FamilyRequest(DeepseekSparseAttention, options), config=_config(),
                            context={"cp_size": cp_size},
                        )
                    else:
                        DeepseekSparseAttention(config=_config(), context={"cp_size": cp_size}, **options)
        for value in (0, -1, True, 2.0):
            with self.subTest(kv_cp_size=value), self.assertRaises(ValueError):
                DeepseekSparseAttention(config=_config(), kv_cp_size=value)
        with self.assertRaises(ValueError):
            DeepseekSparseAttention(config=_config(), kv_cp_strategy="p2p")
        for name, value in (("ulysses_cp_size", 2), ("kv_cp_load_balance", False)):
            with self.subTest(removed_argument=name):
                with self.assertRaises(TypeError):
                    DeepseekSparseAttention(config=_config(), **{name: value})
                with self.assertRaises(TypeError):
                    FamilyRequest(DeepseekSparseAttention, {name: value})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_binding_rejects_an_incompatible_runtime_degree(self) -> None:
        """Feature: DSA runtime binding validation.

        Description: Bind a CP member to a runtime group with a different degree.
        Expectation: The incompatible CP context is rejected before attention execution.
        """
        module = CPDeepseekSparseAttentionModule(config=_config(), kv_cp_size=2)
        context = Mock(size=4, mesh=object())
        with self.assertRaises(ValueError):
            module.bind_parallel(ParallelBinding(mesh_binding=Mock(), cp_context=context))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_attention_mask_does_not_gate_local_or_cp_preflight(self) -> None:
        """Feature: DSA ignores the Transformers attention-mask carrier.

        Description: Pass a mask whose dimensions differ from the local DSA sequence on TP and CP.
        Expectation: Local metadata validation uses sequence lengths, not the external mask shape.
        """
        hidden_states = torch.zeros(1, 8, _config().hidden_size)
        local_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        module = DeepseekSparseAttention(config=_config(), context={"tp_size": 2})
        module._validate_forward_inputs(hidden_states, local_mask, None, None, {})

        cp_module = DeepseekSparseAttention(
            config=_config(), context={"tp_size": 2, "cp_size": 2}, kv_cp_size=2,
        )
        cp_module._cp_context = SimpleNamespace(size=2, rank=1)
        metadata = cp_module.preflight_attention(
            hidden_states, local_mask, None, None, None, {}, use_dense=False,
        )
        self.assertEqual(metadata.query_ends.tolist(), [8])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_comm_overlap_selects_gather_scheduling(self) -> None:
        """Feature: DSA CP key-bank scheduling.

        Description: Select synchronous or overlapped key-bank gathering.
        Expectation: Each mode uses its expected transport and preserves selected banks.
        """
        bank = torch.arange(8, device="cpu").reshape(1, 8, 1, 1)
        local = bank[:, :4]
        metadata = build_cp_sequence_metadata(
            batch_size=1, local_seq_len=4, device=bank.device, cp_size=2, cp_rank=0,
        )
        for overlap in (False, True):
            with self.subTest(cp_comm_overlap=overlap):
                module = DeepseekSparseAttention(
                    config=_config(), context={"cp_size": 2}, kv_cp_size=2, cp_comm_overlap=overlap,
                )
                context = Mock(size=2, mesh=object())
                context.gather_sequence.return_value = (bank, None)
                receipt = Mock()
                receipt.wait.return_value = (bank, None)
                context.launch_gather_sequence.return_value = receipt
                module.bind_parallel(ParallelBinding(mesh_binding=Mock(), cp_context=context))

                index, attention = module.prepare_key_bank(
                    (local, local, local.squeeze(-1)), (local, local, local, local), metadata,
                )

                for tensor in (index[1], attention[1], attention[3]):
                    torch.testing.assert_close(tensor, local, rtol=0, atol=0)
                if overlap:
                    self.assertEqual(context.launch_gather_sequence.call_count, 3)
                    self.assertEqual(receipt.wait.call_count, 3)
                    context.gather_sequence.assert_not_called()
                else:
                    self.assertEqual(context.gather_sequence.call_count, 3)
                    context.launch_gather_sequence.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_sequence_metadata_preserves_packed_documents(self) -> None:
        """Feature: DSA CP TND metadata.

        Description: Project global TND boundaries into compact rank-local query and KV segments.
        Expectation: Query endpoints, causal key ranges and compact bank lengths match each rank's token order.
        """
        for actual_seq_len, expected_ends in ((None, [8, 16]), ([6, 16], [6, 16])):
            with self.subTest(actual_seq_len=actual_seq_len):
                metadata = SparseAttentionSequenceMetadata.build(
                    batch_size=2, local_seq_len=8, device=torch.device("cpu"), actual_seq_len=actual_seq_len,
                )
                self.assertEqual(metadata.kv_prefix_length, 8)
                for endpoints in metadata.sequence_lengths:
                    torch.testing.assert_close(endpoints, torch.tensor(expected_ends, dtype=torch.int32))

        expected = (
            ([4, 8], [4, 10], ((0, 4), (6, 12))),
            ([2, 4, 8], [6, 8, 18], ((0, 6), (6, 8), (6, 16))),
        )
        for rank, (query_ends, key_ends, key_ranges) in enumerate(expected):
            with self.subTest(cp_rank=rank):
                metadata = build_cp_sequence_metadata(
                    batch_size=2, local_seq_len=4, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[6, 16],
                )
                self.assertEqual(metadata.kv_prefix_length, key_ends[-1])
                torch.testing.assert_close(metadata.query_ends, torch.tensor(query_ends, dtype=torch.int32))
                torch.testing.assert_close(metadata.key_ends, torch.tensor(key_ends, dtype=torch.int32))
                self.assertEqual(metadata.key_ranges, key_ranges)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_packed_candidates_match_global_causal_documents(self) -> None:
        """Feature: DSA CP packed causal key selection.

        Description: Compare selected key banks with global token IDs over aligned and split TND documents.
        Expectation: Every local query sees its document prefix and no future or cross-document keys.
        """
        cases = (
            (2, 8, 1, [6, 16]), (2, 8, 2, [8, 16]), (2, 8, 2, [6, 16]),
            (1, 4096, 2, [384, 1280, 2880, 4096]),
            (2, 16, 4, [14, 32]), (3, 8, 2, [6, 14, 24]), (3, 8, 2, [24]),
        )
        for batch, length, cp_size, ends in cases:
            bank = torch.arange(batch * length, device="cpu").reshape(batch, length, 1, 1)
            for rank in range(cp_size):
                for overlap in (False, True):
                    with self.subTest(shape=(batch, length), cp=cp_size, rank=rank, ends=ends, overlap=overlap):
                        metadata, queries, banks = _gather_cp_tokens(
                            bank, cp_size=cp_size, rank=rank, ends=ends, overlap=overlap,
                        )
                        for query_index, query_id in enumerate(queries.tolist()):
                            document_start = max([0] + [end for end in ends if end <= query_id])
                            expected = torch.arange(document_start, query_id + 1, device="cpu")
                            for key_bank in banks:
                                torch.testing.assert_close(
                                    _causal_keys(metadata, key_bank, query_index), expected, rtol=0, atol=0,
                                )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_cross_row_document_first_query_has_one_candidate(self) -> None:
        """Feature: DSA CP documents crossing batch rows.

        Description: Inspect the first query in a packed document spanning two rows.
        Expectation: It selects its own causal prefix and no future tokens.
        """
        bank = torch.arange(8192, device="cpu").reshape(2, 4096, 1, 1)
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                metadata, queries, banks = _gather_cp_tokens(
                    bank, cp_size=2, rank=1, ends=[3584, 8192], overlap=overlap,
                )
                for query_id in (3584, 3585, 6144):
                    query_index = queries.tolist().index(query_id)
                    actual = _causal_keys(metadata, banks[0], query_index)
                    expected = torch.arange(3584, query_id + 1, device="cpu")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(min(actual.numel(), 1024), min(query_id - 3584 + 1, 1024))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_repeated_key_prefixes_accumulate_backward(self) -> None:
        """Feature: DSA CP repeated KV-prefix gradients.

        Description: Reuse gathered causal prefixes from multiple local queries.
        Expectation: Backward accumulates every selected-key contribution at its source token.
        """
        for rank in (0, 1):
            for overlap in (False, True):
                with self.subTest(rank=rank, overlap=overlap):
                    bank = torch.arange(16, dtype=torch.float32, device="cpu").reshape(2, 8, 1, 1).requires_grad_()
                    metadata, queries, banks = _gather_cp_tokens(
                        bank, cp_size=2, rank=rank, ends=[6, 16], overlap=overlap,
                    )
                    loss = sum(
                        _causal_keys(metadata, key_bank, query_index).sum()
                        for key_bank in banks for query_index in range(queries.numel())
                    )
                    loss.backward()
                    expected = torch.zeros(16, device="cpu")
                    for query_id in queries.detach().int().tolist():
                        document_start = 0 if query_id < 6 else 6
                        expected[document_start:query_id + 1] += 3
                    torch.testing.assert_close(bank.grad.flatten(), expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_rejects_mismatched_packed_q_k_boundaries(self) -> None:
        """Feature: DSA CP packed-document pairing.

        Description: Provide different TND document endpoints for self-attention Q and KV.
        Expectation: Reject ambiguous document-to-prefix pairing before communication.
        """
        with self.assertRaisesRegex(ValueError, "matching packed Q and KV document boundaries"):
            build_cp_sequence_metadata(
                batch_size=1, local_seq_len=4, device=torch.device("cpu"), cp_size=2, cp_rank=0,
                kwargs={"actual_seq_qlen": [3, 8], "actual_seq_kvlen": [4, 8]},
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cp_compacts_zero_query_documents_in_tnd(self) -> None:
        """Feature: DSA CP zero-query TND fragments.

        Description: Project the 4096-token TND example whose document edges do not align with CP=2.
        Expectation: Drop empty local Q fragments and retain each non-empty query's causal KV range.
        """
        expected = (
            ([384, 1280, 2048], [384, 1280, 2048], ((0, 384), (384, 1280), (1280, 2048))),
            ([832, 2048], [1600, 2816], ((1280, 2880), (2880, 4096))),
        )
        for rank, (query_ends, key_ends, key_ranges) in enumerate(expected):
            with self.subTest(cp_rank=rank):
                metadata = build_cp_sequence_metadata(
                    batch_size=1, local_seq_len=2048, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[384, 1280, 2880, 4096],
                )

                torch.testing.assert_close(metadata.query_ends, torch.tensor(query_ends, dtype=torch.int32))
                torch.testing.assert_close(metadata.key_ends, torch.tensor(key_ends, dtype=torch.int32))
                self.assertEqual(metadata.key_ranges, key_ranges)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_indexer_weights_scale_without_intermediate_bf16_rounding(self) -> None:
        """Feature: DSA indexer merge-weight precision.

        Description: Scale BF16 merge weights for the 32-head, 128-dimension indexer.
        Expectation: FP32 intermediate scaling matches division by the exact scale factor.
        """
        config = _config()
        config.dtype = torch.bfloat16
        config.index_n_heads, config.index_head_dim = 32, 128
        module = DeepseekSparseAttention(config=config)
        raw_weights = torch.linspace(1, 2, 32, device="cpu").to(torch.bfloat16)
        hidden_states = torch.zeros(1, 2, 16, dtype=torch.bfloat16, device="cpu")
        hidden_states[..., 0] = 1
        with torch.no_grad():
            module.indexer.weights_proj.weight.zero_()
            module.indexer.weights_proj.weight[:, 0].copy_(raw_weights)

        index_states, _ = module.prepare_attention(hidden_states, None, use_dense=False)

        expected = (raw_weights.float() / 64).to(torch.bfloat16).expand(1, 2, 32)
        torch.testing.assert_close(index_states[2], expected, rtol=0, atol=0)
