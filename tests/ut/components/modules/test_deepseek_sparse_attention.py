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
    _PendingMainBanks,
    build_cp_sequence_metadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    SparseAttentionSequenceMetadata,
)
from hyper_parallel.components.modules.module_family import FamilyRequest, build_family
from hyper_parallel.distributed.module_binding import ParallelBinding
from hyper_parallel.models.module_build import prepare_model_components


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


from tests.common.mark_utils import arg_mark

class TestDeepseekSparseAttention(unittest.TestCase):
    """Cover family wiring and local numerical behavior without device kernels."""

    def setUp(self) -> None:
        """Keep parameter initialization deterministic and preserve the caller RNG."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(7)

    def test_native_config_selects_local_and_cp_members(self) -> None:
        """The public factory constructs local and CP members directly from config."""
        for cp_size, expected_type in ((1, DeepseekSparseAttentionModule), (2, CPDeepseekSparseAttentionModule)):
            with self.subTest(cp_size=cp_size):
                module = DeepseekSparseAttention(
                    config=_config(), context={"cp_size": cp_size}, kv_cp_size=cp_size,
                )

                self.assertIs(type(module), expected_type)
                self.assertEqual(module.linear_qkv.weight.shape, (20, 16))
                self.assertEqual(module.index_topk, 1024)
                self.assertEqual({parameter.device.type for parameter in module.parameters()}, {"cpu"})

    def test_cp_defaults_match_factory_and_deferred_requests(self) -> None:
        """YAML requests that omit arguments receive the documented constructor defaults."""
        expected = {
            "kv_cp_size": 1, "kv_cp_strategy": "allgather", "cp_comm_overlap": False,
            "main_kv_comm_fusion": False,
        }
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                if deferred:
                    module = build_family(FamilyRequest(DeepseekSparseAttention, {}), config=_config())
                else:
                    module = DeepseekSparseAttention(config=_config())
                self.assertIs(type(module), DeepseekSparseAttentionModule)
                self.assertEqual({name: getattr(module, name) for name in expected}, expected)

    def test_cp_promotion_preserves_parameters_and_eval_mode(self) -> None:
        """Promoting a loaded local member reuses its Parameters and training mode."""
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

    def test_deferred_cp_preparation_preserves_parameters_and_options(self) -> None:
        """CPU and meta construction can record KV CP before the job topology exists."""
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

    def test_cp_rejects_inconsistent_topology_and_unsupported_options(self) -> None:
        """KV AllGather degrees must match the job; unrelated CP options are not exposed."""
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
        with self.assertRaisesRegex(ValueError, "main_kv_comm_fusion must be a bool"):
            DeepseekSparseAttention(config=_config(), main_kv_comm_fusion=1)
        for name, value in (("ulysses_cp_size", 2), ("kv_cp_load_balance", False)):
            with self.subTest(removed_argument=name):
                with self.assertRaises(TypeError):
                    DeepseekSparseAttention(config=_config(), **{name: value})
                with self.assertRaises(TypeError):
                    FamilyRequest(DeepseekSparseAttention, {name: value})

    def test_cp_binding_rejects_an_incompatible_runtime_degree(self) -> None:
        """Physical CP binding also validates a concrete member's requested KV degree."""
        module = CPDeepseekSparseAttentionModule(config=_config(), kv_cp_size=2)
        context = Mock(size=4, mesh=object())
        with self.assertRaises(ValueError):
            module.bind_parallel(ParallelBinding(mesh_binding=Mock(), cp_context=context))

    def test_cp_comm_overlap_selects_gather_scheduling(self) -> None:
        """The renamed option reaches synchronous or asynchronous key-bank transport."""
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

    def test_sequence_metadata_preserves_packed_documents(self) -> None:
        """Local endpoints and CP causal prefixes retain the same document boundaries."""
        for actual_seq_len, expected_ends in ((None, [8, 16]), ([6, 16], [6, 16])):
            with self.subTest(actual_seq_len=actual_seq_len):
                metadata = SparseAttentionSequenceMetadata.build(
                    batch_size=2, local_seq_len=8, device=torch.device("cpu"), actual_seq_len=actual_seq_len,
                )
                self.assertEqual(metadata.kv_prefix_length, 8)
                for endpoints in metadata.sequence_lengths:
                    torch.testing.assert_close(endpoints, torch.tensor(expected_ends, dtype=torch.int32))

        for rank, query_ends, key_ends in ((0, [4, 8], [4, 10]), (1, [2, 4, 8], [6, 8, 18])):
            with self.subTest(cp_rank=rank):
                metadata = build_cp_sequence_metadata(
                    batch_size=2, local_seq_len=4, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[6, 16],
                )
                self.assertEqual(metadata.kv_prefix_length, (rank + 1) * 4)
                torch.testing.assert_close(metadata.query_ends, torch.tensor(query_ends, dtype=torch.int32))
                torch.testing.assert_close(metadata.key_ends, torch.tensor(key_ends, dtype=torch.int32))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_cp_packed_candidates_match_global_causal_documents(self) -> None:
        """
        Feature: Shared packed-attention planning and routing.
        Description: Every local query sees its complete document prefix and no future keys.
        Expectation: Every local query sees only the causal prefix of its own document.
        """
        cases = (
            (2, 8, 1, [6, 16]), (2, 8, 2, [8, 16]), (2, 8, 2, [6, 16]),
            (2, 16, 4, [14, 32]), (3, 8, 2, [6, 14, 24]), (3, 8, 2, [24]),
            (1, 8, 2, [3, 8]), (1, 8, 2, [4, 8]), (2, 8, 4, [1, 3, 7, 11, 16]),
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

    def test_cp_cross_row_document_first_query_has_one_candidate(self) -> None:
        """The reported query 3584 selects itself, not 1024 future candidates."""
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

    def test_cp_repeated_key_prefixes_accumulate_backward(self) -> None:
        """Duplicated KV prefixes sum adjoints into the original global keys."""
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_cp_omits_documents_without_local_queries(self) -> None:
        """
        Feature: Shared packed-attention planning and routing.
        Description: Packed documents may belong to disjoint CP owners without padding.
        Expectation: Zero-query documents are absent and active prefixes match the reference.
        """
        for rank, expected_key in ((0, (0, 4)), (1, (4, 8))):
            with self.subTest(cp_rank=rank):
                metadata = build_cp_sequence_metadata(
                    batch_size=1, local_seq_len=4, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[4, 8],
                )
                self.assertEqual(metadata.query_ends.tolist(), [4])
                self.assertEqual(metadata.key_ends.tolist(), [4])
                self.assertEqual(metadata.key_ranges, (expected_key,))

    def test_indexer_weights_scale_without_intermediate_bf16_rounding(self) -> None:
        """The 32-head, 128-dimension scale equals an exact division by 64."""
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_worker_projection_preserves_owner_values_without_gradients(self) -> None:
        """Feature: Detached qr worker selection.

        Description: Reorder query rows across owner-sized projection chunks.
        Expectation: The FP32 projection matches owner values and builds no graph.
        """
        module = DeepseekSparseAttention(config=_config())
        query = torch.randn(1, 7, module.q_lora_rank, requires_grad=True)
        order = torch.tensor([6, 1, 3, 0, 5, 4, 2])
        owner = module.indexer.wq_b(query).view(1, 7, module.num_index_heads, module.index_head_dim)
        worker = module.project_worker_index_query(query[:, order], None, 4)
        torch.testing.assert_close(worker, owner[:, order], rtol=1e-6, atol=1e-7)
        self.assertFalse(worker.requires_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_deferred_cp_preserves_common_routing_options(self) -> None:
        """Feature: Shared planner options through module-family promotion.

        Description: Build before the job topology and promote without reinitializing.
        Expectation: Configuration and parameter identity survive the lifecycle.
        """
        options = {
            "kv_cp_size": 2, "indexer_load_balance": "hybrid", "indexer_balance_block_size": 16,
            "indexer_zigzag_min_length": 32, "indexer_paged_k": True,
            "main_kv_comm": "all_to_all_v", "index_k_comm": "all_to_all_v", "main_kv_comm_fusion": True,
        }
        source = DeepseekSparseAttention(config=_config(), **options)
        original = dict(source.named_parameters())
        promoted, _ = prepare_model_components(source, {"cp_size": 2})
        for name, value in options.items():
            self.assertEqual(getattr(promoted, name), value)
        for name, parameter in promoted.named_parameters():
            self.assertIs(parameter, original[name])

    def test_merged_main_banks_keep_layout_and_accumulate_prefix_gradients(self) -> None:
        """Merged fields preserve repeated document prefixes and field-specific gradients."""
        for gathered in (False, True):
            with self.subTest(gathered=gathered):
                latent = torch.arange(24, dtype=torch.float32).reshape(1, 6, 1, 4).requires_grad_()
                rotary = torch.arange(12, dtype=torch.float32).reshape(1, 6, 1, 2).requires_grad_()
                combined = torch.cat((latent, rotary), dim=-1)
                receipt = Mock()
                receipt.wait.return_value = (combined, None) if gathered else combined
                pending = _PendingMainBanks((receipt,), ((0, 3), (1, 4)) if gathered else None, (4, 2))

                actual = pending.wait()
                rows = [0, 1, 2, 1, 2, 3] if gathered else list(range(6))
                for value, source in zip(actual, (latent, rotary)):
                    torch.testing.assert_close(value, source[:, rows], rtol=0, atol=0)
                    self.assertTrue(value.is_contiguous())
                (actual[0].sum() * 2 + actual[1].sum() * 3).backward()
                counts = torch.tensor([rows.count(row) for row in range(6)]).view(1, 6, 1, 1)
                torch.testing.assert_close(latent.grad, (counts * 2).expand_as(latent).float(), rtol=0, atol=0)
                torch.testing.assert_close(rotary.grad, (counts * 3).expand_as(rotary).float(), rtol=0, atol=0)
                receipt.wait.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    pending.wait()
