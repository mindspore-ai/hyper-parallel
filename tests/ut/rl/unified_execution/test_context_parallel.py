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
"""CPU tests for CP coordinate ownership and explicit support boundaries."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from rl.trainer.build import _TrainerBuilder
from rl.unified_execution.core.contract import build_unified_payload
from rl.unified_execution.runtimes.hyper_parallel.attention import prepare_cp_attention_metadata
from rl.workers.adapters.context_parallel import compute_cp_token_log_probs, slice_cp_targets, gather_cp_log_probs


def mesh(rank=0, size=2):
    """Supply topology without starting a distributed process group."""
    return SimpleNamespace(size=lambda: size, get_local_rank=lambda: rank, get_group=object)


class TestContextParallel(unittest.TestCase):
    """Test geometry independently of the native attention implementation."""

    def test_document_prefix_uses_global_offset(self):
        """A rank crossing packed boundaries gets a distinct KV prefix per document."""
        metadata = prepare_cp_attention_metadata(torch.ones(1, 12), 1, 6, torch.device('cpu'), mesh(1), (8, 4))
        self.assertEqual(metadata.query_indices.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(metadata.key_indices.tolist(), list(range(12)))
        self.assertEqual(metadata.cu_seqlens_q.tolist(), [0, 2, 6])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 8, 12])

    def test_left_right_padding_and_empty_rank(self):
        """Padding positions are excluded while full KV coordinates remain physical."""
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        metadata = prepare_cp_attention_metadata(mask, 2, 4, torch.device('cpu'), mesh(1))
        self.assertEqual(metadata.query_indices.tolist(), [0, 1, 2])
        self.assertEqual(metadata.key_indices.tolist(), [2, 3, 4, 5, 6])
        empty = prepare_cp_attention_metadata(mask[:1] * 0, 1, 4, torch.device('cpu'), mesh(1))
        self.assertEqual(empty.query_indices.numel(), 0)
        self.assertEqual(empty.cu_seqlens_q.tolist(), [0])

    def test_invalid_global_metadata_fails(self):
        """Local masks and invalid packed boundaries cannot be mistaken for global geometry."""
        with self.assertRaisesRegex(ValueError, 'padding mask'):
            prepare_cp_attention_metadata(torch.ones(1, 4), 1, 4, torch.device('cpu'), mesh())
        for documents in ((3, 4), (0, 8), [4, 4]):
            with self.subTest(documents=documents), self.assertRaisesRegex(ValueError, 'document lengths'):
                prepare_cp_attention_metadata(None, 1, 4, torch.device('cpu'), mesh(), documents)

    def test_next_token_target_crosses_cp_boundary(self):
        """The final local logit predicts the first token on the next CP rank."""
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        seen = []

        def forward(**kwargs):
            seen.append(kwargs)
            logits = torch.zeros(1, 4, 16)
            logits.scatter_(-1, (kwargs['input_ids'] + 1).unsqueeze(-1), 10.)
            return SimpleNamespace(logits=logits)

        output = compute_cp_token_log_probs(forward, ids, torch.ones_like(ids), mesh())
        self.assertTrue(bool((output > -.01).all()))
        self.assertEqual(seen[0]['position_ids'].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(seen[0]['attention_mask'].shape, ids.shape)
        compute_cp_token_log_probs(forward, ids, torch.ones_like(ids), mesh(1))
        self.assertEqual(seen[1]['position_ids'].tolist(), [[4, 5, 6, 7]])

    def test_loss_targets_share_padded_query_partition(self):
        """Last query and alignment padding carry zero loss, without dropping a boundary."""
        value = torch.arange(1, 7).reshape(1, 6)
        first = slice_cp_targets(value, mesh(), 4)
        second = slice_cp_targets(value, mesh(1), 4)
        self.assertEqual(first.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(second.tolist(), [[5, 6, 0, 0]])

    def test_target_restore_is_detached_and_ordered(self):
        """Probability restoration has no implicit autograd collective."""
        with self.assertRaisesRegex(ValueError, 'detached'):
            gather_cp_log_probs(torch.ones(1, 3, requires_grad=True), mesh())
        def gather(parts, local, **kwargs):
            del kwargs
            parts[0].copy_(local)
            parts[1].copy_(local + 3)
        with patch('rl.workers.adapters.context_parallel.dist.all_gather', side_effect=gather) as collective:
            actual = gather_cp_log_probs(torch.tensor([[1., 2., 3.]]), mesh())
        self.assertEqual(actual.tolist(), [[1., 2., 3., 4., 5., 6.]])
        collective.assert_called_once()

    @staticmethod
    def config():
        root = Path(__file__).resolve().parents[4]
        path = root / 'hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml'
        return yaml.safe_load(path.read_text(encoding='utf-8'))

    def test_cp_contract_keeps_inference_cp_one_and_experts_cross_cp(self):
        config = self.config()
        del config['rollout']['vllm']['context_parallel_size']
        config['rollout']['vllm']['data_parallel_size'] = 2
        config['train']['accelerator']['ep'] = 4
        payload = build_unified_payload(config)
        self.assertEqual(payload['training_parallelism']['cp'], 2)
        self.assertEqual(payload['parallelism']['cp'], 1)
        self.assertEqual(payload['expert_parallelism']['training'], 4)

    def test_unvalidated_combinations_fail(self):
        for key, value in (('activation_checkpoint', 'full'), ('loss_parallel', True)):
            config = self.config()
            config['train']['accelerator'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Training CP'):
                build_unified_payload(config)
        config = self.config()
        config['train']['plan_overrides'].pop()
        with self.assertRaisesRegex(ValueError, 'CP attention binding'):
            build_unified_payload(config)
        config = self.config()
        del config['consistency']
        with self.assertRaisesRegex(ValueError, 'explicit unified'):
            build_unified_payload(config)
        config = self.config()
        config['train']['accelerator']['ep'] = 1
        with self.assertRaisesRegex(ValueError, 'graph-capable EP'):
            build_unified_payload(config)


class TestCPRequestOwnership(unittest.TestCase):
    """Logical DP remains the sampling domain when requests are shared across TP and CP."""

    def test_all_tp_cp_siblings_share_one_request_rank(self):
        for cp_rank in range(2):
            for tp_rank in range(2):
                with self.subTest(cp=cp_rank, tp=tp_rank):
                    request_mesh = SimpleNamespace(get_group=lambda: 'request-group',
                                                   get_local_rank=lambda: cp_rank * 2 + tp_rank, size=lambda: 4)
                    device_mesh = {'tp': object(), ('cp', 'tp'): SimpleNamespace(flatten=lambda name: request_mesh)}
                    builder = object.__new__(_TrainerBuilder)
                    builder.parallel_dims = SimpleNamespace(tp_size=2, cp_size=2, tp_rank=tp_rank, cp_rank=cp_rank,
                                                             dp_rank=1, dp_size=2, device_mesh=device_mesh)
                    configure = Mock()
                    builder.rollout_engine = SimpleNamespace(configure_trainer_tensor_parallel=configure)
                    builder._configure_rollout_tensor_parallel()
                    configure.assert_called_once_with(group='request-group', tp_rank=cp_rank * 2 + tp_rank,
                                                       tp_size=4, request_rank=1, request_size=2)
