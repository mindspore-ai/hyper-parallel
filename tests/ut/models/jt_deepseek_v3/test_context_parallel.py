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
"""CPU contracts for JT token halos, head statistics and managed projections."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.components.modules.mtp import DeepseekV3MTP
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_data import JTSequenceRuntime
from hyper_parallel.data.batching.runtime_input import RuntimeInputContext
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLAContextParallel
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import (
    _JTTrainingContext, _head_projection_request,
)
from hyper_parallel.models.jt_deepseek_v3.configuration_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Attention, observed_fusion_attention,
)
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer


class TestJTContextParallel(unittest.TestCase):
    """Check contracts independently of real distributed setup."""

    def test_head_projection_keeps_parameters_and_hooks(self):
        """Selective KV projection retains FSDP hooks and Parameter identity."""
        linear = torch.nn.Linear(3, 16, bias=True).double()
        weight = linear.weight
        calls = []
        linear.register_forward_pre_hook(lambda _module, _args: calls.append("pre"))
        linear.register_forward_hook(lambda _module, _args, _output: calls.append("post"))
        _commit_forward_rewrite(_head_projection_request(linear, 4))
        inputs = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        output = linear(inputs, mla_head_range=(1, 3))
        expected = inputs @ weight[4:12].T + linear.bias[4:12]
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertIs(linear.weight, weight)
        self.assertEqual(calls, ["pre", "post"])
        self.assertEqual(weight.grad[:4].count_nonzero().item(), 0)
        self.assertEqual(weight.grad[12:].count_nonzero().item(), 0)
        self.assertGreater(weight.grad[4:12].abs().sum().item(), 0)

    def test_mtp_shift_matches_global_sequence_across_two_depths(self):
        """Halos preserve next targets across ranks and zero only the global tail."""
        context = object.__new__(_JTTrainingContext)
        context.degree, context.cp_group = 2, None
        values = (torch.arange(8).reshape(1, 8), torch.arange(8, 16).reshape(1, 8),
                  torch.tensor([[True, True, False, True, True, False, True, True]]))
        for _depth in range(2):
            shards = [tuple(value.chunk(2, 1)[rank] for value in values) for rank in range(2)]
            firsts = [torch.stack([value[:, 0].long() for value in shard], -1) for shard in shards]

            def gather(outputs: list[torch.Tensor], _input: torch.Tensor, group: Any) -> None:
                """Supply both first-token halos without distributed initialization."""
                del group
                for output, first in zip(outputs, firsts):
                    output.copy_(first)

            results = []
            with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_gather",
                       side_effect=gather):
                for rank in range(2):
                    context.rank = rank
                    results.append(context.shift_inputs(*shards[rank]))
            values = tuple(torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), 1) for value in values)
            for index, expected in enumerate(values):
                torch.testing.assert_close(torch.cat([part[index] for part in results], 1), expected)

    def test_fa_observation_requests_tnd_statistics(self):
        """Per-head maxima follow explicit TND or BNSD statistics layout."""
        query = torch.zeros(1, 2, 3, 4)
        maxima = torch.tensor([2.0, 7.0])
        for packed in (True, False):
            stats = maxima.view(1, 2, 1).expand(3, 2, 8).clone() if packed else (
                maxima.view(1, 2, 1, 1).expand(1, 2, 3, 8).clone())
            output = torch.zeros(3, 2, 4) if packed else torch.zeros_like(query)
            observer = SimpleNamespace(max_logits_val=None, is_causal=True)
            with patch("torch_npu.npu_fusion_attention", return_value=(output, stats)) as kernel:
                observed_fusion_attention(observer, query, query, query, None,
                                          actual_seq_len=(3,) if packed else None)
            self.assertEqual(kernel.call_args.kwargs["softmax_layout"], "TND" if packed else "")
            torch.testing.assert_close(observer.max_logits_val, maxima)

    def test_mtp_loss_uses_global_target_denominator(self):
        """Local MTP sums use global target counts before the model-level CP sum."""
        context = object.__new__(_JTTrainingContext)
        context.cp_group = None

        def loss_function(*, logits: torch.Tensor, shift_labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Use independent CE summation as the token-loss oracle."""
            self.assertEqual(kwargs["num_items_in_batch"], 1)
            return F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(), reduction="sum")

        context.model = SimpleNamespace(config=SimpleNamespace(vocab_size=4), loss_function=loss_function)
        logits = torch.randn(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[1, -100, 2]])
        with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_reduce",
                   side_effect=lambda count, **_kwargs: count.fill_(4)):
            value = context.token_loss(logits, labels, labels >= 0)
        expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / 4
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(torch.autograd.grad(value, logits)[0], torch.autograd.grad(expected, logits)[0])

    def test_head_divisibility_fails_before_communication(self):
        """The TP-local head shard divides evenly over the CP mesh."""
        mesh = SimpleNamespace(size=lambda: 4, ndim=1, get_local_rank=lambda: 0)
        with self.assertRaisesRegex(ValueError, "heads must be divisible"):
            MLAContextParallel(mesh, 2, "expanded_ulysses")
        runtime = MLAContextParallel(mesh, 8, "latent_kv_head")
        self.assertEqual(runtime.head_range, (0, 2))

    def test_qk_scaling_handles_fsdp_shard_crossing_a_head(self):
        """The local row interval can begin inside a head instead of on its boundary."""
        class Shard:
            """Describe only the parameter layout consumed by clipping."""

            shape = (16, 3)
            device_mesh = object()
            placements = ("rows",)

            def __init__(self, value: torch.Tensor) -> None:
                """Store one local parameter slice."""
                self.value = value

            def to_local(self) -> torch.Tensor:
                """Return this rank's six rows."""
                return self.value

        parameter = Shard(torch.ones(6, 3))

        def distribute(factors: torch.Tensor, mesh: Any, placements: tuple) -> Shard:
            """Select a shard deliberately beginning inside a head."""
            self.assertIs(mesh, parameter.device_mesh)
            self.assertEqual(placements, parameter.placements)
            return Shard(factors[6:12])

        with patch.object(jt_optimizer, "DTensor", Shard), patch.object(
                jt_optimizer, "distribute_tensor", side_effect=distribute):
            jt_optimizer._scale_projection(parameter, torch.tensor([0.25, 0.64]), 4, 4, query=True)
        # A 16-row tensor sharded three ways gives rank 1 rows [6:12], crossing head 0 into head 1.
        expected = torch.tensor([0.25, 0.25, 0.8, 0.8, 0.8, 0.8])[:, None].expand(-1, 3)
        torch.testing.assert_close(parameter.value, expected)

    def test_auxiliary_loss_matches_global_router_gradient(self):
        """CP/TP replicas and SP partitions differentiate the same global router objective."""
        torch.manual_seed(5)
        hidden = torch.randn(16, 3, dtype=torch.float64)
        indices = torch.randn(16, 8).topk(2, dim=-1).indices
        counts = torch.bincount(indices.flatten(), minlength=8).float()
        module = SimpleNamespace(config=SimpleNamespace(n_routed_experts=8, moe_aux_loss_coeff=0.01))
        for sequence_parallel in (False, True):
            for loss_parallel in (False, True):
                weight = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
                scores = (hidden @ weight).sigmoid()
                normalized = scores / scores.sum(-1, keepdim=True)
                expected = (normalized.mean(0) * (counts / counts.sum())).sum() * 0.08
                context = object.__new__(_JTTrainingContext)
                context.statistics_group, context.tp_group = "statistics", "tp"
                context.degree, context.tp_size, context.loss_parallel = 2, 2, loss_parallel
                contributions = []
                for cp_rank in range(2):
                    local_scores, local_indices = scores.chunk(2)[cp_rank], indices.chunk(2)[cp_rank]
                    tp_scores = local_scores.chunk(2) if sequence_parallel else (local_scores, local_scores)
                    tp_indices = local_indices.chunk(2) if sequence_parallel else (local_indices, local_indices)
                    mean_sum = sum((part / part.sum(-1, keepdim=True)).mean(0).mul(
                        counts / counts.sum()).sum() * 0.08 for part in tp_scores).detach()

                    def reduce(tensor: torch.Tensor, group: str) -> None:
                        """Provide independently computed global counts or the TP auxiliary sum."""
                        tensor.copy_(counts if group == "statistics" else mean_sum)

                    with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed."
                               "context_parallel.dist.all_reduce", side_effect=reduce):
                        for local_index, local_score in zip(tp_indices, tp_scores):
                            _, value = context.routing_statistics(module, local_index, local_score)
                            contributions.append(value)
                total = sum(contributions)
                torch.testing.assert_close(total.detach() / 2, expected.detach(), rtol=1e-7, atol=1e-12)
                # FSDP divides TP replicas only when the vocabulary-parallel loss is disabled.
                if not loss_parallel:
                    total = total / 2
                actual_gradient = torch.autograd.grad(total, weight, retain_graph=True)[0]
                expected_gradient = torch.autograd.grad(expected, weight)[0]
                self.assertGreater(expected_gradient.norm().item(), 1e-8)
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-7, atol=1e-12)


    def test_packed_mtp_stops_at_each_document_for_two_depths(self):
        """Packed future embeddings and targets never leak from the following document."""
        prediction = DeepseekV3MTP(hidden_size=2, num_layers=2,
                                  decoder_factory=lambda _index: torch.nn.Identity())
        embedding = torch.nn.Embedding(32, 2)
        inputs = torch.tensor([[0, 1, 2, 10, 11, 20]])
        tails = torch.tensor([[False, False, True, False, True, True]])
        embedded, targets = [], []
        embedding.register_forward_pre_hook(lambda _module, args: embedded.append(args[0].clone()))

        def loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Capture supervision independently of decoder arithmetic."""
            targets.append(labels.masked_fill(mask == 0, -100).clone())
            return logits.float().sum() * 0

        prediction(torch.zeros(1, 6, 2), inputs, embedding=embedding, head=torch.nn.Linear(2, 32),
                   labels=inputs + 1, loss_mask=torch.ones_like(inputs), loss_factor=0.3,
                   sequence_end_mask=tails, loss_fn=loss)
        torch.testing.assert_close(embedded[0], torch.tensor([[1, 2, 0, 11, 0, 0]]))
        torch.testing.assert_close(embedded[1], torch.tensor([[2, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(targets[0], torch.tensor([[2, 3, -100, 12, -100, -100]]))
        torch.testing.assert_close(targets[1], torch.tensor([[3, -100, -100, -100, -100, -100]]))

    def test_packed_metadata_and_zero_local_supervision(self):
        """Global boundaries survive CP; only globally empty supervision is rejected."""
        runtime = JTSequenceRuntime()
        info = RuntimeInputContext(local_input_shape=(1, 4), parallel_ranks={"cp": 0},
                                   parallel_sizes={"cp": 2}, options={})
        metadata = runtime.build_runtime_inputs(batch={"cu_seq_lens": torch.tensor([0, 3, 8])}, context=info)
        self.assertEqual(metadata, {"actual_seq_len": (3, 8)})
        context = object.__new__(_JTTrainingContext)
        context.degree, context.rank, context.cp_group = 2, 0, None
        batch = {"input_ids": torch.arange(4).view(1, 4), "shift_labels": torch.full((1, 4), -100), **metadata}
        with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_reduce",
                   side_effect=lambda count, **_kwargs: count.fill_(2)):
            _, result = context.prepare(None, (), batch)
        self.assertEqual(result["actual_seq_len"], (3, 8))
        self.assertEqual(result["sequence_start"], 0)
        with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_reduce"):
            with self.assertRaisesRegex(ValueError, "somewhere in the global"):
                context.prepare(None, (), batch)

    def test_packed_reference_attention_matches_independent_documents(self):
        """CPU packed outputs and gradients equal separate single-document executions."""
        torch.manual_seed(91)
        config = JTDeepseekV3Config.from_dict({
            "hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 2,
            "q_lora_rank": 4, "kv_lora_rank": 4, "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 4, "v_head_dim": 2,
        })
        packed = JTDeepseekV3Attention(config, 0).double()
        separate = deepcopy(packed)
        inputs = torch.randn(1, 7, 8, dtype=torch.float64, requires_grad=True)
        other = inputs.detach().clone().requires_grad_()
        angles = torch.randn(1, 7, 4, dtype=torch.float64)
        positions = (angles.cos(), angles.sin())
        actual = packed(inputs, position_embeddings=positions, actual_seq_len=(3, 7))[0]
        expected = torch.cat([
            separate(other[:, begin:end], position_embeddings=tuple(value[:, begin:end] for value in positions))[0]
            for begin, end in ((0, 3), (3, 7))
        ], dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(inputs.grad, other.grad, atol=1e-7, rtol=1e-6)
        for parameter, reference in zip(packed.parameters(), separate.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=1e-6)
