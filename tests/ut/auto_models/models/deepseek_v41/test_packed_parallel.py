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
"""Packed CP adapter parity for native Engram and replicated experts."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
except ImportError as exc:
    raise unittest.SkipTest(f"DeepSeek-V4 Transformers dependency unavailable: {exc}") from exc

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_engram_compute_fn,
    deepseek_v41_replicated_moe_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.native_module_parity import HyperParallelMoE
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Engram
from tests.common.mark_utils import arg_mark


class TestPackedParallel(unittest.TestCase):
    """Preserve complete-document semantics across the CP boundary."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_engram_cross_shard_forward_and_gradients(self) -> None:
        """Feature: Packed Engram CP history.
        Description: Split a document between two CP ranks with separate PAD packing.
        Expectation: Outputs and summed parameter gradients match the native full row.
        """
        torch.manual_seed(7)
        config = SimpleNamespace(hidden_size=4, hc_mult=2, rms_norm_eps=1e-6)
        assets = {
            "layer_ids": [0], "max_ngram_size": 3, "num_heads": 1, "head_dim": 2,
            "primes": [[[17], [19]]], "num_embeddings": [36],
            "multipliers": [[101, 103, 107]], "token_map": list(range(32)), "pad_token_id": 2,
        }
        reference = DeepseekV41Engram(config, 0, assets)
        source = copy.deepcopy(reference)
        candidate = EngramModule(module=source)
        for name, parameter in candidate.named_parameters():
            self.assertIs(parameter, dict(source.named_parameters())[name])
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 2, 2]])
        starts = torch.tensor([[True, False, True, False, False, False, True, False]])
        inputs = torch.randn(1, 8, 2, 4, requires_grad=True)
        expected = reference(inputs, ids, starts)
        expected.square().sum().backward()
        outputs = []
        input_gradients = []
        for rank in range(2):
            local_slice = slice(rank * 4, (rank + 1) * 4)
            local = inputs.detach()[:, local_slice].clone().requires_grad_()
            mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda: "cp")
            compute = deepseek_v41_engram_compute_fn(
                module=candidate, mesh=None, cp_mesh=mesh, tp_mesh=None, ep_mesh=None,
            )
            with patch("hyper_parallel.components.modules.engram._all_gather_sequence",
                       side_effect=[ids, starts]):
                output = compute(candidate, local, ids[:, local_slice], starts[:, local_slice])
                output.square().sum().backward()
            outputs.append(output)
            input_gradients.append(local.grad)
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
        torch.testing.assert_close(torch.cat(input_gradients, dim=1), inputs.grad)
        for name, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_replicated_moe_matches_native_forward_backward(self) -> None:
        """Feature: EP1 training-aware MoE adapter.
        Description: Execute real experts, auxiliary loss and padding-masked router loads.
        Expectation: Native outputs and gradients are retained without EP communication.
        """
        torch.manual_seed(8)
        config = DeepseekV4Config.from_dict({
            "hidden_size": 8, "moe_intermediate_size": 12, "n_routed_experts": 4,
            "n_shared_experts": 1, "num_experts_per_tok": 2, "swiglu_limit": 1.5,
        })
        config.scoring_func = "sqrtsoftplus"
        config.router_aux_loss_coef = 0.01
        reference = HyperParallelMoE(config)
        reference.is_hash = False
        for parameter in reference.parameters():
            if parameter.requires_grad:
                torch.nn.init.normal_(parameter, std=0.1)
        candidate = copy.deepcopy(reference)
        compute = deepseek_v41_replicated_moe_compute_fn(
            module=candidate, mesh=None, cp_mesh=None, tp_mesh=None, ep_mesh=None,
        )
        inputs = torch.randn(1, 6, 8, requires_grad=True)
        local = inputs.detach().clone().requires_grad_()
        metadata = {
            "router_token_mask": torch.tensor([[True, True, True, True, False, False]]),
            "router_sequence_ids": torch.tensor([[0, 0, 1, 1, 2, 2]]), "router_num_sequences": 3,
        }
        expected = reference(inputs, **metadata)
        output = compute(candidate, local, **metadata)
        expected.square().sum().backward()
        output.square().sum().backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(local.grad, inputs.grad)
        for name, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
        torch.testing.assert_close(candidate.gate.tokens_per_expert, reference.gate.tokens_per_expert)
        self.assertEqual(candidate.gate.tokens_per_expert.sum().item(), 8)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_replicated_router_uses_partition_groups_and_local_metadata(self) -> None:
        """Feature: Replicated MoE distributed statistics.
        Description: Bind DP and CP/TP groups with a TP-local metadata slice.
        Expectation: Only sequence axes reduce the auxiliary loss; DP also reduces bias loads.
        """
        states = torch.ones(1, 2, 4)
        gate = Mock(return_value=(None, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)))
        module = SimpleNamespace(gate=gate, experts=Mock(return_value=states.reshape(2, 4)),
                                 shared_experts=torch.zeros_like, is_hash=False)
        dp = SimpleNamespace(size=lambda: 2, get_group=lambda: "dp")
        cp = SimpleNamespace(size=lambda: 2, get_group=lambda: "cp")
        tp = SimpleNamespace(size=lambda: 2, get_group=lambda: "tp", get_local_rank=lambda: 1)
        mesh = Mock(mesh_dim_names=("dp", "cp", "tp"))
        mesh.__getitem__ = Mock(side_effect={"dp": dp, "cp": cp, "tp": tp}.__getitem__)
        compute = deepseek_v41_replicated_moe_compute_fn(
            module=module, mesh=mesh, cp_mesh=cp, tp_mesh=tp, ep_mesh=None,
        )
        valid = torch.tensor([[True, True, True, False]])
        sequence_ids = torch.tensor([[0, 0, 1, 2]])
        compute(module, states, router_token_mask=valid, router_sequence_ids=sequence_ids,
                router_num_sequences=3)
        actual = gate.call_args.kwargs
        torch.testing.assert_close(actual["token_mask"], valid[:, 2:])
        torch.testing.assert_close(actual["sequence_ids"], sequence_ids[:, 2:])
        self.assertEqual(actual["num_sequences"], 3)
        self.assertEqual(actual["sequence_partition_groups"], ("cp", "tp"))
        self.assertEqual(gate.expert_bias_update_groups, ("dp", "cp", "tp"))
        with self.assertRaisesRegex(ValueError, "requires EP1"):
            deepseek_v41_replicated_moe_compute_fn(
                module=module, mesh=mesh, cp_mesh=cp, tp_mesh=tp, ep_mesh=object(),
            )
