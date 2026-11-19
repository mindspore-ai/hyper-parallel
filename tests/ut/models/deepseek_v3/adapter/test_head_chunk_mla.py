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
"""CPU contracts for model-owned Trainer replacement and global packed metadata."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.data.batching.runtime_input import RuntimeInputContext
from hyper_parallel.models.deepseek_v3.adapter.conversion.head_chunk_mla import DeepseekV3HeadChunkMLA
from hyper_parallel.models.deepseek_v3.adapter.data.packed_mla import (
    DeepseekV3PackedIndexedCollator,
    DeepseekV3PackedMLARuntime,
    register_head_chunk_mla_attention,
)
from hyper_parallel.models.deepseek_v3.adapter.distributed.head_chunk_mla import head_chunk_mla_cp_wrapper


class TestDeepseekV3HeadChunkMLA(unittest.TestCase):
    """No devices, process groups, model downloads or dataset assets."""

    @staticmethod
    def source() -> DeepseekV3Attention:
        """Build a tiny authoritative HF attention without replacing its formulas."""
        config = DeepseekV3Config.from_dict({
            "hidden_size": 128, "num_attention_heads": 8, "num_key_value_heads": 8,
            "q_lora_rank": 48, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16, "v_head_dim": 16, "rope_interleave": False,
            "attention_bias": False, "attention_dropout": 0.0,
        })
        config._attn_implementation = "eager"  # pylint: disable=protected-access
        return DeepseekV3Attention(config, layer_idx=0)

    def test_hf_weight_mapping_and_forward_backward(self) -> None:
        """Loaded weights, HF redundant kwargs and every gradient keep their meaning."""
        torch.manual_seed(17)
        reference = self.source()
        source = copy.deepcopy(reference)
        module = DeepseekV3HeadChunkMLA(module=source, module_fqn="model.layers.0.self_attn",
                                      context={}, head_chunk_size=4, attention_backend="eager")
        self.assertIs(module.o_proj.weight, source.o_proj.weight)
        self.assertEqual(len(module.make_transforms()), 1)
        torch.testing.assert_close(module.linear_qkv.weight,
                                   torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight)),
                                   rtol=0, atol=0)
        hidden = torch.randn(1, 7, 128, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        angles = torch.randn(1, 7, 8).repeat(1, 1, 2)
        positions = (angles.cos(), angles.sin())
        mask = torch.full((1, 1, 7, 7), -torch.inf).triu(1)
        expected = reference(hidden, position_embeddings=positions, attention_mask=mask)[0]
        actual = module(other, position_embeddings=positions, position_ids=torch.arange(7)[None],
                        shift_labels=torch.zeros(1, 7, dtype=torch.long), num_items_in_batch=7)[0]
        probe = torch.randn_like(actual)
        (expected * probe).sum().backward()
        (actual * probe).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-6)
        torch.testing.assert_close(other.grad, hidden.grad, rtol=3e-5, atol=5e-6)
        gradients = {name: param.grad for name, param in reference.named_parameters()}
        for name, param in module.named_parameters():
            expected_grad = (torch.cat((gradients["q_a_proj.weight"], gradients["kv_a_proj_with_mqa.weight"]))
                             if name == "linear_qkv.weight" else gradients[name])
            torch.testing.assert_close(param.grad, expected_grad, rtol=3e-5, atol=5e-6)
        for kwargs in ({"use_cache": True}, {"past_key_values": object()},
                       {"position_ids": torch.arange(7)[None]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module(other, **kwargs)

    def test_cp_binding_retains_staged_forward(self) -> None:
        """CP binding installs no second attention implementation and rejects TP."""
        module = DeepseekV3HeadChunkMLA(module=self.source(), module_fqn="self_attn", context={},
                                      head_chunk_size=4, attention_backend="eager")
        cp = SimpleNamespace(ndim=1, size=lambda: 4)
        requests = head_chunk_mla_cp_wrapper(module, None, None, cp, None)
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0].companion_attrs["head_chunk_cp_mesh"], cp)
        self.assertFalse(hasattr(module, "mla_cp_runtime"))
        with self.assertRaisesRegex(ValueError, "TP"):
            head_chunk_mla_cp_wrapper(module, None, SimpleNamespace(size=lambda: 2), cp, None)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            head_chunk_mla_cp_wrapper(module, None, None, SimpleNamespace(ndim=2), None)

    def test_packed_global_boundaries_and_target_count(self) -> None:
        """A document spanning ranks retains global ends; ignored targets are not shifted again."""
        runtime = DeepseekV3PackedMLARuntime()
        context = RuntimeInputContext((1, 4), {"cp": 1, "tp": 0}, {"cp": 4, "tp": 1}, {})
        batch = {"cu_seq_lens": torch.tensor([0, 3, 11, 16])}
        self.assertEqual(runtime.build_runtime_inputs(batch=batch, context=context),
                         {"actual_seq_len": (3, 11, 16)})
        for boundaries in ([0, 2, 4], [0, 3, 3, 16], [3, 11, 16], [0, 3.5, 16]):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                runtime.build_runtime_inputs(batch={"cu_seq_lens": boundaries}, context=context)
        collate = DeepseekV3PackedIndexedCollator(eod_token_id=1)
        result = collate([{"input_ids": torch.tensor([2, 1, 4, 5]),
                           "labels": torch.tensor([1, 4, 5, -100])}])
        torch.testing.assert_close(result["labels"], torch.tensor([[1, -100, 5, -100]]))
        self.assertEqual(int((result["labels"] != -100).sum()), 2)

    def test_attention_registration_is_explicit(self) -> None:
        """Do not patch HF SDPA or register a CP-local dense mask implementation."""
        with patch("hyper_parallel.models.deepseek_v3.adapter.data.packed_mla.AttentionInterface.register") as register:
            name = register_head_chunk_mla_attention()
        self.assertEqual(name, "head_chunk_mla_native")
        register.assert_called_once_with(name, npu_fusion_attention_forward)
