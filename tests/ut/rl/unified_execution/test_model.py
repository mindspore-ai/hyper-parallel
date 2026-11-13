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
"""CPU protocol tests around the actual HF Qwen3 decoder and HP wrapper."""

from __future__ import annotations

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from rl.unified_execution.contract import build_unified_payload, fingerprint
from rl.unified_execution.runtimes.hyper_parallel.builder import build_inference_model
from rl.unified_execution.runtimes.vllm.model_adapter import HyperParallelQwen3ForCausalLM
from rl.unified_execution.runtimes.vllm.paged_attention import paged_attention_forward
from rl.unified_execution.runtimes.vllm.parallel_context import InferenceParallelContext
from rl.unified_execution.validation import capture_training_structure
from rl.workers.engine.materialization import register_qwen3_materialization
from rl.workers.engine.model import build_role_model
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import Qwen3Config
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state


class _ReferencePagedCore(nn.Module):
    """CPU stand-in for request boundaries owned by vLLM forward metadata."""

    def __init__(self, attention: nn.Module, lengths: tuple[int, ...]) -> None:
        """Store the head layout and explicit request lengths for the CPU reference."""
        super().__init__()
        self.scale = attention.scaling
        self.head_dim = attention.config.head_dim
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.lengths = lengths

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Independently compute each scheduled request with causal attention."""
        outputs, offset = [], 0
        for length in self.lengths:
            query_part = query[offset:offset + length].view(length, self.heads, self.head_dim).transpose(0, 1)
            key_part = key[offset:offset + length].view(length, self.kv_heads, self.head_dim).transpose(0, 1)
            value_part = value[offset:offset + length].view(length, self.kv_heads, self.head_dim).transpose(0, 1)
            # Pylint cannot infer this Torch C-extension callable; real CPU tests execute it.
            output = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                query_part, key_part, value_part, is_causal=True, scale=self.scale, enable_gqa=True,
            )
            outputs.append(output.transpose(0, 1).flatten(1))
            offset += length
        return torch.cat(outputs)


class TestUnifiedModel(unittest.TestCase):
    """The wrapper must retain the original decoder and absolute-position RoPE."""

    def setUp(self) -> None:
        """Build a tiny actual Qwen3 model and a shared empty replacement plan."""
        device = patch(
            "hyper_parallel.models._transformers.auto_model._current_device", return_value=torch.device("cpu"),
        )
        device.start()
        self.addCleanup(device.stop)
        torch.manual_seed(13)
        self.config = Qwen3Config.from_dict({
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "max_position_embeddings": 64, "tie_word_embeddings": True,
        })
        self.reference = build_inference_model(self.config, [], attn_implementation="eager")
        self.payload = build_unified_payload({
            "consistency": {"schema_version": 1, "mode": "architecture"},
            "model": {"name": "qwen3"}, "train": {"plan_overrides": []},
            "rollout": {"vllm": {"enforce_eager": True}},
        })

    def _wrapper(self, lengths: tuple[int, ...]) -> HyperParallelQwen3ForCausalLM:
        """Replace the device-specific core while preserving the production bridge."""
        def install(model: nn.Module, unused_config: object, unused_prefix: str) -> None:
            """Keep the real interface conversion while supplying the CPU reference core."""
            del unused_config, unused_prefix
            ALL_ATTENTION_FUNCTIONS.register("hyper_parallel_vllm_test", paged_attention_forward)
            for layer in model.layers:
                layer.self_attn.vllm_attention = _ReferencePagedCore(layer.self_attn, lengths)
            model.config._attn_implementation = "hyper_parallel_vllm_test"  # pylint: disable=protected-access

        self.config.hyper_rl_unified_execution = self.payload
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=self.config, enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            additional_config={}, quant_config=None,
        )
        with patch("rl.unified_execution.runtimes.vllm.parallel_context.dist.is_initialized", return_value=False), \
                patch("rl.unified_execution.runtimes.vllm.model_adapter.install_paged_attention", side_effect=install):
            wrapper = HyperParallelQwen3ForCausalLM(vllm_config=config)
        wrapper.load_weights(self.reference.state_dict().items())
        return wrapper

    @torch.no_grad()
    def test_flattened_requests_remain_independent(self) -> None:
        """The shared decoder delegates packed request isolation to its core."""
        wrapper = self._wrapper((3, 2))
        tokens = torch.tensor([1, 2, 3, 17, 19])
        positions = torch.tensor([0, 1, 2, 12, 13])
        outputs = []
        for start, end in ((0, 3), (3, 5)):
            result = self.reference.model(
                tokens[start:end][None], position_ids=positions[start:end][None], use_cache=False,
            )
            outputs.append(result.last_hidden_state[0])
        torch.testing.assert_close(wrapper(tokens, positions), torch.cat(outputs), atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_sequence_padding_never_reaches_paged_cache(self) -> None:
        """An odd scheduler length is restored after hidden/head padding."""
        wrapper = self._wrapper((3,))
        wrapper.parallel_context = InferenceParallelContext(tp_size=2, sequence_parallel=True)
        tokens, positions = torch.tensor([2, 5, 7]), torch.arange(3)
        expected = self.reference.model(
            tokens[None], position_ids=positions[None], use_cache=False,
        ).last_hidden_state[0]
        hidden = wrapper(tokens, positions)
        torch.testing.assert_close(hidden, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(wrapper.compute_logits(hidden), self.reference.lm_head(expected),
                                   atol=1e-6, rtol=1e-5)
        embedded = wrapper.embed_input_ids(tokens)
        self.assertEqual(tuple(embedded.shape), (3, self.config.hidden_size))
        torch.testing.assert_close(wrapper(None, positions, inputs_embeds=embedded), hidden, atol=0, rtol=0)

    def test_one_padding_graph_handles_odd_even_and_single_rows(self) -> None:
        """Replay the same FX graph at divisibility boundaries, without Dynamo recompilation."""
        for multiple in (2, 4):
            for shape, dtype in (((128,), torch.int64), ((128, 3), torch.float32)):
                context = SimpleNamespace(
                    parallel_context=InferenceParallelContext(tp_size=multiple, sequence_parallel=True),
                )
                graph = make_fx(
                    partial(HyperParallelQwen3ForCausalLM._pad_rows, context),
                    tracing_mode="symbolic",
                )(torch.ones(shape, dtype=dtype))
                for rows in (1, 2, 3, 7, 8, 31, 64, 127, 128):
                    with self.subTest(tp=multiple, shape=shape, rows=rows):
                        value = torch.arange(rows, dtype=dtype)
                        if len(shape) == 2:
                            value = value[:, None].expand(rows, shape[1]).clone().requires_grad_()
                        actual = graph(value)
                        expected_rows = ((rows + multiple - 1) // multiple) * multiple
                        self.assertEqual(actual.shape[0], expected_rows)
                        torch.testing.assert_close(actual[:rows], value, rtol=0, atol=0)
                        torch.testing.assert_close(actual[rows:], torch.zeros_like(actual[rows:]), rtol=0, atol=0)
                        if value.requires_grad:
                            actual.sum().backward()
                            torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)

    def test_one_attention_graph_restores_dynamic_sp_padding(self) -> None:
        """Attention consumes scheduled rows and restores padded rows without shape branches."""
        for multiple in (2, 4):
            context = SimpleNamespace(
                parallel_context=InferenceParallelContext(tp_size=multiple, sequence_parallel=True),
            )
            module = SimpleNamespace(unified_sp_multiple=multiple,
                                     vllm_attention=lambda query, _key, _value: query)

            def make_forward(module, context):
                """Bind one parallel configuration without changing the traced signature."""
                def forward(value: torch.Tensor) -> torch.Tensor:
                    """Exercise the actual model entry and attention exit padding together."""
                    module.unified_token_count = value.shape[0]
                    padded = HyperParallelQwen3ForCausalLM._pad_rows(context, value)
                    query = padded.reshape(1, -1, 2, 4).transpose(1, 2)
                    return paged_attention_forward(module, query, query, query, None)[0]

                return forward

            graph = make_fx(make_forward(module, context), tracing_mode="symbolic")(torch.ones(128, 8))
            for rows in (1, 2, 3, 7, 8, 31, 127, 128):
                with self.subTest(tp=multiple, rows=rows):
                    value = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
                    actual = graph(value).flatten(2).squeeze(0)
                    self.assertEqual(actual.shape[0], ((rows + multiple - 1) // multiple) * multiple)
                    torch.testing.assert_close(actual[:rows], value, rtol=0, atol=0)
                    torch.testing.assert_close(actual[rows:], torch.zeros_like(actual[rows:]), rtol=0, atol=0)

    def test_empty_attention_schedule_keeps_explicit_head_width(self) -> None:
        """An empty scheduled tensor must not infer an ambiguous reshape dimension."""
        core = Mock(return_value=torch.empty(0, 16))
        output, weights = paged_attention_forward(
            SimpleNamespace(vllm_attention=core), torch.empty(1, 4, 0, 4),
            torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4), None,
        )
        self.assertEqual(tuple(output.shape), (1, 0, 4, 4))
        self.assertIsNone(weights)
        self.assertEqual([tuple(tensor.shape) for tensor in core.call_args.args], [(0, 16), (0, 8), (0, 8)])

    def test_actual_training_structure_is_required_to_match(self) -> None:
        """A valid payload digest cannot conceal a different realized Actor model."""
        report = capture_training_structure(self.reference, self.payload)
        self.payload["training_structure"] = report
        wrapper = self._wrapper((2,))
        self.assertEqual(wrapper.unified_structure_report, report)
        report["structure"]["modules"]["model.norm"] = "different.Norm"
        report["fingerprint"] = fingerprint(report["structure"])
        with self.assertRaisesRegex(ValueError, "differs from the Actor"):
            self._wrapper((2,))


if __name__ == "__main__":
    unittest.main()


class TestQwen3Materialization(unittest.TestCase):
    """Keep non-parameter RoPE state identical without replacing live storage."""

    def setUp(self) -> None:
        """Use the real Qwen3-4B head dimension and rotary base on CPU."""
        self.config = Qwen3Config.from_dict({
            "head_dim": 128, "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        })
        with torch.device("cpu"):
            self.expected = Qwen3RotaryEmbedding(self.config).inv_freq.clone()

    def test_meta_registration_defers_storage_until_materialization(self) -> None:
        """Meta registration keeps meta buffers; the public rebuild fills live storage later."""
        with torch.device("meta"):
            rotary = Qwen3RotaryEmbedding(self.config)
            register_qwen3_materialization(rotary)
        self.assertTrue(rotary.inv_freq.is_meta)
        self.assertTrue(rotary.original_inv_freq.is_meta)
        rotary.to_empty(device="cpu")
        identities = {name: id(buffer) for name, buffer in rotary.named_buffers()}
        rebuild_materialized_state(rotary, MaterializationContext(reason="random_init", device=torch.device("cpu")))
        for name in ("inv_freq", "original_inv_freq"):
            self.assertEqual(id(getattr(rotary, name)), identities[name])
            torch.testing.assert_close(getattr(rotary, name), self.expected, rtol=0, atol=0)

    def test_non_meta_from_config_applies_canonical_recipe(self) -> None:
        """The real inference builder restores device-specific constructor frequencies."""
        config = Qwen3Config.from_dict({
            "vocab_size": 16, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        })
        original = Qwen3RotaryEmbedding.compute_default_rope_parameters
        expected, _ = original(config, device=torch.device("cpu"))

        def constructor_frequency(config: Qwen3Config, device: torch.device | None = None,
                                  seq_len: int | None = None) -> tuple[torch.Tensor, float]:
            """Simulate ambient-device constructor drift while preserving the CPU recipe."""
            value, scale = original(config, device=device, seq_len=seq_len)
            # Model constructors inherit the ambient device; the registered
            # canonical recipe explicitly selects CPU instead.
            return (value + 0.01 if device is None else value), scale

        with patch("hyper_parallel.models._transformers.auto_model._current_device",
                   return_value=torch.device("cpu")), \
                patch.object(Qwen3RotaryEmbedding, "compute_default_rope_parameters",
                             side_effect=constructor_frequency):
            model = build_inference_model(config, [], attn_implementation="eager")
        for name in ("inv_freq", "original_inv_freq"):
            torch.testing.assert_close(getattr(model.model.rotary_emb, name), expected, rtol=0, atol=0)

    def test_bf16_default_keeps_canonical_fp32_buffers(self) -> None:
        """The vLLM parameter dtype default must not reduce rotary-buffer precision."""
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cpu"):
                rotary = Qwen3RotaryEmbedding(self.config)
            register_qwen3_materialization(rotary)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(rotary.inv_freq, self.expected, rtol=0, atol=0)

    def test_role_builder_preserves_registered_moe_rotary_recipe(self) -> None:
        """Native recovery must not overwrite the shared canonical RoPE buffers."""
        for registered in (False, True):
            with self.subTest(registered=registered):
                model = torch.nn.Linear(2, 2)
                model.config = SimpleNamespace(model_type="qwen3_moe")
                model._rl_rope_materialization_registered = registered
                runtime = SimpleNamespace(
                    model=SimpleNamespace(build=lambda model=model, **_kwargs: model),
                    activation_checkpoint=SimpleNamespace(mode="off"), peft=None,
                )
                with patch("rl.workers.engine.model.rebuild_qwen3_moe_rotary_state") as restore:
                    result = build_role_model(runtime, None, frozen=False, token_log_probs_mode="dense")
                self.assertIs(result, model)
                self.assertEqual(restore.call_count, int(not registered))
