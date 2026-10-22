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
"""CPU contracts for Qwen encoder routing and unchanged default behavior."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed._builder.source_shard import build_source_shard_info
from hyper_parallel.distributed.encoder_parallel.attention import EncoderAttentionContext, activate_encoder_attention
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderWorkItem, EncoderSegment
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime, build_encoder_parallel_runtime
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (
    QwenEncoderAttention, QwenVisionBridge, get_encoder_parallel_spec,
)
from hyper_parallel.trainer.config.parallelism import AcceleratorConfig
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.common.mark_utils import arg_mark


class NativeAttention(nn.Module):
    """Small independent SDPA oracle with Qwen's module/parameter contract."""

    def __init__(self) -> None:
        """Initialize native-shaped attention projections."""
        super().__init__()
        self.num_heads, self.head_dim, self.scaling = 2, 4, 0.5
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor], **kwargs: object) -> torch.Tensor:
        """Evaluate each image independently without any padded tokens."""
        del kwargs
        query, key, value = self.qkv(hidden_states).reshape(-1, 3, 2, 4).unbind(1)
        cosine, sine = [value.unsqueeze(1) for value in position_embeddings]
        query = query * cosine + torch.cat((-query[..., 2:], query[..., :2]), -1) * sine
        key = key * cosine + torch.cat((-key[..., 2:], key[..., :2]), -1) * sine
        outputs = []
        for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            outputs.append(functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                *(part[start:stop].transpose(0, 1).unsqueeze(0) for part in (query, key, value)),
                scale=self.scaling,
            ).squeeze(0).transpose(0, 1).reshape(-1, 8))
        return self.proj(torch.cat(outputs))


def make_model() -> nn.Module:
    """Build a tiny native-shaped facade without optional Transformers kernels."""
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5_moe", image_token_id=99)
    model.model = nn.Module()
    visual = nn.Module()
    model.model.visual = visual
    visual.config = SimpleNamespace(num_heads=2)
    visual.spatial_merge_size = 2
    visual.blocks = nn.ModuleList([nn.Module()])
    visual.blocks[0].attn = QwenEncoderAttention(NativeAttention(), "model.visual.blocks.0.attn")
    model._hp_replaced_module_fqns = ("model.visual.blocks.0.attn",)
    return model


def item(index: int, length: int, start: int = 0) -> EncoderWorkItem:
    """Create a synthetic image job."""
    return EncoderWorkItem(index, 0, length, length, start, length, 0, length, {})


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
class TestQwenEncoder(unittest.TestCase):
    """Feature: optional Qwen visual parallelism with isolated image attention."""

    def setUp(self) -> None:
        """Reset the random seed for independent test cases."""
        torch.manual_seed(42)

    def test_inactive_exact_forward_and_gradients(self):
        """Description: no active runtime. Expectation: exact output and gradients."""
        native = NativeAttention()
        replacement = QwenEncoderAttention(copy.deepcopy(native), "vision.attn")
        hidden = torch.randn(5, 8, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        angle = torch.randn(5, 4)
        kwargs = {"cu_seqlens": torch.tensor([0, 2, 5]), "position_embeddings": (angle.cos(), angle.sin())}
        expected, actual = native(hidden, **kwargs), replacement(other, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
        self.assertEqual(list(native.state_dict()), list(replacement.state_dict()))
        for left, right in zip(native.parameters(), replacement.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_active_padding_and_image_isolation(self):
        """Description: two padded image segments. Expectation: real output/gradient match."""
        native = NativeAttention()
        replacement = QwenEncoderAttention(copy.deepcopy(native), "vision.attn")
        source = torch.randn(5, 8, requires_grad=True)
        padded = torch.cat((source[:2].detach(), torch.randn(1, 8), source[2:].detach(),
                            torch.randn(1, 8))).requires_grad_()
        source_pos = (torch.ones(5, 4), torch.zeros(5, 4))
        padded_pos = (torch.ones(7, 4), torch.zeros(7, 4))
        expected = native(source, torch.tensor([0, 2, 5]), source_pos)
        segments = (EncoderSegment(item(0, 2), 0, 2, 3, 0), EncoderSegment(item(1, 3), 0, 3, 4, 3))
        context = EncoderAttentionContext(segments, SimpleNamespace(size=lambda: 1),
                                           dict.fromkeys(("query", "key", "value"), 2), frozenset(["vision.attn"]))
        with activate_encoder_attention(context):
            result = replacement(padded, position_embeddings=padded_pos)
        context.assert_exactly_once()
        actual = torch.cat((result[:2], result[3:6]))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(torch.cat((padded.grad[:2], padded.grad[3:6])), source.grad)
        torch.testing.assert_close(padded.grad[[2, 6]], torch.zeros(2, 8), rtol=0, atol=0)
        for left, right in zip(native.parameters(), replacement.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

    def test_metadata_and_rejections(self):
        """Description: image spans, missing payload/video. Expectation: explicit validation."""
        bridge = QwenVisionBridge(make_model())
        batch = {"input_ids": torch.tensor([[1, 99, 99, 2]]), "pixel_values": torch.zeros(8, 12),
                 "image_grid_thw": torch.tensor([[1, 2, 4]])}
        parsed = bridge.parse_images(batch)
        self.assertEqual((parsed[0].sequence_length, parsed[0].output_rows), (8, 2))
        for changes in ({"input_ids": torch.tensor([[99, 2]])}, {"image_grid_thw": torch.tensor([[1, 3, 4]])}):
            with self.assertRaises(ValueError):
                bridge.parse_images({**batch, **changes})
        with self.assertRaises(NotImplementedError):
            bridge.parse_images({**batch, "video_grid_thw": torch.tensor([[1, 2, 2]])})
        with self.assertRaises(NotImplementedError):
            bridge.validate_training({"micro_batch_size": 1, "cp_size": 2})
        with self.assertRaises(NotImplementedError):
            bridge.validate_training({"micro_batch_size": 1, "activation_checkpoint": "full"})

    def test_default_disabled_and_visual_only_provider(self):
        """Description: default config. Expectation: visual FSDP only, no encoder runtime."""
        config = resolve_component({}, expected_type=AcceleratorConfig, path="accelerator")
        self.assertFalse(config.encoder_parallel.enabled)
        enabled_config = resolve_component({"encoder_parallel": {"enabled": True, "encoder_cp_size": 2}},
                                           expected_type=AcceleratorConfig, path="accelerator")
        self.assertEqual(enabled_config.encoder_parallel, EncoderParallelConfig(True, 2))
        self.assertIsNone(build_encoder_parallel_runtime(config=config.encoder_parallel, mesh_context=None,
                                                         model=None, model_adapter_spec=None, execution_options=None))
        model = make_model()
        self.assertEqual(ShardingPlanner()._encoder_isolated_roots(model, "qwen3_5_moe"), ("model.visual",))
        enabled = ShardingPlanner(encoder_parallel_enabled=True)
        self.assertEqual(enabled._encoder_isolated_roots(model, "qwen3_5_moe"), ("model.visual",))
        with self.assertRaises(ValueError):
            enabled._encoder_isolated_roots(model, "qwen3_5_moe_text")

    def test_visual_plan_is_parameter_only(self):
        """Description: enabled visual isolation. Expectation: no decoder boundary or special handlers."""
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1), init_backend=False)
        try:
            plan = ShardingPlanner(encoder_parallel_enabled=True).plan(make_model(), mesh, tp_size=2)
            self.assertEqual(set(plan.modules), {"model.visual"})
            self.assertFalse(plan.modules["model.visual"].is_boundary)
            self.assertFalse(plan.special_handlers)
            self.assertEqual(len(plan.modules["model.visual"].params), 4)
        finally:
            _DEVICE_MESH_MAP.clear()

    def test_visual_fsdp_only_with_decoder_cp_ep(self):
        """Description: decoder CP/EP enabled. Expectation: complete replicated visual metadata."""
        model = make_model()
        visual = model.model.visual
        visual.pos_embed = nn.Embedding(16, 8)
        visual.patch_embed = nn.Linear(8, 8)
        model.model.language_model = nn.Module()
        layer = nn.Module()
        model.model.language_model.layers = nn.ModuleList([layer])
        layer.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, nn.Linear(8, 8, bias=False))
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(8, 4, bias=False)
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 8))
        layer.mlp.experts.down_proj = nn.Parameter(torch.zeros(4, 8, 8))
        mesh = init_device_mesh("cpu", (4, 2, 1), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=tuple(range(8)), init_backend=False)
        expert_mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("edp", "ep"),
                                       rank_list=tuple(range(8)), init_backend=False)
        try:
            plan = ShardingPlanner().plan(model, mesh, cp_size=2, ep_size=2, sequence_parallel=False)
            spec = plan.modules["model.visual"]
            self.assertFalse(spec.is_boundary)
            self.assertIsNone(spec.inner_wrapper)
            self.assertIsNone(spec.local_compute_fn)
            self.assertEqual(spec.in_src, {})
            self.assertEqual(spec.in_dst, {})
            self.assertEqual(set(spec.params), dict(visual.named_parameters()).keys())
            self.assertFalse(any(name.startswith("model.visual.") for name in plan.modules))
            self.assertTrue(all(placement.is_replicate() for axes in spec.params.values()
                                for placement in axes.values()))
            expert_spec = plan.modules["model.language_model.layers.0.mlp"]
            self.assertTrue(expert_spec.params["experts.gate_up_proj"]["ep"].is_shard())
            self.assertIn("model.language_model.layers.0.self_attn", plan.modules)
            metadata = build_source_shard_info(plan, mesh, expert_source_mesh=expert_mesh)
            self.assertEqual(set(metadata), dict(model.named_parameters()).keys())
            for name, (placements, source_mesh) in metadata.items():
                if name.startswith("model.visual."):
                    self.assertEqual(source_mesh.mesh_dim_names, ("tp",))
                    self.assertTrue(all(placement.is_replicate() for placement in placements))
        finally:
            _DEVICE_MESH_MAP.clear()

    def test_text_model_has_no_visual_isolation(self):
        """Description: text-only adapter. Expectation: no new source-layout declarations."""
        model = nn.Module()
        self.assertEqual(ShardingPlanner()._encoder_isolated_roots(model, "qwen3_5_moe_text"), ())
        self.assertEqual(ShardingPlanner()._encoder_isolated_roots(model, "qwen3_5_text"), ())
        self.assertEqual(ShardingPlanner(derive=False)._encoder_isolated_roots(model, "qwen3_5_moe"), ())

    def test_forward_feature_insertion_and_context_cleanup(self):
        """Description: route features through native call site. Expectation: no rerun; graph retained."""
        model = make_model()
        native_calls = []

        def visual_forward(pixels: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            """Record calls to the unmodified visual path."""
            del kwargs
            native_calls.append(True)
            return SimpleNamespace(pooler_output=pixels)

        def model_forward(pixel_values: torch.Tensor = None, **kwargs: object) -> SimpleNamespace:
            """Consume visual features at the native model call site."""
            del kwargs
            features = model.model.visual(pixel_values).pooler_output
            return SimpleNamespace(loss=features.square().sum())

        model.model.visual.forward = visual_forward
        model.forward = model_forward
        bridge = QwenVisionBridge(model)
        bridge.install_forward()
        original = torch.randn(2, 8, requires_grad=True)
        routed = torch.randn(2, 8, requires_grad=True)
        anchor_source = torch.ones(1, requires_grad=True)
        plan = SimpleNamespace(encode_and_route=lambda _: ([(item(0, 2), routed)], anchor_source.sum() * 0))
        output = model(pixel_values=original, encoder_parallel_plan=plan)
        self.assertEqual(native_calls, [])
        output.loss.backward()
        torch.testing.assert_close(routed.grad, 2 * routed)
        torch.testing.assert_close(anchor_source.grad, torch.zeros(1))
        self.assertIsNone(original.grad)
        self.assertIsNone(bridge.feature_context.get())
        baseline = model(pixel_values=original)
        torch.testing.assert_close(baseline.loss, original.square().sum())
        self.assertEqual(len(native_calls), 1)

    def test_runtime_local_and_input_adapter(self):
        """Description: one-rank full plan. Expectation: features and gradient retained."""
        model = make_model()
        mesh = init_device_mesh("cpu", (1, 1, 1), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=(0,), init_backend=False)
        context = SimpleNamespace(device_mesh=mesh, cp_size=1, tp_size=1, pp_size=1,
                                  cp_rank=0, tp_rank=0, sequence_parallel=False, loss_parallel=False)
        try:
            runtime = EncoderParallelRuntime(
                config=EncoderParallelConfig(True), mesh_context=context, model=model,
                spec=get_encoder_parallel_spec(), execution_options={"micro_batch_size": 1},
            )
            batch = {"input_ids": torch.tensor([[1, 99, 2]]), "pixel_values": torch.randn(4, 8),
                     "image_grid_thw": torch.tensor([[1, 2, 2]])}
            adapted = EncoderParallelInputAdapter(runtime).build(batch=batch, parallel_context=None)
            plan = adapted["encoder_parallel_plan"]

            def encode(inputs: dict[str, torch.Tensor], segments: tuple[EncoderSegment, ...]) -> torch.Tensor:
                """Exercise attention without optional vision embedding kernels."""
                del segments
                return model.model.visual.blocks[0].attn(inputs["pixel_values"],
                                                         position_embeddings=(torch.ones(4, 4), torch.zeros(4, 4)))

            with patch.object(runtime.bridge, "encode_local", side_effect=encode), \
                    patch.object(runtime.bridge, "project", side_effect=lambda value, _: value.mean(0, keepdim=True)):
                records, anchor = plan.encode_and_route(batch)
            self.assertEqual(records[0][1].shape, (1, 8))
            (records[0][1].sum() + anchor).backward()
            self.assertIsNotNone(model.model.visual.blocks[0].attn.qkv.weight.grad)
        finally:
            _DEVICE_MESH_MAP.clear()
