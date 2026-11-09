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
"""CPU regressions for vision attention, checkpoint replay, and mixed precision."""

import copy
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.core.multicore.modules.mega_gate.module import MegaGate
from hyper_parallel.core.multicore.modules.mega_mhc.module import MegaMhc
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.distributed.encoder_parallel.attention import (
    EncoderAttentionContext,
    activate_encoder_attention,
    current_encoder_attention_context,
    encoder_attention_checkpoint_contexts,
)
from hyper_parallel.models._transformers.model_builder import validate_model_init_dtype
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import DeepseekV41VisionBridge
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower, _vision_attention


def _vision_model() -> nn.Module:
    """Construct a tiny two-block vision model with the real V4.1 adapter identity."""
    config = SimpleNamespace(
        v41_vision_hidden_size=8, v41_vision_patch_size=2,
        v41_vision_num_attention_heads=2, v41_vision_rope_theta=10000.0,
        v41_vision_intermediate_size=16, v41_vision_num_hidden_layers=2,
    )
    model = nn.Module()
    model.config = SimpleNamespace(model_type="deepseek_v41")
    model.model = nn.Module()
    model.model.vision = DeepseekV41VisionTower(config)
    model.model.aligner = nn.Linear(8, 8)
    return model


def _attention_oracle(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute dense noncausal attention independently of either attention backend."""
    scores = query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5
    return scores.softmax(dim=-1) @ value


class TestVisionRecompute(unittest.TestCase):
    """Exercise complete block replay and isolation of per-forward metadata."""

    def setUp(self) -> None:
        """Keep CPU replay independent of an optional NPU extension's default."""
        previous_device_type = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, previous_device_type)

    def test_inactive_checkpoint_contexts_leave_attention_inactive(self) -> None:
        """Feature: Vision checkpointing without encoder parallelism.

        Description: Enter both contexts without an active encoder pass.
        Expectation: Neither forward nor replay installs encoder metadata.
        """
        for context in encoder_attention_checkpoint_contexts():
            with context:
                self.assertIsNone(current_encoder_attention_context())

    def test_replay_restores_original_batch_and_keeps_audits_independent(self) -> None:
        """Feature: Encoder attention checkpoint metadata ownership.

        Description: Replay a previous forward inside a different active batch.
        Expectation: Replay uses the original metadata and preserves both batch audits.
        """
        first = EncoderAttentionContext((), object(), {"query": 2}, frozenset({"block"}))
        later = EncoderAttentionContext((), object(), {"query": 4}, frozenset({"later"}))
        with activate_encoder_attention(first):
            forward, replay = encoder_attention_checkpoint_contexts()
            with forward:
                self.assertIs(current_encoder_attention_context(), first)
                first.record("block")
        with activate_encoder_attention(later):
            with self.assertRaisesRegex(RuntimeError, "replay failure"):
                with replay:
                    active = current_encoder_attention_context()
                    self.assertIsNot(active, first)
                    self.assertIs(active.cp_mesh, first.cp_mesh)
                    self.assertEqual(active.declared_heads, first.declared_heads)
                    self.assertEqual(active.seen, {})
                    active.record("block")
                    active.assert_exactly_once()
                    raise RuntimeError("replay failure")
            self.assertIs(current_encoder_attention_context(), later)
        first.assert_exactly_once()
        self.assertEqual(later.seen, {})
        self.assertIsNone(current_encoder_attention_context())

    def test_full_checkpointing_preserves_vision_output_and_all_gradients(self) -> None:
        """Feature: Model-owned full vision activation checkpointing.

        Description: Compare a two-block checkpointed tower with its eager copy.
        Expectation: Both blocks replay fully and outputs and every gradient agree.
        """
        torch.manual_seed(1385)
        model = _vision_model()
        expected = copy.deepcopy(model.model.vision)
        _apply_activation_checkpointing(model, "full")
        tower = model.model.vision
        replay_hooks = [Mock(return_value=None) for _ in tower.blocks]
        for block, hook in zip(tower.blocks, replay_hooks):
            self.assertIsInstance(block, CheckpointWrapper)
            self.assertIs(block.checkpoint_kwargs["context_fn"], encoder_attention_checkpoint_contexts)
            self.assertIs(block.checkpoint_kwargs["early_stop"], False)
            handle = block.get_submodule("_wrapped_module").register_forward_hook(hook)
            self.addCleanup(handle.remove)
        DeepseekV41VisionBridge(model).validate_training({"micro_batch_size": 1, "loss_parallel": False})
        patches = torch.randn(4, 3, 2, 2, requires_grad=True)
        reference_patches = patches.detach().clone().requires_grad_()
        actual = tower(patches, 2, 2)
        reference = expected(reference_patches, 2, 2)
        torch.testing.assert_close(actual, reference)
        actual.square().sum().backward()
        reference.square().sum().backward()
        self.assertEqual([hook.call_count for hook in replay_hooks], [2, 2])
        torch.testing.assert_close(patches.grad, reference_patches.grad)
        for parameter, target in zip(tower.parameters(), expected.parameters()):
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, target.grad)

    def test_encoder_rejects_nested_checkpoint_even_with_supported_block(self) -> None:
        """Feature: Encoder checkpoint collective safety validation.

        Description: Add an independently checkpointed attention inside a safe block.
        Expectation: Validation rejects the nested checkpoint instead of allowing unsafe replay.
        """
        model = _vision_model()
        _apply_activation_checkpointing(model, "full")
        block = model.model.vision.blocks[0]
        wrapped = block.get_submodule("_wrapped_module")
        # assertWarns inspects every loaded module and resolves optional HF/NPU imports.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            wrapped.attn = CheckpointWrapper(wrapped.attn)
        self.assertTrue(any(issubclass(item.category, UserWarning) and "already wrapped" in str(item.message)
                            for item in caught))
        with self.assertRaisesRegex(RuntimeError, "context-preserving full vision-block checkpoints"):
            DeepseekV41VisionBridge(model).validate_training({"micro_batch_size": 1})


class TestVisionAttention(unittest.TestCase):
    """Check padding and the NPU host-side dispatch contract on CPU tensors."""

    def test_padding_prefix_matches_dense_attention_and_gradients(self) -> None:
        """Feature: Encoder vision attention padding isolation.

        Description: Attend five query rows with three valid key/value rows.
        Expectation: Outputs and Q/K/V gradients match an independent dense oracle.
        """
        torch.manual_seed(41)
        tensors = [torch.randn(5, 2, 4, requires_grad=True) for _ in range(3)]
        reference = [tensor.detach().clone().requires_grad_() for tensor in tensors]
        actual = _vision_attention(nn.Module(), *tensors, key_length=3)
        expected = _attention_oracle(
            reference[0].transpose(0, 1), reference[1][:3].transpose(0, 1),
            reference[2][:3].transpose(0, 1),
        ).transpose(0, 1)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for tensor, target in zip(tensors, reference):
            torch.testing.assert_close(tensor.grad, target.grad)
        for tensor in tensors[1:]:
            torch.testing.assert_close(tensor.grad[3:], torch.zeros_like(tensor.grad[3:]))

    def test_npu_dispatch_uses_canonical_layout_and_bidirectional_options(self) -> None:
        """Feature: NPU vision attention layout and argument translation.

        Description: Substitute only the optional native entry with a CPU dense oracle.
        Expectation: Canonical contiguous BNSD inputs and full bidirectional options reach the entry.
        """
        torch.manual_seed(42)
        query, key, value = (torch.randn(5, 2, 4) for _ in range(3))
        expected = _attention_oracle(
            query.transpose(0, 1), key[:3].transpose(0, 1), value[:3].transpose(0, 1),
        ).transpose(0, 1)
        native = Mock(side_effect=lambda q, k, v, *_args, **_kwargs: (_attention_oracle(q, k, v),))
        extension = SimpleNamespace(npu_fusion_attention=native)
        with (
            patch.dict(sys.modules, {"torch_npu": extension}),
            patch.object(torch.Tensor, "device", property(lambda _tensor: SimpleNamespace(type="npu"))),
        ):
            actual = _vision_attention(nn.Module(), query, key, value, key_length=3)
        torch.testing.assert_close(actual, expected)
        native.assert_called_once()
        args, options = native.call_args
        self.assertEqual([tuple(tensor.shape) for tensor in args[:3]],
                         [(1, 2, 5, 4), (1, 2, 3, 4), (1, 2, 3, 4)])
        for tensor in args[:3]:
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.stride()[0], tensor.numel())
        self.assertEqual(args[3:], (2, "BNSD"))
        self.assertIsNone(options["atten_mask"])
        self.assertEqual(options["sparse_mode"], 0)
        self.assertEqual(options["pre_tockens"], 2147483647)
        self.assertEqual(options["next_tockens"], 2147483647)
        self.assertEqual(options["keep_prob"], 1.0)
        self.assertEqual(options["scale"], 0.5)


class TestFusedModulePrecision(unittest.TestCase):
    """Validate model dtype checks against the modules' real FP32 declarations."""

    def test_bf16_conversion_preserves_declared_fp32_parameters(self) -> None:
        """Feature: BF16 model validation with fused-module FP32 parameters.

        Description: Convert real MegaGate and MegaMhc modules, then corrupt a declared bias.
        Expectation: Correct mixed state passes and the corrupted FP32 declaration is rejected.
        """
        model = nn.Module()
        model.gate = MegaGate(hidden_size=128, num_experts=4, top_k=2,
                              scoring_func="softmax", routed_scaling_factor=1.0, vision_enabled=True)
        model.mhc = MegaMhc(hidden_size=128)
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        model.to(dtype=torch.bfloat16)
        validate_model_init_dtype(model, "bfloat16")
        for name, module in model.named_children():
            for local_name, parameter in module.named_parameters():
                expected = torch.float32 if local_name in module.fp32_parameter_names else torch.bfloat16
                self.assertEqual(parameter.dtype, expected)
                self.assertEqual(id(parameter), identities[f"{name}.{local_name}"])
        model.gate.bias.data = model.gate.bias.data.to(torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "gate.bias"):
            validate_model_init_dtype(model, "bfloat16")
