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
"""Unit tests for LlamaFactory activation optimization integration."""

# pylint: disable=wrong-import-position,protected-access
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

# Stub transformers and accelerate if not installed
try:
    from transformers import Seq2SeqTrainer
except ImportError:
    transformers_stub = ModuleType("transformers")

    class Seq2SeqTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs

    transformers_stub.Seq2SeqTrainer = Seq2SeqTrainer
    sys.modules["transformers"] = transformers_stub

if "accelerate.accelerator" not in sys.modules:
    accelerate_stub = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
    accelerator_stub = ModuleType("accelerate.accelerator")
    accelerator_stub.fsdp2_prepare_model = lambda accelerator, model: model
    accelerate_stub.accelerator = accelerator_stub
    sys.modules["accelerate.accelerator"] = accelerator_stub

from hyper_parallel.integration.llamafactory.activation import (
    _build_policy_fn,
    find_transformer_blocks,
    setup_activation_optimization,
)
from hyper_parallel.integration.llamafactory.utils import HyperParallelArguments
from hyper_parallel.platform.torch.activation_checkpoint import CheckpointWrapper


# ---------------------------------------------------------------------------
# Helper models for testing
# ---------------------------------------------------------------------------


class FakeDecoderLayer(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class FakeLlamaModel(nn.Module):
    """Mimics HF model.model with gradient_checkpointing attribute."""

    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        # HF PreTrainedModel containers mark themselves with this attribute;
        # find_transformer_blocks uses it as the sole discovery signal.
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([FakeDecoderLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeCausalLM(nn.Module):
    """Mimics HF causal LM wrapping a container that has gradient_checkpointing."""

    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        self.model = FakeLlamaModel(num_layers, dim)
        self.lm_head = nn.Linear(dim, 100)
        self.config = type("Config", (), {"use_cache": True})()

    def forward(self, x):
        return self.lm_head(self.model(x))


class FakeGPT2Model(nn.Module):
    """Mimics model.transformer with gradient_checkpointing attribute."""

    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        self.gradient_checkpointing = False
        self.h = nn.ModuleList([FakeDecoderLayer(dim) for _ in range(num_layers)])


class FakeGPT2LM(nn.Module):
    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        self.transformer = FakeGPT2Model(num_layers, dim)


class FakeUnknownModel(nn.Module):
    """Model with no gradient_checkpointing attribute anywhere."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)


class FakeVisualTower(nn.Module):
    """Vision encoder container with its own gradient_checkpointing attribute."""

    def __init__(self, num_blocks=2, dim=16):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([FakeDecoderLayer(dim) for _ in range(num_blocks)])


class FakeMultimodalLM(nn.Module):
    """Mimics a VL model: visual tower + LLM both have gradient_checkpointing."""

    def __init__(self, visual_blocks=2, llm_layers=4, dim=16):
        super().__init__()
        self.visual = FakeVisualTower(visual_blocks, dim)
        self.model = FakeLlamaModel(llm_layers, dim)
        self.lm_head = nn.Linear(dim, 100)
        self.config = type("Config", (), {"use_cache": True})()


# ---------------------------------------------------------------------------
# Tests: find_transformer_blocks
# ---------------------------------------------------------------------------


class TestFindTransformerBlocks:
    """Tests for ``find_transformer_blocks`` across HF model topologies."""

    def test_llama_structure(self):
        model = FakeCausalLM(num_layers=4)
        results = find_transformer_blocks(model)
        assert len(results) == 1, f"Expected 1 container, got {len(results)}"
        parent, attr, blocks, path = results[0]
        assert len(blocks) == 4, f"Expected 4 blocks, got {len(blocks)}"
        assert path == "model.layers", f"Expected 'model.layers', got {path!r}"
        assert parent is model.model
        assert attr == "layers"

    def test_gpt2_structure(self):
        model = FakeGPT2LM(num_layers=3)
        results = find_transformer_blocks(model)
        assert len(results) == 1, f"Expected 1 container, got {len(results)}"
        parent, attr, blocks, path = results[0]
        assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"
        assert path == "transformer.h", f"Expected 'transformer.h', got {path!r}"
        assert parent is model.transformer
        assert attr == "h"

    def test_multimodal_returns_all_containers(self):
        """VL models with multiple gc-enabled containers should return all."""
        model = FakeMultimodalLM(visual_blocks=2, llm_layers=4)
        results = find_transformer_blocks(model)
        assert len(results) == 2, f"Expected 2 containers, got {len(results)}"
        paths = [r[3] for r in results]
        assert "visual.blocks" in paths, f"Missing visual tower in {paths}"
        assert "model.layers" in paths, f"Missing LLM tower in {paths}"
        block_counts = {r[3]: len(r[2]) for r in results}
        assert block_counts["visual.blocks"] == 2, f"Visual blocks={block_counts}"
        assert block_counts["model.layers"] == 4, f"LLM layers={block_counts}"

    def test_unknown_model_returns_empty(self):
        model = FakeUnknownModel()
        results = find_transformer_blocks(model)
        assert not results, f"Expected empty list, got {results}"


# ---------------------------------------------------------------------------
# Tests: HyperParallelArguments activation validation
# ---------------------------------------------------------------------------


class TestActivationArgValidation:
    """Tests for activation-related fields on ``HyperParallelArguments``."""

    def test_default_activation_mode(self):
        args = HyperParallelArguments()
        args.validate()
        assert args.activation_mode == "none"

    def test_valid_recompute_mode(self):
        args = HyperParallelArguments(activation_mode="recompute")
        args.validate()

    def test_valid_swap_mode(self):
        args = HyperParallelArguments(activation_mode="swap")
        args.validate()

    def test_invalid_activation_mode(self):
        args = HyperParallelArguments(activation_mode="invalid")
        with pytest.raises(ValueError, match="activation_mode"):
            args.validate()

    def test_from_dict_with_activation(self):
        config = {
            "activation_mode": "recompute",
        }
        args = HyperParallelArguments.from_dict(config)
        assert args.activation_mode == "recompute"

    def test_from_dict_swap_with_options(self):
        config = {
            "activation_mode": "swap",
            "activation_swap_inputs": False,
        }
        args = HyperParallelArguments.from_dict(config)
        assert args.activation_mode == "swap"
        assert args.activation_swap_inputs is False


# ---------------------------------------------------------------------------
# Tests: _build_policy_fn
# ---------------------------------------------------------------------------


class TestBuildPolicyFn:
    """Tests for ``_build_policy_fn`` per-op routing in swap mode."""

    def test_recompute_returns_none(self):
        args = HyperParallelArguments(activation_mode="recompute")
        assert _build_policy_fn(args) is None

    def test_none_returns_none(self):
        args = HyperParallelArguments(activation_mode="none")
        assert _build_policy_fn(args) is None

    def test_swap_returns_callable(self):
        args = HyperParallelArguments(activation_mode="swap")
        fn = _build_policy_fn(args)
        assert fn is not None
        assert callable(fn)

    def test_swap_policy_matmul_ops_must_swap(self):
        """Matmul-family ops should be marked MUST_SWAP, other ops MUST_RECOMPUTE."""
        from hyper_parallel.core.activation_checkpoint import CheckpointPolicy  # pylint: disable=C0415

        args = HyperParallelArguments(activation_mode="swap")
        fn = _build_policy_fn(args)

        matmul_ops = [
            torch.ops.aten.matmul.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.mm.default,
        ]
        for op in matmul_ops:
            result = fn(None, op)
            assert result == CheckpointPolicy.MUST_SWAP, (
                f"Expected MUST_SWAP for matmul op {op}, got {result}"
            )

        non_matmul_ops = [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.relu.default,
            torch.ops.aten.softmax.int,
        ]
        for op in non_matmul_ops:
            result = fn(None, op)
            assert result == CheckpointPolicy.MUST_RECOMPUTE, (
                f"Expected MUST_RECOMPUTE for non-matmul op {op}, got {result}"
            )


# ---------------------------------------------------------------------------
# Tests: setup_activation_optimization
# ---------------------------------------------------------------------------


class TestSetupActivationOptimization:
    """Tests for ``setup_activation_optimization`` entry point."""

    def test_none_mode_noop(self):
        model = FakeCausalLM(num_layers=4)
        original_layer_type = type(model.model.layers[0])
        result = setup_activation_optimization(model, HyperParallelArguments())
        assert result is model
        assert isinstance(model.model.layers[0], original_layer_type)

    def test_recompute_wraps_blocks(self):
        model = FakeCausalLM(num_layers=4)
        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)
        # After wrapping, blocks should be CheckpointWrapper instances
        for block in model.model.layers:
            assert isinstance(block, CheckpointWrapper)

    def test_recompute_disables_use_cache(self):
        model = FakeCausalLM(num_layers=4)
        assert model.config.use_cache is True
        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)
        assert model.config.use_cache is False

    def test_unknown_model_raises(self):
        model = FakeUnknownModel()
        args = HyperParallelArguments(activation_mode="recompute")
        with pytest.raises(ValueError, match="gradient_checkpointing"):
            setup_activation_optimization(model, args)

    def test_multimodal_wraps_all_containers(self):
        """All gc-enabled containers (visual + LLM) should get wrapped."""
        model = FakeMultimodalLM(visual_blocks=2, llm_layers=4)
        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)

        for block in model.visual.blocks:
            assert isinstance(block, CheckpointWrapper), (
                f"Visual block {type(block).__name__} should be wrapped"
            )
        for block in model.model.layers:
            assert isinstance(block, CheckpointWrapper), (
                f"LLM block {type(block).__name__} should be wrapped"
            )

    def test_swap_wraps_blocks(self):
        model = FakeCausalLM(num_layers=4)
        args = HyperParallelArguments(activation_mode="swap")
        setup_activation_optimization(model, args)
        for block in model.model.layers:
            assert isinstance(block, CheckpointWrapper)
            assert hasattr(block, "_swap_group_order")

    def test_block_count_preserved(self):
        model = FakeCausalLM(num_layers=6)
        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)
        assert len(model.model.layers) == 6

    def test_frozen_blocks_skipped(self):
        """Frozen blocks should not be wrapped with CheckpointWrapper."""
        model = FakeCausalLM(num_layers=4)
        # Freeze the first two blocks
        for param in model.model.layers[0].parameters():
            param.requires_grad_(False)
        for param in model.model.layers[1].parameters():
            param.requires_grad_(False)

        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)

        assert not isinstance(model.model.layers[0], CheckpointWrapper), (
            "Frozen block 0 should not be wrapped"
        )
        assert not isinstance(model.model.layers[1], CheckpointWrapper), (
            "Frozen block 1 should not be wrapped"
        )
        assert isinstance(model.model.layers[2], CheckpointWrapper), (
            "Trainable block 2 should be wrapped"
        )
        assert isinstance(model.model.layers[3], CheckpointWrapper), (
            "Trainable block 3 should be wrapped"
        )

    def test_frozen_blocks_skipped_swap(self):
        """Frozen blocks should not be wrapped in swap mode either."""
        model = FakeCausalLM(num_layers=4)
        for param in model.model.layers[0].parameters():
            param.requires_grad_(False)

        args = HyperParallelArguments(activation_mode="swap")
        setup_activation_optimization(model, args)

        assert not isinstance(model.model.layers[0], CheckpointWrapper), (
            "Frozen block 0 should not be wrapped"
        )
        for i in range(1, 4):
            assert isinstance(model.model.layers[i], CheckpointWrapper), (
                f"Trainable block {i} should be wrapped"
            )

    def test_all_frozen_no_wrap(self):
        """If all blocks are frozen, none should be wrapped."""
        model = FakeCausalLM(num_layers=3)
        for param in model.parameters():
            param.requires_grad_(False)

        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)

        for i in range(3):
            assert not isinstance(model.model.layers[i], CheckpointWrapper), (
                f"Frozen block {i} should not be wrapped"
            )

    def test_state_dict_transparent(self):
        """CheckpointWrapper should not change state_dict keys."""
        model = FakeCausalLM(num_layers=2)
        original_keys = set(model.state_dict().keys())

        args = HyperParallelArguments(activation_mode="recompute")
        setup_activation_optimization(model, args)

        new_keys = set(model.state_dict().keys())
        assert new_keys == original_keys
