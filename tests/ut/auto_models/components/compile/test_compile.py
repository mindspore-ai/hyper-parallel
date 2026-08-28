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
"""Unit tests for decoder-layer compilation."""

import unittest
from unittest.mock import patch
from typing import Any

import torch
from torch import nn

from hyper_parallel.auto_models.components.compile import apply_compile, get_compile_layers
from hyper_parallel.auto_models.trainer.config import CompileConfig


class _DecoderLayer(nn.Module):
    """Minimal decoder layer used to identify compile segments."""


class _OtherLayer(nn.Module):
    """Unrelated repeated layer that must not be selected."""


class _RecordingLayer(nn.Module):
    """Minimal layer recording compile keyword arguments."""

    def __init__(self) -> None:
        """Initialize an empty compile-call list."""
        super().__init__()
        self.compile_calls: list[dict[str, Any]] = []

    def compile(self, **kwargs: Any) -> None:
        """Record one compile invocation."""
        self.compile_calls.append(kwargs)


class _ModelWithNoSplitMetadata(nn.Module):
    """Model declaring decoder blocks without prescribing their path."""

    _no_split_modules = {_DecoderLayer.__name__}

    def __init__(self) -> None:
        """Build decoder blocks under a deliberately arbitrary path."""
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
        self.other_blocks = nn.ModuleList([_OtherLayer(), _OtherLayer()])


class _ModelWithCommonPaths(nn.Module):
    """Model exposing more than one conventional layer path."""

    def __init__(self) -> None:
        """Build common containers used to verify path priority."""
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DecoderLayer()])
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_OtherLayer()])


class _ModelWithDeclaredLayers(_ModelWithCommonPaths):
    """Model overriding conventional paths with an explicit contract."""

    def __init__(self) -> None:
        """Build one explicitly declared layer."""
        super().__init__()
        self.declared_layer = _OtherLayer()

    def get_compile_layers(self) -> list[nn.Module]:
        """Return the explicitly selected layer."""
        return [self.declared_layer]


class TestGetCompileLayers(unittest.TestCase):
    """Test model-owned decoder-layer discovery."""

    def test_uses_no_split_metadata_without_model_paths(self) -> None:
        """Discover decoder blocks by class metadata at any module path."""
        model = _ModelWithNoSplitMetadata()

        layers = get_compile_layers(model)

        self.assertEqual(
            [name for name, _ in layers],
            ["backbone.blocks.0", "backbone.blocks.1"],
        )
        self.assertEqual(
            [layer for _, layer in layers],
            list(model.backbone.blocks),
        )

    def test_prefers_common_paths_in_declared_order(self) -> None:
        """Prefer model.layers over later conventional paths."""
        model = _ModelWithCommonPaths()

        layers = get_compile_layers(model)

        self.assertEqual([name for name, _ in layers], ["model.layers.0"])
        self.assertIs(layers[0][1], model.model.layers[0])

    def test_uses_transformer_h_when_model_layers_is_absent(self) -> None:
        """Use the GPT-style path when the primary path is unavailable."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([_DecoderLayer()])

        layers = get_compile_layers(model)

        self.assertEqual([name for name, _ in layers], ["transformer.h.0"])

    def test_prefers_model_getter_over_common_paths(self) -> None:
        """Prefer the model-owned contract over conventional paths."""
        model = _ModelWithDeclaredLayers()

        layers = get_compile_layers(model)

        self.assertEqual([name for name, _ in layers], ["declared_layer"])
        self.assertIs(layers[0][1], model.declared_layer)

    def test_apply_compile_discovers_layers_when_enabled(self) -> None:
        """Discover and compile common-path layers when compile is enabled."""
        model = _ModelWithCommonPaths()
        model.model.layers = nn.ModuleList([_RecordingLayer(), _RecordingLayer()])
        config = CompileConfig(
            enabled=True,
            backend="eager",
        )

        with (
            patch.object(
                torch._dynamo.config,  # pylint: disable=protected-access
                "cache_size_limit",
                torch._dynamo.config.cache_size_limit,  # pylint: disable=protected-access
            ),
            patch(
                "hyper_parallel.auto_models.components.compile.compile."
                "_install_dynamo_mapping_get_polyfill"
            ),
        ):
            result = apply_compile(model, config)

        self.assertIs(result, model)
        self.assertEqual([len(layer.compile_calls) for layer in model.model.layers], [1, 1])
        self.assertFalse(hasattr(model.transformer.h[0], "compile_calls"))

    def test_rejects_model_without_layer_metadata(self) -> None:
        """Require an explicit layer contract when model metadata is absent."""
        model = nn.Module()
        model.blocks = nn.ModuleList([_DecoderLayer()])

        with self.assertRaisesRegex(ValueError, "no decoder-layer compile contract"):
            get_compile_layers(model)


if __name__ == "__main__":
    unittest.main()
