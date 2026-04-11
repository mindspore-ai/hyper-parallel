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
"""Unit tests for ``Platform.is_linear_module`` / ``Platform.is_embedding_module``.

Tests cover:
- TorchPlatform correctly identifies nn.Linear and nn.Embedding
- TorchPlatform rejects non-matching module types
- Platform base class raises NotImplementedError
"""
import os
import unittest

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform.platform import Platform
from hyper_parallel.platform.torch.platform import TorchPlatform


class TestTorchPlatformIsLinearModule(unittest.TestCase):
    """Tests for TorchPlatform.is_linear_module."""

    def test_linear_is_recognized(self):
        """
        Feature: TorchPlatform.is_linear_module identifies nn.Linear
        Description: pass nn.Linear(4, 8) to is_linear_module
        Expectation: returns True
        """
        module = nn.Linear(4, 8)
        self.assertTrue(TorchPlatform.is_linear_module(module))

    def test_embedding_is_not_linear(self):
        """
        Feature: TorchPlatform.is_linear_module rejects nn.Embedding
        Description: pass nn.Embedding(100, 64) to is_linear_module
        Expectation: returns False
        """
        module = nn.Embedding(100, 64)
        self.assertFalse(TorchPlatform.is_linear_module(module))

    def test_layernorm_is_not_linear(self):
        """
        Feature: TorchPlatform.is_linear_module rejects nn.LayerNorm
        Description: pass nn.LayerNorm(8) to is_linear_module
        Expectation: returns False
        """
        module = nn.LayerNorm(8)
        self.assertFalse(TorchPlatform.is_linear_module(module))

    def test_identity_is_not_linear(self):
        """
        Feature: TorchPlatform.is_linear_module rejects nn.Identity
        Description: pass nn.Identity() to is_linear_module
        Expectation: returns False
        """
        module = nn.Identity()
        self.assertFalse(TorchPlatform.is_linear_module(module))

    def test_conv1d_is_not_linear(self):
        """
        Feature: TorchPlatform.is_linear_module rejects nn.Conv1d
        Description: pass nn.Conv1d(3, 8, 1) to is_linear_module
        Expectation: returns False
        """
        module = nn.Conv1d(3, 8, 1)
        self.assertFalse(TorchPlatform.is_linear_module(module))


class TestTorchPlatformIsEmbeddingModule(unittest.TestCase):
    """Tests for TorchPlatform.is_embedding_module."""

    def test_embedding_is_recognized(self):
        """
        Feature: TorchPlatform.is_embedding_module identifies nn.Embedding
        Description: pass nn.Embedding(100, 64) to is_embedding_module
        Expectation: returns True
        """
        module = nn.Embedding(100, 64)
        self.assertTrue(TorchPlatform.is_embedding_module(module))

    def test_linear_is_not_embedding(self):
        """
        Feature: TorchPlatform.is_embedding_module rejects nn.Linear
        Description: pass nn.Linear(4, 8) to is_embedding_module
        Expectation: returns False
        """
        module = nn.Linear(4, 8)
        self.assertFalse(TorchPlatform.is_embedding_module(module))

    def test_layernorm_is_not_embedding(self):
        """
        Feature: TorchPlatform.is_embedding_module rejects nn.LayerNorm
        Description: pass nn.LayerNorm(8) to is_embedding_module
        Expectation: returns False
        """
        module = nn.LayerNorm(8)
        self.assertFalse(TorchPlatform.is_embedding_module(module))

    def test_embedding_bag_is_not_embedding(self):
        """
        Feature: TorchPlatform.is_embedding_module rejects nn.EmbeddingBag
        Description: pass nn.EmbeddingBag(100, 64) to is_embedding_module
        Expectation: returns False (only plain Embedding is supported)
        """
        module = nn.EmbeddingBag(100, 64)
        self.assertFalse(TorchPlatform.is_embedding_module(module))


class TestPlatformBaseRaises(unittest.TestCase):
    """Tests for Platform base class NotImplementedError."""

    def test_base_is_linear_module_raises(self):
        """
        Feature: Platform.is_linear_module is abstract
        Description: call Platform.is_linear_module directly
        Expectation: raises NotImplementedError
        """
        with self.assertRaises(NotImplementedError):
            Platform.is_linear_module(nn.Linear(4, 8))

    def test_base_is_embedding_module_raises(self):
        """
        Feature: Platform.is_embedding_module is abstract
        Description: call Platform.is_embedding_module directly
        Expectation: raises NotImplementedError
        """
        with self.assertRaises(NotImplementedError):
            Platform.is_embedding_module(nn.Embedding(100, 64))


class TestGetPlatformModuleTypeCheck(unittest.TestCase):
    """Tests for module type check via get_platform() instance."""

    def test_get_platform_is_linear_module(self):
        """
        Feature: get_platform().is_linear_module works end-to-end
        Description: use get_platform() and check nn.Linear
        Expectation: returns True
        """
        from hyper_parallel.platform import get_platform
        plat = get_platform()
        self.assertTrue(plat.is_linear_module(nn.Linear(4, 8)))

    def test_get_platform_is_embedding_module(self):
        """
        Feature: get_platform().is_embedding_module works end-to-end
        Description: use get_platform() and check nn.Embedding
        Expectation: returns True
        """
        from hyper_parallel.platform import get_platform
        plat = get_platform()
        self.assertTrue(plat.is_embedding_module(nn.Embedding(100, 64)))


if __name__ == "__main__":
    unittest.main()
