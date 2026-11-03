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
"""Tests for model initialization after meta-device materialization."""

import unittest
from unittest.mock import patch

from torch import nn

from hyper_parallel.core.activation_memory import CheckpointExcludeWrapper
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights


class _DeferredResetModule(nn.Module):
    """Module whose packed parameters are reset after materialization."""

    def __init__(self) -> None:
        super().__init__()
        self._hp_reset_after_materialization = True
        self.reset_count = 0

    def reset_parameters(self) -> None:
        self.reset_count += 1


class _ModelWithNativeInitialization(nn.Module):
    """Model with a native init hook and a wrapped deferred-reset child."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = CheckpointExcludeWrapper(child, swap=True)

    def initialize_weights(self) -> None:
        """Stand in for the Transformers model-native initialization pass."""


class ModelBuilderInitializationTest(unittest.TestCase):
    """Materialization reset hooks should run on the real module once."""

    def test_activation_wrapper_does_not_repeat_deferred_reset(self) -> None:
        child = _DeferredResetModule()
        model = _ModelWithNativeInitialization(child)

        with patch(
            "hyper_parallel.models._transformers.model_builder.get_model_external_state",
            return_value=None,
        ):
            _initialize_model_weights(model)

        self.assertEqual(child.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
