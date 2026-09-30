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
"""CPU contracts for the Qwen3.5-MoE adapter used by Qwen3.6."""

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from hyper_parallel.models.registry import get_model_adapter
from tests.common.mark_utils import arg_mark


class TestQwen35MoeAdapterContracts(unittest.TestCase):
    """Pin the outer and text configuration registration aliases."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_registration_resolves_outer_and_text_identities(self):
        """Validate Qwen3.5-MoE adapter discovery.

        Feature: Qwen3.5-MoE adapter registration.
        Description: Resolve the Transformers outer and text identities through the lazy model registry.
        Expectation: Both identities resolve to their respective Qwen3.5-MoE adapter specifications.
        """
        text_spec = get_model_adapter("qwen3_5_moe_text")
        outer_spec = get_model_adapter("Qwen3_5MoeForConditionalGeneration")

        self.assertIsNotNone(text_spec, f"Qwen3.5-MoE text adapter must be registered, got {text_spec}")
        self.assertIsNotNone(outer_spec, f"Qwen3.5-MoE outer adapter must be registered, got {outer_spec}")
        self.assertEqual(
            text_spec.model_type,
            "qwen3_5_moe_text",
            f"Unexpected text model type: expected=qwen3_5_moe_text, got={text_spec.model_type}",
        )
        self.assertEqual(
            outer_spec.model_type,
            "qwen3_5_moe",
            f"Unexpected outer model type: expected=qwen3_5_moe, got={outer_spec.model_type}",
        )


if __name__ == "__main__":
    unittest.main()
