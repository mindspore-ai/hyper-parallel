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
"""Unit tests for tied parameters in deferred Transformers construction."""

import unittest

from torch import nn

from hyper_parallel.models._transformers.model_builder import (
    _tie_model_weights_before_parallelism,
)


class _DeferredTiedModel(nn.Module):
    """Mimic a Transformers model before its post-load tying step."""

    def __init__(self, resolve_alias: bool = True) -> None:
        """Create distinct embedding and output parameters."""
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.all_tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}
        self.resolve_alias = resolve_alias
        self.tie_calls = 0

    def tie_weights(self, recompute_mapping: bool = True) -> None:
        """Apply the declared alias unless this fixture is configured invalidly."""
        self.tie_calls += 1
        if recompute_mapping:
            raise AssertionError("deferred construction must use the static tied mapping")
        if self.resolve_alias:
            self.lm_head.weight = self.embed_tokens.weight

    def get_parameter_or_buffer(self, target: str) -> nn.Parameter:
        """Resolve one parameter FQN using the Transformers public contract."""
        return self.get_parameter(target)


class TestDeferredTiedWeights(unittest.TestCase):
    """Ensure FSDP receives shared parameter identity, not storage aliases."""

    def test_ties_parameters_before_distributed_wrapping(self) -> None:
        """The declared source becomes the exact parameter used by the target."""
        model = _DeferredTiedModel()
        self.assertIsNot(model.lm_head.weight, model.embed_tokens.weight)

        _tie_model_weights_before_parallelism(model)

        self.assertEqual(model.tie_calls, 1)
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)

    def test_rejects_tie_implementation_that_leaves_distinct_parameters(self) -> None:
        """A broken model tying contract fails before FSDP creates two owners."""
        model = _DeferredTiedModel(resolve_alias=False)

        with self.assertRaisesRegex(ValueError, "did not establish parameter identity"):
            _tie_model_weights_before_parallelism(model)


if __name__ == "__main__":
    unittest.main()
