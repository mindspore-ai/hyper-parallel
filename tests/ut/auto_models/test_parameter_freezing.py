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
"""Parameter freezing during the model build (``freeze_config``).

``freeze_config`` used to be a stub that only logged "not implemented". It is now
applied after module replacement and **before** plan/FSDP derivation, which is the
only point where it is effective: FSDP decides from ``requires_grad`` which
parameters join the gradient reduction, so freezing afterwards would leave the
tower in the all-reduce while its weights never move. Freezing also stops autograd
at the tower's output, removing its backward pass and saved activations -- which is
what makes an A/B against a pipeline that freezes the same tower meaningful.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from torch import nn

from tests.common.mark_utils import arg_mark

# transformer-heavy import chain: on CI pythons without liblzma the lazy
# transformers -> torchvision import dies, so guard as the sibling tests do.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None


def _build_model():
    """Build a nested stand-in whose towers mirror the real FQN shape.

    The real model nests the towers under ``model`` (``KimiK25ForConditionalGeneration.model``),
    so the FQNs look like ``model.vision_tower.layers.0``; keep that shape here or the
    patterns under test would not exercise what production actually matches.
    """
    inner = nn.Module()
    tower = nn.Module()
    tower.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    inner.vision_tower = tower
    inner.mm_projector = nn.Linear(4, 2)
    language_model = nn.Module()
    language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
    inner.language_model = language_model
    root = nn.Module()
    root.model = inner
    return root


def _all_frozen(module):
    """Whether every parameter under ``module`` has ``requires_grad`` cleared."""
    return all(not parameter.requires_grad for parameter in module.parameters())


def _all_trainable(module):
    """Whether every parameter under ``module`` still requires grad."""
    return all(parameter.requires_grad for parameter in module.parameters())


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestApplyParameterFreezing(unittest.TestCase):
    """``_apply_parameter_freezing`` freezes exactly the matched subtrees."""

    @staticmethod
    def _freeze(model, patterns):
        from hyper_parallel.models._transformers.model_builder import (
            _apply_parameter_freezing,
        )

        return _apply_parameter_freezing(model, patterns)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_glob_freezes_only_the_matched_subtree(self):
        """A tower glob freezes that tower and leaves every sibling trainable."""
        model = _build_model()

        self._freeze(model, ["model.vision_tower*"])

        self.assertTrue(_all_frozen(model.model.vision_tower))
        self.assertTrue(_all_trainable(model.model.mm_projector))
        self.assertTrue(_all_trainable(model.model.language_model))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_accepts_a_bare_string_pattern(self):
        """A single pattern need not be wrapped in a list."""
        model = _build_model()

        self._freeze(model, "model.vision_tower*")

        self.assertTrue(_all_frozen(model.model.vision_tower))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_multiple_patterns_and_deep_names(self):
        """Patterns compose, and ``*`` crosses dotted module boundaries."""
        model = _build_model()

        self._freeze(model, ["model.vision_tower.layers*", "model.mm_projector*"])

        self.assertTrue(_all_frozen(model.model.vision_tower))
        self.assertTrue(_all_frozen(model.model.mm_projector))
        self.assertTrue(_all_trainable(model.model.language_model))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_match_raises_instead_of_silently_continuing(self):
        """A typo must fail loudly: a quietly trainable tower invalidates the A/B."""
        with self.assertRaisesRegex(ValueError, "matched no parameter"):
            self._freeze(_build_model(), ["model.vison_tower*"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_pattern_list_is_a_no_op(self):
        """An empty list freezes nothing and does not raise."""
        model = _build_model()

        self._freeze(model, [])

        self.assertTrue(_all_trainable(model.model.vision_tower))
        self.assertTrue(_all_trainable(model.model.language_model))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_freezing_twice_is_idempotent(self):
        """Re-applying the same patterns keeps the state and does not raise."""
        model = _build_model()

        self._freeze(model, ["model.vision_tower*"])
        self._freeze(model, ["model.vision_tower*"])

        self.assertTrue(_all_frozen(model.model.vision_tower))


if __name__ == "__main__":
    unittest.main()
