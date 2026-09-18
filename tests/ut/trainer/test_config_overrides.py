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
"""CLI override contracts for the trainer config entry point.

The override layer spans :mod:`~hyper_parallel.trainer.config.parser` (token and
YAML-value parsing) and :mod:`~hyper_parallel.trainer.config.resolver` (typed
writes into the resolved config tree). These tests pin the user-visible
behaviour of ``--a.b=c`` overrides: annotation-driven coercion, the three write
paths (Target argument / mapping key / dataclass field), and the error messages
for bad tokens, unknown names and type mismatches.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config.resolver import ConfigResolutionError


def _sample_model(  # pylint: disable=unused-argument
    width: int,
    depth: int = 2,
    *,
    activation: str = "gelu",
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Model target whose parameters the override tests rewrite."""


def _sample_optimizer(  # pylint: disable=unused-argument
    learning_rate: float,
    *,
    clip: bool = False,
) -> None:
    """Optimizer target whose parameters the override tests rewrite."""


_SAMPLE_YAML = """
model:
  _target_: {module}._sample_model
  width: 8
  metadata:
    owner: default
optimizer:
  _target_: {module}._sample_optimizer
  learning_rate: 0.01
"""


class TestConfigOverrides(unittest.TestCase):
    """``--a.b=c`` overrides applied to a resolved trainer config."""

    def _parse(self, *overrides: str):
        """Write the sample YAML into a temporary directory and parse it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "train.yaml"
            yaml_path.write_text(_SAMPLE_YAML.format(module=__name__), encoding="utf-8")
            return parse_training_args([str(yaml_path), *overrides])

    def test_reads_yaml_without_overrides(self) -> None:
        """YAML values survive, and the callable default is filled in."""
        config = self._parse()
        self.assertEqual(
            config.model.width, 8,
            msg=f"case: yaml_value_survives, width={config.model.width!r}")
        self.assertEqual(
            config.model.depth, 2,
            msg=f"case: callable_default_is_filled, depth={config.model.depth!r}")
        self.assertEqual(
            config.model.activation, "gelu",
            msg=f"case: keyword_default_is_filled, activation={config.model.activation!r}")

    def test_scalar_overrides_are_coerced(self) -> None:
        """Scalar overrides are coerced against the declared annotations."""
        config = self._parse("--model.width=16", "--model.depth=3", "--model.activation=silu")
        self.assertEqual(
            config.model.width, 16,
            msg=f"case: int_coercion, width={config.model.width!r}")
        self.assertEqual(
            config.model.depth, 3,
            msg=f"case: int_coercion_keyword_default, depth={config.model.depth!r}")
        self.assertEqual(
            config.model.activation, "silu",
            msg=f"case: str_coercion, activation={config.model.activation!r}")

    def test_optimizer_target_argument(self) -> None:
        """``--optimizer.<arg>`` reaches the optimizer target through the shorthand."""
        config = self._parse("--optimizer.learning_rate=0.5")
        self.assertEqual(
            config.optimizer.target.learning_rate, 0.5,
            msg=f"case: optimizer_target_shorthand, target={config.optimizer.target.to_dict()!r}")

    def test_type_mismatch_reports_annotation(self) -> None:
        """A value the annotation cannot accept fails at the override path."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"CLI\.model\.width: expected int, got str",
            msg="case: type_mismatch_reports_annotation",
        ):
            self._parse("--model.width=abc")

    def test_unknown_target_argument_suggests(self) -> None:
        """A misspelled target argument is reported with a suggestion."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unknown target argument 'wdith'; did you mean 'width'\?",
            msg="case: unknown_target_argument_suggests",
        ):
            self._parse("--model.wdith=8")

    def test_nested_mapping_value(self) -> None:
        """An override below a mapping argument rewrites that key."""
        config = self._parse("--model.metadata.owner=tshu")
        self.assertEqual(
            config.model.metadata, {"owner": "tshu"},
            msg=f"case: nested_mapping_write, metadata={config.model.metadata!r}")

    def test_unknown_mapping_key(self) -> None:
        """Writing a key the mapping does not declare is rejected."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unknown mapping key 'size'",
            msg="case: unknown_mapping_key",
        ):
            self._parse("--model.metadata.size=3")

    def test_unselected_component(self) -> None:
        """Overriding a component the YAML left unset is rejected."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"component was not selected",
            msg="case: unselected_component",
        ):
            self._parse("--dataset.foo=1")

    def test_unknown_root_field(self) -> None:
        """An unknown root field names the config type it was looked up on."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unknown field 'nope' on TrainerConfig",
            msg="case: unknown_root_field",
        ):
            self._parse("--nope=1")

    def test_target_replacement_is_rejected(self) -> None:
        """``_target_`` cannot be rewritten through an override."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"changing _target_ through an override is not supported",
            msg="case: target_replacement_is_rejected",
        ):
            self._parse("--model._target_=other.callable")

    def test_bad_token(self) -> None:
        """A token that is not ``--field=value`` is rejected."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"expected a dotted override in '--field=value' form",
            msg="case: bad_token",
        ):
            self._parse("oops")

    def test_invalid_yaml_value(self) -> None:
        """A value that is not valid YAML is reported with the override path."""
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"invalid YAML value",
            msg="case: invalid_yaml_value",
        ):
            self._parse("--model.width=[unclosed")


if __name__ == "__main__":
    unittest.main()
