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
"""Unified log-level configuration: YAML parsing and runtime application."""

import logging
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.runtime.logging import setup_logging


def _sample_model(width: int) -> None:  # pylint: disable=unused-argument
    """Model target required by the trainer config schema."""


def _sample_optimizer(learning_rate: float) -> None:  # pylint: disable=unused-argument
    """Optimizer target required by the trainer config schema."""


_SAMPLE_YAML = """
model:
  _target_: {module}._sample_model
  width: 8
optimizer:
  _target_: {module}._sample_optimizer
  learning_rate: 0.01
{extra}"""


class TestLogLevelConfig(unittest.TestCase):
    """``log_level`` is parsed from YAML/CLI and validated at the boundary."""

    def _parse(self, *overrides: str, extra: str = ""):
        """Write the sample YAML plus ``extra`` lines and parse it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "train.yaml"
            yaml_path.write_text(
                _SAMPLE_YAML.format(module=__name__, extra=extra), encoding="utf-8"
            )
            return parse_training_args([str(yaml_path), *overrides])

    def test_default_log_level_is_warning(self) -> None:
        """The framework stays quiet below WARNING unless configured."""
        config = self._parse()
        self.assertEqual(config.log_level, "warning")

    def test_yaml_log_level_is_accepted(self) -> None:
        """A top-level ``log_level`` key overrides the default."""
        config = self._parse(extra="log_level: info\n")
        self.assertEqual(config.log_level, "info")

    def test_cli_log_level_override_wins(self) -> None:
        """``--log_level=...`` overrides the YAML value."""
        config = self._parse("--log_level=error", extra="log_level: info\n")
        self.assertEqual(config.log_level, "error")

    def test_invalid_log_level_rejected(self) -> None:
        """A value outside the five standard levels fails validation."""
        with self.assertRaises(ValueError):
            self._parse(extra="log_level: verbose\n")


class TestSetupLoggingLevel(unittest.TestCase):
    """``setup_logging`` applies the configured level to the root logger."""

    def setUp(self) -> None:
        """Snapshot the root logger level."""
        self._saved_root_level = logging.getLogger().level

    def tearDown(self) -> None:
        """Restore the root logger level."""
        logging.getLogger().setLevel(self._saved_root_level)

    def test_setup_logging_sets_root_level(self) -> None:
        """The given level lands on the root logger even with handlers present."""
        setup_logging("error")
        self.assertEqual(logging.getLogger().level, logging.ERROR)
        setup_logging("debug")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_setup_logging_defaults_to_warning(self) -> None:
        """No argument keeps the framework default at WARNING."""
        setup_logging()
        self.assertEqual(logging.getLogger().level, logging.WARNING)
