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
"""Parallel degrees must be positive integers — no silent max(1,) clamping.

Regression guard for the combination-test finding that ``tp_size=0`` /
``ep_size=-1`` were silently coerced to ``1`` by the codegen DP projection
(``codegen/spec/project.py``), materializing artifacts for a topology the
user never requested without any error.
"""

import os
import types
import unittest
from unittest import mock

from hyper_parallel.codegen.spec.project import _project_dp_size
from hyper_parallel.trainer.config.parallelism import AcceleratorConfig


class TestAcceleratorConfigValidation(unittest.TestCase):
    """``AcceleratorConfig`` rejects non-positive parallel degrees."""

    def test_defaults_are_valid(self):
        config = AcceleratorConfig()
        self.assertEqual((config.tp_size, config.cp_size,
                          config.ep_size, config.pp_size), (1, 1, 1, 1))

    def test_positive_degrees_accepted(self):
        config = AcceleratorConfig(tp_size=8, cp_size=2, ep_size=4, pp_size=2)
        self.assertEqual(config.tp_size, 8)

    def test_zero_degree_rejected(self):
        for name in ("tp_size", "cp_size", "ep_size", "pp_size"):
            with self.subTest(name=name), self.assertRaises(ValueError) as ctx:
                AcceleratorConfig(**{name: 0})
            self.assertIn(f"accelerator.{name}", str(ctx.exception))

    def test_negative_degree_rejected(self):
        for name in ("tp_size", "cp_size", "ep_size", "pp_size"):
            with self.subTest(name=name), self.assertRaises(ValueError) as ctx:
                AcceleratorConfig(**{name: -1})
            self.assertIn(f"accelerator.{name}", str(ctx.exception))

    def test_bool_rejected(self):
        # bool is an int subclass but is never a valid degree.
        with self.assertRaises(ValueError):
            AcceleratorConfig(tp_size=True)


class TestDpProjectionNoClamp(unittest.TestCase):
    """The DP projection must not paper over invalid degrees."""

    @staticmethod
    def _accelerator(**kwargs):
        base = {"tp_size": 1, "cp_size": 1, "ep_size": 1, "pp_size": 1,
                "dp_size": None}
        base.update(kwargs)
        return types.SimpleNamespace(**base)

    def test_dp_derived_from_world_size(self):
        accel = self._accelerator(tp_size=2, cp_size=1, pp_size=1)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "6"}):
            self.assertEqual(_project_dp_size(accel), 3)

    def test_explicit_dp_wins(self):
        accel = self._accelerator(tp_size=2, dp_size=4)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            self.assertEqual(_project_dp_size(accel), 4)

    def test_non_divisible_world_raises_instead_of_clamping(self):
        accel = self._accelerator(tp_size=2)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "5"}):
            with self.assertRaises(ValueError) as ctx:
                _project_dp_size(accel)
        self.assertIn("not divisible", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
