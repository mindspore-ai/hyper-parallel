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
"""Verify native registration hooks against external registry interfaces."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from tests.ut.integration.verl._contracts import ADAPTERS, SOURCE_ROOT


class TestEntryPoint(unittest.TestCase):
    """Execute the production patch entrypoint without importing VeRL."""

    def test_registration_and_native_construction(self) -> None:
        """Register the real engine and optional-field fixes through explicit external spies."""
        engine_cls = ADAPTERS.engine.HyperParallelEngineWithLMHead
        decorator = Mock(side_effect=lambda engine: engine)
        with patch.object(ADAPTERS.registration, "EngineRegistry") as registry, patch.object(
            ADAPTERS.optional_fields, "PPOTrainer",
        ) as trainer:
            registry.register.return_value = decorator
            spec = spec_from_file_location("hyper_parallel.integration.verl.patch", SOURCE_ROOT / "patch/__init__.py")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            registry.register.assert_called_once_with(
                model_type="language_model", backend="hyper_parallel", device=["cuda", "npu"],
            )
            decorator.assert_called_once_with(engine_cls)
            self.assertIs(trainer._compute_advantage, ADAPTERS.optional_fields._compute_advantage)
            self.assertIs(trainer._compute_metrics, ADAPTERS.optional_fields._compute_metrics)
            with patch.object(engine_cls, "__init__", return_value=None):
                self.assertIsInstance(decorator.call_args.args[0](None, None, None, None), engine_cls)
        self.assertEqual(OmegaConf.create({"tokens": "${cp_token_budget:3,4,2}"}).tokens, 4)
        self.assertNotIn("verl.workers.engine.hyper_parallel", sys.modules)
