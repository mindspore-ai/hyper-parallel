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
"""VeRL external registration selects HyperParallel without loading VeRL's legacy engine."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.environ.get("VERL_SOURCE_ROOT"), "requires an explicit optional VeRL checkout")
class TestEntryPoint(unittest.TestCase):
    """Exercise real framework registration in a clean interpreter."""

    def test_registration_and_native_construction(self) -> None:
        """Import before device assignment, then construct without duplicate registration."""
        root = Path(__file__).resolve().parents[4]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((str(root), environment["VERL_SOURCE_ROOT"],
                                                     environment.get("PYTHONPATH", "")))
        environment["VERL_USE_EXTERNAL_MODULES"] = "hyper_parallel.integration.verl.patch"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        script = """
import sys
from unittest.mock import patch
from hyper_parallel.integration.verl.config import HyperParallelEngineConfig
import verl
from verl.workers.engine.base import EngineRegistry
proxy = EngineRegistry._engines['language_model']['hyper_parallel']['npu']
assert proxy.__module__ == 'hyper_parallel.integration.verl.engine.transformer_impl'
engine_config = HyperParallelEngineConfig(
    recipe_path='hyper_parallel/models/qwen3_moe/recipes/train.yaml',
    trainer_config={'model': {'torch_dtype': 'float32'}},
)
assert engine_config.strategy == 'hyper_parallel'
from hyper_parallel.integration.verl.engine.transformer_impl import HyperParallelEngineWithLMHead
config = engine_config.trainer_config
assert config.model.torch_dtype == 'float32'
assert config.dataset is not None and config.dataloader.get_batch is not None
with patch.object(HyperParallelEngineWithLMHead, '__init__', return_value=None):
    instance = proxy(None, None, None, None)
    assert isinstance(instance, HyperParallelEngineWithLMHead)
assert EngineRegistry._engines['language_model']['hyper_parallel']['npu'] is proxy
assert 'verl.workers.engine.hyper_parallel' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=environment,
                                capture_output=True, text=True, timeout=300, check=False)
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode}; stderr={result.stderr}")
