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
"""Tests for Wan AutoModels config wiring."""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.auto_models.components.models.wan.configuration import (  # pylint: disable=wrong-import-position
    WanTransformer3DTrainingConfig,
)
from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position


class TestWanTransformer3DTrainingConfig(unittest.TestCase):
    """Coverage for Transformers-managed attention configuration."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_preserves_requested_attention_implementation(self):
        config = WanTransformer3DTrainingConfig(attn_implementation="flash_attention_2")

        self.assertEqual(config._attn_implementation, "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
