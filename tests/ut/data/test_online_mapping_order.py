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
"""Verify single-source sampling does not depend on the training horizon."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.data.online.mapping import build_online_mapping_source
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from tests.common.mark_utils import arg_mark


class TestOnlineMappingOrder(unittest.TestCase):
    """Keep source ordering separate from epoch sampling."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_training_horizon_preserves_cyclic_prefix(self):
        """Different training lengths must share the same initial samples.

        Feature: Single-source online sampling.
        Description: Build short and long training views of a real JSONL file.
        Expectation: Both preserve full source order and the legacy cyclic permutation.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(32)))
            expected = list(range(32))
            random.Random(42).shuffle(expected)
            for target in (16, 24, 64):
                with self.subTest(target=target):
                    source = build_online_mapping_source(
                        data_path=str(path), data_config={"random_seed": 42},
                        train_valid_test_num_samples=(target, 0, 0),
                    )
                    self.assertEqual(len(source), 32)
                    self.assertEqual([source[i]["id"] for i in range(32)], list(range(32)))
                    sampler = build_dataset_batch_sampler(
                        total_samples=len(source), micro_batch_size=2, global_batch_size=8,
                        dp_rank=0, dp_world_size=1, sampler_type="cyclic", seed=42,
                    )
                    actual = [source[i]["id"] for batch in sampler for i in batch]
                    self.assertEqual(actual, expected)
