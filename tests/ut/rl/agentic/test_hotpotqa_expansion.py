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
"""CPU checks for expanded splits and conservative rollout error classification."""

import unittest
import json
import pandas as pd
from examples.search_r1.prepare_data import split_samples
from examples.search_r1.container_execution import is_context_overflow


class ExpansionTests(unittest.TestCase):
    """Expansion must never train on earlier held-out questions."""

    def test_frozen_holdout_and_determinism(self) -> None:
        """Keep prior IDs out of training with deterministic new partitions."""
        frame = pd.DataFrame({"question_id": list(map(str, range(20))),
                              "question": [f"Question {i}" for i in range(20)], "answer": ["A"] * 20})
        old_test = frame.iloc[[2, 5]]
        train, test = split_samples(frame, 10, 4, 123, old_test)
        self.assertEqual((len(train), len(test)), (10, 4))
        self.assertFalse(set(train.question_id) & set(test.question_id))
        self.assertTrue(set(old_test.question_id).issubset(set(test.question_id)))
        pd.testing.assert_frame_equal(train, split_samples(frame, 10, 4, 123, old_test)[0])
        with self.assertRaises(ValueError):
            split_samples(pd.concat([frame, frame.iloc[:1]]), 10, 4, 123)
        with self.assertRaises(ValueError):
            split_samples(frame, 19, 4, 123)

    def test_only_explicit_length_rejections_are_task_failures(self) -> None:
        """Server faults and generic bad requests cannot become zero-reward tasks."""
        body = b'{"error": "maximum context length is 8192 tokens"}'
        self.assertTrue(is_context_overflow(400, body))
        self.assertFalse(is_context_overflow(500, body))
        self.assertFalse(is_context_overflow(400, b'invalid tool schema'))
        wrapped = json.dumps({"error": {"type": "gateway_error", "message":
            "vLLM chat completion failed with HTTP 400: maximum context length is 8192"}}).encode()
        self.assertTrue(is_context_overflow(502, wrapped))
        self.assertFalse(is_context_overflow(502, body))
