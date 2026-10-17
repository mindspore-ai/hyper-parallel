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
"""CPU contracts for bounded native-vLLM router payloads and TP transport."""

import base64
import io
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.router_capture import (
    RouterCaptureSpec, decode_router_trace,
)
from rl.roles.rollout.vllm import VLLMGenerationEngine, _VLLMHTTPClient


def _payload(array: np.ndarray) -> str:
    """Encode a native-vLLM NPY response without any service dependency."""
    stream = io.BytesIO()
    np.save(stream, array)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings() -> GenerationSettings:
    """Require routes while leaving sampled probabilities disabled."""
    return GenerationSettings(2, 0.7, 0.9, 0, True, 0, 2, collect_router_trace=True)


class TestRouterCapture(unittest.TestCase):
    """Require exact input alignment and bounded compact integer payloads."""

    def setUp(self) -> None:
        """Use two sparse layers and distinct experts at every valid position."""
        self.spec = RouterCaptureSpec((1, 3), 2, 128)
        self.routes = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int64)

    def test_missing_final_input_row_is_false_without_shifting_prompt_routes(self) -> None:
        """The first prompt token remains route row zero; only the final row is missing."""
        trace = decode_router_trace(_payload(self.routes), 4, self.spec)
        self.assertEqual(trace.expert_ids.dtype, torch.uint8)
        self.assertEqual(trace.expert_ids.device.type, "cpu")
        self.assertEqual(trace.layer_ids, (1, 3))
        torch.testing.assert_close(trace.expert_ids[:3], torch.tensor(self.routes, dtype=torch.uint8))
        torch.testing.assert_close(trace.valid_mask, torch.tensor([True, True, True, False]))
        full = decode_router_trace(_payload(self.routes), 3, self.spec)
        self.assertTrue(full.valid_mask.all())


    def test_choice_order_and_prompt_lengths_remain_aligned(self) -> None:
        """Out-of-order choices carry the routes of their own unpadded prompt."""
        client = _VLLMHTTPClient(None, "http://unused", "moe", 1, self.spec)
        choices = [
            {"index": 1, "token_ids": [20], "routed_experts": _payload(self.routes[:1])},
            {"index": 0, "token_ids": [10], "routed_experts": _payload(self.routes)},
        ]
        records = client._completion_records_from_response([[1, 2, 3], [4]], 1, _settings(), {"choices": choices})
        self.assertEqual([record[0] for record in records], [[10], [20]])
        self.assertEqual([record[3].expert_ids.shape[0] for record in records], [4, 2])
        self.assertNotIn("return_routed_experts", client._completion_payload([[1]], None, _settings()))
        del choices[0]["routed_experts"]
        with self.assertRaisesRegex(ValueError, "payload"):
            client._completion_records_from_response([[1, 2, 3], [4]], 1, _settings(), {"choices": choices})

    def test_left_padding_and_completion_truncation_do_not_shift_routes(self) -> None:
        """Returned route rows exclude left padding and follow the retained completion prefix."""
        trace = decode_router_trace(_payload(np.concatenate((self.routes, self.routes[:1]))), 5, self.spec)
        request = GenerationRequest(torch.tensor([[0, 0, 1, 2]]), torch.tensor([[0, 0, 1, 1]]), _settings())
        result = VLLMGenerationEngine._build_generation_result(request, [([7, 8, 9], None, "length", trace)], 0, 0)
        self.assertEqual(result.router_traces[0].expert_ids.shape[0], 4)
        torch.testing.assert_close(result.router_traces[0].expert_ids, trace.expert_ids[:4])
        with self.assertRaisesRegex(ValueError, "required router trace"):
            VLLMGenerationEngine._build_generation_result(request, [([7], None, "stop")], 0, 0)

    def test_tp_owner_sync_transports_binary_routes_to_its_sibling(self) -> None:
        """Existing timing metadata transport also carries the owner's binary trace."""
        trace = decode_router_trace(_payload(self.routes), 4, self.spec)
        request = GenerationRequest(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.bool), _settings())
        result = VLLMGenerationEngine._build_generation_result(request, [([7, 8], None, "stop", trace)], 1.0, 0)
        tensor_payloads, metadata_payloads = [], []
        owner = object.__new__(VLLMGenerationEngine)
        owner._trainer_tp_rank, owner._trainer_tp_size, owner._trainer_tp_group = 0, 2, "tp"
        sibling = object.__new__(VLLMGenerationEngine)
        sibling._trainer_tp_rank, sibling._trainer_tp_size, sibling._trainer_tp_group = 1, 2, "tp"

        def broadcast(value: torch.Tensor, **_kwargs: Any) -> None:
            """Replicate the owner's tensor buffers without real distributed setup."""
            if owner._trainer_tp_rank == 0:
                tensor_payloads.append(value.clone())
            else:
                value.copy_(tensor_payloads.pop(0))

        def gather(output: list[Any], value: Any, group: Any) -> None:
            """Replicate the existing owner's object metadata including route bytes."""
            del group
            if value is not None:
                metadata_payloads.append(value)
            output[:] = [metadata_payloads[0], None]

        with patch("rl.roles.rollout.vllm.dist.broadcast", side_effect=broadcast), patch(
            "rl.roles.rollout.vllm.dist.all_gather_object", side_effect=gather,
        ):
            owner._broadcast_tp_result(request, result, 0)
            owner._trainer_tp_rank = 1
            replica = sibling._broadcast_tp_result(request, None, 0)
        self.assertIsInstance(metadata_payloads[0]["router_traces"][0]["expert_ids"], bytes)
        torch.testing.assert_close(replica.router_traces[0].expert_ids, trace.expert_ids)
        torch.testing.assert_close(replica.router_traces[0].valid_mask, trace.valid_mask)
