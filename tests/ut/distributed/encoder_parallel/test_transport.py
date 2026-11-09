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
"""CPU image packing, bounded TP transport, and local rotary positions."""

from types import SimpleNamespace
from typing import Any, Iterator, Mapping
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.data.parallel import DataLoaderParallelContext, TPBatchBroadcaster
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderLocalInputs
from hyper_parallel.distributed.encoder_parallel.planning import assign_lpt, build_segments
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import DeepseekV41VisionBridge
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
    _vision_cos_sin,
)


class TestEncoderTransport(unittest.TestCase):
    """Keep complete pixels on the host and preserve shard geometry."""

    def setUp(self) -> None:
        """Build native visual modules without creating process groups."""
        config = SimpleNamespace(
            v41_vision_hidden_size=8, v41_vision_patch_size=2, v41_vision_num_attention_heads=2,
            v41_vision_rope_theta=10000.0, v41_vision_intermediate_size=16,
            v41_vision_num_hidden_layers=2, v41_vision_downsample_ratio=1, hidden_size=8,
        )
        model = nn.Module()
        model.vision = DeepseekV41VisionTower(config)
        model.aligner = DeepseekV41VisionAligner(config)
        self.bridge = DeepseekV41VisionBridge(model)
        self.batch = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "labels": torch.zeros(1, 16, dtype=torch.long),
            "token_types": torch.zeros(1, 16, dtype=torch.long),
            "pixel_values": torch.arange(7 * 12, dtype=torch.float64).reshape(7, 3, 2, 2),
            "image_patch_offsets": torch.tensor([0, 5, 7]),
            "image_vit_grid_hw": torch.tensor([[1, 5], [1, 2]]),
            "image_llm_grid_hw": torch.tensor([[1, 5], [1, 2]]),
            "image_batch_indices": torch.zeros(2, dtype=torch.long),
            "image_token_starts": torch.tensor([0, 8]),
        }
        self.items = self.bridge.parse_images(self.batch)

    def _runtime(self, cp_size: int = 2, tp_size: int = 1) -> SimpleNamespace:
        """Provide pure planning geometry and a recorded finalize operation."""
        return SimpleNamespace(
            bridge=self.bridge, config=EncoderParallelConfig(enabled=True, encoder_cp_size=2),
            mesh_context=SimpleNamespace(cp_size=cp_size, tp_size=tp_size, cp_rank=0,
                                         tp_rank=0, sequence_parallel=False),
            finalize_plan=Mock(),
        )

    def test_cpu_plan_precedes_device_payload_and_excludes_complete_images(self) -> None:
        """Feature: metadata-first image transport.

        Description: Run the real batch broadcaster with a recorded device-movement boundary.
        Expectation: Only selected patches move and native floating-point dtype is retained.
        """
        runtime = self._runtime()
        segments = build_segments(self.items, 0, 2)
        plan = SimpleNamespace(active=True, local_segments=segments, buckets=(self.items,))
        runtime.finalize_plan.return_value = plan
        transport = EncoderParallelBatchTransport(runtime=runtime)
        broadcaster = TPBatchBroadcaster(DataLoaderParallelContext(cp_world_size=2), "cpu")
        moved_fields = []
        move = broadcaster._move_mapping_to_device  # pylint: disable=protected-access

        def _record_move(batch: Mapping[str, Any]) -> dict[str, Any]:
            """Record fields reaching ordinary device movement."""
            moved_fields.extend(batch)
            return move(batch)

        with patch.object(broadcaster, "_move_mapping_to_device", side_effect=_record_move):
            with patch("hyper_parallel.distributed.encoder_parallel.transport.dist.is_initialized", return_value=False):
                device_batch, inputs = transport(self.batch, broadcaster)
        self.assertFalse(set(self.bridge.input_fields).intersection(moved_fields))
        self.assertNotIn("pixel_values", device_batch)
        self.assertIs(inputs["encoder_parallel_plan"], plan)
        patches = inputs["encoder_parallel_inputs"].patches
        torch.testing.assert_close(patches, self.bridge.pack_payload(self.batch, segments))
        self.assertEqual(patches.dtype, torch.float64)
        self.assertEqual(self.batch["pixel_values"].device.type, "cpu")
        self.assertEqual(runtime.finalize_plan.call_args.args[0], self.items)

    def test_tp_destinations_receive_different_buckets_and_cp_shards(self) -> None:
        """Feature: destination-specific TP image transport.

        Description: Replay mocked TP4 broadcasts with two encoder DP buckets.
        Expectation: Each rank receives its own bucket/shard within the staging limit.
        """
        runtime = self._runtime(cp_size=1, tp_size=4)
        transport = EncoderParallelBatchTransport(runtime=runtime, chunk_bytes=32)
        buckets = assign_lpt(self.items, 2, 2)
        broadcasts = []
        for rank in range(4):
            bucket, cp_rank = divmod(rank, 2)
            segments = build_segments(buckets[bucket], cp_rank, 2)
            plan = SimpleNamespace(local_segments=segments, buckets=buckets)
            context = DataLoaderParallelContext(tp_world_size=4, tp_rank=rank)
            broadcaster = TPBatchBroadcaster(context, "cpu")
            cursor = iter(broadcasts)

            def _broadcast(tensor: torch.Tensor, *, source_rank: int = rank,
                           replay: Iterator[torch.Tensor] = cursor, **_kwargs: Any) -> None:
                """Record source chunks or replay them into a receiving rank."""
                if source_rank == 0:
                    broadcasts.append(tensor.clone())
                else:
                    tensor.copy_(next(replay))

            with patch("hyper_parallel.distributed.encoder_parallel.transport.dist.broadcast", side_effect=_broadcast):
                actual = transport._transport_patches(  # pylint: disable=protected-access
                    self.batch if rank == 0 else None, plan, torch.float64, broadcaster,
                )
            torch.testing.assert_close(actual, self.bridge.pack_payload(self.batch, segments))
        self.assertTrue(all(tensor.numel() * tensor.element_size() <= 32 for tensor in broadcasts))

    def test_local_rope_matches_full_grid_with_padding_and_empty_shards(self) -> None:
        """Feature: local rotary position construction.

        Description: Split five patches across eight ranks and also pack a dummy bucket.
        Expectation: Valid RoPE matches the full grid and padding has identity rotation.
        """
        item = self.items[0]
        full_cos, full_sin = _vision_cos_sin(1, 5, 2, 10000.0, torch.device("cpu"))
        for cp_rank in range(8):
            segments = build_segments((item,), cp_rank, 8)
            packed = self.bridge.pack_payload(self.batch, segments)
            actual, cosine, sine = self.bridge._pack_local_inputs(  # pylint: disable=protected-access
                EncoderLocalInputs(packed), segments,
            )
            self.assertIs(actual, packed)
            valid = segments[0].local_valid_length
            torch.testing.assert_close(cosine[:valid], full_cos[cp_rank:cp_rank + valid])
            torch.testing.assert_close(sine[:valid], full_sin[cp_rank:cp_rank + valid])
            torch.testing.assert_close(cosine[valid:], torch.ones_like(cosine[valid:]))
            torch.testing.assert_close(sine[valid:], torch.zeros_like(sine[valid:]))
        dummy = build_segments((), 0, 2)
        torch.testing.assert_close(self.bridge.pack_payload({}, dummy), torch.zeros(1, 3, 2, 2))

    def test_invalid_source_fails_before_plan_or_payload(self) -> None:
        """Feature: image transport input validation.

        Description: Supply incomplete metadata and caller-owned planning fields.
        Expectation: Both inputs fail before plan or payload execution.
        """
        runtime = self._runtime()
        transport = EncoderParallelBatchTransport(runtime=runtime)
        broadcaster = TPBatchBroadcaster(DataLoaderParallelContext(), "cpu")
        for field, value in (("image_patch_offsets", None), ("encoder_parallel_plan", object())):
            with self.subTest(field=field):
                batch = dict(self.batch, **{field: value})
                with patch(
                        "hyper_parallel.distributed.encoder_parallel.transport.dist.is_initialized", return_value=False,
                ):
                    with self.assertRaisesRegex(ValueError, "invalid encoder CPU input"):
                        transport(batch, broadcaster)
        runtime.finalize_plan.assert_not_called()

    def test_inactive_batch_has_no_patch_allocation(self) -> None:
        """Feature: inactive visual input transport.

        Description: Prepare a step with no images on any participant.
        Expectation: The returned inactive plan has no allocated patch payload.
        """
        runtime = self._runtime()
        plan = SimpleNamespace(active=False)
        runtime.finalize_plan.return_value = plan
        transport = EncoderParallelBatchTransport(runtime=runtime)
        batch = {name: value for name, value in self.batch.items() if name not in self.bridge.input_fields}
        broadcaster = TPBatchBroadcaster(DataLoaderParallelContext(), "cpu")
        with patch("hyper_parallel.distributed.encoder_parallel.transport.dist.is_initialized", return_value=False):
            with patch.object(self.bridge, "pack_payload") as pack:
                _, inputs = transport(batch, broadcaster)
        pack.assert_not_called()
        self.assertEqual(inputs, {"encoder_parallel_plan": plan})
