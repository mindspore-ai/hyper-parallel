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
"""Tests for framed CPU payload serialization and integrity validation."""

import hashlib
import pickle
import unittest
from typing import Any
from unittest.mock import patch

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    DataPlaneTransport,
    create_data_groups,
    _decode_payload_segment,
    _encode_payload_segment,
)


def _frame(value: Any) -> bytes:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return b"HPDDP1" + hashlib.sha256(payload).digest() + payload


class TestPayloadCodec(unittest.TestCase):
    """Verify route frames detect corruption and malformed sample identities."""

    def test_round_trip_preserves_order_keys_and_payloads(self) -> None:
        """A valid route segment round-trips without changing payload order."""
        items = (
            (SampleKey(0, 7), {"tokens": [1, 2], "image": b"jpeg"}),
            (SampleKey(4, 3), ("caption", 9)),
        )

        encoded = _encode_payload_segment(items)

        self.assertEqual(_decode_payload_segment(encoded), items)
        self.assertEqual(_encode_payload_segment(()), b"")
        self.assertEqual(_decode_payload_segment(b""), ())

    def test_encoder_rejects_duplicate_sample_keys(self) -> None:
        """One reader route cannot claim the same raw sample twice."""
        duplicate = SampleKey(0, 1)

        with self.assertRaisesRegex(ValueError, "unique SampleKey"):
            _encode_payload_segment(((duplicate, "first"), (duplicate, "second")))

    def test_decoder_rejects_header_and_checksum_corruption(self) -> None:
        """Frame magic and SHA-256 digest are both validated before unpickling."""
        encoded = _encode_payload_segment(((SampleKey(0, 1), {"value": 3}),))
        bad_header = b"BROKEN" + encoded[6:]
        bad_payload = encoded[:-1] + bytes((encoded[-1] ^ 0x01,))

        with self.assertRaisesRegex(ValueError, "invalid frame header"):
            _decode_payload_segment(bad_header)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            _decode_payload_segment(bad_payload)

    def test_decoder_rejects_validly_framed_non_route_payload(self) -> None:
        """A matching digest does not bypass the route schema validation."""
        with self.assertRaisesRegex(ValueError, "invalid route segment"):
            _decode_payload_segment(_frame([(SampleKey(0, 0), "payload")]))

    def test_decoder_rejects_duplicate_keys_from_a_valid_pickle(self) -> None:
        """The receiver independently checks uniqueness instead of trusting the sender."""
        duplicate = SampleKey(1, 5)
        frame = _frame(((duplicate, "first"), (duplicate, "second")))

        with self.assertRaisesRegex(ValueError, "duplicate SampleKey"):
            _decode_payload_segment(frame)


class TestDataPlaneTransport(unittest.TestCase):
    """Verify control metadata and payload bytes use their designated groups."""

    @staticmethod
    def _two_rank_topology() -> DataTopology:
        return DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("dp",),
            rank_list=(0, 1),
            global_rank=0,
            dp_dim_names=("dp",),
        )

    def test_cpu_communication_device_reuses_gloo_control_group(self) -> None:
        """An explicit CPU device must not inherit an accelerator WORLD backend."""
        with (
                patch("hyper_parallel.distributed_data.transport.dist.is_available", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.is_initialized", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
                patch("hyper_parallel.distributed_data.transport.dist.get_rank", return_value=0),
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="hccl") as get_backend,
                patch("hyper_parallel.distributed_data.transport.dist.new_group", return_value="gloo") as new_group,
        ):
            groups = create_data_groups(
                self._two_rank_topology(),
                (0, 1),
                0,
                cpu_backend="gloo",
                payload_backend=None,
                communication_device="cpu",
                enable_payload_exchange=True,
            )

        self.assertEqual(groups.control_group, "gloo")
        self.assertIs(groups.payload_group, groups.control_group)
        self.assertEqual(new_group.call_count, 1)
        get_backend.assert_not_called()

    def test_control_plane_rejects_accelerator_backend_before_group_creation(self) -> None:
        """Object control collectives must use a CPU-capable backend."""
        with (
                patch("hyper_parallel.distributed_data.transport.dist.is_available", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.is_initialized", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
                patch("hyper_parallel.distributed_data.transport.dist.get_rank", return_value=0),
                patch("hyper_parallel.distributed_data.transport.dist.new_group") as new_group,
                self.assertRaisesRegex(ValueError, "cpu_backend must support CPU tensors"),
        ):
            create_data_groups(
                self._two_rank_topology(),
                (0, 1),
                0,
                cpu_backend="hccl",
                payload_backend="hccl",
                communication_device="npu:0",
                enable_payload_exchange=True,
            )

        new_group.assert_not_called()

    def test_payload_a2a_uses_payload_group_after_cpu_size_exchange(self) -> None:
        """Variable split sizes stay on control while framed bytes use the payload group."""
        groups = DataGroups(
            data_plane_ranks=(0, 1),
            control_group="control",
            payload_group="payload",
            model_parallel_group=None,
            planner_rank=0,
            distributed=True,
        )
        transport = DataPlaneTransport(groups, global_rank=0, communication_device="cpu")
        expected = {
            SampleKey(0, 0): {"sample": 0},
            SampleKey(0, 1): {"sample": 1},
        }
        prepared = transport.prepare_exchange({
            0: ((SampleKey(0, 0), expected[SampleKey(0, 0)]),),
            1: ((SampleKey(0, 1), expected[SampleKey(0, 1)]),),
        })
        collective_groups = []

        def fake_all_to_all(output: Any, input_tensor: Any, **kwargs: Any) -> None:
            """Copy local buffers while recording the selected process group."""
            collective_groups.append(kwargs["group"])
            output.copy_(input_tensor)

        with (
                patch.object(transport, "synchronize_error", return_value=None),
                patch(
                    "hyper_parallel.distributed_data.transport.dist.all_to_all_single",
                    side_effect=fake_all_to_all,
                ),
        ):
            received = transport.exchange_prepared(prepared)

        self.assertEqual(received, expected)
        self.assertEqual(collective_groups, ["control", "payload"])


if __name__ == "__main__":
    unittest.main()
