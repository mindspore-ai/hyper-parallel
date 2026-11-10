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
"""Tests for internal payload serialization and sample conservation."""

import pickle
import unittest
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data.api import DistributedDatasetConfig
from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.model_batch import TensorFieldSpec
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    DataPlaneTransport,
    ModelParallelTransport,
    _decode_control,
    create_data_groups,
    _decode_payload_segment,
    _decode_received_payloads,
    _encode_payload_segment,
)


class TestPayloadCodec(unittest.TestCase):
    """Verify route round trips and conservation checks at payload merge."""

    def test_round_trip_preserves_order_keys_and_payloads(self) -> None:
        """Feature: Payload serialization.
        Description: Encode and decode an ordered sample route.
        Expectation: Keys, values, order, and empty routes survive the round trip.
        """
        items = (
            (SampleKey(0, 7), {"tokens": [1, 2], "image": b"jpeg"}),
            (SampleKey(4, 3), ("caption", 9)),
        )

        encoded = _encode_payload_segment(items)

        self.assertEqual(_decode_payload_segment(encoded), items)
        self.assertEqual(_encode_payload_segment(()), b"")
        self.assertEqual(_decode_payload_segment(b""), ())

    def test_tensor_payload_uses_metadata_and_binary_buffer(self) -> None:
        """Feature: Tensor-buffer payload serialization.
        Description: Encode nested tensor and byte leaves without pickling data.
        Expectation: The route round-trips and tensors retain dtype and values.
        """
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        items = ((SampleKey(0, 8), {
            "tokens": base[:, ::2],
            "nested": [torch.tensor([True, False]), b"raw-image", bytearray(b"mutable-image")],
            "scalar": torch.tensor(17, dtype=torch.int64),
            "shape": (3, 2),
        }),)

        encoded = _encode_payload_segment(items)
        decoded = _decode_payload_segment(encoded)

        self.assertTrue(encoded.startswith(b"HPB1"))
        self.assertEqual(decoded[0][0], items[0][0])
        torch.testing.assert_close(decoded[0][1]["tokens"], items[0][1]["tokens"])
        torch.testing.assert_close(decoded[0][1]["nested"][0], items[0][1]["nested"][0])
        torch.testing.assert_close(decoded[0][1]["scalar"], items[0][1]["scalar"])
        self.assertEqual(decoded[0][1]["nested"][1], b"raw-image")
        self.assertIsInstance(decoded[0][1]["nested"][1], bytes)
        self.assertEqual(decoded[0][1]["nested"][2], bytearray(b"mutable-image"))
        self.assertIsInstance(decoded[0][1]["nested"][2], bytearray)
        self.assertIsInstance(decoded[0][1]["nested"], list)
        self.assertEqual(decoded[0][1]["shape"], (3, 2))
        self.assertIsInstance(decoded[0][1]["shape"], tuple)

    def test_wire_format_is_plain_pickle(self) -> None:
        """Feature: Payload serialization.
        Description: Encode a route with the transport codec.
        Expectation: The wire bytes equal one highest-protocol pickle payload.
        """
        items = ((SampleKey(0, 1), {"value": 3}),)

        self.assertEqual(_encode_payload_segment(items), pickle.dumps(items, protocol=pickle.HIGHEST_PROTOCOL))

    def test_decoder_propagates_pickle_errors(self) -> None:
        """Feature: Payload serialization.
        Description: Decode malformed payload bytes.
        Expectation: The original pickle error reaches the caller.
        """
        with self.assertRaises(pickle.UnpicklingError):
            _decode_payload_segment(b"invalid pickle")

    def test_receive_rejects_duplicates_within_and_across_routes(self) -> None:
        """Feature: Payload conservation.
        Description: Merge routes containing duplicate sample keys.
        Expectation: Duplicates fail instead of silently overwriting payloads.
        """
        key = SampleKey(1, 5)
        for segments in (
                (_encode_payload_segment(((key, "first"), (key, "second"))),),
                (_encode_payload_segment(((key, "first"),)), _encode_payload_segment(((key, "second"),))),
        ):
            with self.subTest(segment_count=len(segments)), self.assertRaisesRegex(ValueError, "duplicate payload"):
                _decode_received_payloads(b"".join(segments), tuple(map(len, segments)))

    def test_receive_preserves_empty_routes_and_repeated_dataset_indices(self) -> None:
        """Feature: Payload conservation.
        Description: Merge empty routes and distinct occurrences of one index.
        Expectation: Empty routes are ignored and occurrence keys remain distinct.
        """
        items = ((SampleKey(0, 3, 0), "first"), (SampleKey(0, 3, 1), "second"))
        segments = (b"", _encode_payload_segment(items[:1]), b"", _encode_payload_segment(items[1:]))

        result = _decode_received_payloads(b"".join(segments), tuple(map(len, segments)))

        self.assertEqual(result, dict(items))


class TestDataPlaneTransport(unittest.TestCase):
    """Verify control metadata and payload bytes use their designated groups."""

    def test_control_gather_defers_decode_and_retains_variable_length_buffers(self) -> None:
        """Feature: Deferred control gather completion.
        Description: Gather differently sized objects using a mocked accelerator backend and delayed data work.
        Expectation: Launch neither waits nor decodes the result, and repeated waits return the same objects.
        """
        values = (None, {"metadata": list(range(40))})
        payloads = [pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) for value in values]
        transport = DataPlaneTransport(DataGroups((0, 1), "control", None, None, 0, True), 0)
        work = Mock()

        def gather(outputs: list[torch.Tensor], source: torch.Tensor, **kwargs: Any) -> Any:
            """Populate lengths immediately and bytes only when the consumer waits."""
            self.assertEqual(kwargs["group"], "control")
            if source.dtype == torch.int64:
                for output, payload in zip(outputs, payloads):
                    output.fill_(len(payload))
                self.assertFalse(kwargs.get("async_op", False))
                return None
            self.assertTrue(kwargs["async_op"])

            def complete() -> None:
                """Populate received bytes only after the consumer waits."""
                for output, payload in zip(outputs, payloads):
                    output.zero_()
                    output[:len(payload)].copy_(torch.tensor(list(payload), dtype=torch.uint8))

            work.wait.side_effect = complete
            return work

        with (
            patch("hyper_parallel.distributed_data.transport._control_backend", return_value="hccl"),
            patch(
                "hyper_parallel.distributed_data.transport._require_control_device", return_value=torch.device("cpu"),
            ),
            patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
            patch("hyper_parallel.distributed_data.transport.dist.all_gather", side_effect=gather),
            patch("hyper_parallel.distributed_data.transport._decode_control", wraps=_decode_control) as decode,
        ):
            pending = transport.begin_all_gather_object(values[0])
            work.wait.assert_not_called()
            decode.assert_not_called()
            self.assertEqual(pending.sizes, tuple(map(len, payloads)))
            self.assertEqual(pending.encoded.numel(), max(map(len, payloads)))
            self.assertEqual(pending.wait(), values)
            self.assertEqual(pending.wait(), values)
            work.wait.assert_called_once()
            self.assertEqual(decode.call_count, 2)

    def test_default_communication_backend_is_hccl(self) -> None:
        """Feature: HCCL data-plane default.
        Description: Construct a dataset config without selecting a backend.
        Expectation: NPU training uses HCCL unless Gloo is explicitly requested.
        """
        config = DistributedDatasetConfig(seq_len=8, local_batch_size=1)

        self.assertEqual(config.communication_backend, "hccl")
        self.assertEqual(
            DistributedDatasetConfig(seq_len=8, local_batch_size=1, communication_backend="gloo").communication_backend,
            "gloo",
        )

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
        """Feature: Gloo data groups.
        Description: Build data groups with an explicit CPU device.
        Expectation: The control group is reused for payload exchange.
        """
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

    def test_hccl_control_and_payload_groups_accept_npu_device(self) -> None:
        """Feature: HCCL data groups.
        Description: Build control and payload groups with an NPU device.
        Expectation: Both groups use HCCL-compatible accelerator communication.
        """
        with (
                patch("hyper_parallel.distributed_data.transport.dist.is_available", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.is_initialized", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
                patch("hyper_parallel.distributed_data.transport.dist.get_rank", return_value=0),
                patch("hyper_parallel.distributed_data.transport.dist.new_group", side_effect=["control", "payload"])
                as new_group,
        ):
            groups = create_data_groups(
                self._two_rank_topology(),
                (0, 1),
                0,
                cpu_backend="hccl",
                payload_backend="hccl",
                communication_device="npu:0",
                enable_payload_exchange=True,
            )

        self.assertEqual(groups.control_group, "control")
        self.assertEqual(groups.payload_group, "payload")
        self.assertEqual(new_group.call_count, 2)

    def test_payload_a2a_uses_payload_group_after_cpu_size_exchange(self) -> None:
        """Feature: Payload all-to-all.
        Description: Exchange split sizes and serialized payload bytes.
        Expectation: Size exchange uses control and bytes use the payload group.
        """
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

        with patch(
                "hyper_parallel.distributed_data.transport.dist.all_to_all_single",
                side_effect=fake_all_to_all,
        ):
            received = transport.exchange_prepared(prepared)

        self.assertEqual(received, expected)
        self.assertEqual(collective_groups, ["control", "payload"])

    def test_async_exchange_defers_wait_and_preserves_send_storage(self) -> None:
        """Feature: Asynchronous payload exchange.
        Description: Launch a bulk exchange with a deferred fake Work handle.
        Expectation: Decoding waits once and the prepared send state stays live.
        """
        transport = DataPlaneTransport(DataGroups((0, 1), "control", "payload", None, 0, True), 0)
        key = SampleKey(0, 7)
        prepared = transport.prepare_exchange({1: [(key, {"value": 19})]})
        work = Mock()

        def exchange(output: torch.Tensor, source: torch.Tensor, **kwargs: Any) -> Any:
            """Copy fake control data and complete fake payload data on wait."""
            if kwargs["group"] == "control":
                output.copy_(source)
                return None
            work.wait.side_effect = lambda: output.copy_(source)
            return work

        with patch("hyper_parallel.distributed_data.transport.dist.all_to_all_single", side_effect=exchange):
            pending = transport.begin_exchange_prepared(prepared)
            work.wait.assert_not_called()
            self.assertIs(pending.prepared, prepared)
            self.assertEqual(pending.wait(), {key: {"value": 19}})
            self.assertEqual(pending.wait(), {key: {"value": 19}})
            work.wait.assert_called_once()

    def test_singleton_exchange_still_rejects_duplicate_occurrences(self) -> None:
        """Feature: Singleton payload exchange.
        Description: Decode duplicate keys without a distributed collective.
        Expectation: The local fast path enforces the same duplicate check.
        """
        transport = DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), global_rank=0)
        key = SampleKey(0, 1)
        prepared = transport.prepare_exchange({0: ((key, "first"), (key, "second"))})

        with self.assertRaisesRegex(ValueError, "duplicate payload"):
            transport.exchange_prepared(prepared)

    def test_unknown_target_is_not_silently_dropped(self) -> None:
        """Feature: Payload routing.
        Description: Prepare a route to a rank outside the data plane.
        Expectation: Invalid targets fail before collective entry.
        """
        transport = DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), global_rank=0)

        with self.assertRaisesRegex(ValueError, "outside the data plane"):
            transport.prepare_exchange({1: ((SampleKey(0, 1), "sample"),)})

    def test_missing_control_group_fails_at_construction(self) -> None:
        """Feature: Transport validation.
        Description: Construct a multi-rank transport without a process group.
        Expectation: Members fail early while non-members remain valid.
        """
        groups = DataGroups((0, 1), None, None, None, 0, False)

        with self.assertRaisesRegex(ValueError, "initialized process group"):
            DataPlaneTransport(groups, global_rank=0)
        self.assertFalse(DataPlaneTransport(groups, global_rank=2).is_member)


class TestModelParallelTransport(unittest.TestCase):
    """Verify declared layouts, dynamic dimensions and EOF use tensor collectives."""

    @staticmethod
    def _transport(rank: int, spec: Any) -> ModelParallelTransport:
        topology = DataTopology.from_layout(
            mesh_shape=(1, 2), mesh_dim_names=("dp", "mp"), rank_list=(0, 1),
            global_rank=rank, dp_dim_names=("dp",),
        )
        return ModelParallelTransport(topology, DataGroups((0,), None, None, "model", 0, True),
                                      model_batch_spec=spec)

    def test_dynamic_tensor_batches_round_trip_without_object_serialization(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: Replay changing lengths, empty tensors, nested fields and EOF on a peer.
        Expectation: Tensor values and nested fields round-trip without object serialization.
        """
        spec = {
            "input_ids": TensorFieldSpec((1, 4), torch.int64),
            "nested": [TensorFieldSpec((None,), torch.int32), "causal", None],
        }
        sender, receiver = self._transport(0, spec), self._transport(1, spec)
        for offsets in ([0, 1, 4], [0, 4], []):
            with self.subTest(offsets=offsets):
                batch = {"nested": [torch.tensor(offsets, dtype=torch.int32), "causal", None],
                         "input_ids": torch.arange(4).reshape(1, 4)}
                wire = []
                with patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"), \
                        patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list") as objects, \
                        patch("hyper_parallel.distributed_data.transport.broadcast_control_object") as serialized, \
                        patch("hyper_parallel.distributed_data.transport.pickle.dumps") as pickle_dump, \
                        patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast:
                    broadcast.side_effect = lambda tensor, wire=wire, **_: wire.append(tensor.clone())
                    sender.broadcast(batch)
                    self.assertEqual(len(wire), 3 if offsets else 2)
                    self.assertEqual(wire[0].tolist(), [1, len(offsets)])
                    pending = iter(wire)
                    broadcast.side_effect = lambda tensor, pending=pending, **_: tensor.copy_(next(pending))
                    received = receiver.broadcast(None)
                    objects.assert_not_called()
                    serialized.assert_not_called()
                    pickle_dump.assert_not_called()
                torch.testing.assert_close(received["input_ids"], batch["input_ids"])
                torch.testing.assert_close(received["nested"][0], batch["nested"][0])
                self.assertEqual(received["nested"][1:], ["causal", None])

    def test_eof_broadcast_is_one_small_tensor(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: EOF reaches peers without any payload or object collective.
        Expectation: One status tensor reports EOF without entering payload collectives.
        """
        spec = {"x": TensorFieldSpec((4,), torch.float32)}
        wire = []
        with patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"), \
                patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast:
            broadcast.side_effect = lambda tensor, **_: wire.append(tensor.clone())
            self.assertIsNone(self._transport(0, spec).broadcast(None))
            self.assertEqual(wire[0].tolist(), [0])
            broadcast.side_effect = lambda tensor, **_: tensor.copy_(wire[0])
            self.assertIsNone(self._transport(1, spec).broadcast(None))
            self.assertEqual(broadcast.call_count, 2)

    def test_invalid_source_notifies_peer_before_payload_collectives(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: Changed keys, dtype, fixed shapes and static values fail on all peers.
        Expectation: Source and receiver both fail after exchanging only the error header.
        """
        spec = {"x": TensorFieldSpec((2,), torch.int64), "mode": "causal"}
        batches = [
            {"y": torch.ones(2, dtype=torch.int64), "mode": "causal"},
            {"x": torch.ones(2), "mode": "causal"},
            {"x": torch.ones(3, dtype=torch.int64), "mode": "causal"},
            {"x": torch.ones(2, dtype=torch.int64), "mode": "other"},
        ]
        for batch in batches:
            with self.subTest(batch=batch):
                wire = []
                with patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"), \
                        patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast:
                    broadcast.side_effect = lambda tensor, wire=wire, **_: wire.append(tensor.clone())
                    with self.assertRaises(ValueError):
                        self._transport(0, spec).broadcast(batch)
                    self.assertEqual(wire[0].tolist(), [-1])
                    broadcast.side_effect = lambda tensor, wire=wire, **_: tensor.copy_(wire[0])
                    with self.assertRaisesRegex(RuntimeError, "Constructor failed"):
                        self._transport(1, spec).broadcast(None)
                    self.assertEqual(broadcast.call_count, 2)

    def test_model_parallel_requires_an_explicit_layout(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: Unknown Python batch structures cannot silently restore object broadcasts.
        Expectation: Missing layout declarations raise a descriptive ValueError.
        """
        with self.assertRaisesRegex(ValueError, "model_batch_spec"):
            self._transport(0, None)

    def test_pure_dp_needs_no_layout_or_collectives(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: Pure-DP callers retain their existing arbitrary batch contract.
        Expectation: Pure DP returns the original batch without communication.
        """
        topology = DataTopology.from_layout((1,), ("dp",), (0,), 0, dp_dim_names=("dp",))
        transport = ModelParallelTransport(topology, DataGroups((0,), None, None, None, 0, False))
        batch = {"caption": "dynamic value"}
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast:
            self.assertIs(transport.broadcast(batch), batch)
            broadcast.assert_not_called()

    def test_invalid_tensor_specs_are_rejected_locally(self) -> None:
        """Feature: Tensor-only model broadcast.
        Description: Reject ambiguous dimensions, unsupported dtype values and device policies.
        Expectation: Invalid dimensions, dtypes and device policies raise ValueError.
        """
        for shape, dtype, device in [((-1,), torch.int64, "device"), ((True,), torch.int64, "device"),
                                     ((1,), "int64", "device"), ((1,), torch.int64, "cuda:0")]:
            with self.subTest(shape=shape, dtype=dtype, device=device), self.assertRaises(ValueError):
                TensorFieldSpec(shape, dtype, device)


if __name__ == "__main__":
    unittest.main()
