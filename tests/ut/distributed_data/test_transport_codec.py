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
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    DataPlaneTransport,
    ModelParallelTransport,
    _decode_model_batch,
    _encode_model_batch,
    create_data_groups,
    _decode_payload_segment,
    _decode_received_payloads,
    _encode_payload_segment,
)
from tests.common.mark_utils import arg_mark


class TestPayloadCodec(unittest.TestCase):
    """Verify route round trips and conservation checks at payload merge."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_tensor_payload_uses_metadata_and_binary_buffer(self) -> None:
        """Feature: Tensor-buffer payload serialization.
        Description: Encode nested tensor and byte leaves without pickling data.
        Expectation: The route round-trips and tensors retain dtype and values.
        """
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        items = ((SampleKey(0, 8), {
            "tokens": base[:, ::2],
            "nested": [torch.tensor([True, False]), b"raw-image"],
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
        self.assertEqual(decoded[0][1]["shape"], (3, 2))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_wire_format_is_plain_pickle(self) -> None:
        """Feature: Payload serialization.
        Description: Encode a route with the transport codec.
        Expectation: The wire bytes equal one highest-protocol pickle payload.
        """
        items = ((SampleKey(0, 1), {"value": 3}),)

        self.assertEqual(_encode_payload_segment(items), pickle.dumps(items, protocol=pickle.HIGHEST_PROTOCOL))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_decoder_propagates_pickle_errors(self) -> None:
        """Feature: Payload serialization.
        Description: Decode malformed payload bytes.
        Expectation: The original pickle error reaches the caller.
        """
        with self.assertRaises(pickle.UnpicklingError):
            _decode_payload_segment(b"invalid pickle")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_unknown_target_is_not_silently_dropped(self) -> None:
        """Feature: Payload routing.
        Description: Prepare a route to a rank outside the data plane.
        Expectation: Invalid targets fail before collective entry.
        """
        transport = DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), global_rank=0)

        with self.assertRaisesRegex(ValueError, "outside the data plane"):
            transport.prepare_exchange({1: ((SampleKey(0, 1), "sample"),)})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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
    """Verify tensor leaves use direct collectives while metadata stays serialized."""

    @staticmethod
    def _topology() -> DataTopology:
        return DataTopology.from_layout(
            mesh_shape=(1, 2), mesh_dim_names=("dp", "mp"), rank_list=(0, 1),
            global_rank=0, dp_dim_names=("dp",),
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_codec_preserves_nested_structure_and_tensor_values(self) -> None:
        """Feature: Model batch codec.
        Description: Encode and decode a nested batch containing tensors.
        Expectation: Container structure and tensor values are restored.
        """
        batch = {"input_ids": torch.tensor([[1, 2]]), "meta": ["caption", None]}

        schema, tensors = _encode_model_batch(batch)
        restored = _decode_model_batch(schema, [tensor.clone() for tensor in tensors])

        self.assertEqual(restored["meta"], ["caption", None])
        self.assertTrue(torch.equal(restored["input_ids"], batch["input_ids"]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_broadcast_sends_tensor_leaves_directly(self) -> None:
        """Feature: Gloo model broadcast.
        Description: Broadcast a model batch with tensor leaves.
        Expectation: The schema uses object broadcast and tensors use direct broadcast.
        """
        groups = DataGroups(
            data_plane_ranks=(0, 1), control_group=None, payload_group=None,
            model_parallel_group="model", planner_rank=0, distributed=True,
        )
        transport = ModelParallelTransport(self._topology(), groups)
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list") as object_broadcast, \
                patch("hyper_parallel.distributed_data.transport.dist.broadcast") as tensor_broadcast, \
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"):
            transport.broadcast({"input_ids": torch.tensor([1, 2]), "label": "text"})

        object_broadcast.assert_called_once()
        tensor_broadcast.assert_called_once()
        self.assertEqual(object_broadcast.call_args.kwargs["group"], "model")
        self.assertEqual(tensor_broadcast.call_args.kwargs["group"], "model")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_hccl_broadcast_encodes_schema_as_device_tensor(self) -> None:
        """Feature: HCCL model broadcast.
        Description: Broadcast a Python schema through an HCCL model group.
        Expectation: The schema uses tensor encoding and avoids object collectives.
        """
        groups = DataGroups(
            data_plane_ranks=(0, 1), control_group=None, payload_group=None,
            model_parallel_group="model", planner_rank=0, distributed=True,
        )
        transport = ModelParallelTransport(self._topology(), groups, communication_device="npu:0")
        batch = {"label": "text"}
        schema, _ = _encode_model_batch(batch)
        with (
                patch("hyper_parallel.distributed_data.transport._control_backend", return_value="hccl"),
                patch("hyper_parallel.distributed_data.transport.broadcast_control_object", return_value=schema)
                as control_broadcast,
                patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list") as object_broadcast,
        ):
            self.assertEqual(transport.broadcast(batch), batch)

        control_broadcast.assert_called_once_with(
            schema,
            src=0,
            group="model",
            device=torch.device("npu:0"),
        )
        object_broadcast.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_receiver_rebuilds_nested_batch_and_eof(self) -> None:
        """Feature: Model batch broadcast.
        Description: Receive nested tensors and an end-of-stream value.
        Expectation: The receiver rebuilds the batch and preserves the sentinel.
        """
        topology = DataTopology.from_layout(
            mesh_shape=(1, 2), mesh_dim_names=("dp", "mp"), rank_list=(0, 1),
            global_rank=1, dp_dim_names=("dp",),
        )
        groups = DataGroups((0,), None, None, "model", 0, True)
        transport = ModelParallelTransport(topology, groups)
        batch = {"inputs": (torch.tensor([1, 2]), [torch.tensor([[3.0]]), "text"])}
        schema, tensors = _encode_model_batch(batch)
        pending = iter(tensors)

        def broadcast_schema(payload: list, **_kwargs: Any) -> None:
            """Supply the constructor's schema to this receiver."""
            payload[0] = schema

        def broadcast_tensor(tensor: torch.Tensor, **_kwargs: Any) -> None:
            """Populate tensor leaves in encoder order."""
            tensor.copy_(next(pending))

        with (
                patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list",
                      side_effect=broadcast_schema),
                patch("hyper_parallel.distributed_data.transport.dist.broadcast",
                      side_effect=broadcast_tensor) as broadcast,
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"),
        ):
            received = transport.broadcast(None)
            self.assertEqual(broadcast.call_count, 2)
            schema, _ = _encode_model_batch(None)
            self.assertIsNone(transport.broadcast(None))
            self.assertEqual(broadcast.call_count, 2)

        self.assertTrue(torch.equal(received["inputs"][0], batch["inputs"][0]))
        self.assertTrue(torch.equal(received["inputs"][1][0], batch["inputs"][1][0]))
        self.assertEqual(received["inputs"][1][1], "text")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_accelerator_broadcast_preserves_cpu_only_fields(self) -> None:
        """Feature: Model broadcast field placement.
        Description: Stage CPU-only fields for an accelerator collective.
        Expectation: The source batch retains CPU placement after transport.
        """
        groups = DataGroups((0,), None, None, "model", 0, True)
        transport = ModelParallelTransport(self._topology(), groups, communication_device="cuda:0")
        schema, _ = _encode_model_batch({"input_ids": torch.tensor([1, 2])})
        staged = Mock()
        staged.contiguous.return_value = staged
        with (
                patch("hyper_parallel.distributed_data.transport.broadcast_control_object", return_value=schema),
                patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast,
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="nccl"),
                patch.object(torch.Tensor, "to", return_value=staged) as move,
        ):
            batch = {"input_ids": torch.tensor([1, 2])}
            self.assertIs(transport.broadcast(batch), batch)

        move.assert_called_once_with(torch.device("cuda:0"))
        broadcast.assert_called_once_with(staged, src=0, group="model")
        staged.cpu.assert_not_called()
        self.assertEqual(batch["input_ids"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
