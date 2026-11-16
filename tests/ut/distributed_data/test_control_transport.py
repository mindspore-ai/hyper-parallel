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
"""Control-object gathering preserves rank order with bounded host transfers."""

import pickle
import unittest
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.distributed_data import transport
from hyper_parallel.distributed_data.schema import SampleKey, SampleMetadata
from tests.common.mark_utils import arg_mark


class TestControlGather(unittest.TestCase):
    """Exercise accelerator control encoding using CPU tensors and fake collectives."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_accelerator_gather_preserves_objects_with_two_host_transfers(self) -> None:
        """Feature: Contiguous control gathering.
        Description: Gather unequal payloads across different group sizes and local ranks.
        Expectation: Objects retain rank order with two bulk host copies and no scalar reads.
        """
        values = (
            None,
            {"text": "多模态", "bytes": b"\x00\xff", "nested": [1, (True, None)]},
            (SampleKey(7, 11, 3), SampleMetadata(17, features={"P": 17, "D": 0})),
            b"x" * 4097,
            (),
            [1, 2, 3],
            2 ** 65,
            {"empty": {}, "float": 1.23456789012345},
        )
        for backend in ("hccl", "nccl"):
            for count in (1, 2, len(values)):
                expected = values[:count]
                for local_rank in sorted({0, count - 1}):
                    with self.subTest(backend=backend, count=count, local_rank=local_rank):
                        self._check_accelerator_gather(expected, local_rank, backend)

    def _check_accelerator_gather(self, expected: tuple, local_rank: int, backend: str) -> None:
        payloads = tuple(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) for value in expected)
        lengths = tuple(map(len, payloads))
        transfers = []
        tensor_cpu = torch.Tensor.cpu
        group = object()

        def gather(output: torch.Tensor, source: torch.Tensor, **kwargs: Any) -> None:
            """Supply each rank's wire bytes without a real process group."""
            self.assertIs(kwargs["group"], group)
            if source.dtype == torch.int64:
                self.assertEqual(source.tolist(), [lengths[local_rank]])
                output.copy_(torch.tensor(lengths, dtype=torch.int64))
                return
            self.assertEqual(source.numel(), max(lengths))
            self.assertEqual(source[:lengths[local_rank]].numpy().tobytes(), payloads[local_rank])
            output.fill_(255)
            rows = output.view(len(expected), -1)
            for row, payload in zip(rows, payloads):
                row[:len(payload)].copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))

        def copy_to_cpu(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            """Count host-copy boundaries while keeping test tensors on CPU."""
            transfers.append((tensor.dtype, tensor.numel()))
            return tensor_cpu(tensor, *args, **kwargs)

        with (
                patch.object(transport, "_require_control_device", return_value=torch.device("cpu")),
                patch.object(transport.dist, "get_world_size", return_value=len(expected)),
                patch.object(transport.dist, "all_gather_into_tensor", side_effect=gather) as collective,
                patch.object(transport.dist, "all_gather", side_effect=AssertionError("Unexpected list gather")),
                patch.object(torch.Tensor, "cpu", new=copy_to_cpu),
                patch.object(torch.Tensor, "item", side_effect=AssertionError("Unexpected scalar host read")),
        ):
            received = transport.all_gather_control_object(
                expected[local_rank], group=group, device="cpu", backend=backend,
            )

        self.assertEqual(received, expected)
        self.assertEqual(collective.call_count, 2)
        self.assertEqual(transfers, [(torch.int64, len(expected)), (torch.uint8, len(expected) * max(lengths))])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_gloo_keeps_cpu_object_gather(self) -> None:
        """Feature: Gloo control compatibility.
        Description: Gather CPU objects without selecting an accelerator device.
        Expectation: The existing object collective and returned rank order are preserved.
        """
        expected = ({"rank": 0}, None, ("third", 3))
        group = object()

        def gather(output: list, value: Any, **kwargs: Any) -> None:
            """Supply object results in process-group rank order."""
            self.assertIs(kwargs["group"], group)
            self.assertEqual(value, expected[0])
            output[:] = expected

        with (
                patch.object(transport.dist, "get_world_size", return_value=len(expected)),
                patch.object(transport.dist, "all_gather_object", side_effect=gather) as collective,
                patch.object(transport.dist, "all_gather_into_tensor") as tensor_collective,
                patch.object(transport, "_require_control_device") as require_device,
        ):
            received = transport.all_gather_control_object(expected[0], group=group, backend="gloo")

        self.assertEqual(received, expected)
        collective.assert_called_once()
        tensor_collective.assert_not_called()
        require_device.assert_not_called()
