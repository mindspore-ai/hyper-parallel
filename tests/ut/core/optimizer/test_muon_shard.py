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

"""Asynchronous Muon gather reconstruction tests without accelerator hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.optimizer.muon_shard import launch_muon_gather, prepare_muon_gather

from tests.common.mark_utils import arg_mark


class TestMuonAsyncGather(unittest.TestCase):
    """Check that unpacking consumes completed communication and preserves uneven layouts."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cached_preparation_reads_current_inputs_and_clears_padding(self):
        """
        Feature: Muon optimizer correctness.
        Description: A reused send slot must contain this step's values, including zero padding.
        Expectation: Values and execution order match the reference.
        """
        buffers, plan = {}, {}
        stage = SimpleNamespace(split_sizes=(2, 1), pad_shape=(2, 2))
        with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", return_value=2):
            first = prepare_muon_gather([torch.ones(2, 2)], MagicMock(), 0, [stage], {0}, buffers, plan)
            address = first.send_buffer.data_ptr()
            first.send_buffer.fill_(99)
            second = prepare_muon_gather([torch.full((2, 2), 3.)], MagicMock(), 0, [stage], {0}, buffers, plan)
        self.assertEqual(second.send_buffer.data_ptr(), address)
        torch.testing.assert_close(second.send_buffer[:4], torch.full((4,), 3.))
        torch.testing.assert_close(second.send_buffer[4:], torch.zeros_like(second.send_buffer[4:]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cached_preparation_rebuilds_layout_for_new_shape(self):
        """
        Feature: Muon optimizer correctness.
        Description: Shape changes cannot reuse obsolete copy lengths or reconstruction dimensions.
        Expectation: Values and execution order match the reference.
        """
        buffers, plan = {}, {}
        with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", return_value=2):
            prepare_muon_gather([torch.ones(2, 2)], MagicMock(), 0,
                                [SimpleNamespace(split_sizes=(2, 2), pad_shape=(2, 2))], {0}, buffers, plan)
            second = prepare_muon_gather([torch.full((3, 2), 4.)], MagicMock(), 0,
                                        [SimpleNamespace(split_sizes=(3, 3), pad_shape=(3, 2))],
                                        {0}, buffers, plan)
        self.assertEqual(second.metadata[0].actual_numel, 6)
        self.assertEqual(second.metadata[0].split_sizes, (3, 3))
        torch.testing.assert_close(second.send_buffer[:6], torch.full((6,), 4.))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cached_preparation_tracks_remote_padding_changes(self):
        """
        Feature: Muon optimizer correctness.
        Description: A changed remote shard size invalidates offsets even if local shapes match.
        Expectation: Values and execution order match the reference.
        """
        buffers, plan = {}, {}
        inputs = [torch.ones(2, 2), torch.full((1, 2), 7.)]
        with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", return_value=2):
            first = prepare_muon_gather(inputs, MagicMock(), 0, [
                SimpleNamespace(split_sizes=(2, 2), pad_shape=(2, 2)),
                SimpleNamespace(split_sizes=(1, 1), pad_shape=(1, 2)),
            ], {0, 1}, buffers, plan)
            second = prepare_muon_gather(inputs, MagicMock(), 0, [
                SimpleNamespace(split_sizes=(2, 300), pad_shape=(300, 2)),
                SimpleNamespace(split_sizes=(1, 1), pad_shape=(1, 2)),
            ], {0, 1}, buffers, plan)
        self.assertGreater(second.metadata[1].offset, first.metadata[1].offset)
        self.assertEqual(second.metadata[0].split_sizes, (2, 300))
        offset = second.metadata[1].offset
        torch.testing.assert_close(second.send_buffer[offset:offset + 2], torch.full((2,), 7.))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_uneven_gather_waits_before_unpack(self):
        """
        Feature: Muon optimizer correctness.
        Description: Populate the output only on wait, exposing reads of unfinished communication.
        Expectation: Values and execution order match the reference.
        """
        for tensor_dim in (0, 1):
            with self.subTest(tensor_dim=tensor_dim):
                local = torch.tensor([[1., 2.], [3., 4.]])
                remote = torch.tensor([[5., 6.]])
                expected = torch.cat((local, remote), dim=0)
                if tensor_dim == 1:
                    local = local.T
                    expected = expected.T
                work = MagicMock()

                def _submit(output, source, *, remote_shard=remote, pending_work=work, **kwargs):
                    self.assertTrue(kwargs["async_op"])

                    def _complete():
                        output[:source.numel()].copy_(source)
                        output[source.numel():].zero_()
                        output[source.numel():source.numel() + 2].copy_(remote_shard.flatten())
                    pending_work.wait.side_effect = _complete
                    return pending_work

                with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", return_value=2), \
                        patch("hyper_parallel.core.optimizer.muon_shard.dist.all_gather_into_tensor",
                              side_effect=_submit):
                    pending = launch_muon_gather(
                        [local], MagicMock(), tensor_dim,
                        [SimpleNamespace(split_sizes=(2, 1), pad_shape=(2, 2))], {0}, {},
                    )
                    work.wait.assert_not_called()
                    actual = pending.finish()[0]
                work.wait.assert_called_once()
                torch.testing.assert_close(actual, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_preparation_packs_inputs_without_submitting_collectives(self):
        """
        Feature: Muon optimizer correctness.
        Description: The lookahead launch must not repeat per-parameter copies or momentum work.
        Expectation: Values and execution order match the reference.
        """
        local = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", return_value=2), \
                patch("hyper_parallel.core.optimizer.muon_shard.dist.all_gather_into_tensor") as gather:
            prepared = prepare_muon_gather(
                [local], MagicMock(), 0,
                [SimpleNamespace(split_sizes=(2, 2), pad_shape=(2, 2))], {0},
            )
            gather.assert_not_called()
            local.zero_()
            prepared.launch(MagicMock(), {})
        gather.assert_called_once()
        torch.testing.assert_close(prepared.send_buffer[:4], torch.arange(4, dtype=torch.float32))
