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
"""E0 ownership, LPT, and decoder SP layout contracts."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.exchange import FeatureExchangeMode, select_feature_exchange
from hyper_parallel.distributed.encoder_parallel.mesh import decoder_coordinates, encoder_coordinates
from hyper_parallel.distributed.encoder_parallel.planning import (
    DecoderSequenceLayout,
    assign_lpt,
    build_destinations,
    build_segments,
    required_omni_global_alignment,
)
from tests.common.mark_utils import arg_mark


def _e0_items():
    return (
        EncoderWorkItem(0, 0, 576, 64, 96, 74, 0, 576, {}),
        EncoderWorkItem(1, 0, 256, 36, 32, 44, 576, 832, {}),
        EncoderWorkItem(2, 0, 144, 16, 200, 22, 832, 976, {}),
    )


def _mesh(sequence_parallel=True):
    return SimpleNamespace(cp_size=2, tp_size=4, cp_rank=0, tp_rank=0,
                           sequence_parallel=sequence_parallel)


class TestEncoderPlanning(unittest.TestCase):
    """Check E0's exact image and token ownership without process groups."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_e0_lpt_and_route(self):
        """Feature: E0 work assignment and SP destination planning.

        Description: Assign A/B/C and derive all eight decoder token intervals.
        Expectation: Buckets and destination ranks match the E0 route table.
        """
        items = _e0_items()
        buckets = assign_lpt(items, 4, 2)
        self.assertEqual(tuple(tuple(item.item_id for item in bucket) for bucket in buckets),
                         ((0,), (1,), (2,), ()))
        layouts = {rank: DecoderSequenceLayout.build(mesh_context=_mesh(), cp_local_length=128,
                                                     cp_rank=rank // 4, tp_rank=rank % 4)
                   for rank in range(8)}
        self.assertEqual([(value.global_start, value.global_stop) for value in layouts.values()],
                         [(rank * 32, (rank + 1) * 32) for rank in range(8)])
        self.assertEqual(build_destinations(items, layouts), {0: (3, 4, 5), 1: (1, 2), 2: (6,)})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nosp_fanout_and_exchange_modes(self):
        """Feature: decoder routing without sequence parallelism.

        Description: Derive CP2/NOSP owners and CP1 feature exchange mode.
        Expectation: Each CP owner fans out to TP ranks and CP1 uses gather.
        """
        layouts = {rank: DecoderSequenceLayout.build(mesh_context=_mesh(False), cp_local_length=128,
                                                     cp_rank=rank // 4, tp_rank=rank % 4)
                   for rank in range(8)}
        self.assertEqual(build_destinations(_e0_items(), layouts),
                         {0: tuple(range(8)), 1: tuple(range(4)), 2: tuple(range(4, 8))})
        self.assertIs(select_feature_exchange(_mesh()), FeatureExchangeMode.PARENT_VARIABLE_A2A)
        self.assertIs(select_feature_exchange(SimpleNamespace(cp_size=1, tp_size=4,
                                                              sequence_parallel=False)),
                      FeatureExchangeMode.TP_VARIABLE_ALL_GATHER)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_layout_validation_and_coordinates(self):
        """Feature: decoder SP token layout validation.

        Description: Resolve E0 coordinates and validate local hidden geometry.
        Expectation: Valid views slice correctly and invalid lengths fail early.
        """
        self.assertEqual(required_omni_global_alignment(_mesh()), 8)
        self.assertEqual([encoder_coordinates(rank, 2) for rank in range(8)],
                         [(rank // 2, rank % 2) for rank in range(8)])
        self.assertEqual([decoder_coordinates(rank, 4) for rank in range(8)],
                         [(rank // 4, rank % 4) for rank in range(8)])
        with self.assertRaisesRegex(ValueError, "HP-ENCODER-004"):
            DecoderSequenceLayout.build(mesh_context=_mesh(), cp_local_length=130)
        layout = DecoderSequenceLayout.build(mesh_context=_mesh(), cp_local_length=128, tp_rank=3)
        self.assertTrue(torch.equal(layout.localize(torch.arange(128).unsqueeze(0)),
                                    torch.arange(96, 128).unsqueeze(0)))
        with self.assertRaisesRegex(ValueError, "HP-ENCODER-004"):
            layout.validate_hidden(torch.zeros(1, 31, 8))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_segments_and_invalid_spans(self):
        """Feature: padded image segments and decoder span validation.

        Description: Split five patches over ECP2 and plan an invalid image span.
        Expectation: Segments pad to six rows and out-of-domain spans fail.
        """
        item = EncoderWorkItem(0, 0, 5, 2, 0, 4, 0, 5, {})
        segments = build_segments((item,), 1, 2)
        self.assertEqual((segments[0].local_start, segments[0].local_valid_length,
                          segments[0].local_padded_length),
                         (3, 2, 3))
        self.assertTrue(build_segments((), 0, 2)[0].is_dummy)
        layouts = {0: DecoderSequenceLayout.build(mesh_context=_mesh(), cp_local_length=128)}
        with self.assertRaisesRegex(ValueError, "HP-ENCODER-016"):
            build_destinations((EncoderWorkItem(1, 0, 2, 1, 255, 3, 0, 2, {}),), layouts)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_config_rejects_invalid_cp(self):
        """Feature: Encoder-CP configuration validation.

        Description: Reject invalid degrees and removed single-choice settings.
        Expectation: Only enabled and encoder_cp_size remain public inputs.
        """
        for size in (0, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "HP-ENCODER-002"):
                EncoderParallelConfig(encoder_cp_size=size)
        for name, value in (("mode", "sequence_sharded"),
                            ("balance_by", "attention_flops"),
                            ("validate_metadata", True)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unknown encoder_parallel keys"):
                EncoderParallelConfig.from_mapping({name: value})
