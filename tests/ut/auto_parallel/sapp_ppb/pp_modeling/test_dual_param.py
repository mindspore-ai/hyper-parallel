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
"""Tests for dual parameter in SappSolver._reserved_stage_positions.

The ``dual`` flag controls DualPipe-V scheduling: when ``True``,
HEAD and TAIL share stage 0 instead of the default placement where
HEAD is at ``(0, 0)`` and TAIL at ``(vpp-1, pp-1)``.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/pp_modeling/test_dual_param.py
"""

import pytest


class TestDualReservedStagePositions:
    """Verify dual parameter changes _reserved_stage_positions in SappSolver."""

    @pytest.fixture()
    def make_solver(self):
        """Fixture that returns a factory for creating SappSolver instances.

        The factory accepts the same keyword arguments as SappSolver and
        pre-populates ``layers_sorted`` with a single BODY layer.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415

        body = Layer("unit", "body", Layer.type_enum.BODY, 4, time=2)

        def factory(**kwargs):
            layers_sorted = {
                Layer.type_enum.HEAD: [],
                Layer.type_enum.BODY: [body],
                Layer.type_enum.TAIL: [],
            }
            return SappSolver(
                num_of_interleave=2,
                max_memory=80000,
                layers=[body],
                layers_sorted=layers_sorted,
                **kwargs,
            )
        return factory

    def test_dual_false_tail_at_last_stage(self, make_solver) -> None:
        """Verify TAIL at last stage when dual=False and num_of_interleave>1."""
        solver = make_solver(num_of_stage=4, num_of_micro_batch=4, dual=False)
        positions = solver._reserved_stage_positions()
        assert (0, 0) in positions
        assert (1, 3) in positions

    def test_dual_true_tail_at_stage_zero(self, make_solver) -> None:
        """Verify TAIL at stage zero when dual=True."""
        solver = make_solver(num_of_stage=4, num_of_micro_batch=4, dual=True)
        positions = solver._reserved_stage_positions()
        assert (0, 0) in positions
        assert (1, 0) in positions

    def test_dual_false_single_interleave(self, make_solver) -> None:
        """Verify reserved positions when num_of_interleave=1."""
        from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415

        body = Layer("unit", "body", Layer.type_enum.BODY, 4, time=2)
        layers_sorted = {
            Layer.type_enum.HEAD: [],
            Layer.type_enum.BODY: [body],
            Layer.type_enum.TAIL: [],
        }
        solver = SappSolver(
            num_of_stage=4,
            num_of_interleave=1,
            num_of_micro_batch=4,
            max_memory=80000,
            layers=[body],
            layers_sorted=layers_sorted,
            dual=False,
        )
        positions = solver._reserved_stage_positions()
        assert (0, 0) in positions
        assert (0, 3) in positions
