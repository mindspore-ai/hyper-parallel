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
"""Tests for Bug 1 fix: simulate_manual() passes comm_time as keyword argument.

Before the fix, ``simulate_manual()`` called ``self.simulation()`` with
positional arguments.  When ``comm_time`` was inserted between
``constant_mem`` and ``show`` in the ``simulation()`` signature, the
positional mapping shifted:

- ``self.constant_memory_`` (int) → ``constant_mem`` ✓
- ``show`` (bool) → ``comm_time`` ✗ (bool is a float subclass)
- ``file_name`` (str) → ``show`` ✗
- ``sub_fig`` (Figure) → ``file_name`` ✗

Since ``bool(True) > 0.0``, communication simulation was accidentally
enabled when ``show=True`` (the default).

The fix changes ``simulate_manual()`` to use keyword arguments so that
each value lands in the correct parameter regardless of insertion order.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/pp_modeling/test_simulate_manual_comm_time.py
"""

from unittest.mock import MagicMock, patch

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE


@pytest.mark.skipif(not SAPP_PPB_AVAILABLE, reason="sapp_ppb not installed")
class TestSimulateManualCommTime:
    """Verify simulate_manual() passes comm_time=0.0 as keyword argument."""

    @staticmethod
    def _make_pipeline() -> MagicMock:
        """Create a mock SappPipeline with simulate_manual wired to real code."""
        from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        with patch.object(SappPipeline, "__init__", lambda self, *a, **kw: None):
            pipeline = SappPipeline.__new__(SappPipeline)

        body = Layer(name="BODY", ltype=Layer.type_enum.BODY, nb_layer=4, time=5,
                     memory_activation_rec={r: 1.0 for r in Recompute.TYPE})
        head = Layer(name="HEAD", ltype=Layer.type_enum.HEAD, nb_layer=1, time=10)
        tail = Layer(name="TAIL", ltype=Layer.type_enum.TAIL, nb_layer=1, time=10)

        pipeline.layers_sorted_ = {
            Layer.type_enum.HEAD: [head],
            Layer.type_enum.BODY: [body],
            Layer.type_enum.TAIL: [tail],
        }
        pipeline.constant_memory_ = 500
        pipeline.num_of_micro_batch_ = 4
        pipeline.num_of_stage_ = 4
        pipeline.vpp_less_memory_ = False
        pipeline.use_backward_time_ = False
        pipeline.has_some_memory_info = MagicMock(return_value=True)
        pipeline.get_manual_fw_time = MagicMock(return_value=[[10.0, 10.0, 10.0, 10.0]])
        pipeline.get_manual_recompute_time = MagicMock(return_value=[[0.0, 0.0, 0.0, 0.0]])
        pipeline.get_manual_memory_parameter = MagicMock(return_value=[[200.0, 200.0, 200.0, 200.0]])
        pipeline.get_manual_memory_activation = MagicMock(return_value=[[100.0, 100.0, 100.0, 100.0]])
        pipeline.debug_print_manual_theoretical_memory = MagicMock()
        return pipeline, body

    def test_comm_time_is_zero_when_show_true(self) -> None:
        """When show=True (default), comm_time must remain 0.0, not True."""
        pipeline, body = self._make_pipeline()
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        each_layer_per_recompute = {
            body: {r: [[1, 1, 1, 1]] for r in Recompute.TYPE}
        }

        with patch.object(pipeline, "simulation", return_value=100.0) as mock_sim:
            pipeline.simulate_manual(
                each_layer_per_recompute=each_layer_per_recompute,
                show=True,
                interleave_num=1,
            )
            kwargs = mock_sim.call_args.kwargs
            assert kwargs.get("comm_time") == 0.0, (
                f"comm_time should be 0.0, got {kwargs.get('comm_time')}"
            )

    def test_show_keyword_not_misrouted_to_comm_time(self) -> None:
        """show=True must go to 'show' parameter, not to 'comm_time'."""
        pipeline, body = self._make_pipeline()
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        each_layer_per_recompute = {
            body: {r: [[1, 1, 1, 1]] for r in Recompute.TYPE}
        }

        with patch.object(pipeline, "simulation", return_value=100.0) as mock_sim:
            pipeline.simulate_manual(
                each_layer_per_recompute=each_layer_per_recompute,
                show=True,
                interleave_num=1,
            )
            kwargs = mock_sim.call_args.kwargs
            assert kwargs["comm_time"] == 0.0
            assert kwargs["show"] is True

    def test_show_false_does_not_affect_comm_time(self) -> None:
        """When show=False, comm_time must still be 0.0."""
        pipeline, body = self._make_pipeline()
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        each_layer_per_recompute = {
            body: {r: [[1, 1, 1, 1]] for r in Recompute.TYPE}
        }

        with patch.object(pipeline, "simulation", return_value=100.0) as mock_sim:
            pipeline.simulate_manual(
                each_layer_per_recompute=each_layer_per_recompute,
                show=False,
                interleave_num=1,
            )
            kwargs = mock_sim.call_args.kwargs
            assert kwargs["comm_time"] == 0.0
            assert kwargs["show"] is False

    def test_constant_mem_passed_as_keyword(self) -> None:
        """constant_mem must be passed as keyword with the correct value."""
        pipeline, body = self._make_pipeline()
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        each_layer_per_recompute = {
            body: {r: [[1, 1, 1, 1]] for r in Recompute.TYPE}
        }

        with patch.object(pipeline, "simulation", return_value=100.0) as mock_sim:
            pipeline.simulate_manual(
                each_layer_per_recompute=each_layer_per_recompute,
                show=True,
                interleave_num=1,
            )
            kwargs = mock_sim.call_args.kwargs
            assert kwargs["constant_mem"] == 500, (
                f"Expected constant_mem=500, got {kwargs['constant_mem']}"
            )
