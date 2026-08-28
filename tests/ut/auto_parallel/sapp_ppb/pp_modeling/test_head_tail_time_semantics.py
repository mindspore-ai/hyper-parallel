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
"""Tests for Bug 4 fix: HEAD/TAIL time uses forward_time_ + backward_time_rec_[NONE].

Before the fix, ``get_manual_time()`` and ``get_simulator_time()`` used
``head.time_`` / ``tail.time_`` for HEAD/TAIL layers, which equals only
``forward_time_``.  The correct value is
``forward_time_ + backward_time_rec_[NONE]`` — consistent with how BODY
layers are handled (``forward_time_ + backward_time_rec_[r]``) and with
the ILP constraint ``_max_stage_bound_head_tail()`` which already uses
the correct formula.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/pp_modeling/test_head_tail_time_semantics.py
"""

import os

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_32layers.json"
)

_PP = 4
_NUM_BODY_LAYERS = 32
_MICRO_BATCH = 4
_MEMORY_LIMIT = 80000


def _build_layers_with_loader():
    from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

    yaml_config = YamlOptimizationConfig(
        num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
        micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
    )
    layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
    return layer_builder.layers_sapp_ppb


def _make_pipeline_isolated_head_tail():
    """Build a minimal SappPipeline with isolated HEAD/TAIL for testing.

    Returns:
        Tuple of (pipeline, head_layer, body_layer, tail_layer, pp_degree).
    """
    from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415

    pp = 2
    mb = 2
    head_time = 90.0
    body_time = 10.0
    tail_time = 180.0

    head = Layer(name="HEAD", ltype=Layer.type_enum.HEAD, nb_layer=1, time=head_time)
    body = Layer(name="BODY", ltype=Layer.type_enum.BODY, nb_layer=2, time=body_time)
    tail = Layer(name="TAIL", ltype=Layer.type_enum.TAIL, nb_layer=1, time=tail_time)

    pipeline = SappPipeline(
        model_name="test_head_tail_time",
        num_of_stage=pp,
        num_of_micro_batch=mb,
        max_memory=_MEMORY_LIMIT,
        layers=[head, body, tail],
    )
    return pipeline, head, body, tail, pp


@pytest.mark.skipif(not SAPP_PPB_AVAILABLE, reason="sapp_ppb not installed")
class TestGetManualTimeHeadTailSemantics:
    """Verify get_manual_time uses forward_time_ + backward_time_rec_[NONE] for HEAD/TAIL."""

    def test_manual_time_head_isolated_equals_forward_plus_backward(self) -> None:
        """get_manual_time() HEAD contribution = forward_time_ + backward_time_rec_[NONE].

        BODY layers are all assigned to stage 1 so that stage 0 contains
        only HEAD, making the HEAD contribution directly observable.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415

        pipeline, head, body, tail, pp = _make_pipeline_isolated_head_tail()

        each_layer = {
            body: {rec: [[0, 0] for _ in range(pp)] for rec in Recompute.TYPE}
        }
        for rec in Recompute.TYPE:
            each_layer[body][rec][0][1] = 2

        manual_time = pipeline.get_manual_time(each_layer, interleave_num=1)

        expected = head.forward_time_ + head.backward_time_rec_[Recompute.TYPE.NONE]
        assert manual_time[0][0] == expected
        assert expected > head.time_

    def test_manual_time_tail_isolated_equals_forward_plus_backward(self) -> None:
        """get_manual_time() TAIL contribution = forward_time_ + backward_time_rec_[NONE].

        BODY layers are all assigned to stage 0 so that stage 1 contains
        only TAIL, making the TAIL contribution directly observable.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415

        pipeline, head, body, tail, pp = _make_pipeline_isolated_head_tail()

        each_layer = {
            body: {rec: [[0, 0] for _ in range(pp)] for rec in Recompute.TYPE}
        }
        for rec in Recompute.TYPE:
            each_layer[body][rec][0][0] = 2

        manual_time = pipeline.get_manual_time(each_layer, interleave_num=1)

        expected = tail.forward_time_ + tail.backward_time_rec_[Recompute.TYPE.NONE]
        assert manual_time[0][1] == expected
        assert expected > tail.time_

    def test_manual_time_stage0_exact_value(self) -> None:
        """get_manual_time() stage 0 = HEAD(fw+bw) + BODY(fw+bw, NONE only).

        With HEAD time=90, BODY time=10, 1 BODY layer (NONE) per stage:
          HEAD(fw+bw) = 90 + 180 = 270
          BODY(fw+bw, NONE) = 10 + 20 = 30
          stage 0 = 300
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415

        pipeline, head, body, tail, pp = _make_pipeline_isolated_head_tail()

        each_layer = {
            body: {rec: [[0, 0] for _ in range(pp)] for rec in Recompute.TYPE}
        }
        each_layer[body][Recompute.TYPE.NONE][0][0] = 1
        each_layer[body][Recompute.TYPE.NONE][0][1] = 1

        manual_time = pipeline.get_manual_time(each_layer, interleave_num=1)

        head_contrib = head.forward_time_ + head.backward_time_rec_[Recompute.TYPE.NONE]
        body_none = body.forward_time_ + body.backward_time_rec_[Recompute.TYPE.NONE]
        assert manual_time[0][0] == head_contrib + body_none

    def test_manual_time_last_stage_exact_value(self) -> None:
        """get_manual_time() last stage = TAIL(fw+bw) + BODY(fw+bw, NONE only).

        With TAIL time=180, BODY time=10, 1 BODY layer (NONE) per stage:
          TAIL(fw+bw) = 180 + 360 = 540
          BODY(fw+bw, NONE) = 10 + 20 = 30
          stage 1 = 570
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415

        pipeline, head, body, tail, pp = _make_pipeline_isolated_head_tail()

        each_layer = {
            body: {rec: [[0, 0] for _ in range(pp)] for rec in Recompute.TYPE}
        }
        each_layer[body][Recompute.TYPE.NONE][0][0] = 1
        each_layer[body][Recompute.TYPE.NONE][0][1] = 1

        manual_time = pipeline.get_manual_time(each_layer, interleave_num=1)

        tail_contrib = tail.forward_time_ + tail.backward_time_rec_[Recompute.TYPE.NONE]
        body_none = body.forward_time_ + body.backward_time_rec_[Recompute.TYPE.NONE]
        assert manual_time[0][1] == tail_contrib + body_none
