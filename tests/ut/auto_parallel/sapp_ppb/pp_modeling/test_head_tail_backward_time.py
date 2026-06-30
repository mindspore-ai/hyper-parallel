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
"""Tests for HEAD/TAIL backward_time_rec_ handling after layer post-processing.

HEAD/TAIL layers loaded from native JSON have ``backward_time_rec_`` set to
``None`` for all recompute types because the JSON only provides ``{"type":
"HEAD", "time": 90}`` — no per-recompute backward time or memory fields.

``_apply_recompute_considered()`` sets ``recompute_considered_[NONE]`` to
``True`` and calls ``compute_internal_time()``, which fills
``backward_time_rec_[NONE]`` with ``(1 + DEFAULT_COEF[NONE]) * backward_time_``.
Non-NONE types remain ``None`` because ``recompute_considered_[rec]`` is
``False`` for HEAD/TAIL — the ILP solver never reads these entries.

Previously, ``_sanitize_layer_rec_none()`` replaced ``None`` with ``0``, but
this was redundant because all ILP expression paths guard with
``recompute_considered_[rec]``.  The sanitize function has been removed; the
test now asserts ``None`` for non-NONE types instead of ``0``.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/pp_modeling/test_head_tail_backward_time.py
"""

import os

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_32layers.json"
)

_PP = 8
_NUM_BODY_LAYERS = 32
_MICRO_BATCH = 8
_MEMORY_LIMIT = 80000


@pytest.mark.skipif(not SAPP_PPB_AVAILABLE, reason="sapp_ppb not installed")
class TestHeadTailBackwardTime:
    """Verify HEAD/TAIL layers have non-zero backward_time_rec_[NONE]."""

    def test_head_backward_time_none_is_positive(self) -> None:
        """HEAD layer backward_time_rec_[NONE] must be > 0."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        head = [lay for lay in layers if lay.type_ == layer_cls.type_enum.HEAD][0]
        assert head.backward_time_rec_[Recompute.TYPE.NONE] > 0, (
            f"HEAD backward_time_rec_[NONE] should be > 0, "
            f"got {head.backward_time_rec_[Recompute.TYPE.NONE]}"
        )

    def test_tail_backward_time_none_is_positive(self) -> None:
        """TAIL layer backward_time_rec_[NONE] must be > 0."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        tail = [lay for lay in layers if lay.type_ == layer_cls.type_enum.TAIL][0]
        assert tail.backward_time_rec_[Recompute.TYPE.NONE] > 0, (
            f"TAIL backward_time_rec_[NONE] should be > 0, "
            f"got {tail.backward_time_rec_[Recompute.TYPE.NONE]}"
        )

    def test_head_backward_time_matches_formula(self) -> None:
        """HEAD backward_time_rec_[NONE] = (1 + DEFAULT_COEF[NONE]) * 2 * time_."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        head = [lay for lay in layers if lay.type_ == layer_cls.type_enum.HEAD][0]
        expected = (1 + Recompute.DEFAULT_COEF[Recompute.TYPE.NONE]) * 2 * head.time_
        assert head.backward_time_rec_[Recompute.TYPE.NONE] == expected, (
            f"HEAD backward_time_rec_[NONE] should be {expected}, "
            f"got {head.backward_time_rec_[Recompute.TYPE.NONE]}"
        )

    def test_tail_backward_time_matches_formula(self) -> None:
        """TAIL backward_time_rec_[NONE] = (1 + DEFAULT_COEF[NONE]) * 2 * time_."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        tail = [lay for lay in layers if lay.type_ == layer_cls.type_enum.TAIL][0]
        expected = (1 + Recompute.DEFAULT_COEF[Recompute.TYPE.NONE]) * 2 * tail.time_
        assert tail.backward_time_rec_[Recompute.TYPE.NONE] == expected, (
            f"TAIL backward_time_rec_[NONE] should be {expected}, "
            f"got {tail.backward_time_rec_[Recompute.TYPE.NONE]}"
        )

    def test_head_tail_non_none_recompute_backward_time_is_none(self) -> None:
        """HEAD/TAIL non-NONE recompute types have backward_time_rec_ == None.

        Since ``_sanitize_layer_rec_none`` was removed (its None→0
        replacement was redundant — all ILP expression paths guard with
        ``recompute_considered_[rec]``), non-considered recompute types
        retain their original ``None`` values.  This is safe because
        HEAD/TAIL have ``recompute_considered_[rec] == False`` for all
        non-NONE types, so the ILP solver never reads these entries.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        for layer in layers:
            if layer.type_ in (layer_cls.type_enum.HEAD, layer_cls.type_enum.TAIL):
                for rec in Recompute.TYPE:
                    if rec != Recompute.TYPE.NONE:
                        assert layer.backward_time_rec_[rec] is None, (
                            f"{layer.type_.name} backward_time_rec_[{rec}] should be None, "
                            f"got {layer.backward_time_rec_[rec]}"
                        )

    def test_body_backward_time_unaffected(self) -> None:
        """BODY layer backward_time_rec_ is not affected by the HEAD/TAIL fix."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _get_pipeline_layer_class  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder  # pylint: disable=C0415

        yaml_config = YamlOptimizationConfig(
            num_layer=_NUM_BODY_LAYERS, pp_degree=_PP,
            micro_batch_num=_MICRO_BATCH, memory_limit=_MEMORY_LIMIT,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        layers = layer_builder.layers_sapp_ppb

        layer_cls = _get_pipeline_layer_class()
        body = [lay for lay in layers if lay.type_ == layer_cls.type_enum.BODY][0]
        assert body.backward_time_rec_[Recompute.TYPE.NONE] > 0

    def test_apply_recompute_considered_fills_backward_time(self) -> None:
        """_apply_recompute_considered with NONE=True fills backward_time_rec_[NONE].

        This simulates the exact scenario from JSON loading: a HEAD layer
        whose memory_activation_rec_ fields are all None (because the JSON
        does not provide them), resulting in all recompute_considered_ being
        False and backward_time_rec_ being None.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _apply_recompute_considered  # pylint: disable=C0415

        layer = Layer(
            name="HEAD", ltype=Layer.type_enum.HEAD, nb_layer=1, time=90,
            backward_time_rec={r: None for r in Recompute.TYPE},
            memory_activation_rec={r: None for r in Recompute.TYPE},
        )
        assert layer.backward_time_rec_[Recompute.TYPE.NONE] is None, (
            "Before _apply_recompute_considered, backward_time_rec_[NONE] is None"
        )

        _apply_recompute_considered(
            layer, {r: (r == Recompute.TYPE.NONE) for r in Recompute.TYPE},
        )

        assert layer.backward_time_rec_[Recompute.TYPE.NONE] > 0, (
            f"After _apply_recompute_considered, backward_time_rec_[NONE] "
            f"should be > 0, got {layer.backward_time_rec_[Recompute.TYPE.NONE]}"
        )

    def test_apply_recompute_considered_preserves_forward_time(self) -> None:
        """_apply_recompute_considered does not overwrite existing forward_time_."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import _apply_recompute_considered  # pylint: disable=C0415

        layer = Layer(
            name="HEAD", ltype=Layer.type_enum.HEAD, nb_layer=1, time=90,
            backward_time_rec={r: None for r in Recompute.TYPE},
            memory_activation_rec={r: None for r in Recompute.TYPE},
        )
        original_forward_time = layer.forward_time_
        assert original_forward_time is not None

        _apply_recompute_considered(
            layer, {r: (r == Recompute.TYPE.NONE) for r in Recompute.TYPE},
        )

        assert layer.forward_time_ == original_forward_time, (
            f"forward_time_ should remain {original_forward_time}, "
            f"got {layer.forward_time_}"
        )
