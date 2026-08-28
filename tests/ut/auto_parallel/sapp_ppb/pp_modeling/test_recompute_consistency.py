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
"""Tests for recompute_considered consistency validation across BODY groups.

Covers:
- Single BODY group: validation skipped (no cross-group comparison needed)
- Multiple BODY groups with identical masks: passes
- Multiple BODY groups with different masks: raises ValueError
- HEAD/TAIL mask differences do not affect BODY consistency check
"""

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import (
    LayerBuilder,
    SAPP_PPB_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not SAPP_PPB_AVAILABLE,
    reason="sapp_ppb not available",
)


def _make_layer(name: str, ltype: str, recompute_considered: dict) -> object:
    """Create a minimal layer-like object for testing."""
    from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer

    layer = object.__new__(Layer)
    layer.name_ = name
    layer.type_ = Layer.type_enum[ltype]
    layer.recompute_considered_ = dict(recompute_considered)
    return layer


def _default_considered(none_only: bool = False) -> dict:
    """Return a recompute_considered dict matching sapp-ppb Recompute.TYPE."""
    from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute

    if none_only:
        return {r: (r == Recompute.TYPE.NONE) for r in Recompute.TYPE}
    return {r: (r in (Recompute.TYPE.NONE, Recompute.TYPE.SLCT)) for r in Recompute.TYPE}


class TestValidateRecomputeConsistency:
    """LayerBuilder._validate_recompute_consistency."""

    def test_single_body_group_skips(self) -> None:
        layers = [
            _make_layer("HEAD", "HEAD", _default_considered(none_only=True)),
            _make_layer("encoder", "BODY", _default_considered()),
            _make_layer("TAIL", "TAIL", _default_considered(none_only=True)),
        ]
        LayerBuilder._validate_recompute_consistency(layers)

    def test_multi_body_consistent_masks_passes(self) -> None:
        mask = _default_considered()
        layers = [
            _make_layer("HEAD", "HEAD", _default_considered(none_only=True)),
            _make_layer("encoder", "BODY", mask),
            _make_layer("decoder", "BODY", dict(mask)),
            _make_layer("TAIL", "TAIL", _default_considered(none_only=True)),
        ]
        LayerBuilder._validate_recompute_consistency(layers)

    def test_multi_body_inconsistent_masks_raises(self) -> None:
        """Multiple BODY groups with different recompute_considered masks raises ValueError."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute

        mask_a = _default_considered()
        mask_b = {r: (r in (Recompute.TYPE.NONE, Recompute.TYPE.FULL)) for r in Recompute.TYPE}
        layers = [
            _make_layer("HEAD", "HEAD", _default_considered(none_only=True)),
            _make_layer("encoder", "BODY", mask_a),
            _make_layer("decoder", "BODY", mask_b),
            _make_layer("TAIL", "TAIL", _default_considered(none_only=True)),
        ]
        with pytest.raises(ValueError, match="recompute_considered mask"):
            LayerBuilder._validate_recompute_consistency(layers)

    def test_head_tail_differences_do_not_affect_body(self) -> None:
        """HEAD/TAIL mask differences should not trigger BODY consistency check."""
        from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute

        body_mask = _default_considered()
        head_mask = {r: True for r in Recompute.TYPE}
        tail_mask = {r: False for r in Recompute.TYPE}
        layers = [
            _make_layer("HEAD", "HEAD", head_mask),
            _make_layer("encoder", "BODY", dict(body_mask)),
            _make_layer("decoder", "BODY", dict(body_mask)),
            _make_layer("TAIL", "TAIL", tail_mask),
        ]
        LayerBuilder._validate_recompute_consistency(layers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
