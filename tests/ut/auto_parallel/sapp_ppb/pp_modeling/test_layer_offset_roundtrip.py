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
"""Tests for Bug 3 fix: layer_offset uses naive integer-division nass.

Before the fix, ``_extract_layer_offset_from_ilp()`` used
``_correct_nass()`` to distribute the integer-division remainder across
stages.  However, the original PPB round-trip — ``yaml_from_internal()``
and ``internal_from_yaml()`` — both use naive integer-division nass
(``nb_layer // (pp * vpp)``) without remainder correction.

Using corrected nass for offset calculation but naive nass for
consumption breaks the round-trip:

    Example: 10 layers, PP=4
      - naive nass  = [2, 2, 2, 2]  → total=8
      - corrected   = [3, 3, 2, 2]  → total=10
      - offset with corrected nass  = [0, 0, 0, 0]  (if gass=3)
      - round-trip: offset=0 + naive_nass=2 → only 8 layers, not 10

The fix removes ``_correct_nass()`` from ``_extract_layer_offset_from_ilp()``
so that offset is computed with the same naive nass used by
``internal_from_yaml()`` and ``print_yaml_results()``.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/pp_modeling/test_layer_offset_roundtrip.py
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE


class _Var:
    """Object exposing varValue for yaml_from_internal."""

    def __init__(self, value: Any) -> None:
        """Store a fake solver variable value."""
        self.varValue = value  # pylint: disable=invalid-name


def _make_lp_variables(
    pp: int, vpp: int, none_values: List[List[int]],
) -> Dict[Any, List[List[_Var]]]:
    """Build fake solver lp_variables with NONE=recompute given, others=0."""
    import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

    variables: Dict[Any, List[List[_Var]]] = {}
    for rec in Recompute.TYPE:
        if rec == Recompute.TYPE.NONE:
            variables[rec] = [
                [_Var(none_values[i][s]) for s in range(pp)]
                for i in range(vpp)
            ]
        else:
            variables[rec] = [
                [_Var(0) for _ in range(pp)]
                for _ in range(vpp)
            ]
    return variables


@pytest.mark.skipif(not SAPP_PPB_AVAILABLE, reason="sapp_ppb not installed")
class TestLayerOffsetRoundtrip:
    """Verify yaml_from_internal / internal_from_yaml round-trip consistency.

    The key invariant: the ``nass`` passed to ``yaml_from_internal`` must
    be the same naive integer-division nass used by ``internal_from_yaml``
    for the round-trip to reconstruct the original solver variables.
    """

    def test_roundtrip_10_layers_pp4(self) -> None:
        """10 layers, PP=4: round-trip with naive nass preserves solver values."""
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 4
        vpp = 1
        nb_layer = 10
        none_values = [[3, 3, 2, 2]]

        naive_nass = [[nb_layer // pp] * pp for _ in range(vpp)]
        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, naive_nass)

        internal_back = Recompute.internal_from_yaml(vpp, pp, dict(yaml_out), naive_nass)

        for rec in Recompute.TYPE:
            for i in range(vpp):
                for s in range(pp):
                    original = int(lp_variables[rec][i][s].varValue)
                    reconstructed = internal_back[rec][i][s]
                    assert original == reconstructed, (
                        f"Round-trip mismatch for rec={rec}, i={i}, s={s}: "
                        f"original={original}, reconstructed={reconstructed}"
                    )

    def test_roundtrip_7_layers_pp3(self) -> None:
        """7 layers, PP=3: round-trip with naive nass preserves solver values."""
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 3
        vpp = 1
        nb_layer = 7
        none_values = [[3, 2, 2]]

        naive_nass = [[nb_layer // pp] * pp for _ in range(vpp)]
        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, naive_nass)
        internal_back = Recompute.internal_from_yaml(vpp, pp, dict(yaml_out), naive_nass)

        for rec in Recompute.TYPE:
            for i in range(vpp):
                for s in range(pp):
                    original = int(lp_variables[rec][i][s].varValue)
                    reconstructed = internal_back[rec][i][s]
                    assert original == reconstructed

    def test_roundtrip_exact_division_8_layers_pp4(self) -> None:
        """8 layers, PP=4 (exact division): round-trip is trivially consistent."""
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 4
        vpp = 1
        nb_layer = 8
        none_values = [[2, 2, 2, 2]]

        naive_nass = [[nb_layer // pp] * pp for _ in range(vpp)]
        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, naive_nass)
        internal_back = Recompute.internal_from_yaml(vpp, pp, dict(yaml_out), naive_nass)

        for rec in Recompute.TYPE:
            for i in range(vpp):
                for s in range(pp):
                    original = int(lp_variables[rec][i][s].varValue)
                    reconstructed = internal_back[rec][i][s]
                    assert original == reconstructed

    def test_offset_equals_gass_minus_naive_nass(self) -> None:
        """offset[i][s] = gass[i][s] - nass[i][s] with naive nass.

        This is the core invariant of yaml_from_internal.
        """
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 4
        vpp = 1
        nb_layer = 10
        none_values = [[3, 3, 2, 2]]

        naive_nass = [[nb_layer // pp] * pp for _ in range(vpp)]
        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, naive_nass)

        for i in range(vpp):
            for s in range(pp):
                gass = sum(
                    int(lp_variables[rec][i][s].varValue)
                    for rec in Recompute.TYPE
                )
                expected_offset = gass - naive_nass[i][s]
                actual_offset = yaml_out[Recompute.OFFSET][i][s]
                assert actual_offset == expected_offset, (
                    f"offset[{i}][{s}] should be {expected_offset} "
                    f"(gass={gass} - nass={naive_nass[i][s]}), "
                    f"got {actual_offset}"
                )

    def test_corrected_nass_breaks_roundtrip(self) -> None:
        """Demonstrate that using corrected nass in yaml_from_internal
        but naive nass in internal_from_yaml breaks the round-trip.

        This is the core of Bug 3: the original code used corrected_nass
        in _extract_layer_offset_from_ilp but the consumer (internal_from_yaml)
        uses naive nass.
        """
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 4
        vpp = 1
        nb_layer = 10
        none_values = [[3, 3, 2, 2]]

        naive_nass = [[nb_layer // pp] * pp for _ in range(vpp)]
        corrected_nass = [[3, 3, 2, 2]]

        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, corrected_nass)

        internal_back = Recompute.internal_from_yaml(vpp, pp, dict(yaml_out), naive_nass)

        for rec in [Recompute.TYPE.NONE]:
            for i in range(vpp):
                for s in range(pp):
                    original = int(lp_variables[rec][i][s].varValue)
                    reconstructed = internal_back[rec][i][s]
                    if original != reconstructed:
                        return  # mismatch confirms the bug scenario
        pytest.fail(
            "Expected round-trip mismatch when using corrected_nass for "
            "yaml_from_internal but naive_nass for internal_from_yaml"
        )

    def test_roundtrip_vpp2_10_layers_pp3(self) -> None:
        """10 layers, PP=3, VPP=2: round-trip with naive nass preserves solver values."""
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        pp = 3
        vpp = 2
        nb_layer = 10
        none_values = [[2, 2, 2], [2, 1, 1]]

        naive_nass = [[nb_layer // (pp * vpp)] * pp for _ in range(vpp)]
        lp_variables = _make_lp_variables(pp, vpp, none_values)

        yaml_out = Recompute.yaml_from_internal(vpp, pp, lp_variables, naive_nass)
        internal_back = Recompute.internal_from_yaml(vpp, pp, dict(yaml_out), naive_nass)

        for rec in Recompute.TYPE:
            for i in range(vpp):
                for s in range(pp):
                    original = int(lp_variables[rec][i][s].varValue)
                    reconstructed = internal_back[rec][i][s]
                    assert original == reconstructed
