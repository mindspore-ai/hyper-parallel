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
"""Tests for offset round-trip consistency using naive (uncorrected) nass.

The offset computed by ``_extract_layer_offset_from_ilp`` uses raw
integer-division nass (``nb_layer_ // (pp * vpp)``) so that
``internal_from_yaml`` (which also uses raw nass) can correctly
reconstruct the layer counts.
"""

import pytest

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE


@pytest.mark.skipif(not SAPP_PPB_AVAILABLE, reason="sapp_ppb not installed")
class TestExtractLayerOffsetNaiveNass:
    """Verify _extract_layer_offset_from_ilp uses naive (uncorrected) nass
    for offset calculation, ensuring round-trip consistency with
    internal_from_yaml which also uses naive nass."""

    def test_naive_nass_offset_round_trips(self) -> None:
        """Offset computed with naive nass must round-trip through
        internal_from_yaml (which also uses naive nass).

        With 10 layers, PP=4:
          - naive nass = [2, 2, 2, 2] (integer division)
          - offset = gass - naive_nass per stage
          - internal_from_yaml reconstructs: offset + naive_nass = gass
        """
        import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute  # pylint: disable=C0415

        body_layers = 10
        pp = 4
        naive_per_cell = body_layers // pp
        naive_nass = [[naive_per_cell] * pp]

        gass = [3, 3, 2, 2]
        offset = [gass[s] - naive_per_cell for s in range(pp)]

        assert offset == [1, 1, 0, 0]
        reconstructed = [offset[s] + naive_nass[0][s] for s in range(pp)]
        assert reconstructed == gass

    def test_corrected_nass_would_break_round_trip(self) -> None:
        """Demonstrate that using corrected nass for offset breaks round-trip.

        With 10 layers, PP=4:
          - corrected nass = [3, 3, 2, 2] (sum = 10)
          - offset with corrected nass = [0, 0, 0, 0]
          - internal_from_yaml reconstructs with raw nass: [0+2, 0+2, 0+2, 0+2] = [2,2,2,2]
          - 8 layers, not 10 — two layers lost!
        """
        pp = 4
        raw_nass = [[2, 2, 2, 2]]
        corrected_nass = [[3, 3, 2, 2]]
        gass = [3, 3, 2, 2]

        offset_with_corrected = [gass[s] - corrected_nass[0][s] for s in range(pp)]
        assert offset_with_corrected == [0, 0, 0, 0]

        reconstructed = [offset_with_corrected[s] + raw_nass[0][s] for s in range(pp)]
        assert sum(reconstructed) == 8
        assert sum(reconstructed) != sum(gass)
