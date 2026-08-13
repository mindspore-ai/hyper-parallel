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
"""Verify every name listed in sapp_ppb.__all__ can be resolved.

Catches cases where the ``_EXPORTS`` mapping references a module that
has been renamed or deleted (e.g. ``pp_simulator`` → ``pp_sim_adapter``).
"""
import pytest

import hyper_parallel.auto_parallel.sapp_ppb as pkg


@pytest.mark.parametrize("name", pkg.__all__)
def test_public_symbol_importable(name):
    obj = getattr(pkg, name)
    assert obj is not None


def test_all_names_in_exports():
    for name in pkg.__all__:
        assert name in pkg._EXPORTS, (
            f"{name!r} is in __all__ but missing from _EXPORTS"
        )
