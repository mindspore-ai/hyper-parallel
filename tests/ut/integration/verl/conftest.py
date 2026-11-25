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
"""Bind external contracts only while an adapter unit test is running."""

import pytest
from omegaconf import OmegaConf

from tests.ut.integration.verl._contracts import contract_modules


@pytest.fixture(autouse=True)
def external_contracts():
    """Keep all fake VeRL imports and adapter aliases local to this test."""
    had_resolver = OmegaConf.has_resolver("cp_token_budget")
    try:
        with contract_modules():
            yield
    finally:
        if not had_resolver:
            OmegaConf.clear_resolver("cp_token_budget")
