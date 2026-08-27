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
"""Structurally matched tensor-parallel parameter template for MHC modules."""

from typing import Dict

from hyper_parallel.core.dtensor.placement_types import Replicate

from .sharding_config import ModuleShardingSpec


_MHC_MODULES = frozenset({
    "attn_mhc_pre_module",
    "mlp_mhc_pre_module",
    "merge_mhc_module",
})


def _direct_params(module):
    return {
        name: {"tp": Replicate()}
        for name, _ in module.named_parameters(recurse=False)
    }


def _is_mhc_module_fqn(fqn: str) -> bool:
    return any(part in _MHC_MODULES for part in fqn.split("."))


def matches_mhc_template(model) -> bool:
    """Return whether *model* contains an independently owned MHC subtree."""
    for fqn, module in model.named_modules():
        if not _is_mhc_module_fqn(fqn):
            continue
        if any(module.named_parameters(recurse=False)):
            return True
    return False


def build_mhc_specs(model) -> Dict[str, ModuleShardingSpec]:
    """Materialize replicated parameter specs for MHC modules.

    MHC mixes recurrent streams locally inside each sequence-parallel shard.
    Its coefficients and small projections are therefore replicated across TP
    ranks.  Each physical owner is registered separately so an outer module
    never claims parameters from a nested module subtree.
    """
    specs = {}
    for fqn, module in model.named_modules():
        if not _is_mhc_module_fqn(fqn):
            continue
        params = _direct_params(module)
        if params:
            specs[fqn] = ModuleShardingSpec(
                params=params,
                is_boundary=False,
            )
    return specs
