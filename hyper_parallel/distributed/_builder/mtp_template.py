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
"""Structurally matched tensor-parallel boundary template for MTP layers."""

import re
from typing import Any, Dict

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


_MTP_LAYER = re.compile(r"(?:^|\.)layers\.\d+$")
_MTP_CHILDREN = frozenset({"mtp_block", "prev_norm", "emb_norm", "prev_proj"})


def _matching_mtp_layers(model: Any) -> Dict[str, Any]:
    """Find decoder layers that expose the full MTP structural contract."""
    matches = {}
    for fqn, module in model.named_modules():
        if not _MTP_LAYER.search(fqn):
            continue
        children = dict(module.named_children())
        if not _MTP_CHILDREN.issubset(children):
            continue
        if any(children["prev_proj"].named_parameters(recurse=False)):
            matches[fqn] = module
    return matches


def matches_mtp_template(model: Any) -> bool:
    """Return whether *model* contains an MTP layer with a previous-state projection."""
    return bool(_matching_mtp_layers(model))


def build_mtp_specs(model: Any) -> Dict[str, ModuleShardingSpec]:
    """Materialize sequence-parallel projection specs for MTP layers.

    MTP concatenates the shifted token embedding and the previous hidden state
    inside each sequence-parallel shard.  ``prev_proj`` therefore keeps its
    weight replicated while preserving the sequence shard on its input and
    output activations.

    Args:
        model: Model containing structurally identifiable MTP layers.

    Returns:
        Fully declared sharding specs keyed by ``prev_proj`` module FQN.
    """
    specs = {}
    for layer_fqn, layer in _matching_mtp_layers(model).items():
        prev_proj = layer.prev_proj
        params = {
            name: {"tp": Replicate()}
            for name, _ in prev_proj.named_parameters(recurse=False)
        }
        specs[f"{layer_fqn}.prev_proj"] = ModuleShardingSpec(
            params=params,
            in_src={"input": {"tp": Shard(1)}},
            in_dst={"input": {"tp": Shard(1)}},
            out_src={"output": {"tp": Shard(1)}},
            out_dst={"output": {"tp": Shard(1)}},
        )
    return specs
