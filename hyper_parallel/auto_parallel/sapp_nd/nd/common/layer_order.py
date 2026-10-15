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
"""Where the layers of a pipeline partition sit, in model order.

A partition lists each stage's chunks and each chunk's layers.  Model order
is the order of the layers in the model, which a layer's kind and its
recompute range count in: chunk by chunk across the stages, and back up
them in the second chunk of a V schedule.
"""
from typing import Any, List, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

# The layer type the partition places for each recompute option.
_TYPES = {
    "none": LayerType.NOT_REC_LAYER,
    "full": LayerType.FULL_REC_LAYER,
    "selective": LayerType.SEL_REC_LAYER,
}


def get_model_order(cfg: Any, stages: List) -> List[Tuple[int, int, int]]:
    """Positions ``(stage, chunk, index)`` of the regular layers, in model order.

    Model order runs chunk by chunk across the stages, and back up them in
    the second chunk of a V schedule; walking the stages one at a time is
    only the same order without interleaving.  This is the order the memory
    backbone gives layers their groups in.
    """
    positions = [
        (stage_id, chunk_id, lay_id)
        for chunk_id in range(cfg.vp)
        for stage_id in range(cfg.p)
        for lay_id, layer in enumerate(stages[stage_id][chunk_id])
        if layer not in (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER)
    ]
    if cfg.pp_sched == "zero_bubble_v":
        # First-chunk entries less the embedding, as the memory backbone counts.
        first = sum(len(stage[0]) for stage in stages) - 1
        positions = positions[:first] + positions[first:][::-1]
    return positions


def stated_recompute(ccfg: Any) -> Optional[Tuple[Any, ...]]:
    """Return the recompute ranges a config states, or ``None`` when it states none.

    An empty tuple is a statement too: no layer recomputes.
    """
    ranges = getattr(ccfg, "recompute_ranges", None)
    return ranges if isinstance(ranges, tuple) else None


def layer_switches(ccfg: Any) -> Optional[Tuple[Any, ...]]:
    """Return each layer's own recompute switches in model order, or ``None`` where ``rec_op`` holds the one setting."""
    switches = getattr(ccfg, "layer_switches", None)
    return switches if isinstance(switches, tuple) else None


def layer_recompute_types(ranges: Sequence[Any], layers: int) -> List[LayerType]:
    """Return the layer type of each of *layers* layers in model order.

    Each layer takes the option of the recompute range that covers it, and
    a layer no range covers is not recomputed.
    """
    types = [LayerType.NOT_REC_LAYER] * layers
    for item in ranges:
        stop = layers if item.count is None else min(layers, item.first + item.count)
        for index in range(item.first, stop):
            types[index] = _TYPES[item.option]
    return types
