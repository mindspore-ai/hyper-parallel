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
"""Derive visual encoder coordinates from the existing decoder mesh."""

import math
from typing import Any


def derive_reshaped_submesh(root_mesh: Any, *, source_dims: tuple[str, ...],
                           target_sizes: tuple[int, ...], target_names: tuple[str, ...]) -> Any:
    """Reshape a named submesh while preserving its physical rank order.

    Args:
        root_mesh: Existing decoder mesh.
        source_dims: Decoder axes forming one visual parent group.
        target_sizes: Encoder-DP and Encoder-CP degrees.
        target_names: Names assigned to the derived axes.
    """
    source = root_mesh[source_dims]
    if math.prod(target_sizes) != math.prod(source.mesh_shape) or len(target_sizes) != len(target_names):
        raise ValueError("[HP-ENCODER-002] encoder mesh shape does not match decoder parent")
    flat = source.flatten("encoder_parent")
    return flat._unflatten("encoder_parent", target_sizes, target_names)  # pylint: disable=protected-access


def encoder_coordinates(parent_rank: int, encoder_cp_size: int) -> tuple[int, int]:
    """Return Encoder-DP and Encoder-CP coordinates in parent rank order.

    Args:
        parent_rank: Linear rank within the decoder CP-by-TP parent.
        encoder_cp_size: Number of sequence shards per visual bucket.
    """
    return divmod(parent_rank, encoder_cp_size)


def decoder_coordinates(parent_rank: int, decoder_tp_size: int) -> tuple[int, int]:
    """Return decoder CP and TP coordinates in parent rank order.

    Args:
        parent_rank: Linear rank within the decoder CP-by-TP parent.
        decoder_tp_size: Number of decoder TP ranks per CP rank.
    """
    return divmod(parent_rank, decoder_tp_size)
