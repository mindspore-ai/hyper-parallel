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

import torch.distributed as dist


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
        raise ValueError("encoder mesh shape does not match decoder parent")
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


def select_encoder_domain_size(*, decoder_size: int, job_size: int,
                               encoder_cp_size: int, resource_domain: str) -> int:
    """Select the smallest legal whole-decoder domain, or an explicitly requested domain.

    Resource selection does not change the task or execution protocol. Contiguous
    complete decoder parents preserve sample identity and enable small-group transport.
    """
    if resource_domain not in ("auto", "decoder", "global"):
        raise ValueError("encoder resource_domain must be auto, decoder or global")
    size = {"auto": math.lcm(decoder_size, encoder_cp_size),
            "decoder": decoder_size, "global": job_size}[resource_domain]
    if size % decoder_size or size % encoder_cp_size or job_size % size:
        raise ValueError("encoder resource domain must contain whole decoder parents, "
                         "divide the job size and be divisible by encoder_cp_size")
    return size


def collect_decoder_metadata(value: Any, *, decoder_size: int, parent_mesh: Any,
                             device: Any = None) -> tuple[Any, ...]:
    """Collect one small envelope per decoder leader using the existing resource group.

    A local domain already has identical metadata on its CP/TP replicas. Shared
    domains broadcast leader envelopes instead of gathering empty contributions
    from every replica, retaining backend-portable object transport.

    Args:
        value: Local envelope; only each decoder leader's value is consumed.
        decoder_size: Number of ranks per complete decoder parent.
        parent_mesh: Selected encoder resource domain.
        device: Collective staging device, separate from the CPU metadata.

    Returns:
        Decoder envelopes in resource-domain leader order.
    """
    if parent_mesh.size() == decoder_size:
        return (value,)
    if not dist.is_initialized():
        raise RuntimeError("Independent encoder domains require initialized process groups")
    group = parent_mesh.get_group()
    members = dist.get_process_group_ranks(group)
    envelopes = []
    for source in range(0, parent_mesh.size(), decoder_size):
        envelope = [value if parent_mesh.get_local_rank() == source else None]
        dist.broadcast_object_list(envelope, src=members[source], group=group, device=device)
        envelopes.append(envelope[0])
    return tuple(envelopes)
