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
"""Distributed driver for shape utils on a real ``DeviceMesh``."""
from hyper_parallel import (
    destroy_process_group,
    get_platform,
    init_device_mesh,
    init_process_group,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.utils.shape_utils import compute_local_shape_and_global_offset

platform = get_platform()


def test_shape_utils_alias_shard_uneven_split_matches_chunk():
    """
    Feature: uneven split semantics match ``torch.chunk`` on a real mesh.
    Description: With ``("dp",) = 4`` and a global size of 10, the first two
        ranks (0, 1) receive 3 elements each; the remaining ranks (2, 3) get 2.
        Exercises the real ``Layout.from_device_mesh`` + ``DeviceMesh.get_local_rank``
        path that UT mocks. Even-split parity is already pinned by UT.
    Expectation: rank 0/1 → ``[3]``; rank 2/3 → ``[2]``.
    """
    init_process_group()
    try:
        mesh = init_device_mesh(device_type=platform.device_type(),
                                mesh_shape=(4,), mesh_dim_names=("dp",))
        local_shape = compute_local_shape_and_global_offset(
            global_shape=(10,), device_mesh=mesh, placement=("dp",),
        )
        rank = platform.get_rank()
        expected = 3 if rank < 2 else 2
        assert local_shape == [expected], (
            f"alias shard uneven split mismatch on rank={rank}: "
            f"expected=[{expected}], got={local_shape}"
        )
    finally:
        destroy_process_group()


def test_shape_utils_placement_objects_uneven_match_alias_string():
    """
    Feature: ``Shard`` / ``Replicate`` Placement objects under an uneven shard.
    Description: On ``("dp", "tp") = (2, 2)``, shard a global size of 7 along
        ``dp`` (size 2). ``torch.chunk`` gives rank 0 → 4, rank 1 → 3. Verifies
        that the Placement-object code path (``Layout.placement_to_tensor_map``)
        produces the same uneven local shape as the alias-string path —
        guarding both the Placement→alias translation and the uneven branch
        on a real mesh.
    Expectation: rank 0 → ``[4, 16]``; rank 1 → ``[3, 16]``; placement and alias agree.
    """
    init_process_group()
    try:
        mesh = init_device_mesh(device_type=platform.device_type(),
                                mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
        global_shape = (7, 16)

        local_via_placement = compute_local_shape_and_global_offset(
            global_shape=global_shape, device_mesh=mesh,
            placement=[Shard(0), Replicate()],
        )
        local_via_alias = compute_local_shape_and_global_offset(
            global_shape=global_shape, device_mesh=mesh,
            placement=("dp", "None"),
        )
        rank = platform.get_rank()
        assert local_via_placement == local_via_alias, (
            f"Placement vs alias mismatch on rank={rank}: "
            f"placement={local_via_placement}, alias={local_via_alias}"
        )
        # Ranks share dp coord 0 (ranks 0, 1) get 4; dp coord 1 (ranks 2, 3) get 3.
        expected_dim0 = 4 if rank // 2 == 0 else 3
        assert local_via_placement == [expected_dim0, 16], (
            f"Uneven shard mismatch on rank={rank}: "
            f"expected=[{expected_dim0}, 16], got={local_via_placement}"
        )
    finally:
        destroy_process_group()
