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
"""Four-rank Gloo oracle for Xing40 TP and FSDP QK clipping."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import DeviceMesh, Shard, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import StridedShard
from hyper_parallel.models.xing40.adapter.optim import clip_qk


def _check_mesh(rank, ranks):
    """Compare each main shard against an independent global per-row formula."""
    mesh = DeviceMesh("cpu", ranks, mesh_dim_names=("fsdp_shard", "tp"))
    coordinate = list(mesh.get_coordinate() or ())
    assert len(coordinate) == 2, f"expected coordinate dimensions=2, got={len(coordinate)}"
    dp_rank = coordinate[0]
    tp_rank = coordinate[1]
    assert mesh.rank_list[dp_rank * 2 + tp_rank] == rank, (
        f"expected rank={rank}, got={mesh.rank_list[dp_rank * 2 + tp_rank]}"
    )
    query = torch.arange(1, 49, dtype=torch.float32).reshape(16, 3)
    key_value = query + 50
    placements = (StridedShard(0, split_factor=2), Shard(0))
    maxima_cases = (
        ([[200., 80., 50., 400.], [150., 300., 80., 250.]], [200., 300., 80., 400.]),
        ([[20., 80., 50., 99.], [90., 100., 80., 10.]], [90., 100., 80., 99.]),
    )
    for maxima, global_maxima in maxima_cases:
        q_main = distribute_tensor(query.clone(), mesh, placements)
        kv_main = distribute_tensor(key_value.clone(), mesh, placements)
        q_model = distribute_tensor(query.to(torch.bfloat16), mesh, placements)
        kv_model = distribute_tensor(key_value.to(torch.bfloat16), mesh, placements)
        q_model.main_param, kv_model.main_param = q_main, kv_main
        module = SimpleNamespace(
            q_b_proj=SimpleNamespace(weight=q_model), kv_b_proj=SimpleNamespace(weight=kv_model),
            qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2,
            max_logits=torch.tensor(maxima[dp_rank][tp_rank * 2:tp_rank * 2 + 2]),
        )
        clip_qk([module], 100.)
        expected_q, expected_kv = query.clone(), key_value.clone()
        for row in range(16):
            head, within_head = divmod(row, 4)
            scale = 100. / max(100., global_maxima[head])
            expected_q[row] *= scale ** 0.5 if within_head < 2 else scale
            expected_kv[row] *= scale ** 0.5 if within_head < 2 else 1.
        # Strided FSDP splits each TP half across DP. Rank numbering must not
        # determine either head ownership or which DP replica supplies maxima.
        start = (tp_rank * 2 + dp_rank) * 4
        actual_q, actual_kv = q_main.to_local(), kv_main.to_local()
        torch.testing.assert_close(actual_q, expected_q[start:start + 4], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_kv, expected_kv[start:start + 4], rtol=1e-6, atol=1e-6)
        if max(global_maxima) <= 100.:
            assert torch.equal(actual_q, query[start:start + 4]), (
                f"expected Q unchanged=True, got=False for rank={rank}, start={start}"
            )
            assert torch.equal(actual_kv, key_value[start:start + 4]), (
                f"expected KV unchanged=True, got=False for rank={rank}, start={start}"
            )
        assert module.max_logits is None, (
            f"expected max_logits=None, got non-None for rank={rank}"
        )

    # Head-local maxima cannot describe a column-sharded projection. Reject it
    # before issuing a collective or silently assigning heads to arbitrary rows.
    wrong_weight = distribute_tensor(torch.ones(16, 4), mesh, (Shard(0), Shard(1)))
    module.q_b_proj.weight = wrong_weight
    module.kv_b_proj.weight = wrong_weight
    module.max_logits = torch.ones(2)
    with pytest.raises(ValueError, match="contiguous projection-head TP shards"):
        clip_qk([module], 100.)


def _check_clipping(rank, world_size):
    """Exercise ordinary and non-monotonic TP group membership."""
    assert world_size == 4, f"expected world_size=4, got={world_size}"
    for ranks in ([[0, 1], [2, 3]], [[1, 0], [3, 2]]):
        _check_mesh(rank, ranks)


def test_xing40_qk_clip_tp_worker() -> None:
    """Check global head identity over ordinary and non-monotonic TP meshes."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        _check_clipping(dist.get_rank(), dist.get_world_size())
    finally:
        dist.destroy_process_group()
