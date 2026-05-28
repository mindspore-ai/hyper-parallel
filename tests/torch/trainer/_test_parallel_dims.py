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
"""Distributed driver for ``ParallelDims`` and rank-aware logging cases."""
import io
import logging

from hyper_parallel import (
    destroy_process_group,
    get_platform,
    init_process_group,
)
from hyper_parallel.trainer.parallel_dims import ParallelDims
from hyper_parallel.trainer.utils.logging import (
    get_logger,
    init_logger,
)

platform = get_platform()


def _world_size() -> int:
    return platform.get_world_size()


def test_parallel_dims_pure_fsdp_mesh_and_rank_logging():
    """
    Feature: ParallelDims.build_mesh under pure 1D FSDP + rank-aware logging.
    Description: With ``dp_shard == world_size`` and all other dims = 1,
        ``mesh["fsdp"]``, ``mesh["dp"]`` and ``mesh["loss"]`` must all resolve
        and contain every rank exactly once. The same PG also verifies that
        ``info_rank0`` / ``warning_rank0`` consult the live process group's
        rank (not the env fallback) and therefore fire only on rank 0.
    Expectation: All three aliases yield a 1D mesh whose rank list equals
        ``[0, 1, ..., world_size - 1]``. ``info_rank0`` writes only on rank 0;
        plain ``logger.info`` writes on every rank.
    """
    init_process_group()
    try:
        world = _world_size()
        pd = ParallelDims(dp_shard=world, world_size=world)
        mesh = pd.build_mesh(platform.device_type())

        for alias in ("fsdp", "dp", "loss"):
            sub_mesh = mesh[alias]
            assert sub_mesh.mesh_shape == (world,), (
                f"alias '{alias}' wrong shape: expected ({world},), got {sub_mesh.mesh_shape}"
            )
            assert tuple(sub_mesh.rank_list) == tuple(range(world)), (
                f"alias '{alias}' wrong rank list: expected {tuple(range(world))}, "
                f"got {tuple(sub_mesh.rank_list)}"
            )

        # Rank-aware logging on a live PG — _get_rank() must resolve via the
        # platform, not LOCAL_RANK / RANK env vars.
        stream = io.StringIO()
        init_logger(stream=stream, fmt="rank=%(rank)s %(levelname)s %(message)s")
        logger = get_logger("st_trainer_logging")
        logger.setLevel(logging.DEBUG)

        logger.info_rank0("rank0_msg")
        logger.warning_rank0("warn_rank0_msg")
        logger.info("every_rank_msg")

        output = stream.getvalue()
        rank = platform.get_rank()
        assert "every_rank_msg" in output, (
            f"plain logger.info must fire on every rank (rank={rank}), got {output!r}"
        )
        if rank == 0:
            assert "rank0_msg" in output, f"info_rank0 must fire on rank 0, got {output!r}"
            assert "warn_rank0_msg" in output, f"warning_rank0 must fire on rank 0, got {output!r}"
        else:
            assert "rank0_msg" not in output, (
                f"info_rank0 must NOT fire on rank={rank}, got {output!r}"
            )
            assert "warn_rank0_msg" not in output, (
                f"warning_rank0 must NOT fire on rank={rank}, got {output!r}"
            )
    finally:
        destroy_process_group()


def test_parallel_dims_tp_plus_fsdp_mesh():
    """
    Feature: ParallelDims.build_mesh with TP × FSDP composition.
    Description: ``dp_shard=2, tp=2`` on a 4-card group must build a 2D mesh
        whose ``"fsdp"`` axis groups ranks by tp position and whose ``"dp"``
        alias equals ``"fsdp"`` (no replicate axis, no CP).
    Expectation: ``mesh["fsdp"].rank_list`` matches the dp_shard slice for
        this rank; ``mesh["dp"]`` resolves to the same group.
    """
    init_process_group()
    try:
        world = _world_size()
        assert world >= 4, f"This case needs at least 4 cards, got world_size={world}"
        pd = ParallelDims(dp_shard=2, tp=2, world_size=world)
        mesh = pd.build_mesh(platform.device_type())

        rank = platform.get_rank()
        # Canonical order: dp_shard outer, tp inner. So fsdp partners share the same tp coord.
        # With dp_shard=2, tp=2 → mesh shape (2, 2). Ranks {0,1,2,3} laid out:
        #   (dp=0,tp=0)=0  (dp=0,tp=1)=1
        #   (dp=1,tp=0)=2  (dp=1,tp=1)=3
        # fsdp axis groups by tp coord: {0,2} and {1,3}.
        tp_coord = rank % 2
        expected_fsdp_ranks = tuple(r for r in range(4) if r % 2 == tp_coord)

        fsdp_mesh = mesh["fsdp"]
        assert tuple(fsdp_mesh.rank_list) == expected_fsdp_ranks, (
            f"fsdp rank list wrong on rank={rank}: "
            f"expected={expected_fsdp_ranks}, got={tuple(fsdp_mesh.rank_list)}"
        )

        # No replicate axis and no CP → "dp" alias must equal "fsdp"
        dp_mesh = mesh["dp"]
        assert tuple(dp_mesh.rank_list) == expected_fsdp_ranks, (
            f"dp rank list must equal fsdp under pure TP×FSDP: "
            f"expected={expected_fsdp_ranks}, got={tuple(dp_mesh.rank_list)}"
        )
    finally:
        destroy_process_group()


def test_parallel_dims_hsdp_dp_combines_replicate_and_shard():
    """
    Feature: ParallelDims.build_mesh under HSDP.
    Description: ``dp_replicate=2, dp_shard=2`` on 4 cards must register a
        ``"dp"`` alias that combines both axes — exactly the group used for
        loss / token all-reduce.
    Expectation: ``mesh["dp"].rank_list`` covers all 4 ranks on every rank.
    """
    init_process_group()
    try:
        world = _world_size()
        assert world >= 4, f"This case needs at least 4 cards, got world_size={world}"
        pd = ParallelDims(dp_replicate=2, dp_shard=2, world_size=world)
        mesh = pd.build_mesh(platform.device_type())

        dp_mesh = mesh["dp"]
        # ``_register_flatten_aliases`` calls ``flatten()``, which always
        # produces a 1D mesh (see ``DeviceMesh._create_flatten_mesh``); accept
        # only that shape so a regression replacing the flatten path with a
        # raw 2D submesh would be caught here.
        assert dp_mesh.mesh_shape == (4,), (
            f"HSDP dp alias must flatten replicate×shard into a 1D mesh, "
            f"got mesh_shape={dp_mesh.mesh_shape}"
        )
        rank_set = set(dp_mesh.rank_list)
        assert rank_set == {0, 1, 2, 3}, (
            f"HSDP dp alias must cover all 4 ranks, got rank_list={tuple(dp_mesh.rank_list)}"
        )
    finally:
        destroy_process_group()
