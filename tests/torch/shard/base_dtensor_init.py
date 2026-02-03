"""Distributed tensor initialization consistency tests."""
import torch
import torch.distributed as dist
from hyper_parallel.core.device_mesh import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor import ones, empty, full, zeros
from tests.torch.utils import init_dist

init_dist()

def build_device_mesh():
    world_size = dist.get_world_size()
    return init_device_mesh("npu", (world_size,), mesh_dim_names=("dp",))

def assert_equal(x,y):
    assert (x == y).all()

def _run_init_op(init_op, dist_init_op, eq_op, *args, **kwargs):
    """
        Run init operation and verify DP group consistency.

        Args:
            init_op: Initialization function to test.
            *args: Positional args for init_op.
            **kwargs: Keyword args for init_op.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # 1d mesh test
    device_mesh = build_device_mesh()
    placements_list = [[Shard(0)], [Shard(1)], [Shard(2)], [Replicate()]]

    # even sharding
    tensor_size = [4, 8, 12]
    for placements in placements_list:
        local_tensor_size = tensor_size.copy()
        if isinstance(placements[0], Shard):
            shard_dim = placements[0].dim
            local_tensor_size[shard_dim] //= world_size

        dist_tensor = dist_init_op(
            tensor_size,
            *args,
            **kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )
        ones_expected = init_op(local_tensor_size, *args, **kwargs)
        eq_op(ones_expected, dist_tensor.to_local())

    # uneven sharding
    tensor_size = [5, 10, 15]
    for placements in placements_list:
        dist_tensor = dist_init_op(
            tensor_size,
            *args,
            **kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )
        if isinstance(placements[0], Shard):
            shard_dim = placements[0].dim
            exp_tensor_list = list(
                torch.chunk(
                    init_op(tensor_size, *args, **kwargs),
                    world_size,
                    dim=shard_dim,
                )
            )

            def check_per_rank_chunk(rank, local_tensor, tensor_list):
                if rank < len(tensor_list):
                    eq_op(tensor_list[rank], local_tensor)

            check_per_rank_chunk(rank, dist_tensor.to_local(), exp_tensor_list)
        else:
            exp_tensor = init_op(tensor_size, *args, **kwargs)
            eq_op(exp_tensor, dist_tensor.to_local())

    # empty shape
    local_tensor = dist_init_op(
        [], *args, **kwargs, device_mesh=device_mesh, placements=[Replicate()]
    ).to_local()
    expected_tensor = init_op([], *args, **kwargs)
    eq_op(expected_tensor, local_tensor)

def test_ones():
    _run_init_op(
        torch.ones,
        ones,
        assert_equal,
    )

def test_empty():
    _run_init_op(
        torch.empty,
        empty,
        lambda x, y: (x.shape == y.shape)
                     and (x.dtype == y.dtype)
                     and (x.layout == y.layout),
    )

def test_full():
    _run_init_op(
        torch.full,
        full,
        assert_equal,
        123.4,
    )

def test_zeros():
    _run_init_op(
        torch.zeros,
        zeros,
        assert_equal,
    )
