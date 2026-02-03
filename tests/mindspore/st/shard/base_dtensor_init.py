"""
MindSpore distributed tensor initialization tests.
"""
import mindspore
from mindspore.communication import get_rank, get_group_size
import mindspore.communication.management as D
from hyper_parallel.core.device_mesh import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor import ones, empty, full, zeros

D.init()

def build_device_mesh():
    world_size = get_group_size()
    return init_device_mesh("npu", (world_size,), mesh_dim_names=("dp",))

def assert_equal(x,y):
    assert (x == y).all()

def _run_init_op(init_op, dist_init_op, eq_op, *args, **kwargs):
    """
        Run initialization operation and verify results across ranks.

        Args:
            init_op: Initialization function to test.
            *args: Positional arguments for init_op.
            **kwargs: Keyword arguments for init_op.
    """
    world_size = get_group_size()
    rank = get_rank()
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
                mindspore.mint.chunk(
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
        mindspore.mint.ones,
        ones,
        assert_equal,
    )

def test_empty():
    _run_init_op(
        mindspore.mint.empty,
        empty,
        lambda x, y: (x.shape == y.shape)
                     and (x.dtype == y.dtype)
    )

def test_full():
    _run_init_op(
        mindspore.mint.full,
        full,
        assert_equal,
        123.4,
    )

def test_zeros():
    _run_init_op(
        mindspore.mint.zeros,
        zeros,
        assert_equal,
    )
