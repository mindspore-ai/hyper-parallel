# Copyright 2025 Huawei Technologies Co., Ltd
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
"""parallel_reshape test"""

import pytest
from hyper_parallel import Layout

from hyper_parallel.core.shard.ops.parallel_reshape import ReshapeDistributedOp


op = ReshapeDistributedOp("Reshape")

def test_reshape_layout_not_change_sharded_axis():
    """
    Feature: Reshape do not change sharded axis
    Description: Reshape do not change sharded axis
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (1024, 512, 512)
    dst_shape = (1024, 2, 256, 512)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    # dp(1) maps to mesh dim 0. mp(0) maps to mesh dim 1.
    # Input dim 0 is sharded by dp (1).
    # Output dim 0 corresponds to Input dim 0.
    expected_map = (1, -1, -1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [512, 2, 256, 512]  # Expected local dst shape
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_merge_sharded_axis():
    """
    Feature: Reshape merge shared axis with not shared axis
    Description: Reshape merge shared axis with not shared axis
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (4, 4, 8)
    dst_shape = (16, 8)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    # Input dim 0 (4) sharded by dp(1).
    # Output dim 0 (16) = Input dim 0 (4) * Input dim 1 (4).
    # Sharding is preserved on the major dimension of the merge, so Output dim 0 gets dp(1).
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [8, 8]
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_split_sharded_axis():
    """
    Feature: Reshape split shared asix
    Description: Reshape do not change sharded axis
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # FIX: Input shape is (32, 128) - 2 dims. Layout should set 2 dims, not 3.
    x_layout = x_layout("dp", "None")
    src_shape = (32, 128)
    dst_shape = (4, 8, 128)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    # Input dim 0 (32) sharded by dp(1).
    # Output: 32 -> 4 * 8.
    # dp(size 2) splits 32 into [0, 16), [16, 32).
    # Dst dim 0 (4) splits 32 into [0, 8), [8, 16), [16, 24), [24, 32).
    # dp=0 (0..15) covers Dst dim 0 indices 0, 1.
    # dp=1 (16..31) covers Dst dim 0 indices 2, 3.
    # So Dst dim 0 is sharded by dp(1).
    expected_map = (1, -1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [2, 8, 128]
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_multi_axes_shared():
    """
    Feature: Reshape split, merge, resize axes
    Description: Reshape split, merge, resize axes
    Expectation: Success
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    # Layout mapping: cp->0, dp->2, mp->1
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Tensor map: (0, 2, -1, 1, -1)
    x_layout = x_layout("cp", "dp", "None", "mp", "None")
    src_shape = (32, 6, 128, 28, 10)
    dst_shape = (4, 8, 2, 384, 280)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    # Dst dim 0 (4) comes from Src dim 0 (32, sharded by cp). Sharded by cp(0).
    # Dst dim 2 (2) comes from Src dim 1 (6, sharded by dp). Sharded by dp(2).
    # Dst dim 4 (280) comes from Src dim 3 (28, sharded by mp) merged with dim 4. Sharded by mp(1).
    expected_map = (0, -1, 2, -1, 1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [2, 8, 1, 384, 140]
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected"
                                                         f" expected_local_dst_shape {expected_local_dst_shape} got"
                                                         f" {local_dst_shape}")


def test_reshape_layout_can_not_reshape1():
    """
    Feature: Reshape can not be shared
    Description: Can not be reshaped
    Expectation: Fail
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "None", "mp")
    src_shape = (4, 8, 4, 12)
    dst_shape = (4, 8, 12, 4)

    # Sharding on minor dimension (d3) which is being transposed/mixed with d2. Requires redistribution.
    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_can_not_reshape2():
    """
    Feature: Reshape can not be shared
    Description: Can not be reshaped
    Expectation: Fail
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp", "None")
    src_shape = (4, 8, 12, 7)
    dst_shape = (4, 8, 2, 42)

    # Src d2 (12) sharded by mp (size 4). Local size 3.
    # Dst D2 (2). Size is smaller than mesh size (4). Cannot shard D2 on mp (4) without partial blocks.
    # Requires redistribution or complex mapping not supported by standard tensor_map.
    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_dynamic_shape1():
    """
    Feature: Reshape parallel op with dynamic shape
    Description: Reshape parallel op with dynamic shape
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None", "None")
    src_shape = (1024, -1, 256, 512)
    dst_shape = (1024, -1, 512)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    expected_map = (1, -1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [512, -1, 512]  # Expected local dst shape
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_dynamic_shape2():
    """
    Feature: Reshape parallel op with dynamic shape
    Description: Reshape parallel op with dynamic shape
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (1024, -1, 512)
    dst_shape = (1024, -1, 256, 512)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    expected_map = (1, -1, -1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [512, -1, 256, 512]  # Expected local dst shape
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_dynamic_shape3():
    """
    Feature: Reshape parallel op with dynamic shape
    Description: Reshape parallel op with dynamic shape
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (-1, 256, 512)
    dst_shape = (-1, 512)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [-1, 512]  # Expected local dst shape
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected "
                                                         f"expected_local_dst_shape {expected_local_dst_shape} got "
                                                         f"{local_dst_shape}")


def test_reshape_layout_dynamic_shape4():
    """
    Feature: Reshape parallel op with dynamic shape
    Description: Reshape parallel op with dynamic shape
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (-1, 512)
    dst_shape = (-1, 256, 512)

    # Splitting a dynamic dimension that is sharded. Alignment cannot be guaranteed.
    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_axis_shard_twice():
    """
    Feature: Reshape split, merge, resize axes
    Description: Reshape split, merge, resize axes
    Expectation: Success
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout(("cp", "dp"), "None", "None", "mp", "None")
    src_shape = (32, 6, 128, 28, 10)
    dst_shape = (4, 8, 2, 384, 280)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    # Dst dim 0 (4) inherits sharding from Src dim 0 (32) which is ("cp", "dp").
    # cp(0) and dp(2) both shard dim 0.
    expected_map = ((0, 2), -1, -1, -1, 1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [1, 8, 2, 384, 140]
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected"
                                                         f" expected_local_dst_shape {expected_local_dst_shape} got"
                                                         f" {local_dst_shape}")


op_torch = ReshapeDistributedOp("reshape")

def test_torch_reshape_basic_split():
    """
    Feature: PyTorch style reshape split sharded axis
    Description: Test splitting a sharded dimension into multiple dimensions (preserving layout)
    Expectation: Success
    """
    # Mesh: (2, 4), Layout: dp(0), mp(1)
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Input Layout: [dp, mp] -> [2, 4] split
    x_layout = x_layout("dp", "mp")

    # Input: [32, 64]
    src_shape = (32, 64)
    # Reshape to: [32, 16, 4] -> We split the second dim (64) which is sharded by 'mp'
    dst_shape = (32, 16, 4)

    # Note: reshapeDistributedOp requires (dst_shape, src_shape) in extra_args
    output_layout, local_dst_shape = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))

    # Logic analysis:
    # Merged Input Blocks: [32 (dp)], [64 (mp)]
    # Reverse Match Dst:
    # 1. dst_dim=4: 64 % 4 == 0 -> remainder 16. Consumed part of 'mp' block. Map: -1 (unsharded inner part)
    # 2. dst_dim=16: 16 % 16 == 0 -> remainder 1. Consumed rest of 'mp' block. Map: mp (inherits sharding)
    # 3. dst_dim=32: 32 % 32 == 0 -> remainder 1. Consumed 'dp' block. Map: dp

    expected_map = (1, 0, -1) # (dp, mp, None) -> based on mesh indices: dp=1 (mesh[0]), mp=0 (mesh[1])?
    # Wait, check Layout def in test file:
    # base_mesh_shape = (2, 4) -> index 0 is size 2, index 1 is size 4.
    # In x_layout("dp", "mp"):
    # "dp" maps to mesh axis 0? Usually aliases are mapped sequentially or by dict.
    # Assuming standard Layout behavior:
    # If "dp" is alias[0], "mp" is alias[1].
    # map value: device_matrix_rank - 1 - axis_index.
    # Typically: index 0 (dp) -> map 1, index 1 (mp) -> map 0 (if using reversed mapping common in Ascend/MindSpore tensor map).
    # Let's trust the existing test convention: expected_map = (1, -1) implies 1 is dp.


    assert output_layout.tensor_map == expected_map, \
        f"Expected {expected_map}, got {output_layout.tensor_map}"

    # Local shape:
    # dim 0 (32): sharded by dp(2) -> 16
    # dim 1 (16): sharded by mp(4) -> 4
    # dim 2 (4): not sharded -> 4
    expected_local_dst_shape = [16, 4, 4]
    assert local_dst_shape == expected_local_dst_shape, \
        f"Expected local shape {expected_local_dst_shape}, got {local_dst_shape}"


def test_torch_reshape_dynamic_target():
    """
    Feature: PyTorch style reshape with -1 in target
    Description: Test passing -1 in destination shape to infer dimension automatically
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp") # map: (1, 0)

    src_shape = (32, 64)
    # Reshape: Merge dims and flatten, then split? Or just change view.
    # Let's try: [32, -1] -> implies [32, 64]
    dst_shape = (32, -1)

    output_layout, local_dst_shape = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))

    # Logic: _handle_dynamic_shape converts dst to [32, 64]
    expected_map = (1, 0)
    assert output_layout.tensor_map == expected_map

    # Local shape should handle the resolved shape
    expected_local_dst_shape = [16, 16] # 32/2, 64/4
    assert local_dst_shape == expected_local_dst_shape


def test_torch_reshape_flatten_unsharded():
    """
    Feature: PyTorch style reshape merge unsharded dims
    Description: Merging an unsharded dimension into a sharded one is NOT fully supported if it changes sharding cuts,
                 but merging unsharded dims into other unsharded dims or preserving blocks works.
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Layout: dp, None, mp -> [1, -1, 0]
    x_layout = x_layout("dp", "None", "mp")

    src_shape = (16, 4, 8)
    # We want to reshape to (16, 32). Merging dim 1 (4) and dim 2 (8).
    # Since dim 1 is -1 (unsharded) and dim 2 is mp (sharded),
    # `_merge_unshared_axis` logic:
    # Iterate backwards:
    # Ax 2 (8, mp): New block [8], map 0.
    # Ax 1 (4, -1): Merged into Ax 0? No, `_merge_unshared_axis` merges -1 into the PREVIOUS shared axis (from the left logic)
    # OR it effectively accumulates size.
    # In your code:
    # Loop reversed.
    # Ax 2 (map 0): insert(8), insert(0). cur_vol reset to 1.
    # Ax 1 (map -1): cur_vol *= 4.
    # Ax 0 (map 1): insert(16*4), insert(1).
    # Merged Result: [64, 8] with maps [1, 0].
    # Wait, `_merge_unshared_axis` logic in your code:
    # if tensor_map[axis] != -1: insert(merged_size)...
    # So for (16, 4, 8) with (1, -1, 0):
    # i=2: map=0. insert(8), insert(0). vol=1.
    # i=1: map=-1. vol=1*4=4.
    # i=0: map=1. insert(16*4), insert(1). vol=1.
    # Result: Shape [64, 8], Map [1, 0].

    # Target: (16, 32)
    # Match reversed:
    # dst=32. cur=8 (map 0). 8 % 32 != 0.
    # ERROR condition in your code: if cur_size % shape != 0 -> raise.
    # So (16, 4, 8) -> (16, 32) is IMPOSSIBLE with this logic because the cut 'mp' is at size 8 boundary.
    # You can't have a dimension of size 32 where the first 8 are sharded one way and the rest another way.

    # Let's try a valid case: Reshape (16, 4, 8) -> (64, 8)
    # This matches the merged blocks exactly.
    dst_shape = (64, 8)

    output_layout, _ = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))
    assert output_layout.tensor_map == (1, 0)

    # Let's try Reshape (16, 4, 8) -> (16, 2, 2, 8)
    # Valid split of unsharded dim
    dst_shape_2 = (16, 2, 2, 8)
    output_layout_2, _ = op_torch.infer_layout((x_layout,), (dst_shape_2, src_shape))
    # 8 matches 8 (map 0).
    # 2 matches part of 64?
    # Wait, merged block is 64 (map 1).
    # 64 % 2 == 0 -> rem 32. map -1.
    # 32 % 2 == 0 -> rem 16. map -1.
    # 16 % 16 == 0 -> rem 1. map 1.
    # Result map: (1, -1, -1, 0).
    assert output_layout_2.tensor_map == (1, -1, -1, 0)


def test_torch_reshape_fail_missing_input_shape():
    """
    Feature: PyTorch style reshape exception
    Description: Verify that failing to provide input shape raises ValueError
    Expectation: ValueError
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")

    dst_shape = (32, 64)

    # Intentionally omitted src_shape
    with pytest.raises(ValueError, match="reshape requires output shape and input shape."):
        op_torch.infer_layout((x_layout,), (dst_shape,))


def test_torch_reshape_fail_mismatch_total_size():
    """
    Feature: PyTorch style reshape validation
    Description: Verify shape element count mismatch validation
    Expectation: ValueError
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")

    src_shape = (10, 10)
    dst_shape = (20, 20) # 100 != 400

    with pytest.raises(ValueError):
        op_torch.infer_layout((x_layout,), (dst_shape, src_shape))


op_view = ReshapeDistributedOp("view")

def test_view_layout_flatten_contiguous():
    """
    Feature: View operator flattening
    Description: Test flattening multiple dimensions where the inner dimension is not sharded.
                 Common case: x.view(batch_size, -1)
    Expectation: Success, preserving the sharding of the split point.
    """
    # Mesh: (2, 4), Layout: dp(0), mp(1)
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Input: [batch, seq, hidden] -> [32, 16, 8]
    # Layout: dp, mp, None -> Tensor Map: (1, 0, -1)
    x_layout = x_layout("dp", "mp", "None")

    src_shape = (32, 16, 8)
    # Target: Flatten the last two dimensions -> [32, 128]
    dst_shape = (32, 128)


    output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

    expected_map = (1, 0) # dp, mp
    assert output_layout.tensor_map == expected_map, \
        f"View flatten failed. Expected {expected_map}, got {output_layout.tensor_map}"

    expected_local_dst_shape = [16, 32] # 32/2=16, 128/4=32
    assert local_dst_shape == expected_local_dst_shape


def test_view_layout_unflatten_split():
    """
    Feature: View operator unflattening (inverse of flatten)
    Description: Test expanding a sharded dimension into two, where the outer part keeps sharding
                 and the inner part becomes None (unsharded).
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Input: [32, 128]
    # Layout: dp, mp -> Tensor Map: (1, 0)
    x_layout = x_layout("dp", "mp")

    src_shape = (32, 128)
    # Target: Unflatten to [32, 16, 8]
    # We are splitting the 128 (mp) dimension into 16 and 8.
    dst_shape = (32, 16, 8)


    output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

    expected_map = (1, 0, -1)
    assert output_layout.tensor_map == expected_map, \
        f"View unflatten failed. Expected {expected_map}, got {output_layout.tensor_map}"

    # Local Shape:
    # 32 (dp=2) -> 16
    # 16 (mp=4) -> 4
    # 8 (None) -> 8
    expected_local_dst_shape = [16, 4, 8]
    assert local_dst_shape == expected_local_dst_shape


def test_view_layout_dynamic_shape_inference():
    """
    Feature: View operator with -1 inference
    Description: Support x.view(-1, C) style calls similar to PyTorch
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    # Layout: dp, mp, None -> (1, 0, -1)
    x_layout = x_layout("dp", "mp", "None")

    src_shape = (8, 4, 16) # Total 512
    # Target: [8, -1]. Should infer to [8, 64]
    dst_shape = (8, -1)

    output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

    # Logic:
    # Merged input: [8, 64] with Map [1, 0] (since 4(mp) and 16(None) merge into 64(mp))
    # Target inferred: [8, 64]
    # Match: Exact match.

    expected_map = (1, 0)
    assert output_layout.tensor_map == expected_map

    # Local: 8/2=4, 64/4=16
    expected_local_dst_shape = [4, 16]
    assert local_dst_shape == expected_local_dst_shape


def test_view_layout_fail_shape_mismatch():
    """
    Feature: View operator validation
    Description: Ensure view raises error if total elements don't match (PyTorch behavior)
    Expectation: ValueError
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")

    src_shape = (10, 20)
    dst_shape = (10, 30) # 200 != 300

    with pytest.raises(ValueError, match="total elements number"):
        op_view.infer_layout((x_layout,), (dst_shape, src_shape))
