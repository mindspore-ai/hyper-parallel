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

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_reshape import ReshapeDistributedOp

op = ReshapeDistributedOp("Reshape")


def test_reshape_layout_not_change_sharded_axis():
    """
    Feature: Reshape do not change sharded axis
    Description: Reshape do not change sharded axis
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (1024, 512, 512)
    dst_shape = (1024, 2, 256, 512)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
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
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (4, 4, 8)
    dst_shape = (16, 8)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
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
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (32, 128)
    dst_shape = (4, 8, 128)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
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
    base_device_matrix = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("cp", "dp", "None", "mp", "None")
    src_shape = (32, 6, 128, 28, 10)
    dst_shape = (4, 8, 2, 384, 280)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
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
    base_device_matrix = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "None", "mp")
    src_shape = (4, 8, 4, 12)
    dst_shape = (4, 8, 12, 4)

    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_can_not_reshape2():
    """
    Feature: Reshape can not be shared
    Description: Can not be reshaped
    Expectation: Fail
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp", "None")
    src_shape = (4, 8, 12, 7)
    dst_shape = (4, 8, 2, 42)

    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_dynamic_shape1():
    """
    Feature: Reshape parallel op with dynamic shape
    Description: Reshape parallel op with dynamic shape
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
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
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
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
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
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
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    src_shape = (-1, 512)
    dst_shape = (-1, 256, 512)

    with pytest.raises(ValueError):
        _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))


def test_reshape_layout_axis_shard_twice():
    """
    Feature: Reshape split, merge, resize axes
    Description: Reshape split, merge, resize axes
    Expectation: Success
    """
    base_device_matrix = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout(("cp", "dp"), "None", "None", "mp", "None")
    src_shape = (32, 6, 128, 28, 10)
    dst_shape = (4, 8, 2, 384, 280)

    output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
    expected_map = ((0, 2), -1, -1, -1, 1)  # Expected output tensor map
    assert output_layout.tensor_map == expected_map, (f"Reshape do not change sharded axis failed. Expected  "
                                                      f"expected_map {expected_map} bug got {output_layout.tensor_map}")
    expected_local_dst_shape = [1, 8, 2, 384, 140]
    assert local_dst_shape == expected_local_dst_shape, (f"Reshape do not change sharded axis failed. Expected"
                                                         f" expected_local_dst_shape {expected_local_dst_shape} got"
                                                         f" {local_dst_shape}")
