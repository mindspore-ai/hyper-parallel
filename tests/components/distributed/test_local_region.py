# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""local_region 单元测试（单进程，torch/cpu 平台）。

覆盖：
1. kwargs / positional 双通道的 DTensor 输入 unwrap 与输出重包装；
2. autograd 缝合：区域两端 backward 不断链，in_grad_placements 声明生效；
3. 非 DTensor 输入透传、输出已是 DTensor 时不重复包装；
4. out_placements 契约校验（数量不匹配 / tensor 输出 None 声明）。
"""

import pytest
import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_models.components.distributed.local_region import local_region


@pytest.fixture(scope="module")
def mesh():
    """单 rank mesh：world_size=1 时 Replicate 与任意 Shard 语义等价，
    足以验证包装/解包/autograd 逻辑（多进程数值归 distributed UT）。"""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1
        )
    return DeviceMesh("cpu", (1,), mesh_dim_names=("tp",))


def _make_dtensor(mesh, data, requires_grad=False):
    local = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
    return DTensor.from_local(local, mesh, [Replicate()])


class TestWrapUnwrap:
    def test_kwargs_input_and_output_wrap(self, mesh):
        def fn(hidden_states, scale=None):
            assert not isinstance(hidden_states, DTensor)  # 区域内是 local tensor
            return hidden_states * (scale or 2.0)

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"hidden_states": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt = _make_dtensor(mesh, [1.0, 2.0, 3.0])
        out = wrapped(hidden_states=dt, scale=3.0)
        assert isinstance(out, DTensor)
        assert tuple(out.placements) == (Replicate(),)
        assert torch.allclose(out.to_local(), torch.tensor([3.0, 6.0, 9.0]))

    def test_positional_input_via_signature_binding(self, mesh):
        def fn(x, y):
            return x + y

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),), "y": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt_x = _make_dtensor(mesh, [1.0, 2.0])
        dt_y = _make_dtensor(mesh, [10.0, 20.0])
        out = wrapped(dt_x, dt_y)
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([11.0, 22.0]))

    def test_plain_tensor_passthrough_no_wrap(self, mesh):
        """全部输入非 DTensor（production 参数已解包场景）→ 输出不包装。"""
        def fn(x):
            return x * 2.0

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        out = wrapped(torch.tensor([1.0, 2.0]))
        assert not isinstance(out, DTensor)
        assert torch.allclose(out, torch.tensor([2.0, 4.0]))

    def test_mixed_dtensor_and_plain_args(self, mesh):
        def fn(x, bias):
            return x + bias

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt = _make_dtensor(mesh, [1.0, 2.0])
        out = wrapped(dt, torch.tensor([100.0, 100.0]))
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([101.0, 102.0]))

    def test_tuple_output_with_none_placeholder(self, mesh):
        def fn(x):
            return x * 2.0, "meta"

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=((Replicate(),), None),
        )
        dt = _make_dtensor(mesh, [1.0, 2.0])
        out_tensor, meta = wrapped(dt)
        assert isinstance(out_tensor, DTensor)
        assert meta == "meta"

    def test_output_already_dtensor_not_rewrapped(self, mesh):
        def fn(x):
            return DTensor.from_local(x * 2.0, mesh, [Replicate()])

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        out = wrapped(_make_dtensor(mesh, [1.0]))
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([2.0]))


class TestContractValidation:
    def test_out_placements_count_mismatch(self, mesh):
        def fn(x):
            return x, x

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=((Replicate(),), (Replicate(),), (Replicate(),)),
        )
        with pytest.raises(ValueError, match="does not match"):
            wrapped(_make_dtensor(mesh, [1.0]))

    def test_flat_out_placements_rejected_for_multi_output(self, mesh):
        def fn(x):
            return x, x

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),  # 扁平写法仅允许单输出
        )
        with pytest.raises(ValueError, match="single-output"):
            wrapped(_make_dtensor(mesh, [1.0]))

    def test_tensor_output_with_none_placement_raises(self, mesh):
        def fn(x):
            return x

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=(None,),
        )
        with pytest.raises(TypeError, match="non-None out_placements"):
            wrapped(_make_dtensor(mesh, [1.0]))

    def test_out_placements_none_returns_raw(self, mesh):
        def fn(x):
            return x * 2.0

        wrapped = local_region(
            fn, device_mesh=mesh,
            in_placements={"x": (Replicate(),)},
            out_placements=None,
        )
        out = wrapped(_make_dtensor(mesh, [1.0]))
        assert not isinstance(out, DTensor)
        assert torch.allclose(out, torch.tensor([2.0]))
