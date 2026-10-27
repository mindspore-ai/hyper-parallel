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
"""MoonEP规划、显式GMM反向和真实非重入重算的CPU回归。"""

import inspect
import itertools
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.expert_parallel.moonep import (
    MoonEPExecutor, configure_moonep, wrap_moonep_checkpoint,
)
from hyper_parallel.distributed.expert_parallel.moonep.capacity import solve_capacity_exact
from hyper_parallel.distributed.expert_parallel.moonep.candidates import (
    device_candidate, device_candidates, device_plan_metrics, device_plan_metrics_batch,
)
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import checkpoint_route_ids
from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import MoonEPGradientExtension
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    expert_backward_values, expert_forward_values, grouped_values, grouped_weight_grad,
)
from hyper_parallel.distributed.expert_parallel.moonep import planning
from hyper_parallel.distributed.expert_parallel.moonep import runtime as moonep_runtime
from hyper_parallel.distributed.expert_parallel.moonep.planning import compile_plan
from tests.common.mark_utils import arg_mark


class _Experts(torch.nn.Module):
    """真实参数边界的微型专家；例如两组8×8矩阵。"""

    def __init__(self):
        """随机初始化，不从文件加载；例如每个专家具有gate/up/down三个投影。"""
        super().__init__()
        self.gate = torch.nn.Parameter(torch.randn(2, 8, 16) * 0.1)
        self.down = torch.nn.Parameter(torch.randn(2, 8, 8) * 0.1)
        self._moonep_executor = MoonEPExecutor(
            self, lambda: (self.gate, self.down), ep_group=None, ep_size=1, num_experts=2, top_k=2,
        )

    def forward(self, x, ids, scores):
        """进入实际模块边界；例如5个token展开为10条选择。"""
        return self._moonep_executor(x, ids, scores)


class _Block(torch.nn.Module):
    """一个checkpoint区域只有一次专家调用；例如三层顺序模型中的中间层。"""

    def __init__(self):
        """建立router、专家和输出线性；例如所有参数都需要训练。"""
        super().__init__()
        self.router = torch.nn.Linear(8, 2)
        self.experts = _Experts()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        """可微路由与残差；例如重放使用相同ID但重新计算score梯度。"""
        probability = self.router(x).softmax(-1)
        ids = checkpoint_route_ids(self.router, probability, 2)
        return self.linear(self.experts(x, ids, probability.gather(-1, ids))) + x


class TestMoonEPLifecycle(unittest.TestCase):
    """不初始化分布式、不依赖NPU，检查真实autograd与checkpoint。"""

    def test_executor_has_single_planner_path(self):
        """执行器只使用 CPU 小表规划；例如不再接受后端和规划模式开关。"""
        expert = _Experts()
        self.assertFalse(hasattr(expert._moonep_executor, 'planner_backend'))  # pylint: disable=protected-access
        parameters = inspect.signature(MoonEPExecutor).parameters
        for removed in ('planner_backend', 'planning', 'planner_overlap_sort'):
            self.assertNotIn(removed, parameters)

    def test_mesh_ep_group_is_reused_for_replica_communication(self):
        """三个 EP 用途共用 mesh 组；例如权重和梯度都引用 token 的同一个组。"""
        expert = _Experts()
        mesh_group = object()
        with mock.patch.object(moonep_runtime.dist, 'get_rank', return_value=0):
            expert._moonep_executor = MoonEPExecutor(  # pylint: disable=protected-access
                expert, lambda: (expert.gate, expert.down), ep_group=mesh_group,
                ep_size=2, num_experts=2, top_k=2,
            )
        with mock.patch.object(moonep_runtime.dist, 'new_group', side_effect=AssertionError('重复建组')), \
                mock.patch.object(moonep_runtime.dist, 'all_gather_object',
                                  side_effect=AssertionError('不应交换建组元数据')):
            configure_moonep(expert)
        executor = expert._moonep_executor  # pylint: disable=protected-access
        self.assertIs(executor.weight_group, mesh_group)
        self.assertIs(executor.gradient_group, mesh_group)

    def test_fsdp_prefetch_keeps_moonep_expert_units(self):
        """原生 FSDP 链保留专家；例如 L0 预取 L1，L2 反向预取 L1。"""
        units = [SimpleNamespace(set_modules_to_forward_prefetch=mock.Mock(),
                                 set_modules_to_backward_prefetch=mock.Mock()) for _ in range(5)]
        units[1]._moonep_executor = object()
        units[3]._moonep_executor = object()
        manager = SimpleNamespace(config=SimpleNamespace(forward_prefetch_depth=1,
                                                         backward_prefetch_depth=1))
        FSDP2Manager._configure_prefetch(manager, units)  # pylint: disable=protected-access
        for index, unit in enumerate(units):
            unit.set_modules_to_forward_prefetch.assert_called_once_with(units[index + 1:index + 2])
            unit.set_modules_to_backward_prefetch.assert_called_once_with(units[max(0, index - 1):index])

    def test_ag2_reaches_previous_expert_from_current_expert(self):
        """AG2 在 E7 入口已预取 E6；例如列表顺序是 D7、E6。"""
        units = [SimpleNamespace(set_modules_to_forward_prefetch=mock.Mock(),
                                 set_modules_to_backward_prefetch=mock.Mock()) for _ in range(4)]
        dense6, expert6, dense7, expert7 = units
        manager = SimpleNamespace(config=SimpleNamespace(forward_prefetch_depth=1,
                                                         backward_prefetch_depth=2))
        FSDP2Manager._configure_prefetch(manager, units)  # pylint: disable=protected-access
        expert7.set_modules_to_backward_prefetch.assert_called_once_with([dense7, expert6])
        dense7.set_modules_to_backward_prefetch.assert_called_once_with([expert6, dense6])

    def test_native_backward_prefetch_submits_ep_extension_after_ag(self):
        """保留原生 AG 并按顺序接 EP；例如 B7 的预取先提交 B6 AG 再执行扩展。"""
        order = []
        target = SimpleNamespace(
            hsdp_state=SimpleNamespace(module_name='expert', prefetch=lambda: order.append('efs_ag')),
            _all_gather_extension=SimpleNamespace(
                fsdp_after_backward_prefetch=lambda _: order.append('ep_alltoallv')),
        )
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.hsdp_state = SimpleNamespace(module_name='current')
        scheduler.reshard_after_forward = False
        scheduler.backward_prefetch_cells = [SimpleNamespace(hsdp_scheduler=target)]
        scheduler._hsdp_backward_pre_hook(None, None)
        self.assertEqual(order, ['efs_ag', 'ep_alltoallv'])

    def test_dense_prefetch_extends_every_expert_ag(self):
        """FSDP 列表中的专家都迁移；例如 D7 提交 E6/D6/E5 AG 后接续 E6/E5 EP。"""
        order = []
        targets = []
        for layer in ('expert_6', 'dense_6', 'expert_5'):
            extension = (SimpleNamespace(
                fsdp_after_backward_prefetch=lambda _, layer=layer: order.append(f'ep_{layer}'))
                         if layer.startswith('expert') else None)
            target = SimpleNamespace(
                hsdp_state=SimpleNamespace(module_name=layer,
                                           prefetch=lambda layer=layer: order.append(f'ag_{layer}')),
                _all_gather_extension=extension,
            )
            targets.append(SimpleNamespace(hsdp_scheduler=target))
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.hsdp_state = SimpleNamespace(module_name='dense_7')
        scheduler.reshard_after_forward = False
        scheduler.backward_prefetch_cells = targets
        scheduler._hsdp_backward_pre_hook(None, None)
        self.assertEqual(order, ['ag_expert_6', 'ag_dense_6', 'ag_expert_5',
                                 'ep_expert_6', 'ep_expert_5'])

    def test_fsdp_prefetch_depth_does_not_reserve_moonep_slots(self):
        """专家 AG hook 决定预取时机；例如 FSDP 列表有三项也只按实际请求建槽。"""
        expert = _Experts()
        # 模拟已包装的 FSDP；本测试只验证配置与槽分配，不执行真实通信。
        expert.hsdp_scheduler = SimpleNamespace(backward_prefetch_cells=[object()] * 3)
        with mock.patch.object(moonep_runtime.MoonEPRuntime, '_configure_groups'):
            runtime = configure_moonep(expert)
        self.assertFalse(runtime.standalone_prefetch)
        with mock.patch.object(runtime, '_record_all_gather_workspace_demand'):
            runtime._ensure_slots(expert.gate, expert.down)
        self.assertEqual(len(runtime.slots), 1)
        with self.assertRaisesRegex(ValueError, '由 AG hook 触发'):
            other = _Experts()
            other.hsdp_scheduler = SimpleNamespace(backward_prefetch_cells=[object()] * 3)
            configure_moonep(other, prefetch_depth=3)

    def test_fsdp_waits_for_replica_gradients_before_reduce_scatter(self):
        """扩展归并先于 FSDP 原生归约；例如 B7 的 e0 回传先完成再提交 RS。"""
        order = []
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.hsdp_state = SimpleNamespace(module_name='experts', post_backward=lambda: order.append('rs'))
        scheduler._gradient_extension = SimpleNamespace(
            fsdp_post_gradient_reduce=lambda _: order.append('ep_grad_merge'),
        )
        scheduler._fsdp_group_post_pending = None
        scheduler._hsdp_backward_hook(None, None, None)
        self.assertEqual(order, ['ep_grad_merge', 'rs'])

    def test_gradient_extension_waits_only_before_fsdp_reduce(self):
        """先发起 BF16 副本回传，参数累积后再等待；例如转置 gate 的第 1 行回 home。"""
        gate_param = torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.bfloat16))
        down_param = torch.nn.Parameter(torch.zeros(2, 2, 2, dtype=torch.bfloat16))
        executor = SimpleNamespace(
            read_weights=lambda: (gate_param.transpose(1, 2), down_param), gradient_group=object(),
        )
        runtime = configure_moonep(_Experts())
        extension = MoonEPGradientExtension(executor, runtime)
        scheduler = SimpleNamespace(hsdp_state=SimpleNamespace(hsdp_params=[
            SimpleNamespace(unsharded_param=gate_param), SimpleNamespace(unsharded_param=down_param),
        ]))
        plan = SimpleNamespace(
            home_count=2, home_positions=torch.tensor([0]), home_rows=torch.tensor([0]),
            remote_positions=torch.tensor([1]), weight_send=[1], weight_recv=[1],
            send_ids=torch.tensor([1]), any_migration=True,
        )
        gate_grad = torch.arange(16, dtype=torch.bfloat16).reshape(2, 2, 4)
        down_grad = torch.arange(8, dtype=torch.bfloat16).reshape(2, 2, 2)
        waits = []

        def exchange(output, source, *_args, **_kwargs):
            output.copy_(source)
            return SimpleNamespace(wait=lambda: waits.append(True))

        with mock.patch.object(moonep_runtime.dist, 'all_to_all_single', side_effect=exchange):
            local_gate, local_down = extension.fsdp_pre_gradient_reduce(
                scheduler, (gate_grad, down_grad, plan),
            )
            self.assertEqual(waits, [])
            gate_param.grad = local_gate.transpose(1, 2).contiguous()
            down_param.grad = local_down.clone()
            extension.fsdp_post_gradient_reduce(scheduler)
        self.assertEqual(len(waits), 1)
        self.assertEqual(extension.pending, [])
        torch.testing.assert_close(gate_param.grad[1], gate_grad[1].transpose(0, 1))
        torch.testing.assert_close(down_param.grad[1], down_grad[1])
        torch.testing.assert_close(gate_param.grad[0], gate_grad[0].transpose(0, 1))

    def test_weight_workspace_is_bounded_and_reused(self):
        """AG与EP按需共池且增长不破坏旧视图；例如预算8元素时9元素不能扩池。"""
        expert = _Experts()
        runtime = configure_moonep(expert)
        template = torch.empty(1, dtype=torch.bfloat16)
        first = runtime._borrow_weight_workspace("send", template, 4, 8)
        first.fill_(3)
        self.assertEqual(runtime.weight_workspace["send"].numel(), 4)
        second = runtime._borrow_weight_workspace("send", template, 3, 8)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(second, torch.full((3,), 3, dtype=torch.bfloat16))
        grown = runtime._borrow_weight_workspace("send", template, 7, 8)
        self.assertNotEqual(grown.data_ptr(), first.data_ptr())
        self.assertEqual(runtime.weight_workspace["send"].numel(), 8)
        torch.testing.assert_close(first, torch.full((4,), 3, dtype=torch.bfloat16))
        oversized = runtime._borrow_weight_workspace("send", template, 9, 8)
        self.assertNotEqual(oversized.data_ptr(), grown.data_ptr())
        self.assertEqual(runtime.weight_workspace["send"].numel(), 8)
        empty = runtime._borrow_weight_workspace("send", template, 0, 8)
        self.assertEqual(empty.numel(), 0)
        received = runtime._borrow_weight_workspace("recv", template, 2, 8)
        self.assertEqual(runtime.weight_workspace["recv"].numel(), 2)
        self.assertNotEqual(received.untyped_storage().data_ptr(), grown.untyped_storage().data_ptr())
        runtime.finish_step()
        self.assertEqual(runtime.weight_workspace, {})

    def test_workspace_demand_survives_without_retaining_tensor(self):
        """跨步只保存实际需求；例如上步增长到8元素，需求7使下步直接申请7。"""
        runtime = configure_moonep(_Experts())
        template = torch.empty(1, dtype=torch.bfloat16)
        runtime._borrow_weight_workspace("send", template, 4, 8)
        old = runtime._borrow_weight_workspace("send", template, 7, 8)
        old.fill_(5)
        runtime.finish_step()
        self.assertEqual(runtime.weight_workspace, {})
        self.assertEqual(runtime.weight_workspace_demand, {"send": 7})
        runtime._borrow_weight_workspace("send", template, 2, 8)
        self.assertEqual(runtime.weight_workspace["send"].numel(), 7)
        torch.testing.assert_close(old, torch.full((7,), 5, dtype=torch.bfloat16))
        runtime._borrow_weight_workspace("send", template, 9, 8)
        self.assertEqual(runtime.weight_workspace_demand["send"], 7)

    def test_workspace_ag_requirement_prevents_backward_growth(self):
        """已知AG需求参与首次分配；例如前向只发2元素，反向需6元素而预算8。"""
        runtime = configure_moonep(_Experts())
        runtime.weight_workspace_demand["send"] = 6
        template = torch.empty(1, dtype=torch.bfloat16)
        first = runtime._borrow_weight_workspace("send", template, 2, 8)
        first.fill_(3)
        backward = runtime._borrow_weight_workspace("send", template, 6, 8)
        self.assertEqual(first.data_ptr(), backward.data_ptr())
        self.assertEqual(runtime.statistics["weight_workspace_allocations"], 1)
        torch.testing.assert_close(backward[:2], torch.full((2,), 3, dtype=torch.bfloat16))

    def test_ag_demand_is_read_after_lazy_dtype_initialization(self):
        """首个专家前向读取已初始化dtype；例如构造时尚未知的128元素在建槽时生效。"""
        expert = _Experts()
        runtime = configure_moonep(expert)
        parameter = SimpleNamespace(shard_world_size=2, hsdp_placement=SimpleNamespace(dim=0),
                                    all_gather_process_group=object(), offload_to_cpu=False,
                                    param_dtype=torch.bfloat16, orig_dtype=torch.float32,
                                    padded_sharded_param_size=torch.Size((1, 8, 16)))
        expert.hsdp_scheduler = SimpleNamespace(hsdp_state=SimpleNamespace(
            comm_fusion_policy=SimpleNamespace(enable_comm_fusion=False), hsdp_params=[parameter]))
        self.assertEqual(runtime.weight_workspace_demand, {})
        runtime._ensure_slots(expert.gate.bfloat16(), expert.down.bfloat16())
        self.assertEqual(runtime.weight_workspace_demand, {"send": 128})

    def test_balanced_candidates_do_not_create_remote_quotas(self):
        """省去循环零盈余判断仍保持平衡；例如全零或各卡等负载时没有迁移。"""
        for ranks in (1, 2, 4, 8):
            for count in (0, 3):
                totals = torch.full((ranks * 2,), count, dtype=torch.int64)
                actual = device_candidates(totals, ranks)
                expected = torch.zeros((ranks * 2, ranks), dtype=torch.int64)
                expected[torch.arange(ranks * 2), torch.arange(ranks * 2) // 2] = count
                torch.testing.assert_close(actual, expected.expand(2, -1, -1), rtol=0, atol=0)

    def test_receive_order_preserves_source_and_row_order(self):
        """分段平移等价于接收标签的稳定排序；例如交错来源和零长段也保持原行次序。"""
        generator = torch.Generator().manual_seed(923)
        for ranks, experts in ((1, 1), (2, 2), (4, 16), (8, 128)):
            for sample in range(12):
                counts = torch.randint(0, 9, (ranks, experts), generator=generator)
                if sample == 0:
                    counts.zero_()
                offsets, lengths = planning._receive_order_metadata(counts)
                labels = [expert for row in counts.tolist() for expert, count in enumerate(row)
                          for _ in range(count)]
                expected = torch.tensor(sorted(range(len(labels)), key=labels.__getitem__), dtype=torch.long)
                actual = torch.arange(len(labels)) + torch.repeat_interleave(
                    torch.tensor(offsets), torch.tensor(lengths), output_size=len(labels))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def setUp(self):
        """CPU测试固定checkpoint默认设备；例如机器安装torch_npu也不初始化NPU。"""
        self.previous_device = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type('cpu')

    def tearDown(self):
        """恢复全局默认值；例如后续NPU测试仍使用其原始设置。"""
        DefaultDeviceType.set_device_type(self.previous_device)

    def _run_model(self, recompute, depth, evaluate=False):
        """执行连续两步并收集完整梯度；例如eval→train不能残留执行键。"""
        torch.manual_seed(45)
        model = torch.nn.Sequential(*[_Block() for _ in range(3)])
        if recompute:
            for index, block in enumerate(model):
                model[index] = wrap_moonep_checkpoint(block)
        runtime = configure_moonep(model, prefetch_depth=depth)
        inputs = torch.randn(5, 8, requires_grad=True)
        if evaluate:
            with torch.no_grad():
                model(inputs)
            self.assertEqual(runtime.trace, [])
            self.assertEqual(runtime.executions, {})
        records = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            inputs.grad = None
            loss = model(inputs).square().sum()
            loss.backward()
            runtime.finish_step()
            records.append([loss.detach(), inputs.grad.clone(), *[p.grad.clone() for p in model.parameters()]])
        self.assertEqual(runtime.version, 2)
        self.assertEqual(runtime.trace, [])
        self.assertEqual(runtime.tickets, {})
        self.assertEqual(runtime.statistics['plans'], 9 if evaluate else 6)
        self.assertEqual(runtime.statistics['recomputations'], 6 if recompute else 0)
        self.assertLessEqual(runtime.statistics['max_live_tickets'], 2)
        self.assertEqual(len(runtime.slots), runtime.statistics['max_live_tickets'])
        return records

    def test_checkpoint_prefetch_and_eval_train_match(self):
        """四种生命周期组合与评估切换数值一致；例如R2不能再次规划或重复交换权重。"""
        expected = self._run_model(False, 0)
        for checkpoint, depth in itertools.product((False, True), (0, 1)):
            with self.subTest(checkpoint=checkpoint, depth=depth):
                actual = self._run_model(checkpoint, depth, evaluate=True)
                for step_a, step_b in zip(actual, expected):
                    for value, reference in zip(step_a, step_b):
                        torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-6)
        # 独立脚本路径只预取紧邻前一专家；较大的旧 depth 值不再预留空槽。
        self._run_model(False, 4)

    def test_explicit_backward_matches_independent_autograd(self):
        """比较dX、两组dW及前向；例如限幅端点和空专家都经过独立torch公式。"""
        for activation, limit in (('clipped', 0.0), ('clipped', 0.3), ('swiglu', 0.0)):
            with self.subTest(activation=activation, limit=limit):
                torch.manual_seed(9)
                x = torch.randn(5, 4, requires_grad=True)
                gate = torch.randn(3, 4, 6, requires_grad=True)
                down = torch.randn(3, 3, 4, requires_grad=True)
                boundaries = torch.tensor([2, 2, 5])
                outputs = []
                begin = 0
                for expert, end in enumerate(boundaries.tolist()):
                    left, right = (x[begin:end] @ gate[expert]).chunk(2, -1)
                    if limit > 0:
                        left, right = left.clamp(max=limit), right.clamp(-limit, limit)
                    outputs.append((torch.nn.functional.silu(left) * right) @ down[expert])
                    begin = end
                expected = torch.cat(outputs)
                gradient = torch.randn_like(expected)
                derivatives = torch.autograd.grad(expected, (x, gate, down), gradient)
                actual, saved = expert_forward_values(x, gate, down, boundaries, limit, grouped_values, activation)
                gradients = expert_backward_values(gradient, saved, gate, down, grouped_values, grouped_weight_grad)
                torch.testing.assert_close(actual, expected)
                for value, reference in zip(gradients, derivatives):
                    torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-6)

    def test_executor_permutation_does_not_save_computed_storage(self):
        """真实执行器恢复排列只保存索引；例如10条交错路由不会多保留一份[10,8]专家输出。"""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(73)
                model = _Experts().to(dtype=dtype)
                runtime = configure_moonep(model, prefetch_depth=0)
                x = torch.randn(5, 8, dtype=dtype, requires_grad=True)
                ids = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
                scores = torch.rand(5, 2, dtype=dtype, requires_grad=True)
                saved_pointers, computed_pointers = [], []
                original_apply = moonep_runtime._ExpertFunction.apply

                def capture_computed(*args):
                    """记录实际GMM输出的存储地址；例如不持有输出本身，避免测试改变其生命周期。"""
                    result = original_apply(*args)
                    computed_pointers.append(result.untyped_storage().data_ptr())
                    return result

                def capture_saved(tensor):
                    """观察整个执行器的autograd保存；例如score求导需要恢复后的输出，但不需要原排序输出。"""
                    saved_pointers.append(tensor.untyped_storage().data_ptr())
                    return tensor

                with mock.patch.object(moonep_runtime._ExpertFunction, 'apply', side_effect=capture_computed), \
                        torch.autograd.graph.saved_tensors_hooks(capture_saved, lambda value: value):
                    actual = model(x, ids, scores)
                self.assertEqual(len(computed_pointers), 1)
                self.assertNotIn(computed_pointers[0], saved_pointers)

                # 独立逐token公式同时检查恢复排列、输入梯度、router梯度和两组专家参数梯度。
                expected_rows = []
                for token in range(x.shape[0]):
                    values = []
                    for slot in range(ids.shape[1]):
                        expert = int(ids[token, slot])
                        gate, up = (x[token] @ model.gate[expert]).float().chunk(2, -1)
                        intermediate = (torch.nn.functional.silu(gate) * up).to(dtype)
                        values.append((intermediate @ model.down[expert]) * scores[token, slot])
                    expected_rows.append(values[0] + values[1])
                expected = torch.stack(expected_rows)
                gradient = torch.randn_like(actual)
                parameters = (x, scores, model.gate, model.down)
                expected_gradients = torch.autograd.grad(expected, parameters, gradient)
                actual_gradients = torch.autograd.grad(actual, parameters, gradient)
                tolerance = 3e-3 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                for value, reference in zip(actual_gradients, expected_gradients):
                    torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)
                runtime.finish_step()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_bfloat16_output_with_float32_router_scores(self):
        """Feature: MoonEP mixed precision routing.
        Description: Run BF16 experts with FP32 router scores on CPU.
        Expectation: Output keeps BF16 dtype and both gradient paths are finite.
        """
        torch.manual_seed(91)
        model = _Experts().to(dtype=torch.bfloat16)
        runtime = configure_moonep(model, prefetch_depth=0)
        hidden = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
        indices = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
        scores = torch.rand(5, 2, dtype=torch.float32, requires_grad=True)

        output = model(hidden, indices, scores)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        for name, gradient in (("hidden", hidden.grad), ("scores", scores.grad),
                               ("gate", model.gate.grad), ("down", model.down.grad)):
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        runtime.finish_step()


class TestMoonEPCapacity(unittest.TestCase):
    """独立穷举验证最小迁移与木桶目标，检查通信元数据守恒。"""

    def test_compact_home_gradient_matches_scatter_and_keeps_small_storage(self):
        """全部 home 专家活跃时一次紧凑拷贝等于原散加；缺专家时仍走原路径。"""
        counts = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        allocation = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
        plan = compile_plan(counts, allocation, torch.tensor([0, 1]), 0)
        self.assertTrue(plan.home_dense)
        gate = torch.randn(2, 8, 16)
        down = torch.randn(2, 8, 8)
        original = moonep_runtime.local_home_gradients(gate, down, plan)
        compact = moonep_runtime.local_home_gradients(gate, down, plan, compact_home=True)
        for actual, expected in zip(compact, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())

        sparse_counts = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 1]])
        sparse_allocation = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 1]])
        missing = compile_plan(sparse_counts, sparse_allocation, torch.tensor([0]), 0)
        self.assertFalse(missing.home_dense)
        fallback = moonep_runtime.local_home_gradients(gate[:1], down[:1], missing, compact_home=True)
        sparse_reference = moonep_runtime.local_home_gradients(gate[:1], down[:1], missing)
        for actual, expected in zip(fallback, sparse_reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_small_exact_solver_against_exhaustive_search(self):
        """遍历81组专家计数；例如B=1禁止同一卡同时接收两个不同远程专家。"""
        for totals in itertools.product(range(3), repeat=4):
            targets = ((sum(totals) + 1) // 2, sum(totals) // 2)
            objectives = []
            for to_zero in itertools.product(*(range(value + 1) for value in totals)):
                if sum(to_zero) != targets[0]:
                    continue
                outgoing0 = sum(totals[e] > to_zero[e] for e in (0, 1))
                outgoing1 = sum(to_zero[e] > 0 for e in (2, 3))
                if max(outgoing0, outgoing1) > 1:
                    continue
                moved = sum(totals[e] - to_zero[e] for e in (0, 1)) + sum(to_zero[2:])
                objectives.append((moved, max(outgoing0, outgoing1), outgoing0 + outgoing1))
            result = solve_capacity_exact(totals, 2, 1)
            self.assertEqual((result.moved, result.peak_copies, result.total_copies), min(objectives))

    def test_exact_solver_records_three_stages_with_shared_budget(self):
        """小型可解规划逐阶段记录状态；总预算在三阶段间递减。"""
        events = []
        result = solve_capacity_exact([3, 1, 2, 2], 2, 2,
                                      progress=lambda event, details: events.append((event, details)),
                                      time_limit=30)
        self.assertEqual(result.allocation.shape, (4, 2))
        starts = [details for event, details in events if event == 'milp_start']
        ends = [details for event, details in events if event == 'milp_end']
        self.assertEqual([entry['phase'] for entry in starts],
                         ['migration', 'peak_copies', 'total_copies'])
        self.assertEqual([entry['status'] for entry in ends], [0, 0, 0])
        self.assertGreater(starts[0]['time_limit_s'], starts[-1]['time_limit_s'])

    def test_planner_uses_bounded_fill_fallback_without_solver(self):
        """四卡容量难例由填满接收卡兜底；例如迁移 12 条而不进入整数规划。"""
        totals = torch.tensor([0, 0, 4, 4, 4, 4, 4, 4])
        counts = totals.repeat(4, 1) // 4
        _, allocation, used_fallback = planning.select_allocation(counts, 2)
        self.assertTrue(used_fallback)
        expected = planning._cpu_candidates(counts.sum(0).numpy(), 4, (False, True))[1]
        np.testing.assert_array_equal(allocation.numpy(), expected)
        self.assertEqual(planning._cpu_plan_metrics(allocation.tolist(), totals.tolist(), [2] * 4)[1], 12)

    def test_compiled_dispatch_and_weight_splits_are_reciprocal(self):
        """每条路由只发送一次且权重收发互逆；例如热e0被分到两张卡。"""
        routes = [torch.tensor([0, 0, 0, 0, 1, 2]), torch.tensor([0, 0, 0, 2, 3, 3])]
        counts = torch.stack([torch.bincount(ids, minlength=4) for ids in routes])
        allocation = torch.from_numpy(solve_capacity_exact(counts.sum(0).tolist(), 2, 2).allocation)
        plans = [compile_plan(counts, allocation, ids, rank) for rank, ids in enumerate(routes)]
        for target, plan in enumerate(plans):
            labels = []
            for source, source_plan in enumerate(plans):
                self.assertEqual(plan.token_recv[source], source_plan.token_send[target])
                self.assertEqual(plan.weight_recv[source], source_plan.weight_send[target])
                begin = sum(source_plan.token_send[:target])
                order = source_plan.dispatch_order[begin:begin + source_plan.token_send[target]]
                labels.extend(routes[source][order].tolist())
            sorted_labels = torch.tensor(labels)[plan.sort_order]
            self.assertEqual(torch.bincount(sorted_labels, minlength=4).tolist(), allocation[:, target].tolist())
            self.assertEqual(sorted_labels.tolist(), sorted(labels))

    def test_vectorized_candidate_conservation_and_certificate(self):
        """随机计数验证设备区间交集；例如R4/E8单归属规则始终满足每卡最多两个副本。"""
        generator = torch.Generator().manual_seed(23)
        for ranks in (2, 4, 8):
            for _ in range(40):
                totals = torch.randint(0, 30, (ranks * 2,), generator=generator)
                target = [int(totals.sum()) // ranks + int(r < int(totals.sum()) % ranks)
                          for r in range(ranks)]
                for rule in (False, True):
                    candidate = device_candidate(totals, ranks, rule)
                    self.assertEqual(candidate.sum(0).tolist(), target)
                    self.assertEqual(candidate.sum(1).tolist(), totals.tolist())
                    self.assertTrue(bool((candidate >= 0).all()))
                    metrics = device_plan_metrics(candidate, totals, torch.full((ranks,), 2)).tolist()
                    if rule:
                        self.assertEqual(metrics[0], 1)
                        for receiver in range(ranks):
                            remote_homes = {e // 2 for e in range(ranks * 2)
                                            if e // 2 != receiver and candidate[e, receiver] > 0}
                            self.assertLessEqual(len(remote_homes), 1)
                    if metrics[5]:
                        self.assertEqual(metrics[1], metrics[4])

    def test_two_rank_selection_uses_one_equivalent_candidate(self):
        """两卡仅需一条 CPU 候选；例如奇数总量仍按8/7分配且迁移最少。"""
        generator = torch.Generator().manual_seed(91)
        for _ in range(30):
            counts = torch.randint(0, 12, (2, 6), generator=generator)
            totals = counts.sum(0)
            partial = device_candidate(totals, 2, False)
            filled = device_candidate(totals, 2, True)
            torch.testing.assert_close(partial, filled, rtol=0, atol=0)
            with mock.patch.object(planning, '_cpu_candidates', wraps=planning._cpu_candidates) as candidate_call, \
                    mock.patch.object(planning, '_cpu_candidate_metrics',
                                      wraps=planning._cpu_candidate_metrics) as metric_call:
                copied_counts, allocation, used_fallback = planning.select_allocation(counts, 3)
            candidate_call.assert_called_once()
            metric_call.assert_called_once()
            self.assertFalse(used_fallback)
            torch.testing.assert_close(copied_counts, counts, rtol=0, atol=0)
            torch.testing.assert_close(allocation.sum(1), totals, rtol=0, atol=0)
            total = int(totals.sum())
            targets = torch.tensor([(total + 1) // 2, total // 2])
            torch.testing.assert_close(allocation.sum(0), targets, rtol=0, atol=0)
            moved = int(allocation[:3, 1].sum() + allocation[3:, 0].sum())
            minimum = int((totals.reshape(2, 3).sum(1) - targets).clamp_min(0).sum())
            self.assertEqual(moved, minimum)

    def test_cpu_small_table_matches_reference_candidate_and_selection(self):
        """CPU 小表候选与离线参考逐值一致；例如 EP8 的零负载与热点。"""
        generator = torch.Generator().manual_seed(8921)
        for ranks in (2, 4, 8):
            for sample in range(35):
                counts = torch.randint(0, 48, (ranks, ranks * 4), generator=generator)
                if sample == 0:
                    counts.zero_()
                elif sample == 1:
                    counts[:, 1:] = 0
                totals = counts.sum(0)
                rules = (False,) if ranks == 2 else (False, True)
                expected = device_candidates(totals, ranks, rules)
                actual = planning._cpu_candidates(totals.numpy(), ranks, rules)
                np.testing.assert_array_equal(actual, expected.numpy())
                for capacity in (0, 1, ranks * 4):
                    device_metrics = device_plan_metrics_batch(
                        expected, totals, torch.full((ranks,), capacity)).numpy()
                    cpu_metrics = planning._cpu_candidate_metrics(actual, totals.numpy(), capacity)
                    np.testing.assert_array_equal(cpu_metrics, device_metrics)
                    if capacity == ranks * 4:
                        self.assertEqual(tuple(cpu_metrics[0]), planning._cpu_plan_metrics(
                            actual[0].tolist(), totals.tolist(), [capacity] * ranks))
                if sample == 0:
                    returned_counts, allocation, used_fallback = planning.select_allocation(counts, ranks * 4)
                    torch.testing.assert_close(returned_counts, counts, rtol=0, atol=0)
                    torch.testing.assert_close(allocation, expected[0], rtol=0, atol=0)
                    self.assertFalse(used_fallback)
        # FP32 排序的不可分辨整数必须沿用稳定次序；例如 2**24 与 2**24+1。
        large = torch.tensor([2**24, 2**24 + 1, 1, 0, 0, 0, 0, 0], dtype=torch.long)
        np.testing.assert_array_equal(planning._cpu_candidates(large.numpy(), 4, (False, True)),
                                      device_candidates(large, 4).numpy())

    def test_cpu_small_table_rejects_infeasible_capacity(self):
        """容量不足时明确报错；例如 EP2、B1 无法承接两个远程专家。"""
        counts = torch.tensor([[5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        with self.assertRaisesRegex(RuntimeError, '无法满足均衡及副本容量'):
            planning.select_allocation(counts, 1)

    def test_two_rank_analytic_certificate_matches_general_path(self):
        """穷举81种两卡小计数，对照原设备证书；例如B=P和B>P都不改变分配。"""
        for entries in itertools.product(range(3), repeat=4):
            counts = torch.tensor(entries, dtype=torch.long).reshape(2, 2)
            expected = device_candidate(counts.sum(0), 2, False)
            for capacity in (1, 2):
                metrics = device_plan_metrics(expected, counts.sum(0), torch.full((2,), capacity))
                self.assertEqual(int(metrics[5]), 1)
                actual_counts, actual, used_fallback = planning.select_allocation(counts, capacity)
                torch.testing.assert_close(actual_counts, counts, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertFalse(used_fallback)

    def test_analytic_certificate_keeps_count_validation(self):
        """解析证书仍拒绝非法源计数；例如列和为正也不能接受源rank的-1。"""
        invalid = [torch.ones(2, 2, dtype=torch.float32), torch.ones(4, dtype=torch.long),
                   torch.empty(0, 2, dtype=torch.long), torch.empty(2, 0, dtype=torch.long),
                   torch.tensor([[-1, 3], [2, 1]])]
        for counts in invalid:
            with self.subTest(shape=tuple(counts.shape)):
                with self.assertRaises(ValueError):
                    planning.select_allocation(counts, 2)
        # R4仍需两条迁移规则和通用容量校验，解析分支不能扩张到不成立的拓扑。
        counts = torch.ones(4, 8, dtype=torch.long)
        with mock.patch.object(planning, '_cpu_candidate_metrics',
                               wraps=planning._cpu_candidate_metrics) as metrics:
            planning.select_allocation(counts, 2)
        self.assertEqual(metrics.call_count, 1)

    def test_cpu_list_metrics_match_device_metrics(self):
        """穷举加随机对照全部六项指标；例如负配额的M也必须保留设备版语义。"""
        for entries in itertools.product(range(-1, 3), repeat=4):
            allocation = torch.tensor(entries, dtype=torch.long).reshape(2, 2)
            totals = allocation.sum(1)
            for capacities in ([-1, 1], [0, 0], [1, 1]):
                expected = device_plan_metrics(allocation, totals, torch.tensor(capacities)).tolist()
                actual = planning._cpu_plan_metrics(allocation.tolist(), totals.tolist(), capacities)
                self.assertEqual(actual, tuple(expected))
        generator = torch.Generator().manual_seed(37)
        for ranks in (1, 2, 4, 8):
            for _ in range(40):
                allocation = torch.randint(-2, 8, (ranks * 2, ranks), generator=generator)
                totals = torch.randint(-2, 16, (ranks * 2,), generator=generator)
                capacities = torch.randint(-1, 5, (ranks,), generator=generator)
                expected = device_plan_metrics(allocation, totals, capacities).tolist()
                actual = planning._cpu_plan_metrics(allocation.tolist(), totals.tolist(), capacities.tolist())
                self.assertEqual(actual, tuple(expected))

    def test_weight_spans_match_global_weights_without_output_reallocation(self):
        """独立全局权重对照三段组表；例如rank1按远程e0、本地e3、远程e6排列。

        输入为随机专家分配和全局权重，期望输出直接按全局专家号索引得到。
        接收权重故意使用打包缓冲的非连续视图，同时验证最终连续槽切片不换存储。
        """
        generator = torch.Generator().manual_seed(72)
        max_spans = 0
        for ranks in (1, 2, 4, 8):
            experts = ranks * 3
            for _ in range(30):
                allocation = torch.randint(0, 5, (experts, ranks), generator=generator)
                counts = torch.zeros(ranks, experts, dtype=torch.long)
                counts[0] = allocation.sum(1)
                global_gate = torch.randn(experts, 4, 6, generator=generator)
                global_down = torch.randn(experts, 3, 4, generator=generator)
                for rank in range(ranks):
                    ids = torch.repeat_interleave(torch.arange(experts), counts[rank])
                    plan = compile_plan(counts, allocation, ids, rank)
                    home = torch.arange(rank * 3, (rank + 1) * 3)
                    remote = torch.tensor(
                        [expert for expert in range(experts)
                         if expert // 3 != rank and allocation[expert, rank] > 0],
                        dtype=torch.long,
                    )
                    active = torch.nonzero(allocation[:, rank], as_tuple=True)[0]
                    home_gate, home_down = global_gate[home], global_down[home]
                    packed = torch.cat((global_gate[remote].flatten(1),
                                        global_down[remote].flatten(1)), dim=1)
                    remote_gate = packed[:, :24].reshape(len(remote), 4, 6)
                    remote_down = packed[:, 24:].reshape(len(remote), 3, 4)
                    target_gate = torch.empty(experts + 2, 4, 6)
                    target_down = torch.empty(experts + 2, 3, 4)
                    pointers = target_gate.data_ptr(), target_down.data_ptr()
                    for is_home, begin, end, rows in plan.weight_spans:
                        self.assertTrue(target_gate[begin:end].is_contiguous())
                        self.assertTrue(target_down[begin:end].is_contiguous())
                        torch.index_select(home_gate if is_home else remote_gate, 0, rows,
                                           out=target_gate[begin:end])
                        torch.index_select(home_down if is_home else remote_down, 0, rows,
                                           out=target_down[begin:end])
                    self.assertEqual((target_gate.data_ptr(), target_down.data_ptr()), pointers)
                    torch.testing.assert_close(target_gate[:len(active)], global_gate[active], rtol=0, atol=0)
                    torch.testing.assert_close(target_down[:len(active)], global_down[active], rtol=0, atol=0)
                    self.assertLessEqual(len(plan.weight_spans), 3)
                    max_spans = max(max_spans, len(plan.weight_spans))
        self.assertEqual(max_spans, 3)

    def test_infeasible_capacity_fails_explicitly(self):
        """容量不允许负载均衡时明确报错；例如两个卡B=0不能迁移四条e0。"""
        with self.assertRaises(ValueError):
            solve_capacity_exact([4, 0], 2, 0)


class TestMoonEPOptimizations(unittest.TestCase):
    """独立列表算法及内存布局反例；例如相同的专家并列时仍必须选择较小编号。"""

    @staticmethod
    def reference_candidate(totals, ranks, fill):
        """直接用 Python 供需匹配和逐专家填充；例如 surplus=3、deficit=5 两规则迁移3/5。"""
        values = totals.tolist()
        home_count = len(values) // ranks
        target_load = [sum(values) // ranks + int(rank < sum(values) % ranks) for rank in range(ranks)]
        balance = [sum(values[rank * home_count:(rank + 1) * home_count]) - target_load[rank]
                   for rank in range(ranks)]
        quotas = [[0] * ranks for _ in range(ranks)]
        for _ in range(ranks - 1):
            home = max(range(ranks), key=lambda rank: balance[rank])
            target = min(range(ranks), key=lambda rank: balance[rank])
            amount = (-balance[target] if fill else min(balance[home], -balance[target])) if balance[home] > 0 else 0
            quotas[home][target] += amount
            balance[home] -= amount
            balance[target] += amount
        result = [[0] * ranks for _ in values]
        remaining = list(values)
        for home in range(ranks):
            experts = list(range(home * home_count, (home + 1) * home_count))
            experts.sort(key=lambda expert: -float(torch.tensor(values[expert], dtype=torch.float32)))
            targets = sorted(range(ranks),
                             key=lambda rank: -float(torch.tensor(quotas[home][rank], dtype=torch.float32)))
            for target in targets:
                need = quotas[home][target]
                for expert in experts:
                    amount = min(remaining[expert], need)
                    result[expert][target] += amount
                    remaining[expert] -= amount
                    need -= amount
            for expert in experts:
                result[expert][home] += remaining[expert]
        return torch.tensor(result, dtype=torch.int64)

    def test_batched_planner_matches_independent_sequential_rules(self):
        """穷举4卡小例和随机大数并列例；例如零负载、余数目标和 FP32 并列均保持原选择。"""
        generator = torch.Generator().manual_seed(234)
        cases = [(torch.tensor(values, dtype=torch.int64), 4) for values in itertools.product(range(3), repeat=4)]
        cases += [(torch.randint(0, 100, (ranks * 3,), generator=generator), ranks)
                  for ranks in (2, 4, 8) for _ in range(30)]
        cases.append((torch.tensor([2**24, 2**24 + 1, 1, 0, 0, 0, 0, 0]), 4))
        for totals, ranks in cases:
            actual = device_candidates(totals, ranks)
            for index, fill in enumerate((False, True)):
                expected = self.reference_candidate(totals, ranks, fill)
                torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
            capacities = torch.full((ranks,), 2)
            metrics = device_plan_metrics_batch(actual, totals, capacities)
            for index in range(2):
                expected = planning._cpu_plan_metrics(actual[index].tolist(), totals.tolist(), capacities.tolist())
                self.assertEqual(metrics[index].tolist(), list(expected))

    def test_shared_executor_guard_resets_between_completed_microbatches(self):
        """仅同时存活的重复调用需回退；例如 no_sync 微批1反向完，微批2仍可异步。"""
        model = _Experts()
        runtime = configure_moonep(model)
        executor = model._moonep_executor
        first = runtime.new_key(executor)
        runtime.executions[first] = executor, None
        second = runtime.new_key(executor)
        runtime.executions[second] = executor, None
        self.assertIn(executor.layer, runtime.shared_executors)
        runtime.complete(second)
        self.assertIn(executor.layer, runtime.shared_executors)
        runtime.complete(first)
        self.assertFalse(runtime.shared_executors)
        third = runtime.new_key(executor)
        runtime.executions[third] = executor, None
        self.assertNotIn(executor.layer, runtime.shared_executors)
        runtime.complete(third)
        runtime.finish_step()

    def test_pack_preserves_transposes_duplicates_and_empty_rows(self):
        """检查两矩阵打包的完整字节顺序；例如专家2可重复发送，DeepSeek转置也保留布局。"""
        for dtype, transposed in itertools.product((torch.float32, torch.bfloat16), (False, True)):
            gate = torch.arange(4 * 3 * 6, dtype=dtype).reshape(4, 3, 6)
            down = torch.arange(4 * 3 * 3, dtype=dtype).reshape(4, 3, 3)
            if transposed:
                gate, down = gate.transpose(1, 2), down.transpose(1, 2)
            for ids in ([2, 0, 2], [], [1], [3, 2, 1, 0], [2, 0, 2, 0, 3]):
                rows = torch.tensor(ids, dtype=torch.long)
                expected = torch.cat((gate.index_select(0, rows).reshape(len(ids), 18),
                                      down.index_select(0, rows).reshape(len(ids), 9)), dim=1)
                actual = moonep_runtime.pack_expert_rows(gate, down, rows)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(actual.is_contiguous())
