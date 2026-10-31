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
"""MoonEP 规划守恒和单 rank 前反向精度验证。"""

import unittest

import torch
from torch.nn import functional

from hyper_parallel.distributed.expert_parallel.moonep.legacy import MoonEPExecutor, plan_moonep


class TestMoonEP(unittest.TestCase):
    """使用独立逐专家计算验证脚本 EP 的数值语义。"""

    def test_skewed_plan_balances_entries_and_limits_weight_copies(self) -> None:
        """热专家偏斜时，每 rank 仍获得相同真实条目，且副本数量有界。

        Example:
            四个 rank 各提交八个条目，检查分发和接收的专家标签完全对应。
        """
        routes = torch.tensor([
            [0, 0, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 0, 4, 4, 5, 5],
            [0, 0, 1, 1, 1, 1, 6, 7],
            [0, 0, 0, 0, 0, 0, 6, 7],
        ])
        counts = torch.stack([torch.bincount(row, minlength=8) for row in routes])
        plans = [plan_moonep(counts, rank, routes[rank]) for rank in range(4)]
        for rank, plan in enumerate(plans):
            with self.subTest(rank=rank):
                self.assertEqual(plan.allocation.sum(0).tolist(), [8] * 4)
                self.assertEqual(plan.allocation.sum(1).tolist(), counts.sum(0).tolist())
                self.assertEqual(sum(plan.send_counts), 8)
                self.assertEqual(sum(plan.recv_counts), 8)
                self.assertEqual(len(plan.received_experts), 8)
                self.assertLessEqual(len(plan.remote_experts), 2)
                self.assertTrue(plan.move_weights)
                expected_labels = []
                for source in range(4):
                    self.assertEqual(plan.recv_counts[source], plans[source].send_counts[rank])
                    self.assertEqual(plan.weight_recv[source], plans[source].weight_send[rank])
                    offset = sum(plans[source].send_counts[:rank])
                    selected = plans[source].dispatch_order[offset:offset + plans[source].send_counts[rank]]
                    expected_labels.extend(routes[source][selected].tolist())
                self.assertEqual(plan.received_experts.tolist(), expected_labels)

    def test_receiver_uses_one_remote_home_when_donor_surplus_is_too_small(self) -> None:
        """接收方缺口大于发送方盈余时，仍只接收一个远程归属 rank。

        Example:
            四卡各有五条选择，归属负载为 ``[0, 4, 8, 8]``；
            rank 0 从归属 rank 2 一次接收五条，rank 2 后续再补足自身缺口。
        """
        routes = torch.tensor([
            [2, 2, 2, 2, 4],
            [4, 4, 4, 5, 5],
            [5, 5, 6, 6, 6],
            [6, 7, 7, 7, 7],
        ])
        counts = torch.stack([torch.bincount(row, minlength=8) for row in routes])
        plans = [plan_moonep(counts, rank, routes[rank]) for rank in range(4)]
        self.assertEqual(counts.sum(0).reshape(4, 2).sum(1).tolist(), [0, 4, 8, 8])
        self.assertEqual(plans[0].allocation.sum(0).tolist(), [5] * 4)
        self.assertEqual({expert // 2 for expert in plans[0].remote_experts}, {2})
        for plan in plans:
            self.assertLessEqual(len({expert // 2 for expert in plan.remote_experts}), 1)
            self.assertLessEqual(len(plan.remote_experts), 2)

    def test_balanced_plan_keeps_home_experts(self) -> None:
        """均衡路由跳过远程权重交换，并把条目发回专家所属 rank。

        Example:
            两个 rank 分别提交 ``[0, 2, 0, 2]``，每个 rank 接收自己的两个专家条目。
        """
        routes = torch.tensor([[0, 2, 0, 2], [0, 2, 0, 2]])
        counts = torch.stack([torch.bincount(row, minlength=4) for row in routes])
        plans = [plan_moonep(counts, rank, routes[rank]) for rank in range(2)]
        for rank, plan in enumerate(plans):
            with self.subTest(rank=rank):
                self.assertFalse(plan.move_weights)
                self.assertEqual(plan.remote_experts, [])
                self.assertEqual(plan.send_counts, [2, 2])
                self.assertEqual(plan.recv_counts, [2, 2])
                self.assertEqual(plan.received_experts.tolist(), [rank * 2] * 4)

    def test_uneven_and_empty_sources_keep_global_entries(self) -> None:
        """源 rank 条目数不同时，目标容量仍均衡且接收标签正确。

        Example:
            rank 1 没有本地条目时，它仍接收 rank 0 的两个热专家条目。
        """
        for routes in (
                (torch.tensor([0, 0, 1, 3, 3]), torch.tensor([0])),
                (torch.tensor([0, 0, 0, 0]), torch.empty(0, dtype=torch.long))):
            with self.subTest(source_lengths=tuple(row.numel() for row in routes)):
                counts = torch.stack([torch.bincount(row, minlength=4) for row in routes])
                plans = [plan_moonep(counts, rank, routes[rank]) for rank in range(2)]
                targets = [counts.sum().item() // 2 + int(rank < counts.sum().item() % 2)
                           for rank in range(2)]
                for rank, plan in enumerate(plans):
                    self.assertEqual(plan.allocation.sum(0).tolist(), targets)
                    self.assertEqual(sum(plan.send_counts), routes[rank].numel())
                    self.assertEqual(sum(plan.recv_counts), targets[rank])
                    received = []
                    for source in range(2):
                        self.assertEqual(plan.recv_counts[source], plans[source].send_counts[rank])
                        offset = sum(plans[source].send_counts[:rank])
                        selected = plans[source].dispatch_order[offset:offset + plans[source].send_counts[rank]]
                        received.extend(routes[source][selected].tolist())
                    self.assertEqual(plan.received_experts.tolist(), received)

    def test_single_rank_matches_independent_forward_and_gradients(self) -> None:
        """对照逐专家线性层，验证输出、输入、router 及权重梯度。

        Example:
            五个 token 分别选择两个专家，脚本执行器与逐条目公式前反向一致。
        """
        torch.manual_seed(84)
        executor = MoonEPExecutor(ep_group=None, ep_size=1, num_experts=3, top_k=2,
                                  max_local_num_tokens=None, swiglu_limit=1.5)
        hidden = torch.randn(5, 8, requires_grad=True)
        scores = torch.rand(5, 2, requires_grad=True)
        ids = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]])
        gate_up = torch.randn(3, 8, 12, requires_grad=True)
        down = torch.randn(3, 6, 8, requires_grad=True)
        actual = executor(hidden, ids, scores, expert_weights=(gate_up, down))
        oracle = torch.zeros_like(actual)
        for token in range(5):
            for slot in range(2):
                expert = int(ids[token, slot])
                gate, up = (hidden[token] @ gate_up[expert]).float().chunk(2)
                activation = functional.silu(gate.clamp(max=1.5)) * up.clamp(-1.5, 1.5)
                value = activation.to(hidden.dtype) @ down[expert]
                oracle[token] = oracle[token] + scores[token, slot] * value
        upstream = torch.randn_like(actual)
        tensors = (hidden, scores, gate_up, down)
        actual_grads = torch.autograd.grad(actual, tensors, upstream, retain_graph=True)
        oracle_grads = torch.autograd.grad(oracle, tensors, upstream)
        torch.testing.assert_close(actual, oracle, rtol=1e-5, atol=1e-5)
        for observed, expected in zip(actual_grads, oracle_grads):
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
