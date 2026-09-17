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
"""Check backward storage reuse and reject unsafe scheduler changes on CPU."""

import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue, TaskType


class TestBackwardStorage(unittest.TestCase):
    """Exercise real physical queues, including changes that break lifetime ordering."""

    @staticmethod
    def _config(ep: int = 4, cores: int = 20, rank: int = 0, split: tuple = (128, 128)) -> tuple:
        """Build a complete reordered backward plan without creating device tensors."""
        topology = TaskSplitValue(tp=1, ep=ep, seq_size=4096, all_expert_num=48, top_k=8)
        graph = build_backward_graph(topology, dispatch_sv=split[0], combine_sv=split[1],
                                     hidden_size=5120, intermediate_size=1792, num_cube_cores=cores)
        graph.propagate_splits(topology)
        return build_config_for_rank(graph, topology, rank, cores), topology

    def test_all_managed_plans_join_receive_readers_before_reuse(self) -> None:
        """Feature: Reuse receive storage across managed communication plans.

        Description: Build all EP4/EP8 ranks with 20/24 Cube workers and equal push dispatch/combine task sizes.
        Expectation: Every validated schedule permits reuse through its existing event chain.
        """
        for ep in (4, 8):
            for cores in (20, 24):
                for split in ((128, 128), (512, 512), (1024, 1024)):
                    for rank in range(ep):
                        with self.subTest(ep=ep, cores=cores, split=split, rank=rank):
                            cfg, topology = self._config(ep, cores, rank, split)
                            self.assertTrue(can_reuse_backward_dispatch(cfg, topology.single_rank_expert_num, cores))

    def test_changed_worker_order_disables_reuse(self) -> None:
        """Feature: Conservative fallback for a reordered Cube queue.

        Description: Move a W2-gradient task behind its activation-gradient task.
        Expectation: The plan requires separate receive and input-gradient storage.
        """
        cfg, topology = self._config()
        cfg.cube_task_indices[0], cfg.cube_task_indices[20] = cfg.cube_task_indices[20], cfg.cube_task_indices[0]
        self.assertFalse(can_reuse_backward_dispatch(cfg, topology.single_rank_expert_num, 20))

    def test_changed_events_disable_reuse(self) -> None:
        """Feature: Conservative fallback for incomplete reader joins.

        Description: Weaken an activation completion threshold or bypass the SwiGLU dependency.
        Expectation: Neither modified plan may share the transient buffers.
        """
        for mutation in ("threshold", "dependency"):
            with self.subTest(mutation=mutation):
                cfg, topology = self._config()
                if mutation == "threshold":
                    task = cfg.all_tasks[cfg.cube_task_indices[20]]
                    cfg.all_event_num_triggers[task.trigger_event] = 19
                else:
                    task = next(task for task in cfg.all_tasks
                                if task.task_type == TaskType.TASK_SWI_GLU_GRAD)
                    task.dependent_event = 0
                self.assertFalse(can_reuse_backward_dispatch(cfg, topology.single_rank_expert_num, 20))

    def test_additional_receive_reader_disables_reuse(self) -> None:
        """Feature: Conservative fallback when another kernel reads received gradients.

        Description: Change a W1-gradient input to consume the receive slot.
        Expectation: The unchecked reader prevents storage reuse.
        """
        cfg, topology = self._config()
        task = next(task for task in cfg.all_tasks
                    if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.outputs[0].input_position == 18)
        task.inputs[0].input_position = 0
        self.assertFalse(can_reuse_backward_dispatch(cfg, topology.single_rank_expert_num, 20))

    def test_mixed_worker_queue_disables_reuse(self) -> None:
        """Feature: Conservative fallback for another physical worker queue.

        Description: Add a mixed-worker task outside the checked Cube and Vector queues.
        Expectation: The unchecked queue prevents storage reuse.
        """
        cfg, topology = self._config()
        cfg.task_index_num[2] = 1
        self.assertFalse(can_reuse_backward_dispatch(cfg, topology.single_rank_expert_num, 20))
