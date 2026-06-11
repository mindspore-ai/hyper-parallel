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
"""Build pipeline scheduler chains (1F1B, VPP, VPP-less-memory)."""
from __future__ import annotations

from typing import Callable, List

from hyper_parallel.auto_parallel.sapp_ppb.simulator.sim_block import BlockSim, HeadBlockSim, MicroBlockSim

BuilderFn = Callable[..., List[BlockSim]]


class PipelineBuilder:
    r"""Build pipeline scheduler"""
    @staticmethod
    def _inter_merge(a: list[MicroBlockSim], b: list[MicroBlockSim], delta: int = 0) -> list[MicroBlockSim]:
        r"""merge forward and backward chain for 1f1b"""
        res = []
        if delta >= 0:
            res.extend(a[:delta])
            a = a[delta:]
        else:
            res.extend(b[:-delta])
            b = b[-delta:]
        stable_count = 0
        while a:
            block = a.pop(0)
            block.phase = 'stable'
            res.append(block)
            stable_count += 1
            if b:
                block = b.pop(0)
                block.phase = 'stable'
                res.append(block)
                stable_count += 1
            else:
                break
        if stable_count:
            res[-1].phase = 'cooldown'
        if a:
            res.extend(a)
        elif b:
            res.extend(b)
        return res

    @staticmethod
    def _build_chain(line: list[MicroBlockSim], p: int) -> list[BlockSim]:
        r"""build pipeline chain"""
        # pylint: disable=E1120
        head = HeadBlockSim(p)
        left = head
        for item in line:
            left.right = item
            item.left = left
            left = item
        if p == 0:
            head.right.pre = head
        return line

    @staticmethod
    # pylint: disable=W0613
    def build_1f1b(pp: int, micro_num: int, vp: int, p: int,
                   forward_time: List[float], backward_time: List[float],
                   block_mem: List[float], block_mem_par: List[float]) -> List[BlockSim]:
        """Build a 1F1B schedule chain for one pipeline rank.

        Args:
            pp: Total number of pipeline stages.
            micro_num: Number of micro-batches.
            vp: Virtual-pipeline degree (unused here, kept for a common builder signature).
            p: Pipeline-stage index this chain belongs to.
            forward_time: Per-chunk forward times; only ``forward_time[0]`` is used.
            backward_time: Per-chunk backward times; only ``backward_time[0]`` is used.
            block_mem: Per-chunk activation memory; only ``block_mem[0]`` is used.
            block_mem_par: Per-chunk parameter memory; only ``block_mem_par[0]`` is used.

        Returns:
            The ordered chain of :class:`BlockSim` nodes.
        """
        forward_time = forward_time[0]
        backward_time = backward_time[0]
        block_mem = block_mem[0]
        block_mem_par = block_mem_par[0]
        for_line = [MicroBlockSim(p, 'f', i, 0, forward_time, mem=block_mem, mem_par=block_mem_par, phase='warmup')
                    for i in range(micro_num)]
        back_line = [MicroBlockSim(p, 'b', i, 0, backward_time, mem=block_mem, mem_par=block_mem_par, phase='cooldown')
                     for i in range(micro_num)]
        line = PipelineBuilder._inter_merge(for_line, back_line, pp - p - 1)
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def build_virtualpipeline(pp: int, micro_num: int, vp: int, p: int,
                              forward_time: List[float], backward_time: List[float],
                              block_mem: List[float],
                              block_mem_par: List[float]) -> List[BlockSim]:
        """Build a virtual-pipeline (VPP) 1F1B chain for one pipeline rank."""
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                bi = vp - 1 - i
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i],
                                                   mem=block_mem[i], mem_par=block_mem_par[i],
                                                   phase='warmup') for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, bi, backward_time[bi],
                                                    mem=block_mem[bi], mem_par=block_mem_par[bi],
                                                    phase='cooldown') for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i],
                                               mem=block_mem[i], mem_par=block_mem_par[i],
                                               phase='warmup') for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, bi, backward_time[bi],
                                                mem=block_mem[bi], mem_par=block_mem_par[bi],
                                                phase='cooldown') for m in range(pp)])
        line = PipelineBuilder._inter_merge(for_line, back_line, (vp + 1) * pp - 2 * p - 2 + r * (vp - 1))
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def build_virtualpipeline2(pp: int, micro_num: int, vp: int, p: int,
                               forward_time: List[float], backward_time: List[float],
                               block_mem: List[float],
                               block_mem_par: List[float]) -> List[BlockSim]:
        """Build a VPP 1F1B chain using the less-memory scheduler variant."""
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                bi = vp - 1 - i
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i],
                                                   mem=block_mem[i], mem_par=block_mem_par[i],
                                                   phase='warmup') for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, bi, backward_time[bi],
                                                    mem=block_mem[bi], mem_par=block_mem_par[bi],
                                                    phase='cooldown') for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i],
                                               mem=block_mem[i], mem_par=block_mem_par[i],
                                               phase='warmup') for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, bi, backward_time[bi],
                                                mem=block_mem[bi], mem_par=block_mem_par[bi],
                                                phase='cooldown') for m in range(pp)])

        line = PipelineBuilder._inter_merge(for_line, back_line, vp * pp - p - 1)
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def get_builder(method: str = '1f1b') -> BuilderFn:
        """Return the schedule-builder callable for a given schedule ``method``.

        Args:
            method: One of ``'1f1b'``, ``'vpp'``, ``'vpp2'``.

        Raises:
            ValueError: If ``method`` is not one of the supported values.
        """
        if method == '1f1b':
            return PipelineBuilder.build_1f1b
        if method == 'vpp':
            return PipelineBuilder.build_virtualpipeline
        if method == 'vpp2':
            return PipelineBuilder.build_virtualpipeline2
        raise ValueError(f"`method` only support ['1f1b', 'vpp', 'vpp2'], but got {method}")
