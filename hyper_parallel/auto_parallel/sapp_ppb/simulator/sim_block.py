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
"""Simulator block primitives: compute blocks, comm (send/receive) blocks, head sentinels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from sapp_ppb.simulator.utils import color_mix, dfs_builder


@dataclass
class BlockSim:
    r"""base block sim class"""
    stage: int # p
    state: str # s
    id: int # m
    chunk: int # v
    time: float
    type: str
    start: float = None
    end: float = None
    pre: BlockSim = field(repr=False, default=None)
    left: BlockSim = field(repr=False, default=None)
    # pylint: disable=E0601
    right: MicroBlockSim = field(repr=False, default=None)
    depend_pre: BlockSim = field(repr=False, default=None)
    depend_left: BlockSim = field(repr=False, default=None)
    finish = False
    in_queue = False
    flag = False
    _color = '0;38'
    father: BlockSim = field(repr=False, default=None)

    @property
    def label(self) -> tuple:
        """Identity tuple ``(type, state, id, chunk, stage)`` used in log messages."""
        res = (self.type, self.state, self.id, self.chunk, self.stage)
        return res

    @property
    def color_label(self) -> str:
        """ANSI-coloured version of :attr:`label` for terminal output."""
        return f"\033[{self._color}m{self.label}\033[0m"

    @dfs_builder(False)
    def build_without_comm(self) -> None:
        r"""Build pipeline timeline without comm blocks and dependency."""
        self.pre.build_without_comm()
        self.left.build_without_comm()
        self.start = max(self.pre.end, self.left.end)
        self.end = self.start + self.time

    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.depend_pre.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_pre.end, self.depend_left.end)
        self.end = self.start + self.time

    def reset_time(self) -> None:
        r"""reset time"""
        self.start = None
        self.end = None
        self.finish = False

    # pylint: disable=W0613
    def loc_size(self, x: float = 0, equal_wide: bool = False,
                 mode: str = 'compact') -> Tuple[float, float, float, float]:
        """Return ``(x, y, dx, dy)`` for plotting this block at x-coordinate ``x``."""
        x = x if self.start is None else self.start
        dx = 1 if equal_wide else self.time
        res = x, self.stage + 0.5, dx, 1
        return res

    def loop(self, comm: bool = False) -> List["BlockSim"]:
        """Return the first dependency cycle discovered via DFS, or an empty list."""
        if self.flag and not self.in_queue:
            return []
        res = []
        if self.in_queue:
            loop = [self]
            block = self.father
            while block.father and block is not self:
                block = block.father
                loop.append(block)
            return loop
        self.flag = True
        self.in_queue = True
        depends = [self.depend_pre, self.depend_left] if comm else [self.pre, self.left]
        for dep in depends:
            if dep:
                dep.father = self
                res.extend(dep.loop(comm=comm))
                dep.father = None
        self.in_queue = False
        return res

    def comm_loop(self) -> list[BlockSim]:
        r"""recursively check comm loop"""
        return self.loop(True)


@dataclass
class HeadBlockSim(BlockSim):
    r"""sim block of head"""
    stage: int # p
    type: str = 'h'
    id: int = field(repr=False, init=False)
    state: str = field(repr=False, init=False)
    chunk: int = field(repr=False, init=False)
    time: float = 0.
    start: float = 0.
    end: float = 0.
    finish = True

    @property
    def label(self) -> tuple:
        """Identity tuple ``(type, stage)`` for head sentinel blocks."""
        return (self.type, self.stage)

    @property
    def repr(self) -> str:
        """Multi-line representation listing every block chained to the right of this head."""
        s_list = []
        block = self
        while block:
            s_list.append(repr(block))
            block = block.right
        return '\n'.join(s_list)

    # pylint: disable=W0613
    def draw(self, ax: plt.Axes, *args: object, **kwargs: object) -> None:
        """No-op: the head sentinel is not rendered."""
        return

    def build_without_comm(self) -> None:
        """No-op: the head sentinel has no non-comm dependencies."""
        return

    def build_with_comm(self) -> None:
        """No-op: the head sentinel has no comm dependencies."""
        return

    def reset_time_recursive(self) -> None:
        """No-op: the head sentinel has no times to reset."""
        return


@dataclass
class MicroBlockSim(BlockSim):
    r"""compute sim block"""
    type: str = 'c'
    mem: float = 0.
    mem_par: float = 0.
    phase: str = None
    # pylint: disable=E0601
    send_block: SendBlockSim = field(repr=False, default=None)
    # pylint: disable=E0601
    rec_block: RecBlockSim = field(repr=False, default=None)

    def __post_init__(self) -> None:
        """Choose the ANSI colour based on forward / backward state."""
        self._color = '1;34' if self.state == 'f' else '1;33'

    # pylint: disable=W0613
    def draw(self, ax: plt.Axes, *args: object, **kwargs: object) -> None:
        """Render this compute block as a coloured rectangle on ``ax``."""
        x, y, dx, dy = self.loc_size(kwargs.get('index', 0), kwargs.get('equal_wide', False))
        color = (167 / 255, 184 / 255, 231 / 255) if self.state == 'f' else (255 / 255, 213 / 255, 143 / 255)
        mix_color = (240 / 255, 255 / 255, 245 / 255) if self.state == 'f' else (255 / 255, 240 / 255, 255 / 255)
        color = color_mix(mix_color, color, w1=self.chunk / 3)
        if self.phase == 'warmup' and kwargs.get('phase', False):
            edgecolor = 'lightblue'
        elif self.phase == 'cooldown' and kwargs.get('phase', False):
            edgecolor = 'orange'
        else:
            edgecolor = 'black'
        rect = Rectangle((x, y - dy / 2), dx, dy, facecolor=color, edgecolor=edgecolor, linewidth=0.4)
        if dx > 0.008 * kwargs.get('width', 0):
            ax.text(rect.xy[0] + dx / 2, rect.xy[1] + dy / 2, str(self.id),
                    ha='center', va='center', color='black', fontdict={'fontsize': 9})
        ax.add_patch(rect)

    def reset_time_recursive(self) -> None:
        r"""reset block time"""
        if self.finish:
            self.pre.reset_time_recursive()
            self.left.reset_time_recursive()
            self.reset_time()


@dataclass
class CommBlockSim(BlockSim):
    r"""sim comm block"""
    host: MicroBlockSim = field(repr=False, default=None)
    dual: CommBlockSim = field(repr=False, default=None)

    def get_triangle(self, x: float, y: float, dx: float,
                     dy: float) -> List[List[float]]:
        """Return the three triangle vertices used by :meth:`draw`."""
        raise NotImplementedError

    # pylint: disable=W0613
    def draw(self, ax: plt.Axes, *args: object, **kwargs: object) -> None:
        """Render this communication block as a coloured triangle on ``ax``."""
        color = (167 / 255, 184 / 255, 231 / 255) if self.state == 'f' else (255 / 255, 213 / 255, 143 / 255)
        mix_color = (240 / 255, 255 / 255, 255 / 255) if self.state == 'f' else (255 / 255, 240 / 255, 255 / 255)
        color = color_mix(mix_color, color, w1=1.2 * self.chunk / 3)
        index, equal_wide, mode = (kwargs.get('index', 0), kwargs.get('equal_wide', False),
                                   kwargs.get('mode', 'compact'))
        x, y, dx, dy = self.loc_size(index, equal_wide, mode)
        xy = self.get_triangle(x, y, dx, dy)
        tri = Polygon(xy, closed=True, facecolor=color, edgecolor='black', linewidth=0.4)
        ax.add_patch(tri)


@dataclass
class SendBlockSim(CommBlockSim):
    r"""sim send comm block"""
    type: str = 's'
    _color = '35'

    def loc_size(self, x: float = 0, equal_wide: bool = False,
                 mode: str = 'compact') -> Tuple[float, float, float, float]:
        """Return ``(x, y, dx, dy)`` for a send block relative to its compute host."""
        host_x, _, hostdx_, _ = self.host.loc_size(x, equal_wide)
        x, y, _, _ = super().loc_size(x, equal_wide)
        dx_ = self.time
        dy_ = min(np.sqrt(self.time) * 0.6, 0.6)
        if mode == 'compact':
            x = host_x + hostdx_ - dx_
        res = x, y, dx_, dy_
        return res

    def get_triangle(self, x: float, y: float, dx: float,
                     dy: float) -> List[List[float]]:
        """Return the three triangle vertices for a send block pointing to the right."""
        return [[x, y - dy / 2], [x, y + dy / 2], [x + dx, y]]

    # pylint: disable=W0613
    def draw_comm(self, ax: plt.Axes, *args: object, **kwargs: object) -> None:
        """Draw the send→receive connector arrow between paired blocks on ``ax``."""
        index_from, index_to = (kwargs.get('index_from', 0), kwargs.get('index_to', 0))
        equal_wide, mode = (kwargs.get('equal_wide', False), kwargs.get('mode', 'compact'))
        x, y, dx, _ = self.loc_size(index_from, equal_wide, mode)
        x_, y_, dx_, _ = self.dual.loc_size(index_to, equal_wide, mode)
        ax.annotate(None, xy=(x_ - dx_ / 2, y_), xytext=(x + dx / 2, y),
                    arrowprops={"ec": 'grey', "arrowstyle": '->', "shrinkA": 2, "shrinkB": 2})

    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.dual.depend_left.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_left.end, self.dual.depend_left.end)
        self.end = self.start + self.time

    def loop(self, comm: bool = False) -> List["BlockSim"]:
        """Delegate to :meth:`comm_loop` when ``comm`` is ``True``, else fall back to base."""
        if comm:
            return self.comm_loop()
        return super().loop(comm)

    def comm_loop(self) -> list[BlockSim]:
        r"""recursively check comm loop"""
        if self.flag and not self.in_queue:
            return []
        res = []
        if self.in_queue:
            loop = [self]
            block = self.father
            while block.father and block is not self:
                block = block.father
                loop.append(block)
            return loop
        self.flag = True
        self.in_queue = True
        depends = [self.dual.depend_left, self.depend_left]
        for dep in depends:
            if dep:
                dep.father = self
                res.extend(dep.comm_loop())
                dep.father = None
        self.in_queue = False
        return res


@dataclass
class RecBlockSim(CommBlockSim):
    r"""sim receive comm block"""
    type: str = 'r'
    _color = '32'

    def loc_size(self, x: float = 0, equal_wide: bool = False,
                 mode: str = 'compact') -> Tuple[float, float, float, float]:
        """Return ``(x, y, dx, dy)`` for a receive block relative to its compute host."""
        host_x, _, _, _ = self.host.loc_size(x, equal_wide)
        x, y, _, _ = super().loc_size(x, equal_wide)
        dx_ = self.time
        dy_ = min(np.sqrt(self.time) * 0.6, 0.6)
        if mode == 'compact':
            x = host_x
        res = x, y, -dx_, -dy_
        return res

    def get_triangle(self, x: float, y: float, dx: float,
                     dy: float) -> List[List[float]]:
        """Return the three triangle vertices for a receive block pointing to the left."""
        return [[x, y], [x - dx, y + dy / 2], [x - dx, y - dy / 2]]

    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.dual.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_left.end, self.dual.start)
        self.end = self.start + self.time
