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
"""Exception types raised by the pipeline simulator when dependency loops are detected."""
from __future__ import annotations

import matplotlib.pyplot as plt

from hyper_parallel.auto_parallel.sapp_ppb.simulator.plot_manager import PlotMgr
from hyper_parallel.auto_parallel.sapp_ppb.simulator.sim_block import BlockSim, MicroBlockSim
from hyper_parallel.auto_parallel.sapp_ppb.utils.logger import logger


class CausalError(Exception):
    """Raised when the block pipeline (without comm) contains a dependency loop."""

    def __init__(self, msg: str, blocks: list[list[MicroBlockSim]], loop: list[BlockSim]) -> None:
        """Create the error, draw the offending loop and log a diagnostic message.

        Args:
            msg: Human-readable description of the loop.
            blocks: Full 2-D grid of simulator blocks, ``[pp_rank][block_idx]``.
            loop: Sequence of blocks participating in the dependency loop.
        """
        super().__init__()
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_loop(blocks, loop, 0, False, False, True)
        self.canvas.ax[0].set_title("Block pipeline dependency")
        logger.error("%s", self.canvas.msg)

    def __str__(self) -> str:
        """Show the diagnostic plot and return the error message."""
        plt.show()
        return f"{self.msg}"


class CausalCommError(Exception):
    """Raised when the block pipeline with communication contains a dependency loop."""

    def __init__(self, msg: str, blocks: list[list[MicroBlockSim]], loop: list[BlockSim]) -> None:
        """Create the error, draw the offending loop and log a diagnostic message.

        Args:
            msg: Human-readable description of the loop.
            blocks: Full 2-D grid of simulator blocks, ``[pp_rank][block_idx]``.
            loop: Sequence of blocks (compute + comm) participating in the dependency loop.
        """
        super().__init__()
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_comm_loop(blocks, loop, 0)
        self.canvas.ax[0].set_title("Block comm pipeline dependency")
        logger.error("%s", self.canvas.msg)

    def __str__(self) -> str:
        """Show the diagnostic plot and return the error message."""
        plt.show()
        return f"{self.msg}"
