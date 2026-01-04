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
"""
Memory parameter management module for Pipeline Parallel (PP) optimization.

Defines Memory class to:
- Store fixed memory parameters (activation/recomputation/layer/static memory)
- Calculate stage-specific memory limits (stage0/others/last stage)
- Print/write memory parameters to file
- Update memory limits when base parameters change
"""

from fast_tuner.utils.logger import logger


class Memory:
    """
    Memory parameter management for Pipeline Parallel (PP) memory constraint calculation.

    Class Attributes (fixed memory parameters, unit: MB):
        select_mem0/12/select_mem: Selective recomputation memory per layer
        re_comp_mem0/12/re_comp_mem: Full recomputation memory per layer
        act_mem0/12/act_mem: Activation memory per layer
        layer_mem012/layer_mem: Layer parameter memory
        static_mem0: Static memory for stage 0 (excluding layer/activation)
        static_mem: Static memory for middle stages
        lm_head_mem: LM head memory for last stage

    Instance Attributes:
        mem_lim: Global memory limit per stage (total available memory)
        mem_lim_stage0: Memory limit for stage 0 (mem_lim - static_mem0)
        mem_lim_others: Memory limit for middle stages (mem_lim - static_mem)
        mem_lim_last: Memory limit for last stage (mem_lim - lm_head_mem)
    """
    # Fixed memory parameters (unit: MB)
    select_mem0 = 76
    select_mem12 = 76
    select_mem = 77
    re_comp_mem0 = 3
    re_comp_mem12 = 3
    re_comp_mem = 5
    act_mem0 = 79
    act_mem12 = 79
    act_mem = 79
    layer_mem012 = 349
    layer_mem = 340
    static_mem0 = 734
    static_mem = 116
    lm_head_mem = 690

    def __init__(self, mem_lim):
        """
        Initialize Memory instance with global memory limit.

        Args:
            mem_lim: Global memory limit per stage (total available memory, unit: MB)
        """
        self.mem_lim = mem_lim
        self.mem_lim_stage0 = mem_lim - self.static_mem0
        self.mem_lim_others = mem_lim - self.static_mem
        self.mem_lim_last = mem_lim - self.lm_head_mem

    def update_up_mem(self):
        """
        Update stage-specific memory limits after changing static/lm_head parameters.

        Recalculates:
        - mem_lim_stage0: mem_lim - static_mem0
        - mem_lim_others: mem_lim - static_mem
        - mem_lim_last: mem_lim - lm_head_mem
        """
        self.mem_lim_stage0 = self.mem_lim - self.static_mem0
        self.mem_lim_others = self.mem_lim - self.static_mem
        self.mem_lim_last = self.mem_lim - self.lm_head_mem

    def print_mem(self):
        """Print all memory parameters (fixed + stage-specific limits) to logger."""
        logger.info(
            'select_mem0=%s, select_mem12=%s, select_mem=%s, '
            're_comp_mem0=%s, re_comp_mem12=%s, re_comp_mem=%s, '
            'act_mem0=%s, act_mem12=%s, act_mem=%s, '
            'layer_mem012=%s, layer_mem=%s, '
            'static_mem0=%s, static_mem=%s, lm_head_mem=%s, '
            'mem_lim_stage0=%s, mem_lim_others=%s, mem_lim_last=%s',
            self.select_mem0, self.select_mem12, self.select_mem,
            self.re_comp_mem0, self.re_comp_mem12, self.re_comp_mem,
            self.act_mem0, self.act_mem12, self.act_mem,
            self.layer_mem012, self.layer_mem,
            self.static_mem0, self.static_mem, self.lm_head_mem,
            self.mem_lim_stage0, self.mem_lim_others, self.mem_lim_last
        )

    def write_memory_to_file(self, mem_file):
        """
        Write all fixed memory parameters to a text file.

        Args:
            mem_file: Path to output memory parameter file (UTF-8 encoding)
        """
        with open(mem_file, 'w', encoding='utf-8') as file:
            file.write(f'select_mem0={self.select_mem0}\n')
            file.write(f'select_mem12={self.select_mem12}\n')
            file.write(f'select_mem={self.select_mem}\n')
            file.write(f're_comp_mem0={self.re_comp_mem0}\n')
            file.write(f're_comp_mem12={self.re_comp_mem12}\n')
            file.write(f're_comp_mem={self.re_comp_mem}\n')
            file.write(f'act_mem0={self.act_mem0}\n')
            file.write(f'act_mem12={self.act_mem12}\n')
            file.write(f'act_mem={self.act_mem}\n')
            file.write(f'layer_mem012={self.layer_mem012}\n')
            file.write(f'layer_mem={self.layer_mem}\n')
            file.write(f'static_mem0={self.static_mem0}\n')
            file.write(f'static_mem={self.static_mem}\n')
            file.write(f'lm_head_mem={self.lm_head_mem}\n')
        logger.info(f'Write memory info to {mem_file}')

    def get_mem(self):
        """
        Get all fixed memory parameters as a formatted string.

        Returns:
            str: Newline-separated key=value pairs of fixed memory parameters
        """
        mem = (
            f'select_mem0={self.select_mem0}\n'
            f'select_mem12={self.select_mem12}\n'
            f'select_mem={self.select_mem}\n'
            f're_comp_mem0={self.re_comp_mem0}\n'
            f're_comp_mem12={self.re_comp_mem12}\n'
            f're_comp_mem={self.re_comp_mem}\n'
            f'act_mem0={self.act_mem0}\n'
            f'act_mem12={self.act_mem12}\n'
            f'act_mem={self.act_mem}\n'
            f'layer_mem012={self.layer_mem012}\n'
            f'layer_mem={self.layer_mem}\n'
            f'static_mem0={self.static_mem0}\n'
            f'static_mem={self.static_mem}\n'
            f'lm_head_mem={self.lm_head_mem}'
        )
        return mem
