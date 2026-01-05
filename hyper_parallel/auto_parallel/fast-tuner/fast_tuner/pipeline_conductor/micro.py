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
Micro-batch scheduling and peak memory calculation module for Pipeline Parallel (PP) optimization.

Core components:
- Micro: Data class for micro-batch task metadata (part/vpp/state/micro_id/split)
- SortMicro: Generates micro-batch execution order (forward/backward) for each pipeline stage
- PeakNum: Calculates peak memory consumption and related metrics (activation/recomputation) per stage
"""
from fast_tuner.pipeline_conductor.memory import Memory


class Micro:
    """
    Data class for micro-batch task metadata in Pipeline Parallel (PP) scheduling.

    Class Attributes (type hints):
        part: Micro-batch part index (int)
        vpp: Virtual PP interleave index (int)
        state: Task state ('f' for forward, 'b' for backward) (str)
        micro_id: Micro-batch ID (int)
        split: Sequence split index (int)

    Instance Attributes:
        part: Micro-batch part index
        vpp: Virtual PP interleave index
        state: Task state ('f'/'b')
        micro_id: Micro-batch ID
        split: Sequence split index
    """
    part = int
    vpp = int
    state = 'f'
    micro_id = int
    split = int

    def __init__(self, part, vpp, state, micro_id, split):
        """
        Initialize Micro instance with task metadata.

        Args:
            part: Micro-batch part index
            vpp: Virtual PP interleave index
            state: Task state ('f' for forward, 'b' for backward)
            micro_id: Micro-batch ID
            split: Sequence split index
        """
        self.part = part
        self.vpp = vpp
        self.state = state
        self.micro_id = micro_id
        self.split = split


class SortMicro:
    """
    Generates micro-batch execution order for each pipeline stage in PP optimization.

    Class Attributes:
        parts: Number of micro-batch parts (int)
        num_vpp: Number of virtual PP interleaves (int)
        num_stage: Number of pipeline stages (int)
        distribution: Micro-batch distribution across parts (list)
        low_mem: Whether low memory mode is enabled (bool)
        seq_split: Number of sequence splits (int)
        is_f_then_b: Whether to execute all forward then backward (bool, default False)

    Instance Attributes:
        forward: List of forward micro-batch tasks (Micro instances)
        backward: List of backward micro-batch tasks (Micro instances)
        warmup_num: Warmup micro-batch count per stage (list)
        final_orders: Final execution order per stage (list of lists of Micro)
    """
    parts = int
    num_vpp = int
    num_stage = int
    distribution = []
    low_mem = bool
    seq_split = int
    is_f_then_b = False

    def __init__(self, parts, num_vpp, num_stage, distribution, low_mem, seq_split):
        """
        Initialize SortMicro and generate micro-batch execution order.

        Args:
            parts: Number of micro-batch parts
            num_vpp: Number of virtual PP interleaves
            num_stage: Number of pipeline stages
            distribution: Micro-batch distribution across parts
            low_mem: Whether low memory mode is enabled
            seq_split: Number of sequence splits
        """
        self.forward = []
        self.backward = []
        self.warmup_num = []
        self.final_orders = []
        self.parts = parts
        self.num_vpp = num_vpp
        self.num_stage = num_stage
        self.distribution = distribution
        self.low_mem = low_mem
        self.seq_split = seq_split
        self.build_f_b_sort()
        self.set_warmup_num()
        self.set_micro_sort()

    def build_f_b_sort(self):
        """
        Build forward/backward micro-batch task lists.

        Generates:
        - forward list: All forward tasks (ordered by part/vpp/micro_id/split)
        - backward list: All backward tasks (reverse ordered by part/vpp/micro_id/split)
        """
        for part in range(self.parts):
            for vpp in range(self.num_vpp):
                for micro_id in range(self.distribution[part]):
                    for split in range(self.seq_split):
                        micro = Micro(part, vpp, 'f', micro_id, split)
                        self.forward.append(micro)
            for vpp in range(self.num_vpp - 1, -1, -1):
                for micro_id in range(self.distribution[part]):
                    for split in range(self.seq_split - 1, -1, -1):
                        micro = Micro(part, vpp, 'b', micro_id, split)
                        self.backward.append(micro)

    def set_warmup_num(self):
        """
        Calculate warmup micro-batch count per pipeline stage.

        Warmup count is the number of forward tasks to execute before starting backward tasks,
        adjusted for low memory mode and last stage special case.
        """
        for stage in range(self.num_stage):
            if self.low_mem:
                warmup = min(
                    ((self.num_vpp - 1) * self.distribution[0] + (self.num_stage - stage - 1)) * self.seq_split,
                    len(self.forward)
                )
            else:
                warmup = min(
                    ((self.num_vpp - 1) * self.distribution[0] + (self.num_stage - stage - 1) * 2) * self.seq_split,
                    len(self.forward)
                )
            # For the last stage, the forward process of the first micro must be completed before the backward process.
            if stage == self.num_stage - 1:
                warmup = warmup + self.seq_split - 1
            self.warmup_num.append(warmup)

    def set_micro_sort(self):
        """
        Generate final micro-batch execution order per stage.

        Combines forward/backward tasks with warmup phase:
        1. Warmup phase: Only forward tasks
        2. Steady phase: Alternate forward + backward tasks
        3. Cleanup phase: Remaining backward tasks
        """
        for stage in range(self.num_stage):
            stage_order = []
            stage_order += self.forward[: self.warmup_num[stage]]
            for i in range(self.warmup_num[stage], len(self.forward)):
                stage_order.append(self.forward[i])
                stage_order.append(self.backward[i - self.warmup_num[stage]])
            stage_order += self.backward[len(self.forward) - self.warmup_num[stage]:]
            self.final_orders.append(stage_order)


class PeakNum:
    """
    Calculates peak memory consumption and related metrics per pipeline stage.

    Tracks peak values for:
    - Recomputation (type1/type2 layers)
    - Selective recomputation (type1/type2 layers)
    - Activation memory (type1/type2 layers)
    - Maximum memory consumption per stage

    Class Attributes:
        sort_micro: SortMicro instance for micro-batch order (SortMicro)

    Instance Attributes:
        peak_num_recompute_type1/2: Peak recomputation count for type1/2 layers per stage (dict)
        peak_num_select_recom_type1/2: Peak selective recomputation count per stage (dict)
        peak_num_act_type1/2: Peak activation count per stage (dict)
        max_mem: Maximum memory consumption per stage (dict)
        micro_num_of_max_mem: Micro-batch index of peak memory per stage (dict)
        sort_micro: SortMicro instance
        num_stage: Number of pipeline stages
        num_vpp: Number of PP interleaves
    """
    sort_micro = SortMicro

    def __init__(self, sort_micro: SortMicro):
        """
        Initialize PeakNum with micro-batch order information.

        Args:
            sort_micro: SortMicro instance containing micro-batch execution order
        """
        self.peak_num_recompute_type1 = {}
        self.peak_num_recompute_type2 = {}
        self.peak_num_select_recom_type1 = {}
        self.peak_num_select_recom_type2 = {}
        self.peak_num_act_type1 = {}
        self.peak_num_act_type2 = {}
        self.max_mem = {}
        self.micro_num_of_max_mem = {}
        self.sort_micro = sort_micro
        self.num_stage = sort_micro.num_stage
        self.num_vpp = sort_micro.num_vpp

    def set_peak_act_recompute_num(self, x_type2, rs_type2, ra_type2, x_type1, rs_type1, ra_type1, memory: Memory):
        """
        Calculate peak memory and related metrics per pipeline stage.

        Computes:
        - Stage-specific static + layer memory
        - Dynamic memory (activation/recomputation) per micro-batch
        - Peak memory consumption and corresponding recomputation/activation counts

        Args:
            x_type2: Type2 layer count per vpp/stage (2D list)
            rs_type2: Selective recomputation count for type2 layers (2D list)
            ra_type2: Full recomputation count for type2 layers (2D list)
            x_type1: Type1 layer count per vpp/stage (2D list)
            rs_type1: Selective recomputation count for type1 layers (2D list)
            ra_type1: Full recomputation count for type1 layers (2D list)
            memory: Memory instance with fixed memory parameters
        """
        for stage in range(self.sort_micro.num_stage):
            # The memory and activation/recalculation counts in each micro.
            self.max_mem[stage] = 0
            if stage == 0:
                static_mem = memory.static_mem0
            elif stage == self.sort_micro.num_stage - 1:
                static_mem = memory.lm_head_mem
            else:
                static_mem = memory.static_mem
            layer_mem = (
                memory.layer_mem012 * sum(x_type1[vpp][stage] for vpp in range(self.num_vpp)) +
                memory.layer_mem * sum(x_type2[vpp][stage] for vpp in range(self.num_vpp))
            )
            mem_stage = static_mem + layer_mem
            num_recom_type1_stage = 0
            num_recom_type2_stage = 0
            num_select_recom_type1_stage = 0
            num_select_recom_type2_stage = 0
            num_act_type1_stage = 0
            num_act_type2_stage = 0
            for i in range(len(self.sort_micro.final_orders[stage])):
                micro_batch = self.sort_micro.final_orders[stage][i]
                vpp = micro_batch.vpp
                act_num_type1 = x_type1[vpp][stage] - rs_type1[vpp][stage] - ra_type1[vpp][stage]
                act_num_type2 = x_type2[vpp][stage] - rs_type2[vpp][stage] - ra_type2[vpp][stage]
                act_mem = memory.act_mem12 * act_num_type1 + memory.act_mem * act_num_type2
                ra_mem = memory.re_comp_mem12 * ra_type1[vpp][stage] + memory.re_comp_mem * ra_type2[vpp][stage]
                rs_mem = memory.select_mem12 * rs_type1[vpp][stage] + memory.select_mem * rs_type2[vpp][stage]

                # The computation time is divided into seq_split parts,
                # which can be regarded as the data being divided into seq_split parts,
                # this corresponds to the segmentation of dynamic memory.
                # layer_mem and static_mem remain unchanged.

                total_mem = (act_mem + ra_mem + rs_mem) / self.sort_micro.seq_split
                if micro_batch.state == 'f':
                    mem_stage += total_mem
                    num_recom_type1_stage += ra_type1[vpp][stage]
                    num_recom_type2_stage += ra_type2[vpp][stage]
                    num_select_recom_type1_stage += rs_type1[vpp][stage]
                    num_select_recom_type2_stage += rs_type2[vpp][stage]
                    num_act_type1_stage += act_num_type1
                    num_act_type2_stage += act_num_type2
                else:
                    mem_stage -= total_mem
                    num_recom_type1_stage -= ra_type1[vpp][stage]
                    num_recom_type2_stage -= ra_type2[vpp][stage]
                    num_select_recom_type1_stage -= rs_type1[vpp][stage]
                    num_select_recom_type2_stage -= rs_type2[vpp][stage]
                    num_act_type1_stage -= act_num_type1
                    num_act_type2_stage -= act_num_type2
                if mem_stage > self.max_mem[stage]:
                    self.max_mem[stage] = mem_stage
                    self.peak_num_recompute_type1[stage] = num_recom_type1_stage
                    self.peak_num_recompute_type2[stage] = num_recom_type2_stage
                    self.peak_num_select_recom_type1[stage] = num_select_recom_type1_stage
                    self.peak_num_select_recom_type2[stage] = num_select_recom_type2_stage
                    self.peak_num_act_type1[stage] = num_act_type1_stage
                    self.peak_num_act_type2[stage] = num_act_type2_stage
                    self.micro_num_of_max_mem[stage] = i + 1
