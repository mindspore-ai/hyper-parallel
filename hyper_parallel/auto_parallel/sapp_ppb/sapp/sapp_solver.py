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
"""Solver Class"""

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

import pulp as lpSolver

import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer
from hyper_parallel.auto_parallel.sapp_ppb.utils.logger import logger

# seqpipe const
TENSOR_FLOAT_16 = 2
TENSOR_FLOAT_32 = 4
const_from_byte_to_mb = 1024 * 1024
# llama intermideate_size
LLAMA_INTERMEDIATE_SIZE = 11008


@dataclass
class PipelineMemoryConstraint:
    """constraint struct"""
    prob: Any
    variables: Any
    layers_sorted: dict[Any]
    num_of_stage: int
    num_of_interleave: int
    micro_batch: int
    memory_limit: int


class SappSolver:
    """solver for pipeline balance"""

    BIG_M = 1000000

    MEM_OVERHEAD_NAME = "memory_overhead"
    TOTAL_SUM = "var_sum_FPi_BPi"
    CHUNKS_SUM = "chunks_sum"
    PREV_DIFF = "prev_diff"
    NEXT_DIFF = "next_diff"
    MAX_STAGE_TIME = "max_stage_time"
    MAX_LAST_CHUNK = "max_last_chunk"
    LAYER_FRONTIER = "layer_frontier"
    REC_FRONTIER = "recompute_frontier"
    PROP_PHASE = IntEnum("Propagation", ["FW", "BW"], start=0)

    def __init__(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            num_of_micro_batch: int,
            max_memory: int,
            layers: list[Layer],
            layers_sorted: dict[Layer.type_enum, list[Layer]],
            vpp_less_memory: bool = False,
            # add dualpipe_v arg
            dual: bool = False,
            constant_memory: int = 0,
            optimization_level: int = 1,
            description: str = "Pipeline_execution_time_minimize",
            extracted_training_params: dict[str, int] = None,
            seq_split_num: int = 1,
    ) -> None:
        """Build the ILP variables and the empty problem skeleton.

        Args:
            num_of_stage: Number of physical pipeline stages.
            num_of_interleave: Virtual-pipeline (VPP) chunk count.
            num_of_micro_batch: Number of micro-batches.
            max_memory: Per-device memory budget (MB).
            layers: Flat list of :class:`Layer` descriptors covering the full model.
            layers_sorted: ``layers`` indexed by HEAD / BODY / TAIL classification.
            vpp_less_memory: Use the less-memory VPP scheduler variant.
            dual: Enable dualpipe-V scheduling support.
            constant_memory: Constant per-stage memory overhead (MB).
            optimization_level: Solver optimization level (``0-2``).
            description: Problem description used when exporting the LP model.
            extracted_training_params: Optional training params for sequence-pipeline mode.
            seq_split_num: Number of sequence splits (``>1`` enables sequence pipeline).
        """

        self.num_of_stage_ = num_of_stage
        self.num_of_interleave_ = num_of_interleave
        self.num_of_micro_batch_ = num_of_micro_batch
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        # Add dualpipe_v
        self.dual_ = dual
        self.constant_memory_ = constant_memory
        self.optimization_level_ = optimization_level
        self.layers_ = layers
        self.layers_sorted_ = layers_sorted

        self.recompute_considered_ = self.find_recompute_considered(
            layers_sorted)
        self.extracted_training_params_ = extracted_training_params
        self.seq_split_num_ = seq_split_num
        self.seq_pipe = self.seq_split_num_ > 1
        if self.seq_pipe:
            self._initialize_seq_pipe_layers()

        self.variables_ = self._create_variables_to_solve_(
            num_of_stage, num_of_interleave, layers_sorted)
        self.problem_ = self._create_problem_(description)

    def _initialize_seq_pipe_layers(self):
        """Update memory and time metadata for sequence pipeline mode."""
        self._update_seq_pipe_memory()
        self.num_of_micro_batch_ *= self.seq_split_num_
        self._update_seq_pipe_time()

    def _update_seq_pipe_memory(self):
        """Update layer memory values for sequence pipeline mode."""
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            self._update_body_seq_memory(layer)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            self._update_head_seq_memory(head)
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            self._update_tail_seq_memory(tail)

    def _update_body_seq_memory(self, layer):
        """Update body layer memory values for sequence pipeline mode."""
        if layer.memory_parameter_ is not None:
            logger.info("Body Layer 1f1b Parameter Memory: %s", layer.memory_parameter_)
            layer.memory_parameter_ = self.compute_seq_mem_parameter(
                layer.memory_parameter_, self.extracted_training_params_)
            logger.info("Body Layer Seq Parameter Memory: %s", layer.memory_parameter_)
        for rec in Recompute.TYPE:
            if not self.recompute_considered_[rec]:
                continue
            if rec.name == "FULL":
                self.recompute_considered_[rec] = False
                layer.memory_activation_rec_[rec] = None
                logger.error("Seqpipe doesn't support full recomputation, "
                             "recompute_activation is set as None for seqpp")
                continue
            logger.info(
                "Body Layer 1f1b %s activation Memory: %s",
                rec,
                layer.memory_activation_rec_[rec],
            )
            layer.memory_activation_rec_[rec] = self.compute_seq_mem_activation(
                layer.memory_activation_rec_[rec],
                self.extracted_training_params_,
                self.seq_split_num_
            )
            logger.info(
                "Body Layer seq %s activation Memory: %s",
                rec,
                layer.memory_activation_rec_[rec],
            )

    def _update_head_seq_memory(self, head):
        """Update head layer memory values for sequence pipeline mode."""
        if head.memory_parameter_ is None:
            return
        logger.info("Head cost 1f1b: %s", head.memory_parameter_)
        head.memory_parameter_ = self.compute_seq_mem_head_cost(
            head.memory_parameter_, self.extracted_training_params_, self.seq_split_num_)
        logger.info("Head cost Seq: %s", head.memory_parameter_)

    def _update_tail_seq_memory(self, tail):
        """Update tail layer memory values for sequence pipeline mode."""
        if tail.memory_parameter_ is None:
            return
        logger.info("Tail cost 1f1b: %s", tail.memory_parameter_)
        tail.memory_parameter_ = self.compute_seq_mem_tail_cost(
            tail.memory_parameter_, self.extracted_training_params_, self.seq_split_num_)
        logger.info("Tail cost seq: %s", tail.memory_parameter_)

    def _update_seq_pipe_time(self):
        """Update layer times for sequence pipeline mode."""
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            self._update_layer_seq_time(layer, "Body")
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            self._update_layer_seq_time(head, "Head")
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            self._update_layer_seq_time(tail, "Tail")

    def _update_layer_seq_time(self, layer, layer_name):
        """Scale one layer's time by the sequence split number."""
        logger.info("%s Layer 1f1b fp time: %s", layer_name, layer.forward_time_)
        logger.info("%s Layer 1f1b bp time:", layer_name)
        for key, value in layer.backward_time_rec_.items():
            logger.output("%s: %s", key, value)
        layer.time_ = layer.time_ / self.seq_split_num_
        layer.forward_time_ = layer.forward_time_ / self.seq_split_num_
        layer.update_internal_time_for_seqpp()
        logger.info("%s Layer seq fp time: %s", layer_name, layer.forward_time_)
        logger.info("%s Layer seq bp time:", layer_name)
        for key, value in layer.backward_time_rec_.items():
            logger.output("%s: %s", key, value)

    @staticmethod
    def compute_forward_in_backward(num_of_stage: int,
                                    micro_batch: int) -> list[int]:
        """Computes the number of forward propagation happening after a backward"""
        n = num_of_stage - 1
        factors = []
        for _ in range(num_of_stage):
            factors.append(abs(n))
            n -= 2
        if micro_batch < 2 * num_of_stage:
            for i in range(num_of_stage // 2):
                factors[i] = 0
        return factors

    @staticmethod
    def compute_lm_forward_in_backward(num_of_stage: int) -> list[int]:
        """Function compute_forward_in_backward in less_memory schedule"""
        return list(range(num_of_stage))

    @staticmethod
    def compute_activation_nums(num_of_stage: int, num_of_interleave: int,
                                micro_batch: int) -> list[list[int]]:
        """compute the number of activation"""
        activation_nums = []

        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[0][s] += max(0, num_of_stage - 2 * s - 1)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] += min(
                    0, num_of_stage - 2 * s - 1)
            for i in range(num_of_interleave):
                for s in range(num_of_stage):
                    activation_nums[i][s] = min(activation_nums[i][s],
                                                micro_batch)
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)

        return activation_nums

    @staticmethod
    def compute_activation_nums_dual(num_of_stage: int, num_of_interleave: int,
                                     micro_batch: int) -> list[list[int]]:
        """compute the number of activation for dualpipe_v"""
        activation_nums = []

        for i in range(num_of_interleave):
            activation_nums.append([])
            for _ in range(num_of_stage):
                activation_nums[i].append(0)
        for s in range(num_of_stage):
            activation_nums[0][s] += max(0, 2 * num_of_stage - s)
        for s in range(num_of_stage):
            activation_nums[num_of_interleave - 1][s] += max(
                    0, s + 1)
        for i in range(num_of_interleave):
            for s in range(num_of_stage):
                activation_nums[i][s] = min(activation_nums[i][s],
                                            micro_batch)

        return activation_nums

    @staticmethod
    def compute_less_activation_nums(
            num_of_stage: int, num_of_interleave: int) -> list[list[int]]:
        """compute number of less_mem activation"""
        activation_nums = []
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] -= s
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)
        return activation_nums

    #######################################################################
    ##                                                                   ##
    ##                            SeqPipe                                ##
    ##                                                                   ##
    #######################################################################
    @staticmethod
    def compute_activation_seq_nums(num_of_stage: int, num_of_interleave: int,
                                    seq_split_num: int, micro_batch: int, less_memory: False) -> list[list[int]]:
        """compute the number of activation for seq chunks"""
        activation_nums = []
        if less_memory:
            act_gap = 1
        else:
            act_gap = 2
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] = seq_split_num

            loop_index = 1
            for stage_index in range(num_of_stage - 2, -1, -1):
                flag_added = False
                for chunk_index in range(num_of_interleave):
                    condition1 = activation_nums[chunk_index][stage_index + 1] % num_of_stage != 0
                    condition2 = activation_nums[chunk_index][stage_index + 1] // num_of_stage < loop_index
                    if condition1 or condition2:
                        for update in range(stage_index + 1):
                            activation_nums[chunk_index][update] += act_gap
                        flag_added = True
                        break
                if not flag_added:
                    for update in range(stage_index + 1):
                        activation_nums[0][update] += act_gap
                    loop_index += 1
            # microbatch
            for i in range(num_of_interleave):
                for s in range(num_of_stage):
                    activation_nums[i][s] = min(activation_nums[i][s],
                                                micro_batch)
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s + seq_split_num - 1)

        logger.output("compute_activation_seq_nums: %s", activation_nums)
        return activation_nums

    @staticmethod
    def compute_seq_mem_activation(original_memory_activation: float,
                                   extracted_training_params: dict[str, int],
                                   seq_split_num: int) -> float:
        """compute activation memory for seqpipe"""
        # context parallel? cp?
        batch_size = extracted_training_params['batch_size']
        heads = extracted_training_params['num_heads']
        seq_length = extracted_training_params['seq_length']
        head_dim = extracted_training_params['head_dim']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # 2*Kv add
        # cp?
        kv_update_mem_byte = 2 * ((TENSOR_FLOAT_16 * batch_size * heads * seq_length * head_dim) / (mp))
        kv_update_mem = kv_update_mem_byte / const_from_byte_to_mb
        # Attention Key,Value
        # cp?
        key_mem_byte = (TENSOR_FLOAT_16 * batch_size * heads * seq_length * head_dim) / (mp)
        key_mem = key_mem_byte / const_from_byte_to_mb
        # cp?
        value_mem_byte = (TENSOR_FLOAT_16 * batch_size * heads * seq_length * head_dim) / (mp)
        value_mem = value_mem_byte / const_from_byte_to_mb

        seq_memory_activation = (original_memory_activation - key_mem - value_mem) / seq_split_num + kv_update_mem
        return seq_memory_activation

    @staticmethod
    def compute_seq_mem_parameter(original_memory_parameter: float, extracted_training_params: dict[str, int]) -> float:
        """compute layer parameter memory for seqpipe"""
        # context parallel? cp?
        batch_size = extracted_training_params['batch_size']
        heads = extracted_training_params['num_heads']
        seq_length = extracted_training_params['seq_length']
        head_dim = extracted_training_params['head_dim']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # cp?
        kv_cache_parameter_mem_byte = 4 * (TENSOR_FLOAT_16 * batch_size * heads * seq_length * head_dim / (mp))
        kv_cache_parameter_mem = kv_cache_parameter_mem_byte / const_from_byte_to_mb
        seq_memory_parameter = original_memory_parameter + kv_cache_parameter_mem
        return seq_memory_parameter

    @staticmethod
    def compute_seq_mem_head_cost(original_head_cost: float,
                                  extracted_training_params: dict[str, int],
                                  seq_split_num: int) -> float:
        """compute head stage extra cost for seqpipe"""
        batch_size = extracted_training_params['batch_size']
        seq_length = extracted_training_params['seq_length']
        hidden_size = extracted_training_params['hidden_size']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        if mp > 1:
            # comm operator Mem (recv+reduceScatter)
            # cp?
            comm_operator_mem_byte = 2 * (TENSOR_FLOAT_16 * batch_size * seq_length * hidden_size / (mp))
            comm_operator_mem = comm_operator_mem_byte / const_from_byte_to_mb
            # StridedSliceGrad Operator Mem
            stridslice_operator_mem_byte = TENSOR_FLOAT_16 * batch_size * seq_length * hidden_size
            stridslice_operator_mem = stridslice_operator_mem_byte / const_from_byte_to_mb
            seq_head_cost = original_head_cost - (1 - 1 / seq_split_num) * (comm_operator_mem + stridslice_operator_mem)
        else:
            # comm operator Mem (recv)
            # cp?
            comm_operator_mem_byte = TENSOR_FLOAT_16 * batch_size * seq_length * hidden_size / (mp)
            comm_operator_mem = comm_operator_mem_byte / const_from_byte_to_mb
            # Grad/MatMul // Grad/Mul Operator Mem
            # cp?
            mul_operator_mem_byte = 1 * (TENSOR_FLOAT_16 * batch_size * seq_length * LLAMA_INTERMEDIATE_SIZE / (mp))
            mul_operator_mem = mul_operator_mem_byte / const_from_byte_to_mb
            seq_head_cost = original_head_cost - (1 - 1 / seq_split_num) * (comm_operator_mem + mul_operator_mem)
        return seq_head_cost

    @staticmethod
    def compute_seq_mem_tail_cost(original_tail_cost: float,
                                  extracted_training_params: dict[str, int],
                                  seq_split_num: int) -> float:
        """compute tail stage extra cost for seqpipe"""
        batch_size = extracted_training_params['batch_size']
        seq_length = extracted_training_params['seq_length']
        vocab_size = extracted_training_params['vocab_size']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # Memory extra introduced by loss op:
        # cp?
        loss_operator_mem_byte = TENSOR_FLOAT_32 * batch_size * seq_length * vocab_size / (mp)
        loss_operator_mem = loss_operator_mem_byte / const_from_byte_to_mb
        # New tail Cost = Old tail Cost - (3-3/k)M + (k-1)(M/k)
        seq_tail_cost = original_tail_cost - (3 - 3 / seq_split_num) * loss_operator_mem + (
            seq_split_num - 1) * (loss_operator_mem / seq_split_num)
        return seq_tail_cost

    def add_total_nb_layer_constraint(self, prob: Any, variables: Any,
                                      sorted_layers: Dict[Layer.type_enum, list[Layer]]) -> Any:
        """Enforce that the sum of assigned layers equals ``layer.nb_layer_`` per BODY layer."""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            prob += (lpSolver.lpSum(
                variables[layer.name_][rec] for rec in Recompute.TYPE
                if self.recompute_considered_[rec]) == layer.nb_layer_)
        return prob

    def add_stage_nb_layer_constraint(self, prob: Any, variables: Any,
                                      sorted_layers: Dict[Layer.type_enum, List[Layer]]) -> Any:
        """Require each non-reserved ``(interleave, stage)`` cell to host at least one layer."""
        layer_type_num = len(sorted_layers[Layer.type_enum.BODY])
        reserved_positions = self._reserved_stage_positions()
        for i in range(self.num_of_interleave_):
            for s in range(self.num_of_stage_):
                if (i, s) in reserved_positions:
                    continue
                terms = []
                for rec in Recompute.TYPE:
                    if not self.recompute_considered_[rec]:
                        continue

                    for ll in range(layer_type_num):
                        terms.append(
                            variables[
                                sorted_layers[Layer.type_enum.BODY][ll].name_
                            ][rec][i][s]
                        )

                prob += lpSolver.lpSum(terms) >= 1
        return prob

    def _reserved_stage_positions(self):
        """Return stage positions reserved for head and tail layers."""
        if self.dual_:
            return {(0, 0), (1, 0)}
        return {(0, 0), (self.num_of_interleave_ - 1, self.num_of_stage_ - 1)}

    def add_multimodal_sequence_constraint(
            self, prob: Any, variables: Any,
            sorted_layers: Dict[Layer.type_enum, List[Layer]]) -> Any:
        """Enforce a stage frontier between successive BODY layer types (multimodal models)."""
        for frontier in range(1, len(sorted_layers[Layer.type_enum.BODY])):
            layer = sorted_layers[Layer.type_enum.BODY][frontier].name_
            for v in range(self.num_of_interleave_):
                for s in range(self.num_of_stage_):
                    prob = self._add_frontier_lower_bound(prob, variables, layer, frontier, v, s)
        return self._add_frontier_upper_bounds(prob, variables, sorted_layers)

    def _add_frontier_lower_bound(self, prob, variables, layer, frontier, interleave, stage):
        """Add the lower bound for one multimodal frontier variable."""
        frontier_sum = self._frontier_layer_sum(variables, layer, interleave, stage)
        if frontier_sum is None:
            return prob
        prob += (
            variables[self.LAYER_FRONTIER][frontier - 1][interleave][stage]
            >= frontier_sum / self.BIG_M
        )
        return prob

    def _frontier_layer_sum(self, variables, layer, interleave, stage):
        """Build the layer sum used by multimodal frontier constraints."""
        if self.dual_:
            return self._dual_frontier_layer_sum(variables, layer, interleave, stage)
        return self._current_layer_sum(variables, layer, interleave, range(stage)) + (
            self._previous_layer_sum(variables, layer, interleave)
        )

    def _dual_frontier_layer_sum(self, variables, layer, interleave, stage):
        """Build the layer sum for dualpipe_v multimodal frontier constraints."""
        if interleave == 0:
            return self._current_layer_sum(variables, layer, interleave, range(stage))
        if interleave == 1:
            return self._current_layer_sum(variables, layer, interleave, range(stage, self.num_of_stage_)) + (
                self._previous_layer_sum(variables, layer, interleave)
            )
        return None

    def _current_layer_sum(self, variables, layer, interleave, stage_range):
        """Sum current interleave variables over a stage range."""
        terms = []
        for rec in Recompute.TYPE:
            if not self.recompute_considered_[rec]:
                continue

            for stage in stage_range:
                terms.append(variables[layer][rec][interleave][stage])

        return lpSolver.lpSum(terms)

    def _previous_layer_sum(self, variables, layer, interleave):
        """Sum variables from previous interleaves."""
        terms = []
        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                for prev_interleave in range(interleave):
                    for stage in range(self.num_of_stage_):
                        terms.append(variables[layer][rec][prev_interleave][stage])
        return lpSolver.lpSum(terms)

    def _add_frontier_upper_bounds(self, prob, variables, sorted_layers):
        """Prevent previous body layer types after each multimodal frontier."""
        for frontier in range(1, len(sorted_layers[Layer.type_enum.BODY])):
            layer = sorted_layers[Layer.type_enum.BODY][frontier - 1].name_
            for stage in range(self.num_of_stage_):
                for interleave in range(self.num_of_interleave_):
                    prob = self._add_frontier_upper_bound(prob, variables, layer, frontier, interleave, stage)
        return prob

    def _add_frontier_upper_bound(self, prob, variables, layer, frontier, interleave, stage):
        """Add one upper bound constraint for a multimodal frontier."""
        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                prob += variables[layer][rec][interleave][stage] <= (
                    1 - variables[self.LAYER_FRONTIER][frontier - 1][interleave][stage]
                ) * self.BIG_M
        return prob

    def add_multimodal_recompute_constraint(
            self, prob: Any, variables: Any,
            sorted_layers: Dict[Layer.type_enum, List[Layer]]) -> Any:
        """Keep recomputation schemes consistent across BODY layer types (MindFormer constraint)."""

        # if (self.recompute_considered_[Recompute.TYPE.FULL] and
        #     self.recompute_considered_[Recompute.TYPE.FULL]):

        considered = Recompute.get_used_list(self.recompute_considered_)
        if len(considered) > 2:
            logger.error("Careful: MindFormer does not allow a fine recomputation scheme "
                         "for heterogeneous models. Pipeline balancing is currently unable to "
                         "comply with MF constraint for more than 1 recomputation type.")
            return prob

        if len(considered) < 2:
            # this constraint is unnecessary if there is no recomputation
            return prob

        most_rec = max(considered)
        layer_type_num = len(sorted_layers[Layer.type_enum.BODY])
        for v in range(self.num_of_interleave_):
            for s in range(self.num_of_stage_):
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec] and rec is not Recompute.TYPE.NONE:
                        for layer_idx in range(0, layer_type_num - 1):
                            prob += variables[self.REC_FRONTIER][v][s][layer_idx] >= (
                                lpSolver.lpSum(
                                    variables[sorted_layers[Layer.type_enum.BODY][next_idx].name_][most_rec][v][s]
                                    for next_idx in range(layer_idx + 1, layer_type_num))) / self.BIG_M

        least_rec = min(considered)
        for layer_idx in range(0, layer_type_num - 1):
            layer = sorted_layers[Layer.type_enum.BODY][layer_idx].name_
            for v in range(0, self.num_of_interleave_):
                for s in range(0, self.num_of_stage_):
                    prob += variables[layer][least_rec][v][s] <= (
                        1 - variables[self.REC_FRONTIER][v][s][layer_idx]
                        ) * self.BIG_M
        return prob

    @staticmethod
    def find_recompute_considered(
            layers_sorted: Dict[Layer.type_enum, List[Layer]]) -> Dict[Recompute.TYPE, bool]:
        """Return the recomputation-considered flags copied from the first BODY layer."""
        return layers_sorted[Layer.type_enum.BODY][0].recompute_considered_

    def max_stage_micro_eq_stage(self, prob: Any,
                                 layers_sorted: Dict[Layer.type_enum, List[Layer]]) -> Any:
        """Apply additional VPP optimisations when ``pp == num_of_micro_batch``."""
        last_chunk = self.num_of_interleave_ - 1

        for i_stage in range(self.num_of_stage_):
            for inter in range(last_chunk):
                prob += self.variables_[self.MAX_STAGE_TIME] >= (
                    self._max_stage_bound_i_bp(layers_sorted, i_stage, inter) +
                    self._max_stage_bound_head_tail(layers_sorted, i_stage,
                                                    -1, inter))

        if self.vpp_less_memory_:
            factors = self.compute_lm_forward_in_backward(self.num_of_stage_)
        else:
            factors = self.compute_forward_in_backward(
                self.num_of_stage_, self.num_of_micro_batch_)

        for i_stage in range(self.num_of_stage_):
            logger.debug(
                "v=%s, s=%s: (BP + HT) + (%s / %s * FP)",
                last_chunk,
                i_stage,
                factors[i_stage],
                self.num_of_micro_batch_,
            )
            prob += self.variables_[self.MAX_LAST_CHUNK] >= (
                self._max_stage_bound_i_bp(layers_sorted, i_stage, last_chunk) +
                self._max_stage_bound_head_tail(layers_sorted, i_stage, last_chunk, last_chunk) +
                (factors[i_stage] / self.num_of_micro_batch_) *
                self._max_stage_bound_i_fp(layers_sorted, i_stage, last_chunk))

        if self.optimization_level_ >= 2:
            logger.debug("Approach 2a")
            prob += self.variables_[self.MAX_STAGE_TIME] >= (
                self.variables_[self.MAX_LAST_CHUNK])

            return self.variables_[self.MAX_STAGE_TIME]
        logger.debug("Approach 2b")
        prob += self.variables_[self.MAX_LAST_CHUNK] >= (
            self.variables_[self.MAX_STAGE_TIME])

        return (self.variables_[self.MAX_STAGE_TIME] +
                self.variables_[self.MAX_LAST_CHUNK])

    def add_performance_constraint(self, prob: Any,
                                   layers_sorted: Dict[Layer.type_enum, List[Layer]],
                                   pipeline_total_time: Any) -> Any:
        """Add the ``pipeline_total_time >= …`` performance constraints."""
        max_stage_time = self.variables_[self.MAX_STAGE_TIME]
        max_stage_time = self.add_max_stage_constraint(prob, layers_sorted, max_stage_time)

        total_sum = self.variables_[self.TOTAL_SUM]
        prob += total_sum >= self._total_sum(layers_sorted)

        if self.optimization_level_ >= 2:
            # approach A
            for v in range(self.num_of_interleave_ - 1):
                prob += self.variables_[self.PREV_DIFF][v] >= (
                    self._prev_diff_sum(layers_sorted, prob, v))

                prob += self.variables_[self.CHUNKS_SUM][v] >= (
                    (self.num_of_interleave_ - v) / self.num_of_interleave_ *
                    self._chunks_sum(layers_sorted, v))

            chunks_sum = lpSolver.lpSum(self.variables_[self.CHUNKS_SUM])
            prev_diff = lpSolver.lpSum(self.variables_[self.PREV_DIFF])

            next_diff = self.variables_[self.NEXT_DIFF]
            prob += next_diff >= (
                self._next_diff_sum(layers_sorted, prob))

            prob += pipeline_total_time >= (
                (total_sum + chunks_sum + prev_diff + next_diff)
                / max(1, (self.num_of_interleave_ - 2))
                + max_stage_time * (self.num_of_micro_batch_ - 2)
            )
        else:
            # approach B
            prob += pipeline_total_time >= max_stage_time
        return prob

    def add_max_stage_constraint(self, prob: Any,
                                 layers_sorted: Dict[Layer.type_enum, List[Layer]],
                                 max_stage_time: Any) -> Any:
        """Add the ``max_stage_time`` lower-bound constraints over every ``(interleave, stage)``."""
        if (self.num_of_interleave_ > 1 and self.optimization_level_ >= 1
                and self.num_of_micro_batch_ == self.num_of_stage_):
            max_stage_time = self.max_stage_micro_eq_stage(prob, layers_sorted)
        else:
            # Constraints on sub-main-part of a stage that it may take (for all stage)
            for i_stage in range(self.num_of_stage_):
                for inter_f in range(self.num_of_interleave_):
                    for inter_b in range(self.num_of_interleave_):
                        prob += max_stage_time >= (
                             self._max_stage_bound_i_fp(layers_sorted, i_stage, inter_f) +
                             self._max_stage_bound_i_bp(layers_sorted, i_stage, inter_b) +
                             self._max_stage_bound_head_tail(layers_sorted, i_stage,
                                                             inter_f, inter_b))

        return max_stage_time

    ############################################
    #            Memory Constraint             #
    ############################################
    def stage_param_memory(self, variables: Any,
                           layers_sorted: Dict[Layer.type_enum, List[Layer]],
                           stage_id: int, num_of_stage: int,
                           num_of_interleave: int) -> Any:
        """Return an LP expression for the parameter memory of ``stage_id``."""
        # Add if dual to decide whether dualpipe_v is used
        if self.dual_:
            bound = lpSolver.LpAffineExpression()
            for inter_id in range(num_of_interleave):
                for layer in layers_sorted[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            bound += (
                                variables[layer.name_][rec][inter_id][stage_id] *
                                layer.memory_parameter_)
            if stage_id == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.memory_parameter_
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.memory_parameter_
        else:
            bound = lpSolver.LpAffineExpression()
            for inter_id in range(num_of_interleave):
                for layer in layers_sorted[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            bound += (
                                variables[layer.name_][rec][inter_id][stage_id] *
                                layer.memory_parameter_)
            if stage_id == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.memory_parameter_
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.memory_parameter_
            if stage_id == num_of_stage - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.memory_parameter_
        return bound

    def stage_active_memory_per_micro(
            self, variables: Any,
            layers_sorted: Dict[Layer.type_enum, List[Layer]],
            stage_id: int, inter_id: int) -> Any:
        """Return an LP expression for the activation memory of ``stage_id`` per micro-batch."""
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (variables[layer.name_][rec][inter_id][stage_id] *
                              layer.memory_activation_rec_[rec])
        return bound

    def stage_active_memory(self, variables: Any,
                            layers_sorted: Dict[Layer.type_enum, List[Layer]],
                            stage_id: int, num_of_interleave: int,
                            activation_nums: List[List[int]]) -> Any:
        """Return the total activation-memory LP expression for ``stage_id``."""
        bound = lpSolver.LpAffineExpression()
        for inter_id in range(num_of_interleave):
            for layer in layers_sorted[Layer.type_enum.BODY]:
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        bound += (
                            variables[layer.name_][rec][inter_id][stage_id] *
                            layer.memory_activation_rec_[rec] *
                            activation_nums[inter_id][stage_id])
        return bound

    def init_overhead_variables(self, variables: Any, s: int) -> Any:
        """Compute the per-stage overhead LP expression used in the VPP memory constraint."""
        bound = lpSolver.LpAffineExpression()
        vf = self.num_of_interleave_ - 1
        vb = self.num_of_interleave_ - 1
        incr_f = True
        if self.vpp_less_memory_:
            for _ in range(self.num_of_interleave_ - 1):
                if incr_f:
                    vf = (vf + 1) % self.num_of_interleave_
                    factor = abs(self.num_of_stage_ - s)
                else:
                    vb = vb - 1
                    factor = s
                incr_f = not incr_f

                logger.debug("%s * (act(%s,%s) - act(%s,%s)", factor, vf, s, vb, s)
                bound += factor * (
                    self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                    - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb))
        else:
            for _ in range(self.num_of_interleave_ - 1):
                if incr_f:
                    vf = (vf + 1) % self.num_of_interleave_
                    logger.debug(
                        "%s * (act(%s,%s) - act(%s,%s)",
                        self.num_of_stage_ - abs(self.num_of_stage_ - 2 * s - 1),
                        vf,
                        s,
                        vb,
                        s,
                    )
                    bound += (self.num_of_stage_ - abs(self.num_of_stage_ - 2 * s - 1)) * (
                        self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                        - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb)
                        )
                else:
                    vb = vb - 1
                    logger.debug(
                        "%s * (act(%s,%s) - act(%s,%s)",
                        max(self.num_of_stage_ - 2 * s - 1, 0),
                        vf + 1,
                        s,
                        vb + 1,
                        s,
                    )
                    bound += max(self.num_of_stage_ - 2 * s - 1, 0) * (
                        self.stage_active_memory_per_micro(variables,
                                                           self.layers_sorted_, s, vf + 1)
                        - self.stage_active_memory_per_micro(variables,
                                                             self.layers_sorted_, s, vb + 1)
                        )
                    logger.debug(
                        "%s * (act(%s,%s) - act(%s,%s)",
                        max(-(self.num_of_stage_ - 2 * s - 1), 0),
                        vf,
                        s,
                        vb,
                        s,
                    )
                    bound += max(-(self.num_of_stage_ - 2 * s - 1), 0) * (
                        self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                        - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb)
                        )
                incr_f = not incr_f

        return bound

    def stage_overhead_memory(self, variables: Any, stage_id: int) -> Any:
        """Return the stage-``stage_id`` memory overhead LP expression."""
        bound = lpSolver.LpAffineExpression()
        for v in range(self.num_of_interleave_ - 1):
            bound += variables[self.MEM_OVERHEAD_NAME][stage_id][v]
        return bound

    def add_pipeline_memory_constraint(self,
                                       constraint: PipelineMemoryConstraint) -> None:
        """Add per-stage memory upper-bound constraints to the solver problem."""
        prob = constraint.prob
        variables = constraint.variables
        layers_sorted = constraint.layers_sorted
        num_of_stage = constraint.num_of_stage
        num_of_interleave = constraint.num_of_interleave
        micro_batch = constraint.micro_batch
        memory_limit = constraint.memory_limit

        if self.vpp_less_memory_:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    num_of_stage, num_of_interleave, self.seq_split_num_, micro_batch, True)
            else:
                activation_nums = self.compute_less_activation_nums(
                    num_of_stage, num_of_interleave)
        # Add if dual to decide whether dualpipe_v is used
        elif self.dual_:
            activation_nums = self.compute_activation_nums_dual(
                num_of_stage, num_of_interleave, micro_batch)

        else:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    num_of_stage, num_of_interleave, self.seq_split_num_, micro_batch, False)
            else:
                activation_nums = self.compute_activation_nums(
                    num_of_stage, num_of_interleave, micro_batch)
        logger.info("activation nums = %s", activation_nums)

        if self.num_of_stage_ == self.num_of_micro_batch_:
            for s in range(num_of_stage):
                prob += memory_limit >= (
                    self.stage_param_memory(variables, layers_sorted, s,
                                            num_of_stage, num_of_interleave) +
                    self.stage_active_memory(variables, layers_sorted, s,
                                             num_of_interleave, activation_nums) +
                    self.constant_memory_)
        else:
            for s in range(num_of_stage):
                prob += variables[self.MEM_OVERHEAD_NAME][s] >= (
                    self.init_overhead_variables(variables, s)
                )
                prob += memory_limit >= (
                    self.stage_param_memory(
                        variables, layers_sorted, s, num_of_stage, num_of_interleave
                    )
                    + self.stage_active_memory(
                        variables, layers_sorted, s, num_of_interleave, activation_nums
                    )
                    + variables[self.MEM_OVERHEAD_NAME][s]
                    + self.constant_memory_
                )

    def get_simulator_memory_activation(self) -> list[float]:
        """Give the activation memory per stage for simulator."""

        memory_active = []
        if self.has_some_memory_info():
            for inter in range(self.num_of_interleave_):
                memory_active.append([])
                for stage in range(self.num_of_stage_):
                    memory_active[inter].append(0)
                    memory_activation = 0
                    for rec in Recompute.TYPE:
                        if not self.recompute_considered_[rec]:
                            continue

                        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                            var_value = self.variables_.get(layer.name_)[rec][inter][stage].varValue
                            memory_activation += var_value * layer.memory_activation_rec_[rec]

                    memory_active[inter][stage] = memory_activation
        return memory_active

    def get_simulator_memory_parameter(self) -> list[float]:
        """Give the parameter memory per stage for simulator."""
        memory_param_stage = [0] * self.num_of_stage_
        if self.has_some_memory_info():
            for inter in range(self.num_of_interleave_):
                for stage in range(self.num_of_stage_):
                    memory_param_stage[stage] += self._get_stage_parameter_memory(inter, stage)

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param_stage[0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param_stage[self.num_of_stage_ -
                                   1] += tail.memory_parameter_
        memory_param = [memory_param_stage] * self.num_of_interleave_
        return memory_param

    def _get_stage_parameter_memory(self, interleave, stage):
        """Calculate BODY-layer parameter memory for one pipeline position."""
        total = 0
        for rec in Recompute.TYPE:
            if not self.recompute_considered_[rec]:
                continue

            for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                if layer.memory_parameter_ is not None:
                    var_value = self.variables_.get(layer.name_)[rec][interleave][stage].varValue
                    total += var_value * layer.memory_parameter_
        return total

    def get_simulator_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_.get(
                                layer.name_)[rec][i][s].varValue * (
                                    layer.forward_time_ +
                                    layer.backward_time_rec_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_
        return time

    def get_simulator_forward_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_[layer.name_][rec][i][
                                s].varValue * (layer.forward_time_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_
        return time

    def get_simulator_recompute_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time_all_rec = []
        time_no_rec = []
        for i in range(self.num_of_interleave_):
            time_all_rec.append([])
            time_no_rec.append([])
            for s in range(self.num_of_stage_):
                time_all_rec[i].append(0)
                time_no_rec[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time_all_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[rec])
                            time_no_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[
                                        Recompute.TYPE.NONE])
        return [[r - n for r, n in zip(ar, nr)]
                for ar, nr in zip(time_all_rec, time_no_rec)]

    def has_some_memory_info(self) -> bool:
        """Check if there is some information for memory constraint."""
        some_info = False
        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                some_info = True
        return some_info

    ############################################
    #            General Constraint            #
    ############################################
    def add_optional_recompute_constraint(
            self, prob: Any, variables: Any,
            sorted_layers: Dict[Layer.type_enum, List[Layer]]) -> None:
        """Pin unused recomputation variables to zero in the ILP."""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if not self.recompute_considered_[rec]:
                    prob += lpSolver.lpSum(variables[layer.name_][rec]) == 0

    def dump_problem(self, folder: Optional[str] = None) -> None:
        """Serialize the pulp LP model to ``<folder>/<auto-generated-name>.lp``."""
        dump_name = "problem_" + str(self.layers_[0].model_name_)
        dump_name += "_" + str(self.max_memory_)
        dump_name += "_" + str(self.num_of_interleave_)
        dump_name += "_" + str(self.num_of_stage_)

        logger.info("dump_problem:out folder = %s", folder)
        if folder is not None:
            dump_name = os.path.join(folder, dump_name)
        dump_name += ".lp"
        logger.info("dump problem file: %s", dump_name)
        self.problem_.writeLP(dump_name)

    def print_results(self) -> None:
        """Log the detailed per-layer solver assignment for the solved problem."""
        if self.has_some_memory_info():
            logger.output("For max memory %s", self.max_memory_)
            logger.output("==============")
        for body_layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = body_layer.name_
            logger.output("For layer: %s", layer_name)
            logger.output("=========")
            logger.output("  Forward Prop time: %s", body_layer.forward_time_)
            for rec in Recompute.TYPE:
                if body_layer.recompute_considered_[rec]:
                    logger.output("  Backward Prop %s time: %s",
                                  Recompute.YAML_NAME[rec], body_layer.backward_time_rec_[rec])
            for inter in range(self.num_of_interleave_):
                for stage in range(self.num_of_stage_):
                    parts = []
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            value = str(int(self.variables_[layer_name][rec][inter][stage].varValue))
                            parts.append(value if rec is Recompute.TYPE.NONE else f"+ {value} {rec.name}")
                    chunk = f" of chunk {inter}" if self.num_of_interleave_ != 1 else ""
                    logger.output("    Assign %s: %s%s  to stage %d",
                                  layer_name, " ".join(parts), chunk, stage)
        for s in range(self.num_of_stage_):
            logger.debug(
                "%s[%s] =%s",
                self.MEM_OVERHEAD_NAME,
                s,
                self.variables_[self.MEM_OVERHEAD_NAME][s].varValue,
            )

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(
                "%s[%s] = %s",
                self.CHUNKS_SUM,
                v,
                self.variables_[self.CHUNKS_SUM][v].varValue,
            )

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(
                "%s[%s] = %s",
                self.PREV_DIFF,
                v,
                self.variables_[self.PREV_DIFF][v].varValue,
            )

        logger.debug("%s = %s", self.NEXT_DIFF, self.variables_[self.NEXT_DIFF].varValue)
        logger.debug("%s = %s", self.TOTAL_SUM, self.variables_[self.TOTAL_SUM].varValue)
        logger.debug("%s = %s", self.MAX_STAGE_TIME, self.variables_[self.MAX_STAGE_TIME].varValue)
        logger.debug("%s = %s", self.MAX_LAST_CHUNK, self.variables_[self.MAX_LAST_CHUNK].varValue)

        for body_layer in range(len(self.layers_sorted_[Layer.type_enum.BODY]) - 1):
            for v in range(self.num_of_interleave_):
                for s in range(self.num_of_stage_):
                    logger.info(
                        "%s[%s][%s][%s] = %s",
                        self.LAYER_FRONTIER,
                        body_layer,
                        v,
                        s,
                        self.variables_[self.LAYER_FRONTIER][body_layer][v][s].varValue,
                    )

    def debug_print_solver_theoretical_memory(self) -> None:
        """Log the solver-implied per-stage theoretical memory (debug aid)."""
        logger.info("%s Solver Theoretical Memory Analysis %s", "=" * 20, "=" * 20)

        if self.vpp_less_memory_:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.seq_split_num_, self.num_of_micro_batch_, True)
            else:
                activation_nums = self.compute_less_activation_nums(
                    self.num_of_stage_, self.num_of_interleave_)
        else:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.seq_split_num_, self.num_of_micro_batch_, False)
            else:
                activation_nums = self.compute_activation_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.num_of_micro_batch_)

        # compute theoretical value for each stage
        for s in range(self.num_of_stage_):
            param_mem = self.stage_param_memory(
                self.variables_,
                self.layers_sorted_,
                s,
                self.num_of_stage_,
                self.num_of_interleave_
            ).value()

            act_mem = self.stage_active_memory(
                self.variables_,
                self.layers_sorted_,
                s,
                self.num_of_interleave_,
                activation_nums
            ).value()

            # overhead = self.variables_[self.MEM_OVERHEAD_NAME][s].varValue * overhead_factors[s]
            overhead = 0
            total = param_mem + act_mem + overhead + self.constant_memory_

            logger.info("Stage %d Solver Memory Analysis:", s)
            logger.info("Parameter Memory:     %.2f", param_mem)
            logger.info("Activation Memory:    %.2f", act_mem)
            logger.info("Memory Overhead:      %.2f", overhead)
            logger.info("Constant Memory:      %.2f", self.constant_memory_)
            logger.info("Total Theoretical Memory: %.2f", total)


    def solve(self, time_limit: int = 90, dump_folder: Optional[str] = None) -> None:
        """Solve the ILP problem using PuLP's bundled CBC backend.

        Args:
            time_limit: Upper bound on solver wall-clock time in seconds.
            dump_folder: Directory to write the LP model to; ``None`` skips the dump.
        """
        logger.info("solve:out folder = %s", dump_folder)
        self.dump_problem(dump_folder)
        solver = lpSolver.getSolver("PULP_CBC_CMD", timeLimit=time_limit)
        self.problem_.solve(solver)

        self.print_results()

        self.debug_print_solver_theoretical_memory()

        for name, result in self.result().items():
            logger.output("%s %s %s", name, result, "\n")

    def result(self) -> dict[str, list[list[str]]]:
        """return schedule distribution for each layer (in the form of a dict)"""
        r = {}
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = layer.name_
            inter = []
            for i in range(self.num_of_interleave_):
                stage = []
                for s in range(self.num_of_stage_):
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            stage.append(
                                str(
                                    self.variables_.get(layer_name)[rec][i]
                                    [s].varValue) + " + ")
                inter.append(stage)
            r[layer_name] = inter
        return r

    def _create_problem_(self, description: str) -> lpSolver.LpProblem:
        """create the problem"""
        prob = lpSolver.LpProblem(description, lpSolver.LpMinimize)
        layers_sorted = self.layers_sorted_
        num_of_stage = self.num_of_stage_
        num_of_interleave = self.num_of_interleave_
        num_of_micro_batch = self.num_of_micro_batch_
        max_memory = self.max_memory_
        # Local variable declaration
        # max time that a "main" stage have to take (var to minimize)
        pipeline_total_time = lpSolver.LpVariable("pipeline_total_time", 0,
                                                  None, lpSolver.LpContinuous)

        # Var to Minimize
        prob += pipeline_total_time

        result = self.add_total_nb_layer_constraint(prob, self.variables_, layers_sorted)
        if result is None:
            raise RuntimeError("add_total_nb_layer_constraint() returned None.")
        # Add if dual to the original layer order constraint
        try:
            prob = self.add_stage_nb_layer_constraint(
                prob, self.variables_, layers_sorted
            )
        except Exception:
            logger.exception("Failed to add stage number layer constraint.")
            raise
        try:
            result = self.add_multimodal_sequence_constraint(prob, self.variables_, layers_sorted)
        except Exception:
            logger.exception("Failed to add multimodal sequence constraint.")
            raise

        #self.add_stage_nb_layer_constraint_dual(prob, self.variables_, layers_sorted)
        #self.add_multimodal_sequence_constraint_dual(prob, self.variables_, layers_sorted)
        try:
            result = self.add_multimodal_recompute_constraint(prob, self.variables_, layers_sorted)
            if result is None:
                raise RuntimeError("add_multimodal_recompute_constraint() returned None.")
        except Exception:
            logger.exception("Failed to add multimodal recompute constraint.")
            raise

        try:
            result = self.add_performance_constraint(prob, layers_sorted, pipeline_total_time)
            if result is None:
                raise RuntimeError("add_performance_constraint() returned None.")
            prob = result
        except Exception:
            logger.exception("Failed to add performance constraint.")
            raise

        constraint = PipelineMemoryConstraint(
            prob=prob,
            variables=self.variables_,
            layers_sorted=layers_sorted,
            num_of_stage=num_of_stage,
            num_of_interleave=num_of_interleave,
            micro_batch=num_of_micro_batch,
            memory_limit=max_memory,
        )
        if self.has_some_memory_info():
            self.add_pipeline_memory_constraint(constraint)
        return prob

    def _create_variables_to_solve_(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            layers: dict[Layer.type_enum, list[Layer]],
    ):
        """create variables to solve"""
        variables = {}

        variables[self.TOTAL_SUM] = lpSolver.LpVariable(
            self.TOTAL_SUM, 0, None, lpSolver.LpContinuous)

        chunks_sum_dict = lpSolver.LpVariable.dicts(
            name=self.CHUNKS_SUM,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous
        )
        chunks_sum_list = list(chunks_sum_dict.values())
        variables[self.CHUNKS_SUM] = chunks_sum_list

        prev_diff_dict = lpSolver.LpVariable.dicts(
            name=self.PREV_DIFF,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous
        )
        prev_diff_list = list(prev_diff_dict.values())
        variables[self.PREV_DIFF] = prev_diff_list

        layer_frontier_dict = lpSolver.LpVariable.dicts(
            name=self.LAYER_FRONTIER,
            indices=(
                range(1, len(self.layers_sorted_[Layer.type_enum.BODY])),
                range(0, self.num_of_interleave_),
                range(0, self.num_of_stage_)),
            lowBound=0,
            upBound=1,
            cat=lpSolver.LpBinary
        )
        layer_frontier_list = list(layer_frontier_dict.values())
        variables[self.LAYER_FRONTIER] = layer_frontier_list

        rec_frontier_dict = lpSolver.LpVariable.dicts(
            name=self.REC_FRONTIER,
            indices=(
                range(0, self.num_of_interleave_),
                range(0, self.num_of_stage_),
                range(0, len(self.layers_sorted_[Layer.type_enum.BODY])-1)),
            lowBound=0,
            upBound=1,
            cat=lpSolver.LpBinary
        )
        rec_frontier_list = list(rec_frontier_dict.values())
        variables[self.REC_FRONTIER] = rec_frontier_list

        variables[self.NEXT_DIFF] = lpSolver.LpVariable(
            self.NEXT_DIFF, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_STAGE_TIME] = lpSolver.LpVariable(
            self.MAX_STAGE_TIME, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_LAST_CHUNK] = lpSolver.LpVariable(
            self.MAX_LAST_CHUNK, 0, None, lpSolver.LpContinuous)

        lp_variable_dict = lpSolver.LpVariable.dicts(
            name=self.MEM_OVERHEAD_NAME,
            indices=(range(0, self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpInteger,
        )
        variables_list = list(lp_variable_dict.values())
        variables[self.MEM_OVERHEAD_NAME] = variables_list

        for layer in layers[Layer.type_enum.BODY]:
            variable_dict = lpSolver.LpVariable.dicts(
                name=layer.name_,
                indices=(
                    range(0, len(Recompute.TYPE)),
                    range(0, num_of_interleave),
                    range(0, num_of_stage),
                ),
                lowBound=0,
                upBound=None,
                cat=lpSolver.LpInteger,
            )
            variable_values = list(variable_dict.values())
            interleave_values = []
            for interleave in variable_values:
                interleave_value = list(interleave.values())
                interleave_values.append(interleave_value)
            variables[layer.name_] = interleave_values

        return variables

    ############################################
    #             Time Constraint              #
    ############################################
    def _max_stage_bound_i_fp(self, layers_sorted, stage_id, inter_f):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (self.variables_[layer.name_][rec][inter_f][stage_id] *
                              layer.forward_time_)
        return bound

    def _max_stage_bound_i_bp(self, layers_sorted, stage_id, inter_b):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (self.variables_[layer.name_][rec][inter_b][stage_id] *
                              layer.backward_time_rec_[rec])
        return bound

    def _max_stage_bound_head_tail(self, layers_sorted, stage_id, inter_f,
                                   inter_b):
        """maximize the stage bound of head and tail"""
        bound = lpSolver.LpAffineExpression()
        if stage_id == 0:
            if inter_f == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_
            if inter_b == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2
        if stage_id == self.num_of_stage_ - 1:
            if inter_f == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_
            if inter_b == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2
        return bound

    def _total_sum(self, layers_sorted):
        """sum up the layer time"""
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    for inter in range(self.num_of_interleave_):
                        for stage in range(self.num_of_stage_):
                            bound += self.variables_[layer.name_][rec][inter][stage] * (
                                layer.forward_time_ +
                                layer.backward_time_rec_[rec])
        return bound

    def body_layer_time(self, prop: "SappSolver.PROP_PHASE", layer: Layer,
                        inter: int, stage: int) -> Any:
        """Return a forward or backward time LP expression for ``layer`` at ``(inter, stage)``."""
        if prop == self.PROP_PHASE.FW:
            bound = lpSolver.lpSum(
                self.variables_[layer.name_][rec][inter][stage] * layer.forward_time_
                for rec in Recompute.TYPE if self.recompute_considered_[rec])
        else:
            bound = lpSolver.lpSum(
                self.variables_[layer.name_][rec][inter][stage] * layer.backward_time_rec_[rec]
                for rec in Recompute.TYPE if self.recompute_considered_[rec])

        return bound

    def micro_batch_time(self, prop: "SappSolver.PROP_PHASE",
                         layers_sorted: Dict[Layer.type_enum, List[Layer]],
                         inter: int, stage: int) -> Any:
        """Return the total micro-batch time LP expression at ``(inter, stage)``."""
        bound = lpSolver.LpAffineExpression()
        if prop == self.PROP_PHASE.FW:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_
        else:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2
        return bound

    def _chunks_sum(self, layers_sorted, v):
        """sum up the warm-up and cool-down time of a given chunk"""
        bound = lpSolver.LpAffineExpression()
        for stage in range(self.num_of_stage_):
            bound += self.micro_batch_time(self.PROP_PHASE.FW, layers_sorted, v, stage)
            bound += self.micro_batch_time(self.PROP_PHASE.BW, layers_sorted, v, stage)
        # normalize
        bound = bound / self.num_of_stage_
        return bound

    def _prev_diff_sum(self, layers_sorted, prob, v):
        """models bubble time for the first diagonal (forward, interleave 0)"""
        max_prev_stages = lpSolver.LpVariable.dicts(
            name="max_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_prev_stages = lpSolver.LpVariable.dicts(
            name="diff_with_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        head_time = 0
        for head in layers_sorted[Layer.type_enum.HEAD]:
            head_time = head.time_

        prob += max_prev_stages[0] >= (self.micro_batch_time(
            self.PROP_PHASE.FW, layers_sorted, v, 0)) - head_time

        for stage in range(1, self.num_of_stage_):
            prob += max_prev_stages[stage] >= max_prev_stages[stage - 1]
            prob += max_prev_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, layers_sorted, v, stage))


            prob += diff_with_prev_stages[stage] >= (
                max_prev_stages[stage - 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, layers_sorted, v, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_prev_stages[s] for s in range(1, self.num_of_stage_))
        return bound

    def _next_diff_sum(self, layers_sorted, prob):
        """models bubble time for the last diagonal (forward, last chunk)"""
        last_chunk = self.num_of_interleave_ - 1
        max_next_stages = lpSolver.LpVariable.dicts(
            name="max_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_next_stages = lpSolver.LpVariable.dicts(
            name="diff_with_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        prob += max_next_stages[self.num_of_stage_ -
                                1] >= (self.micro_batch_time(
                                    self.PROP_PHASE.FW, layers_sorted, last_chunk,
                                    self.num_of_stage_ - 1))

        for stage in reversed(range(0, self.num_of_stage_ - 1)):
            prob += max_next_stages[stage] >= max_next_stages[stage + 1]
            prob += max_next_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, layers_sorted, last_chunk, stage))

            prob += diff_with_next_stages[stage] >= (
                max_next_stages[stage + 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, layers_sorted, last_chunk, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_next_stages[s] for s in range(self.num_of_stage_ - 1))
        return bound
