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
"""High-level orchestrator around :class:`SappSolver`: build, solve, simulate, export YAML."""
import os
import sys
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import yaml

import hyper_parallel.auto_parallel.sapp_ppb.simulator.pp_simulator as sim
import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute
from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver
from hyper_parallel.auto_parallel.sapp_ppb.utils.check_rules import check_yaml_depth_before_loading
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer, filter_layer_type
from hyper_parallel.auto_parallel.sapp_ppb.utils.logger import logger


class SappPipeline:
    """pipeline balancer"""

    def __init__(
            self,
            model_name: str,
            num_of_stage: int,
            num_of_micro_batch: int,
            max_memory: int,
            layers: List[Layer],
            vpp_less_memory: bool = False,
            # Add arg dual
            dual: bool = False,
            num_of_interleave: int = 1,
            constant_memory: int = 0,
            optimization_level: int = 1,
            extracted_training_params: Optional[Dict[str, int]] = None,
            seq_split_num: int = 1,
            use_backward_time: bool = False,
    ) -> None:
        """Cache pipeline parameters and index the input ``layers`` by HEAD / BODY / TAIL.

        Args:
            model_name (str): Model identifier, used for dump filenames and log prefixes.
            num_of_stage (int): Number of physical pipeline stages.
            num_of_micro_batch (int): Number of micro-batches scheduled per iteration.
            max_memory (int): Per-device memory budget in MB.
            layers (List[Layer]): Ordered list of layer descriptors covering HEAD/BODY/TAIL.
            vpp_less_memory (bool, optional): If ``True``, use the less-memory VPP scheduler variant.
                Default: ``False``.
            dual (bool, optional): Enable dualpipe-V scheduling support. Default: ``False``.
            num_of_interleave (int, optional): Virtual-pipeline (VPP) chunk count. Default: ``1``.
            constant_memory (int, optional): Constant per-stage memory overhead (MB). Default: ``0``.
            optimization_level (int, optional): Solver optimization level (``0-2``). Default: ``1``.
            extracted_training_params (Optional[Dict[str, int]], optional): Optional training-config parameters for
                seqpp. Default: ``None``.
            seq_split_num (int, optional): Number of sequence splits; ``>1`` enables sequence pipeline.
                Default: ``1``.
        """
        self.model_name_ = model_name
        self.num_of_stage_ = num_of_stage
        self.num_of_micro_batch_ = num_of_micro_batch
        self.num_of_interleave_ = num_of_interleave
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        # Add arg dual_
        self.dual_ = dual
        self.constant_memory_ = constant_memory
        self.optimization_level = optimization_level
        self.extracted_training_params_ = extracted_training_params
        self.seq_split_num_ = seq_split_num
        self.use_backward_time_ = use_backward_time
        self.seqpipe_ = self.seq_split_num_ > 1
        # logger.output("seq chunk: %s",self.seq_split_num_)

        self.problem_ = None
        self.layers_ = layers
        self.layers_sorted_ = {
            Layer.type_enum.HEAD: filter_layer_type(layers,
                                                    Layer.type_enum.HEAD),
            Layer.type_enum.BODY: filter_layer_type(layers,
                                                    Layer.type_enum.BODY),
            Layer.type_enum.TAIL: filter_layer_type(layers,
                                                    Layer.type_enum.TAIL),
        }

    @property
    def simulator(self):
        """Pipeline simulator instance (available after :meth:`simulate`)."""
        return self._simulator

    def has_some_memory_info(self) -> bool:
        """Check if there is all information for memory constraint."""
        return self.problem_.has_some_memory_info()

    def construct_problem(self, solver: str = "pulp") -> None:
        """Construct the underlying ILP problem using the requested solver backend."""
        if solver == "pulp":
            self.problem_ = self._construct_problem_pulp_()
        elif solver == "other":
            logger.warning(
                "No other solver available..., automatically switch to pulp!!!"
            )
            self.problem_ = self._construct_problem_pulp_()
        else:
            logger.warning(
                "No other solver available..., automatically switch to pulp!!!"
            )
            self.problem_ = self._construct_problem_pulp_()

    def solve_problem(self, time_limit: int = 90, dump_folder: Optional[str] = None) -> None:
        """Solve the ILP, optionally dumping the LP model into ``dump_folder``."""
        self.problem_.solve(time_limit, dump_folder)

    def get_result(self) -> dict[str, list[list[str]]]:
        """Get result distribution of the solution (compact form)."""
        return self.problem_.result()

    def get_memory_activation(self) -> list[float]:
        """Get the activation memory per stage for simulator."""
        return self.problem_.get_simulator_memory_activation()

    def get_memory_parameter(self) -> list[float]:
        """Get the parameter memory per stage for simulator."""
        return self.problem_.get_simulator_memory_parameter()

    def get_fw_time(self) -> list[float]:
        """Get the forward time per stage for simulator."""
        time = self.problem_.get_simulator_forward_time()
        return time

    def get_recompute_time(self) -> list[float]:
        """Get the recompute time per stage for simulator."""
        time = self.problem_.get_simulator_recompute_time()
        return time

    def get_time(self) -> list[float]:
        """Get the time per stage for simulator."""
        return self.problem_.get_simulator_time()

    def naive_layer_per_stage(self,
                              layer_num: int,
                              num_of_interleave: int = 1) -> List[List[int]]:
        """Return the naive layer-to-stage assignment (``layer_num`` evenly split)."""
        logger.output("layer_num = %s", layer_num)
        layer_count = layer_num // (self.num_of_stage_ * num_of_interleave)
        return [[layer_count] * self.num_of_stage_ for _ in range(num_of_interleave)]

    def print_yaml_results(self) -> None:
        """Log the solver output in the MindFormers YAML schema."""

        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            nass = self.naive_layer_per_stage(layer.nb_layer_,
                                              self.num_of_interleave_)
            yaml_format = Recompute.yaml_from_internal(
                self.num_of_interleave_,
                self.num_of_stage_,
                self.problem_.variables_[layer.name_],
                nass,
            )
            logger.output("layer-to-stage assignment baseline is \n\t%s", nass)
            yaml_results = "\nTo put in yaml configuration:"
            for y, v in yaml_format.items():
                yaml_results += f"\n\t{y}: {v}"
            logger.output(yaml_results)

    def get_manual_memory_activation(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage activation memory for a user-supplied layer assignment."""
        memory_active = []
        if self.has_some_memory_info():
            for inter in range(interleave_num):
                memory_active.append([])
                for stage in range(self.num_of_stage_):
                    memory_activation = 0
                    for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                        memory_activation += self._get_layer_memory_activation(
                            each_layer_per_recompute, layer, inter, stage
                        )
                    memory_active[inter].append(memory_activation)
        return memory_active

    @staticmethod
    def _get_layer_memory_activation(each_layer_per_recompute, layer, interleave, stage):
        """Calculate activation memory for one layer at one pipeline position."""
        memory_activation = 0
        unused_recompute_list = Recompute.get_unused_list(each_layer_per_recompute[layer])
        for rec in Recompute.TYPE:
            if rec in unused_recompute_list:
                continue
            value = each_layer_per_recompute[layer][rec][interleave][stage]
            if value > 0:
                memory_activation += value * layer.memory_activation_rec_[rec]
        return memory_activation

    def get_manual_memory_parameter(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage parameter memory for a user-supplied layer assignment."""
        memory_param_stage = [0] * self.num_of_stage_
        for inter in range(interleave_num):
            for stage in range(self.num_of_stage_):
                for rec in Recompute.TYPE:
                    for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                        if layer.memory_parameter_ is None:
                            continue

                        if rec in Recompute.get_unused_list(each_layer_per_recompute[layer]):
                            continue

                        value = each_layer_per_recompute[layer][rec][inter][stage]
                        if value <= 0:
                            continue

                        memory_param_stage[stage] += value * layer.memory_parameter_
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param_stage[0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param_stage[self.num_of_stage_ -
                                   1] += tail.memory_parameter_
        memory_param = [memory_param_stage] * interleave_num
        return memory_param

    def get_manual_time(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage execution time for a user-supplied layer assignment."""
        time = []
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for r in Recompute.TYPE:
                        if each_layer_per_recompute[layer][r][i][s] > 0:
                            time[i][s] += each_layer_per_recompute[layer][r][i][s] * (
                                layer.forward_time_ +
                                layer.backward_time_rec_[r])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.forward_time_ + head.backward_time_rec_[Recompute.TYPE.NONE]
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += (
                tail.forward_time_
                + tail.backward_time_rec_[Recompute.TYPE.NONE]
            )
        return time

    def get_manual_fw_time(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage forward time for a user-supplied layer assignment."""
        time = []
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for r in Recompute.TYPE:
                        if (r not in Recompute.get_unused_list(each_layer_per_recompute[layer])
                            and each_layer_per_recompute[layer][r][i][s] > 0):
                            time[i][s] += each_layer_per_recompute[layer][r][i][s] * (
                                layer.forward_time_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.forward_time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += tail.forward_time_
        return time

    def get_manual_backward_time(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage backward time for a user-supplied layer assignment."""
        time = []
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for r in Recompute.TYPE:
                        if (r not in Recompute.get_unused_list(each_layer_per_recompute[layer])
                            and each_layer_per_recompute[layer][r][i][s] > 0):
                            time[i][s] += each_layer_per_recompute[layer][r][i][s] * (
                                layer.backward_time_rec_[r])
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.backward_time_rec_[Recompute.TYPE.NONE]
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += (
                tail.backward_time_rec_[Recompute.TYPE.NONE]
            )
        return time

    def get_manual_recompute_time(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> List[List[float]]:
        """Return the per-stage recompute-only time for a user-supplied layer assignment."""
        logger.output("each_layer_per_recompute = %s", each_layer_per_recompute)
        time_all_rec = []
        time_no_rec = []
        for i in range(interleave_num):
            time_all_rec.append([])
            time_no_rec.append([])
            for s in range(self.num_of_stage_):
                time_all_rec[i].append(0)
                time_no_rec[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    self._add_manual_recompute_time(
                        each_layer_per_recompute, layer, i, s, time_all_rec, time_no_rec)

        return [[r - n for r, n in zip(ar, nr)]
                for ar, nr in zip(time_all_rec, time_no_rec)]

    def _add_manual_recompute_time(self, each_layer_per_recompute, layer, interleave, stage,
                                   time_all_rec, time_no_rec):
        """Accumulate recompute time for a single layer and stage."""
        logger.output("backward_time_rec_(%s) = %s", layer, layer.backward_time_rec_)
        unused_rec = Recompute.get_unused_list(each_layer_per_recompute[layer])
        for rec in Recompute.TYPE:
            layer_num = each_layer_per_recompute[layer][rec][interleave][stage]
            if rec in unused_rec or layer_num <= 0:
                continue
            if layer.backward_time_rec_[rec] is None:
                raise ValueError("No backward tme is specified for this "
                                 "recomputation. Recomputation "
                                 f"'{Recompute.YAML_NAME[rec]}' is likely not considered")
            logger.output("r = %s; i = %s; s = %s", rec, interleave, stage)
            time_all_rec[interleave][stage] += layer_num * layer.backward_time_rec_[rec]
            time_no_rec[interleave][stage] += layer_num * layer.backward_time_rec_[Recompute.TYPE.NONE]

    def simulate(self, show: bool = True, file_name: Optional[str] = None,
                 sub_fig: Optional[plt.Figure] = None, comm_time: float = 0.0) -> float:
        """Run the simulator on the solved schedule and return its estimated total time."""
        forward_time = self.get_fw_time()
        recompute_overhead = self.get_recompute_time()
        backward_time = self.problem_.get_simulator_backward_time() if self.use_backward_time_ else 0
        stage_mem_par = 0
        stage_mem_act = 0
        if self.has_some_memory_info():
            stage_mem_par = self.get_memory_parameter()
            stage_mem_act = self.get_memory_activation()

        return self.simulation(
            forward_time,
            recompute_overhead,
            stage_mem_par,
            stage_mem_act,
            self.constant_memory_,
            backward_time=backward_time,
            show=show,
            file_name=file_name,
            sub_fig=sub_fig,
            comm_time=comm_time,
        )

    def simulate_naive(self, layers: List[Layer], output_folder: str) -> None:
        """Simulate the naive (even) layer-to-stage assignments for sanity comparison."""
        num_layers = 0
        rec_considered = {}
        for layer in layers:
            if layer.type_ == Layer.type_enum.BODY:
                num_layers = layer.nb_layer_
                rec_considered = layer.recompute_considered_

        all_recomp = {"offset": 0}
        no_recomp = {"offset": 0}
        for rec in [Recompute.TYPE.FULL, Recompute.TYPE.SLCT, Recompute.TYPE.COMM]:
            if rec_considered.get(rec, False):
                all_recomp[Recompute.YAML_NAME[rec]] = True
                no_recomp[Recompute.YAML_NAME[rec]] = False

        self.simulate_yaml(
            yaml_format=all_recomp,
            show=True,
            interleave_num=self.num_of_interleave_,
            file_name=os.path.join(output_folder,
                                   "result_naive_all_recomp.svg"),
        )

        if num_layers % self.num_of_stage_ == 0:
            self.simulate_yaml(
                yaml_format=no_recomp,
                show=True,
                interleave_num=self.num_of_interleave_,
                file_name=os.path.join(output_folder,
                                       "result_naive_no_recomp.svg"),
            )
        else:
            logger.warning("num layer cannot be divided by num stage")

    def simulate_comparison(self, manual_config_file: str, output_folder: str) -> None:
        """Render side-by-side automatic vs manual simulations for every entry in the YAML."""
        with open(manual_config_file, encoding="utf-8") as fp:
            check_yaml_depth_before_loading(fp)
            fp.seek(0)
            data = yaml.safe_load(fp)
        yaml_data = {}
        for manual in data.values():
            yaml_data[Recompute.OFFSET] = manual.get(Recompute.OFFSET)
            if isinstance(yaml_data[Recompute.OFFSET], list) and all(
                    isinstance(item, int) for item in yaml_data[Recompute.OFFSET]):
                yaml_data[Recompute.OFFSET] = [yaml_data[Recompute.OFFSET]]

            for rec in Recompute.YAML_NAME.values():
                yaml_data[rec] = manual.get(rec)
                if isinstance(yaml_data[rec], list) and all(
                        isinstance(item, int) for item in yaml_data[rec]):
                    yaml_data[rec] = [yaml_data[rec]]
            interleave_num = manual.get("interleave_num",
                                        self.num_of_interleave_)
            show = manual.get("show", False)
            file_name = manual.get("file_name")
            full_file_name = os.path.join(output_folder,
                                          file_name) if (file_name) else None

            fig = plt.figure(figsize=(24, 8))
            sub_figs = fig.subfigures(1, 2, wspace=0.07)
            sub_figs[0].suptitle('Automatic', fontsize='x-large')
            try:
                simulate_result = self.simulate(
                    show=False,
                    file_name=os.path.join(output_folder, "Auto_" + file_name),
                    sub_fig=sub_figs[0],
                )
            except Exception:
                logger.exception("Failed to simulate auto pipeline.")
                raise

            if simulate_result is None:
                raise RuntimeError("simulate() returned None.")

            sub_figs[1].suptitle('Manual', fontsize='x-large')
            self.simulate_yaml(yaml_data, False, interleave_num, full_file_name, sub_figs[1])
            plt.savefig(os.path.join(output_folder, "Comparison_" + file_name))
            if show:
                plt.show()

    def simulate_only_manual(self, manual_config_file: str, output_folder: str) -> None:
        """Render only the manual simulation for every entry in ``manual_config_file``."""
        with open(manual_config_file, encoding="utf-8") as fp:
            check_yaml_depth_before_loading(fp)
            fp.seek(0)
            data = yaml.safe_load(fp)
        yaml_data = {}
        for manual in data.values():
            yaml_data[Recompute.OFFSET] = manual.get(Recompute.OFFSET)
            if isinstance(yaml_data[Recompute.OFFSET], list) and all(
                    isinstance(item, int) for item in yaml_data[Recompute.OFFSET]):
                yaml_data[Recompute.OFFSET] = [yaml_data[Recompute.OFFSET]]

            for rec in Recompute.YAML_NAME.values():
                yaml_data[rec] = manual.get(rec)
                if isinstance(yaml_data[rec], list) and all(
                        isinstance(item, int) for item in yaml_data[rec]):
                    yaml_data[rec] = [yaml_data[rec]]
            interleave_num = manual.get("interleave_num",
                                        self.num_of_interleave_)
            show = manual.get("show", False)
            file_name = manual.get("file_name")
            full_file_name = os.path.join(output_folder,
                                          file_name) if (file_name) else None

            fig = plt.figure(figsize=(12, 8))
            self.simulate_yaml(yaml_data, False, interleave_num, full_file_name, fig)
            plt.savefig(os.path.join(output_folder, "manual_file_" + file_name))
            if show:
                plt.show()

    def simulate_yaml(self, yaml_format: Dict[str, Any], show: bool = True,
                      interleave_num: int = 1,
                      file_name: Optional[str] = None,
                      sub_fig: Optional[plt.Figure] = None) -> float:
        """Simulate a manual pipeline configuration encoded as a YAML-compatible dict."""
        layer_num = 0
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_num += layer.nb_layer_
        nass = self.naive_layer_per_stage(layer_num,
                                          num_of_interleave=interleave_num)
        layer_per_recompute = Recompute.internal_from_yaml(
            interleave_num, self.num_of_stage_, yaml_format, nass)
        each_layer_per_recompute = self.split_layer_per_recompute(layer_per_recompute)
        return self.simulate_manual(
            each_layer_per_recompute,
            show,
            interleave_num=interleave_num,
            file_name=file_name,
            sub_fig=sub_fig
        )

    #######################################################################
    ##                                                                   ##
    ##                      Print Solver Model                           ##
    ##                                                                   ##
    #######################################################################
    def _calculate_activation_memory(self, each_layer_per_recompute, v, s):
        """Calculate activation memory for next and current stage"""
        act_mem_next = 0
        act_mem_curr = 0

        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.problem_.recompute_considered_[rec]:
                    if each_layer_per_recompute[layer][rec][v + 1][s] > 0:  # next
                        act_mem_next += (each_layer_per_recompute[layer][rec][v + 1][s] *
                                         layer.memory_activation_rec_[rec])
                    if each_layer_per_recompute[layer][rec][v][s] > 0:    # current
                        act_mem_curr += (each_layer_per_recompute[layer][rec][v][s] *
                                         layer.memory_activation_rec_[rec])

        return act_mem_next, act_mem_curr

    def _compute_parameter_memory_manually_solver(self, each_layer_per_recompute, s, interleave_num=1):
        """Solver memory model: parameter memory"""
        param_mem = 0
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if layer.memory_parameter_ is not None:
                param_mem += self._calculate_layer_parameter_memory(
                    layer, each_layer_per_recompute[layer], s, interleave_num)
        return param_mem

    def _calculate_layer_parameter_memory(self, layer, layer_per_recompute, s, interleave_num):
        """Calculate parameter memory for a single layer"""
        layer_mem = 0
        for inter in range(interleave_num):
            for rec in Recompute.TYPE:
                if self.problem_.recompute_considered_[rec]:
                    if layer_per_recompute[rec][inter][s] > 0:
                        layer_mem += layer_per_recompute[rec][inter][s] * layer.memory_parameter_
        return layer_mem

    def _calculate_activation_memory_solver(self, each_layer_per_recompute, s, interleave_num, activation_nums):
        """Calculate activation memory for a given stage"""
        act_mem = 0
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            for inter in range(interleave_num):
                for rec in Recompute.TYPE:
                    if self.problem_.recompute_considered_[rec]:
                        if each_layer_per_recompute[layer][rec][inter][s] > 0:
                            act_mem += (each_layer_per_recompute[layer][rec][inter][s] *
                                        layer.memory_activation_rec_[rec] *
                                        activation_nums[inter][s])
        return act_mem


    def debug_print_manual_theoretical_memory(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]],
            interleave_num: int = 1) -> None:
        """Log the per-stage theoretical memory implied by the solver model (debug aid)."""
        logger.info("%s Manual Theoretical Memory Analysis %s", "=" * 20, "=" * 20)

        if self.vpp_less_memory_:
            if self.seqpipe_:
                activation_nums = self.problem_.compute_activation_seq_nums(
                    self.num_of_stage_, interleave_num, self.seq_split_num_, self.num_of_micro_batch_, True)
            else:
                activation_nums = self.problem_.compute_less_activation_nums(
                    self.num_of_stage_, interleave_num)
        # Add if dual to decide whether dualpipe_v is used
        elif self.dual_:
            activation_nums = self.problem_.compute_activation_nums_dual(
                self.num_of_stage_, interleave_num, self.num_of_micro_batch_)
        else:
            if self.seqpipe_:
                activation_nums = self.problem_.compute_activation_seq_nums(
                    self.num_of_stage_, interleave_num, self.seq_split_num_, self.num_of_micro_batch_, False)
            else:
                activation_nums = self.problem_.compute_activation_nums(
                    self.num_of_stage_, interleave_num, self.num_of_micro_batch_)

        logger.info("Activation nums = %s", activation_nums)

        # compute for each stage
        for s in range(self.num_of_stage_):

            # parameter memory
            param_mem = self._compute_parameter_memory_manually_solver(each_layer_per_recompute, s, interleave_num)

            # head memory
            if s == 0:
                for head in self.layers_sorted_[Layer.type_enum.HEAD]:
                    if head.memory_parameter_ is not None:
                        param_mem += head.memory_parameter_

            # tail memory
            if s == self.num_of_stage_ - 1:
                for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
                    if tail.memory_parameter_ is not None:
                        param_mem += tail.memory_parameter_

            # act memory
            act_mem = self._calculate_activation_memory_solver(each_layer_per_recompute, s,
                                                               interleave_num, activation_nums)

            # overhead
            overhead = 0

            total = param_mem + act_mem + overhead + self.constant_memory_

            logger.info("Stage %d Manual Memory Analysis:", s)
            logger.info("Parameter Memory:     %.2f", param_mem)
            logger.info("Activation Memory:    %.2f", act_mem)
            logger.info("Memory Overhead:      %.2f", overhead)
            logger.info("Constant Memory:      %.2f", self.constant_memory_)
            logger.info("Total Theoretical Memory: %.2f", total)

    def split_layer_per_recompute(
            self,
            layer_per_recompute: Dict[Recompute.TYPE, List[List[int]]]
    ) -> Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]]:
        """Split aggregate per-recompute layer counts into counts per BODY layer."""
        each_layer_per_recompute = {}
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            rest = layer.nb_layer_
            each_layer_per_recompute[layer] = {r: [] for r in Recompute.TYPE}
            for rec in Recompute.TYPE:
                for i in range(self.num_of_interleave_):
                    each_layer_per_recompute[layer][rec].append([0]*self.num_of_stage_)
                    for s in range(self.num_of_stage_):
                        subtract = min(layer_per_recompute[rec][i][s], rest)
                        layer_per_recompute[rec][i][s] -= subtract
                        rest -= subtract
                        each_layer_per_recompute[layer][rec][i][s] += subtract
        return each_layer_per_recompute

    def fuse_layer_per_recompute(
            self,
            each_layer_per_recompute: Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]]
    ) -> Dict[Recompute.TYPE, List[List[int]]]:
        """Fuse per-layer recompute counts back into aggregate per-recompute-type totals."""
        all_layers_per_recompute = {r: [] for r in Recompute.TYPE}
        for rec in Recompute.TYPE:
            for i in range(self.num_of_interleave_):
                all_layers_per_recompute[rec].append([])
                for s in range(self.num_of_stage_):
                    all_layers_per_recompute[rec][i].append(sum(
                        each_layer_per_recompute[layer][rec][i][s]
                        for layer in self.layers_sorted_[Layer.type_enum.BODY]
                    ))
        return all_layers_per_recompute


    def simulate_manual(
            self,
            each_layer_per_recompute: Optional[Dict[Layer, Dict[Recompute.TYPE, List[List[int]]]]] = None,
            show: bool = True,
            interleave_num: int = 1,
            file_name: Optional[str] = None,
            sub_fig: Optional[plt.Figure] = None) -> float:
        """Run the simulator on a user-supplied per-layer recompute strategy."""
        logger.output("Simulating given strategy: %s", each_layer_per_recompute)

        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if len(each_layer_per_recompute[layer][rec]) != interleave_num:
                    logger.error(
                        "For layer %s with recompute %s, %s does not match interleave number %s",
                        layer,
                        rec,
                        len(each_layer_per_recompute[layer][rec]),
                        interleave_num,
                    )
                    return sys.maxsize

        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if any(x < 0 for sublist in each_layer_per_recompute[layer][rec]
                    for x in sublist):
                    raise ValueError(
                        f"for {rec}, there is strategy less than 0 in "
                        f"{each_layer_per_recompute[layer][rec]}"
                    )

        forward_time = self.get_manual_fw_time(each_layer_per_recompute,
                                               interleave_num)
        recompute_overhead = self.get_manual_recompute_time(
            each_layer_per_recompute, interleave_num)
        backward_time = (
            self.get_manual_backward_time(
                each_layer_per_recompute, interleave_num)
            if self.use_backward_time_
            else 0
        )
        stage_mem_par = 0
        stage_mem_act = 0
        if self.has_some_memory_info():
            stage_mem_par = self.get_manual_memory_parameter(
                each_layer_per_recompute, interleave_num=interleave_num)
            stage_mem_act = self.get_manual_memory_activation(
                each_layer_per_recompute, interleave_num=interleave_num)

        self.debug_print_manual_theoretical_memory(each_layer_per_recompute, interleave_num)

        return self.simulation(
            forward_time,
            recompute_overhead,
            stage_mem_par,
            stage_mem_act,
            constant_mem=self.constant_memory_,
            backward_time=backward_time,
            show=show,
            file_name=file_name,
            sub_fig=sub_fig,
            comm_time=0.0,
        )

    def simulation(
            self,
            forward_time: List[List[float]],
            recompute_overhead: Union[int, List[List[float]]] = 0,
            stage_mem_par: Union[int, List[List[float]]] = 0,
            stage_mem_act: Union[int, List[List[float]]] = 0,
            constant_mem: int = 0,
            backward_time: Union[int, List[List[float]]] = 0,
            show: bool = True,
            file_name: Optional[str] = None,
            sub_fig: Optional[plt.Figure] = None,
            comm_time: float = 0.0,
    ) -> float:
        """Run the low-level :class:`PipelineSimulator` and return its reported end time."""
        use_comm = comm_time > 0.0
        if self.has_some_memory_info():
            logger.output(
                "PipelineSimulator(\n\t%s, %s,"
                "\n\tblock_mem_act=%s,"
                "\n\tblock_mem_par=%s,"
                "\n\tlayer_recompute=%s,"
                "\n\tbackward_time=%s,"
                "\n\tless_memory=%s )",
                forward_time,
                self.num_of_micro_batch_,
                stage_mem_act,
                stage_mem_par,
                recompute_overhead,
                backward_time,
                self.vpp_less_memory_,
            )

            sim_method = "vpp2" if self.vpp_less_memory_ else "vpp"
            simulator = sim.PipelineSimulator(
                forward_time,
                self.num_of_micro_batch_,
                comm_time=comm_time,
                block_mem=stage_mem_act,
                block_mem_par=stage_mem_par,
                constant_mem=constant_mem,
                layer_recompute=recompute_overhead,
                backward_time=backward_time,
                method=sim_method,
                sub_fig=sub_fig
            )
        else:
            logger.output(
                "PipelineSimulator(\n\t%s, %s,"
                "\n\tlayer_recompute=%s,"
                "\n\tbackward_time=%s,"
                "\n\tless_memory=%s )",
                forward_time,
                self.num_of_micro_batch_,
                recompute_overhead,
                backward_time,
                self.vpp_less_memory_,
            )
            simulator = sim.PipelineSimulator(
                forward_time,
                self.num_of_micro_batch_,
                comm_time=comm_time,
                layer_recompute=recompute_overhead,
                backward_time=backward_time,
                less_memory=self.vpp_less_memory_,
                sub_fig=sub_fig
            )

        simulator.run(comm=use_comm)
        self._simulator = simulator
        if file_name:
            simulator.save(file_name)
        if show:
            simulator.show()
        return simulator.end_time

    def _construct_problem_pulp_(self) -> SappSolver:
        """construct the problem using pulp"""
        prob = SappSolver(
            num_of_stage=self.num_of_stage_,
            num_of_micro_batch=self.num_of_micro_batch_,
            num_of_interleave=self.num_of_interleave_,
            max_memory=self.max_memory_,
            vpp_less_memory=self.vpp_less_memory_,
            # Add arg dual
            dual = self.dual_,
            constant_memory=self.constant_memory_,
            layers=self.layers_,
            layers_sorted=self.layers_sorted_,
            optimization_level=self.optimization_level,
            extracted_training_params=self.extracted_training_params_,
            seq_split_num=self.seq_split_num_
        )
        return prob

    def _recompute_considered(self):
        return self.problem_.recompute_considered_


def choose_interleave(
        model_name: str,
        number_of_stage: int,
        number_of_micro_batch: int,
        max_memory: int,
        layers: list[Layer],
) -> tuple[int, int, dict[str, list[list[str]]]]:
    """Simulates different interleaves and returns the best."""
    max_inter = 4
    best_time = int(sys.maxsize)
    best_inter = 1
    best_distribution = {}

    for inter in range(1, max_inter + 1):
        pipe = SappPipeline(
            model_name=model_name,
            num_of_stage=number_of_stage,
            num_of_micro_batch=number_of_micro_batch,
            max_memory=max_memory,
            layers=layers,
            num_of_interleave=inter,
        )

        pipe.construct_problem(solver="pulp")
        pipe.solve_problem()
        time = pipe.simulate(show=False)
        logger.output("for interleave %s, time = %s", inter, time)
        if time < best_time:
            best_time = time
            best_inter = inter
            best_distribution = pipe.get_result()

    return (best_inter, best_time, best_distribution)


def flatten(inter_stage_list: List[List[float]]) -> List[float]:
    """Collapse an ``[interleave][stage]`` matrix into a per-stage list via summation."""
    stage_list = [0] * len(inter_stage_list[0])
    for inter, _ in enumerate(inter_stage_list):
        for stage, _ in enumerate(inter_stage_list[inter]):
            stage_list[stage] += inter_stage_list[inter][stage]
    return stage_list
