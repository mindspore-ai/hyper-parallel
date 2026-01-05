# Copyright 2024 Huawei Technologies Co., Ltd
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
"""pipeline_alg unit test"""

import unittest
import sys
import os
from pathlib import Path
from fast_tuner.pipeline_conductor import pipeline_parallel
from fast_tuner.pipeline_conductor.start_service import InitConfig, ExpertInput
sys.path.append(os.getcwd())


class PipeTestCase(unittest.TestCase):
    """流水线负载均衡算法单元测试用例"""

    def get_yaml_path(self, filename):
        yaml_file_path = Path(__file__).resolve().parents[2] / 'config' / filename
        return yaml_file_path

    def test_micro64_dp16_tp4_pp16_vp1_ep8(self):
        """test1"""
        yaml_file = self.get_yaml_path('test_62_16_1_64_1_0.yaml')
        mind_former_file = ''
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.backward_ratio = 1.839
        expert_input.layer_ratio = 0.397
        expert_input.head_loss = 1.5
        expert_input.recompute_ratio = 1
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 400.0
        model_input.memory.select_mem12 = 400.0
        model_input.memory.select_mem = 765.0
        model_input.memory.re_comp_mem0 = 30.0
        model_input.memory.re_comp_mem12 = 30.0
        model_input.memory.re_comp_mem = 30.0
        model_input.memory.act_mem0 = 479.0
        model_input.memory.act_mem12 = 479.0
        model_input.memory.act_mem = 765.0
        model_input.memory.layer_mem012 = 866.0
        model_input.memory.layer_mem = 10691.0
        model_input.memory.static_mem0 = 3383.0
        model_input.memory.static_mem = 2382.0
        model_input.memory.lm_head_mem = 8675.0
        model_input.memory.mem_lim_stage0 = 51913.0
        model_input.memory.mem_lim_others = 52914.0
        model_input.memory.mem_lim_last = 46621.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 1132.66
        self.assertLessEqual(cur_solution.object_value, object_value + 0.01)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 0.01)

    def test768_4k_gp(self):
        """test2"""
        yaml_file = self.get_yaml_path('768die4k_gp.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 1 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 416.0
        model_input.memory.select_mem12 = 416.0
        model_input.memory.select_mem = 674.0
        model_input.memory.re_comp_mem0 = 46.0
        model_input.memory.re_comp_mem12 = 46.0
        model_input.memory.re_comp_mem = 158.0
        model_input.memory.act_mem0 = 494.0
        model_input.memory.act_mem12 = 494.0
        model_input.memory.act_mem = 673.0
        model_input.memory.layer_mem012 = 1010.0
        model_input.memory.layer_mem = 2644.0
        model_input.memory.static_mem0 = 4243.0
        model_input.memory.static_mem = 2279.0
        model_input.memory.lm_head_mem = 8468.0
        model_input.memory.mem_lim_stage0 = 53101.0
        model_input.memory.mem_lim_others = 55065.0
        model_input.memory.mem_lim_last = 48876.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 3603.39
        self.assertLessEqual(cur_solution.object_value, object_value + 0.01)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 0.01)

    def test512_8k(self):
        """test3"""
        yaml_file = self.get_yaml_path('layers62_micro960_dp4_tp8_pp16_vp1_ep32.yaml')
        mind_former_file = ''
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 3 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 480.0
        model_input.memory.select_mem12 = 480.0
        model_input.memory.select_mem = 737.0
        model_input.memory.re_comp_mem0 = 78.0
        model_input.memory.re_comp_mem12 = 78.0
        model_input.memory.re_comp_mem = 190.0
        model_input.memory.act_mem0 = 614.0
        model_input.memory.act_mem12 = 614.0
        model_input.memory.act_mem = 738.0
        model_input.memory.layer_mem012 = 130.0
        model_input.memory.layer_mem = 7230.0
        model_input.memory.static_mem0 = 5789.0
        model_input.memory.static_mem = 3210.0
        model_input.memory.lm_head_mem = 7939.0
        model_input.memory.mem_lim_stage0 = 51555.0
        model_input.memory.mem_lim_others = 54134.0
        model_input.memory.mem_lim_last = 49405.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 10943.91
        self.assertLessEqual(cur_solution.object_value, object_value + 0.01)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 0.01)

    def test512_8k_no_swap(self):
        """test4"""
        yaml_file = self.get_yaml_path('512_8k_no_swap.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 5 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 832.0
        model_input.memory.select_mem12 = 832.0
        model_input.memory.select_mem = 1347.0
        model_input.memory.re_comp_mem0 = 92.0
        model_input.memory.re_comp_mem12 = 92.0
        model_input.memory.re_comp_mem = 316.0
        model_input.memory.act_mem0 = 988.0
        model_input.memory.act_mem12 = 988.0
        model_input.memory.act_mem = 1346.0
        model_input.memory.layer_mem012 = 1109.0
        model_input.memory.layer_mem = 3670.0
        model_input.memory.static_mem0 = 5618.0
        model_input.memory.static_mem = 2667.0
        model_input.memory.lm_head_mem = 12405.0
        model_input.memory.mem_lim_stage0 = 49678.0
        model_input.memory.mem_lim_others = 52629.0
        model_input.memory.mem_lim_last = 42891.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 5661.55
        self.assertLessEqual(cur_solution.object_value, object_value + 0.01)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 0.01)

    def test512_4k(self):
        """test5"""
        yaml_file = self.get_yaml_path('layers62_micro60_dp32_tp4_pp4_vp1_ep128.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 3 * 60
        expert_input.backward_ratio = 1.627
        expert_input.layer_ratio = 0.71
        expert_input.head_loss = 1.5
        expert_input.recompute_ratio = 0.71
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 415.0
        model_input.memory.select_mem12 = 415.0
        model_input.memory.select_mem = 672.0
        model_input.memory.re_comp_mem0 = 30.0
        model_input.memory.re_comp_mem12 = 30.0
        model_input.memory.re_comp_mem = 142.0
        model_input.memory.act_mem0 = 493.0
        model_input.memory.act_mem12 = 493.0
        model_input.memory.act_mem = 673.0
        model_input.memory.layer_mem012 = 869.0
        model_input.memory.layer_mem = 2111.0
        model_input.memory.static_mem0 = 6704.0
        model_input.memory.static_mem = 2377.0
        model_input.memory.lm_head_mem = 7986.0
        model_input.memory.mem_lim_stage0 = 48592.0
        model_input.memory.mem_lim_others = 52919.0
        model_input.memory.mem_lim_last = 47310.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 5868.49
        self.assertLessEqual(cur_solution.object_value, object_value + 0.01)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 0.01)

    # 此用例运行时间较长，大概40分钟
    def test512_8k_swap(self):
        """test6"""
        yaml_file = self.get_yaml_path('512_8k_swap.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 1 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 832.0
        model_input.memory.select_mem12 = 832.0
        model_input.memory.select_mem = 1347.0
        model_input.memory.re_comp_mem0 = 92.0
        model_input.memory.re_comp_mem12 = 92.0
        model_input.memory.re_comp_mem = 316.0
        model_input.memory.act_mem0 = 988.0
        model_input.memory.act_mem12 = 988.0
        model_input.memory.act_mem = 1346.0
        model_input.memory.layer_mem012 = 1109.0
        model_input.memory.layer_mem = 3670.0
        model_input.memory.static_mem0 = 5438.0
        model_input.memory.static_mem = 2664.0
        model_input.memory.lm_head_mem = 12402.0
        model_input.memory.mem_lim_stage0 = 49858.0
        model_input.memory.mem_lim_others = 52632.0
        model_input.memory.mem_lim_last = 42894.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        object_value = 5562.66
        self.assertLessEqual(cur_solution.object_value, object_value + 20)
        self.assertGreaterEqual(cur_solution.object_value, object_value - 20)

if __name__ == '__main__':
    unittest.main()
