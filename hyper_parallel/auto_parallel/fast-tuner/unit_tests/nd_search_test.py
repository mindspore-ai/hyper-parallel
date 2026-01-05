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
"""nd search unit test"""

import unittest

from fast_tuner.ndsearch.build_initial_spaces import build_initial_spaces
from fast_tuner.ndsearch.expert_filter_configs import expert_filter_configs
from fast_tuner.ndsearch.memory_model import grey_box_memory_prune
from fast_tuner.utils.input_config import InputConfig
from fast_tuner.utils.profiling.profile_parser import ProfileParser


class NdTestCase(unittest.TestCase):
    """ND搜索单元测试用例"""
    def test_memory_prune(self):
        """测试根据dryrun的内存信息对候选配置剪枝功能"""
        # input construct
        input_args = InputConfig('../../config/pretrain_deepseek3_671b.yaml')
        dryrun_info_init = [
            [32, 2, 16, 8, 142786],
            [128, 2, 4, 8, 188944],
            [16, 2, 32, 4, 153480],
            [16, 8, 8, 8, 96950],
            [64, 8, 2, 8, 825586],
            [64, 2, 8, 4, 221281],
            [8, 4, 32, 8, 81943],
            [64, 4, 4, 4, 283005],
            [64, 1, 16, 8, 241679],
            [32, 8, 4, 4, 270048],
            [8, 4, 32, 4, 98967],
            [256, 2, 2, 4, 1656206],
            [64, 4, 4, 8, 162045],
            [8, 8, 16, 8, 68632],
            [4, 8, 32, 8, 54686],
            [128, 1, 8, 4, 299846],
            [8, 8, 16, 4, 101784],
            [32, 8, 4, 8, 148595],
            [256, 1, 4, 4, 363705],
            [32, 4, 8, 8, 116591],
            [4, 8, 32, 4, 71710],
            [32, 4, 8, 4, 181999],
            [32, 2, 16, 4, 175938],
            [16, 2, 32, 8, 136456],
            [64, 8, 2, 4, 1635569],
            [64, 2, 8, 8, 155874],
            [128, 1, 8, 8, 234458],
            [16, 4, 16, 8, 93350],
            [256, 2, 2, 8, 847944],
            [128, 4, 2, 8, 832465],
            [16, 8, 8, 4, 162358],
            [32, 1, 32, 8, 245502],
            [512, 1, 2, 8, 927251],
            [64, 1, 16, 4, 274811],
            [16, 4, 16, 4, 126502],
            [128, 2, 4, 4, 309904],
            [32, 1, 32, 4, 262506],
            [256, 1, 4, 8, 242746],
            [128, 4, 2, 4, 1642448],
            [512, 1, 2, 4, 1687161]
        ]
        test_ep = [4, 8]
        max_expert_parallel = 64
        # execute prune
        memory_aware_configs = grey_box_memory_prune(input_args, dryrun_info_init, test_ep, max_expert_parallel)
        print(f"Dryrun Search space len: {len(memory_aware_configs)}, format: [dp, tp, pp, ep, evaluate_peak_mem]")
        # output check
        assert len(memory_aware_configs) == 10
        for sub_array in memory_aware_configs:
            assert len(sub_array) == 5
            assert sub_array[3] <= 64


    def test_initial_expert_space(self):
        input_args = InputConfig('../../config/pretrain_deepseek3_671b.yaml')
        initial_configs = build_initial_spaces(input_args)
        # 生成搜索空间test
        assert len(initial_configs) == 576030

        # 专家剪枝test
        expert_prune_search_space = expert_filter_configs(initial_configs, input_args, 1024)
        assert len(expert_prune_search_space) == 8753

    # dryrun yaml生成test

    # parser test
    def test_profile_parser(self):
        test = ProfileParser()
        test.config_anal('dp8tp4pp4ep32', [0, 32, 64, 127])
        test.refined_data()
        test.refresh()
        test.config_anal('dp8tp4pp2ep32', [0, 127])
        test.refined_data()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NdTestCase)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
