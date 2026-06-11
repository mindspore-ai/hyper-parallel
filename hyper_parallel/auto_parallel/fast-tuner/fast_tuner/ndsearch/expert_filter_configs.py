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
"""expert filter"""

import math
from fast_tuner.utils.logger import logger


class ExpertFilterManager:
    """
    expert experience filter nd configs
    """
    def __init__(self, input_args, gbs):
        self.expert_filters = []
        self.input_args = input_args
        self.gbs = gbs

    @staticmethod
    def sequential_combination(selected_experiences, candidate_space):
        """Apply a sequence of expert experience filter functions to the candidate space."""
        result = candidate_space
        for exp in selected_experiences:
            result = exp(result)
        return result

    @staticmethod
    def get_cp(config):
        """Extract context parallelism size from a config tuple."""
        return config[0][0][2]

    @staticmethod
    def get_dp(config):
        """Extract data parallelism size from a config tuple."""
        return config[0][0][0]

    @staticmethod
    def get_op(config):
        """Extract operator parallelism size from a config tuple."""
        return config[0][1][1]

    @staticmethod
    def get_tp(config):
        """Extract tensor parallelism size from a config tuple."""
        return config[0][0][1]

    @staticmethod
    def get_pp(config):
        """Extract pipeline parallelism size from a config tuple."""
        return config[0][0][3]

    @staticmethod
    def get_sp_switch(config):
        """Extract the sequence parallelism switch flag from a config tuple."""
        return config[1]

    @staticmethod
    def get_world_size(config):
        """Compute total world size as the product of dp, tp, cp, and pp dimensions."""
        return math.prod(config[0][0])

    @staticmethod
    def get_mbs(config):
        """Extract micro-batch size from a config tuple."""
        return config[0][1][3]

    @staticmethod
    def get_ep(config):
        """Extract expert parallelism size from a config tuple."""
        return config[0][1][0]

    def get_gbs(self):
        """Return the global batch size."""
        return self.gbs

    def get_num_layers(self):
        """Return the number of model layers from input arguments."""
        return self.input_args.num_layers

    def get_mbn(self):
        """Return the micro-batch number from input arguments."""
        return self.input_args.mbn

    def add_experience(self, experience_function):
        """
        Add an expert experience function to the list.

        Args:
            experience_function (callable): Expert experience function to add.
        """
        self.expert_filters.append(experience_function)
        logger.info(f"add experience success:{experience_function.__name__}")

    def remove_experience(self, experience_function):
        """
        Remove an expert experience function from the list.

        Args:
            experience_function (callable): Expert experience function to remove.
        """
        if experience_function in self.expert_filters:
            self.expert_filters.remove(experience_function)
            logger.info(f"remove experience succ:{experience_function.__name__}")
        else:
            logger.info(f"can not find experience:{experience_function.__name__}, cannot remove.")

    def ep_for_torchtitan(self, candidate_space):
        """
        Default for etp=1 scenario.
        """
        configs = []
        for config in candidate_space:
            ep = self.get_ep(config)
            cp = self.get_cp(config)
            tp = self.get_tp(config)
            op = self.get_op(config)
            if ep % (cp * tp) == 0 and (op * cp * tp) % ep == 0:
                configs.append(config)
        return configs

    def ep_for_mindspore(self, candidate_space):
        """Filter configs where operator parallelism is divisible by (dp * tp) / ep for MindSpore compatibility."""
        configs = []
        for config in candidate_space:
            ep = self.get_ep(config)
            dp = self.get_dp(config)
            tp = self.get_tp(config)
            op = self.get_op(config)
            if op % ((dp * tp) / ep) == 0:
                configs.append(config)
        return configs

    def cp_for_deepseek_expert(self, candidate_space):
        """Filter configs where context parallelism is 1 (DeepSeek v2.28 constraint)."""
        # deepseek 2.28 version cp = 1
        return [config for config in candidate_space if self.get_cp(config) == 1]

    def dp_cp_ep_for_megatron_expert(self, candidate_space):
        """Filter configs where data parallelism times context parallelism is divisible by expert parallelism."""
        # megatron dp * cp % ep == 0
        return [config for config in candidate_space
                if self.get_dp(config) * self.get_cp(config) % self.get_ep(config) == 0]

    def pp_for_deepseek(self, candidate_space):
        """Filter configs where pipeline parallelism size is greater than 1 (DeepSeek large-scale requirement)."""
        # 10k-card deepseek training requires pp>1, pp too small causes OOM
        return [config for config in candidate_space if self.get_pp(config) > 1]

    def pp_for_768die(self, candidate_space):
        """Filter configs where pipeline parallelism size does not exceed 32 (768-die constraint)."""
        # 768die, requires pp<=32
        return [config for config in candidate_space if self.get_pp(config) <= 32]

    def tp_for_910b_expert(self, candidate_space):
        """Filter configs where tensor parallelism size does not exceed 8 (910B 8-card-per-node constraint)."""
        # 910b is 8 cards per node, inter-node comm too slow, so tp cannot exceed 8
        return [config for config in candidate_space if self.get_tp(config) <= 8]

    def tp_for_large_scale_expert(self, candidate_space):
        """Filter configs where tensor parallelism size does not exceed 64 (super-node constraint)."""
        # super-node tech, expert experience tp <= 64
        return [config for config in candidate_space if self.get_tp(config) <= 64]

    def tp_for_large_scale_768die(self, candidate_space):
        """Filter configs where tensor parallelism size is not divisible by 3 (768-die constraint)."""
        # 768die, tp must be power of 2
        return [config for config in candidate_space if self.get_tp(config) % 3 != 0]

    def tp_for_yoco_expert(self, candidate_space):
        """Filter configs where tensor parallelism size divides 56 (yoco model constraint)."""
        return [config for config in candidate_space if 56 % self.get_tp(config) == 0]

    def ep_for_large_scale_expert(self, candidate_space):
        """Filter configs where expert parallelism size does not exceed 64."""
        return [config for config in candidate_space if self.get_ep(config) <= 64]

    def sp_for_lm_expert(self, candidate_space):
        """Filter configs ensuring sequence parallelism is enabled at large scale (>= 1000 devices)."""
        # at 1k+ scale sp must be on, below 1k scale supports sp search
        world_size = self.get_world_size(candidate_space[0])
        return [config for config in candidate_space
                if world_size < 1000 or (self.get_tp(config) ==1 or self.get_sp_switch(config))]

    def pp_for_mbs_expert(self, candidate_space):
        """Filter configs where pipeline parallelism size does not exceed the minimum of num_layers and micro-batch count per DP rank."""
        return [config for config in candidate_space if
                self.get_pp(config) <=
                min(self.get_num_layers(), self.get_gbs() // self.get_dp(config) // self.get_mbs(config))]

    def gbs_for_dp_expert(self, candidate_space):
        """Filter configs where global batch size is divisible by data parallelism size."""
        return [config for config in candidate_space if self.get_gbs() % self.get_dp(config) == 0]

def expert_filter_configs(search_spaces, input_args, gbs):
    """
    :param search_spaces: initial search space [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :param input_args: user input model config
    :param gbs: global batch size
    Returns:
        list: Configs after expert experience pruning.
    """
    expert_manager = ExpertFilterManager(input_args, gbs)
    expert_manager.add_experience(expert_manager.cp_for_deepseek_expert)
    expert_manager.add_experience(expert_manager.tp_for_large_scale_expert)
    expert_manager.add_experience(expert_manager.ep_for_large_scale_expert)
    expert_manager.add_experience(expert_manager.sp_for_lm_expert)
    expert_manager.add_experience(expert_manager.pp_for_mbs_expert)
    expert_manager.add_experience(expert_manager.gbs_for_dp_expert)
    #expert_manager.add_experience(expert_manager.pp_for_deepseek)
    #expert_manager.add_experience(expert_manager.dp_cp_ep_for_megatron_expert)
    expert_manager.add_experience(expert_manager.ep_for_torchtitan)
    #expert_manager.add_experience(expert_manager.ep_for_mindspore)
    # add for 768die
    expert_manager.add_experience(expert_manager.pp_for_768die)
    expert_manager.add_experience(expert_manager.tp_for_large_scale_768die)
    # # add for yoco model
    # expert_manager.add_experience(expert_manager.tp_for_yoco_expert)
    valid_configs = expert_manager.sequential_combination(expert_manager.expert_filters, search_spaces)
    return valid_configs
