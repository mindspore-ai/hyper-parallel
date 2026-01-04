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
"""inner para class"""
from fast_tuner.utils.logger import logger


class InputParam:
    '''All information that we need to record'''
    PARAM_MAPPING = {
        "ENV_JSON" : "env_json",
        "DATASET" : "dataset",
        "YAML_PATH" : "yaml_path",
        "SHELL_PATH" : "shell_path",
        "TOML_PATH" : "toml_path",
        "MINDFORMERS_DIR" : "mindformers_dir",
        "MINDSPEED_PATH" : "mindspeed_path",
        "TORCHTITAN_PATH": "torchtitan_path",
        "DRYRUN_DATA_DIR" : "dryrun_data_dir",
        "PROFILE_DATA_DIR" : "profile_data_dir",
        "DRYRUN_LIM" : "dryrun_lim",
        "RANK_NUM" : "rank_num",
        "SOLVER_NAME" : "solver_name",
        "MAX_EXPERT_PARALLEL": "max_expert_parallel",
        "OUTPUT_PATH": "output_path",
        "GBS": "gbs",
        "SELECT_RECOMPUTE": "select_recompute",
        "ALG_PHASE": "alg_phase",
        "PARSER_RESULT": "parser_result",
        "DRYRUN": "dryrun",
        "CHECK": "check",
        "CONFIG": "config",
        "NPUS_PER_NODE": "npus_per_node",
        "NNODES": "nnodes",
        "STRATEGY": "strategy",
    }

    def __init__(self, args):
        self.args = args

    def __getattr__(self, name):
        if name in self.PARAM_MAPPING:
            param_name = self.PARAM_MAPPING[name]
            return getattr(self.args, param_name)
        raise AttributeError(f"'InputParam' object has no attribute '{name}'")

    def print_params(self):
        for key, value in self.PARAM_MAPPING.items():
            logger.info(f"{key}: {value} = {getattr(self, key)}")
