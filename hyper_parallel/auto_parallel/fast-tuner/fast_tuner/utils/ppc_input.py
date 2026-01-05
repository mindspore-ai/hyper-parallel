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
"""interface between nd and pp """
import os
import re
import csv
from typing import List
from pathlib import Path
from fast_tuner.utils.common import GENERAL_TOML
from fast_tuner.utils.profiling.profile_info import ProfileInfo
from fast_tuner.pipeline_conductor.dryrun import DryRun, dryrun_config_error
from fast_tuner.ndsearch.para_for_nd_search import ParaForNd

class ParallelConfig:
    def __init__(self, gbs, config):
        self.dp = config[0]
        self.tp = config[1]
        self.pp = config[2]
        self.vp = 1
        self.ep = config[3]
        self.micro = gbs / self.dp

class PipelineInputConfig:
    def __init__(self, profiling_info: ProfileInfo, config_path, model_args = None):
        self.profiling_info = profiling_info
        self.config_path = config_path
        self.model_args = model_args

class ParallelInput:
    """define the input of parallel"""
    def __init__(self, para, profile_file_dir=None):
        self.candidate_configs: List[PipelineInputConfig] = []

        self.init_configs_info(para, profile_file_dir)

        self.is_lowmem = int(os.getenv('ENABLE_LESS_MEM_VPP', None))
        args = para.args
        self.solver_name = args.solver_name
        self.env_config_json = args.env_json
        self.dryrun_lim = args.dryrun_lim
        self.dryrun = args.dryrun
        self.check = args.check
        self.output_path = args.output_path
        if DryRun.config_file_type == 0:
            self.ms_adapter_file = args.mindformers_dir
        elif DryRun.config_file_type == 1:
            self.ms_adapter_file = args.mindspeed_path
        elif DryRun.config_file_type == 2:
            self.ms_adapter_file = args.torchtitan_path
        else:
            raise TypeError(dryrun_config_error)

    @staticmethod
    def parse_results_by_csv(csv_file):
        """parser the csv results"""
        result_dict = {}
        with open(csv_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # key = "_".join([row['dp'], row['tp'], row['pp'], row['ep']])
                key_columns = list(row.keys())[:-5]  # 获取前N-5列的列名
                key = "_".join([row[col] for col in key_columns])
                value = [float(row['dmratio']), float(row['bfratio']), float(row['re_grow_ration']),
                         float(row['hratio']), float(row['moe_fw'])]
                result_dict[key] = value
        return result_dict

    def get_model_files_dir(self, args, profile_file_dir=None):
        """get the model file dir path"""
        if profile_file_dir is not None:
            files_dir = Path(profile_file_dir)
        elif args.files_dir is not None:
            files_dir = Path(args.files_dir)
        else:
            raise RuntimeError('Must specify either files_dir or profile_file_dir')

        if args.yaml_path:
            model_files = files_dir.glob('*.yaml')
        elif args.shell_path:
            model_files = files_dir.glob('*.sh')
        elif args.toml_path:
            model_files = files_dir.glob('*.toml')
        else:
            raise Exception("No yaml_path or shell_path specified")
        return model_files

    def get_args_info(self, para, config_file):
        if config_file.name.endswith('.toml'):
            para.TOML_PATH = config_file
            model_args = ParaForNd(para)
            return model_args
        return None

    def init_configs_info(self, para, profile_file_dir=None):
        """
        Docstring for init_configs_info
        
        :param self: Description
        :param para: Description
        :param profile_file_dir: Description
        """
        args = para.args
        model_files = self.get_model_files_dir(args, profile_file_dir)
        csv_result = {}
        # 若用户直接输入profiling解析信息---csv文件，则从文件中读入
        if args.parser_result is not None:
            csv_result = self.parse_results_by_csv(args.parser_result)

        pattern = r'（\d+）'
        if args.yaml_path :
            pattern = re.compile(r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)')
            DryRun.config_file_type = 0
        elif args.shell_path:
            pattern = re.compile(r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)')
            DryRun.config_file_type = 1
        elif args.toml_path:
            pattern = re.compile(GENERAL_TOML.replace('{}', r'(\d+)'))
            DryRun.config_file_type = 2

        for config_file in model_files:
            match = pattern.search(config_file.name)
            if match:
                # dp_num, tp_num, pp_num, ep_num = map(int, match.groups())
                # config = [dp_num, tp_num, pp_num, ep_num]
                config = [int(x) for x in match.groups()]
                config_str = "_".join(map(str, config))
                if csv_result.get(config_str):
                    profile_list = csv_result[config_str]
                else:
                    profile_list = []
               # todo: profiling解析的输入有待确认,例如rank
                profile_info = ProfileInfo(args.profile_data_dir, profile_list)
                input_args = self.get_args_info(para, config_file)
                pipeline_input_config = PipelineInputConfig(profile_info, config_file, input_args)
                self.candidate_configs.append(pipeline_input_config)
