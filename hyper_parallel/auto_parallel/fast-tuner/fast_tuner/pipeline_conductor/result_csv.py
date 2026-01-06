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
"""generate result to csv"""

import os
import time
import csv
import yaml

from fast_tuner.utils.logger import logger
from fast_tuner.pipeline_conductor import pp_util
from fast_tuner.pipeline_conductor.dryrun import DryRun, DRYRUN_CONFIG_ERROR

class ResultCsv:
    """
    generate result csv
    """
    def __init__(self, output_path, output_file='pipeline_output', name='test_result'):
        self.header = ['test', 'layers', 'micro', 'dp', 'tp', 'pp', 'ep', 'vp', 'op/fsdp', 'dense:moe', '反向:正向',
                       '重计算增加比率', 'mtp+head', 'moe时长', 'low_mem', '目标值', 'cost', 'x', 'offset',
                       'ra', '内存信息', '内存上限(GB)', '求解器', 'GAP', 'solution_status', 'model_status',
                       '求解耗时/s', 'dryrun_check']
        self.name = name
        timestamp = time.strftime("%Y%m%d%H%M%S")
        csv_dir = os.path.join(os.path.abspath(output_path), output_file)
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        self.path = os.path.join(csv_dir, f'{self.name}_{timestamp}.csv')
        self.create_csv_file()

    def create_csv_file(self):
        """
        create csv file
        """
        with open(self.path, 'w', encoding='utf-8-sig') as file:
            header = ','.join(self.header) + '\n'
            file.write(header)
        logger.info (f'Successfully created {self.path}')

    def config_to_csv(self, candidate, low_mem, solver_name):
        """write detail info to csv"""
        new_row = ['']*len(self.header)
        h = self.header
        new_row[h.index('test')] = candidate.config_path
        new_row[h.index('求解器')] = solver_name
        new_row[h.index('low_mem')] = low_mem
        new_row[h.index('dense:moe')] = candidate.profiling_info.dmratio
        new_row[h.index('反向:正向')] = candidate.profiling_info.bfratio
        new_row[h.index('重计算增加比率')] = candidate.profiling_info.re_grow_ration
        new_row[h.index('mtp+head')] = candidate.profiling_info.hratio
        new_row[h.index('moe时长')] = candidate.profiling_info.moe_fw
        if DryRun.config_file_type == 0:
            self.yaml_to_row(candidate.config_path, new_row)
        elif DryRun.config_file_type == 1:
            self.shell_to_row(candidate.config_path, new_row)
        elif DryRun.config_file_type == 2:
            self.toml_to_row(candidate, new_row)
        else:
            raise TypeError(DRYRUN_CONFIG_ERROR)
        with open(self.path, 'a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerows([new_row])

    def yaml_to_row(self, yaml_file, row):
        """
        trans yaml info to row of csv
        """
        with open(yaml_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            h = self.header
            row[h.index('layers')] = data['model']['model_config']['num_layers']
            if 'mtp_depth' in data['model']['model_config']:
                row[h.index('layers')] += data['model']['model_config']['mtp_depth']
            row[h.index('micro')] = data['parallel_config']['micro_batch_num']
            row[h.index('dp')] = data['parallel_config']['data_parallel']
            row[h.index('tp')] = data['parallel_config']['model_parallel']
            row[h.index('pp')] = data['parallel_config']['pipeline_stage']
            if 'expert_parallel' in data['parallel_config']:
                row[h.index('ep')] = data['parallel_config']['expert_parallel']
            else:
                row[h.index('ep')] = 1
            if 'pp_interleave_num' in data['model']['model_config']:
                row[h.index('vp')] = data['model']['model_config']['pp_interleave_num']
            else:
                row[h.index('vp')] = 1

    def shell_to_row(self, shell_file, row):
        h = self.header
        configs, _ = pp_util.parse_shell(shell_file)
        row[h.index('layers')] = configs.get('NUM_LAYERS')
        row[h.index('micro')] = configs.get('GBS') // (configs.get('MBS') * configs.get('DP'))
        row[h.index('dp')] = configs.get('DP')
        row[h.index('tp')] = configs.get('TP')
        row[h.index('pp')] = configs.get('PP')
        row[h.index('ep')] = configs.get('EP', 1)
        row[h.index('vp')] = configs.get('VPP', 1)

    def toml_to_row(self, candidate, row):
        """
        trans toml info to row of csv
        """
        model_args = candidate.model_args
        h = self.header
        row[h.index('layers')] = model_args.num_layers
        row[h.index('micro')] = model_args.mbn
        row[h.index('dp')] = model_args.dp
        row[h.index('tp')] = model_args.tp
        row[h.index('pp')] = model_args.pp
        row[h.index('ep')] = model_args.ep
        row[h.index('vp')] = 1
        row[h.index('op/fsdp')] = model_args.dp_shard_degree

    def result_to_csv(self, solution):
        """
        fill solution result to csv
        """
        row_cost =[]
        row_no_cost = []
        with open(self.path, 'r', newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            h = self.header
            for row in reader:
                if row[h.index('test')] == str(solution.init_config.config_file):
                    row[h.index('内存信息')] = solution.init_config.memory.get_mem()
                    row[h.index('内存上限(GB)')] = solution.init_config.memory.mem_lim / 1024
                    row[h.index('求解耗时/s')] = solution.run_time
                    row[h.index('solution_status')] = solution.solution_status
                    row[h.index('model_status')] = solution.model_status
                    if solution.solution_status != 'None':
                        row[h.index('目标值')] = solution.object_value
                        row[h.index('x')] = solution.layer_dis.tolist()
                        row[h.index('offset')] = solution.offset.tolist()
                        row[h.index('ra')] = solution.ra_dis.tolist()
                        row[h.index('GAP')] = solution.gap
                        row[h.index('cost')] = solution.object_value * float(row[h.index('moe时长')])
                        row[h.index('dryrun_check')] = solution.check_peak_mem
                if row[h.index('cost')]:
                    row_cost.append(row)
                else:
                    row_no_cost.append(row)
                sorted_rows = sorted(row_cost[1:], key=lambda x: float(x[self.header.index('cost')]))
                sorted_rows = [self.header] + sorted_rows + row_no_cost
        with open(self.path, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerows(sorted_rows)

if __name__ == '__main__':
    csv = ResultCsv('./output')
