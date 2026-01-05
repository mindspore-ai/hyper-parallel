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
"""profile parser result interface"""
import csv
from fast_tuner.utils.logger import logger


class ProfileInfo:
    """
    Porfile info
    """
    def __init__(self, input_data):
        """若input_data为空，则需要解析profiling结果"""
        if not input_data:
            logger.info('did not use the input data!')
            input_data = self.parse_profiling_result()
        self.dmratio = input_data[0]
        self.bfratio = input_data[1]
        self.re_grow_ration = input_data[2]
        self.hratio = input_data[3]
        self.moe_fw = input_data[4]
        logger.info(f'{input_data}')

    @staticmethod
    def generate_csv():
        """定义 CSV 文件的列名"""
        headers = ['dp', 'tp', 'pp', 'ep', 'vp', 'dmratio', 'bfratio', 'hratio', 'moe_bw', 're_grow_ration']

        # 示例数据，这里可以根据实际需求修改或添加更多行数据
        data = [
            [128, 1, 8, 16, 1, 0.1, 0.2, 0.3, 100, 0.34],
            [128, 1, 8, 8, 1, 0.4, 0.5, 0.6, 200, 0.24]
        ]

        # 定义要保存的 CSV 文件路径
        csv_file_path = './config/profiling_result.csv'

        # 打开文件并写入数据
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入列名
            writer.writerow(headers)

            # 写入数据行
            for row in data:
                writer.writerow(row)

        logger.info(f"CSV 文件已生成，路径为: {csv_file_path}")

    def parse_profiling_result(self):
        """todo: 待填充"""
        profiling_data = [0.1, 0.2, 0.3, 100, 2]
        return profiling_data
