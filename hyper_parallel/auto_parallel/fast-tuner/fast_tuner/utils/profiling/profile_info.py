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
    Profile info
    """
    def __init__(self, input_data):
        """If input_data is empty, parse profiling result"""
        if not input_data:
            logger.info('did not use the input data!')
            input_data = self.parse_profiling_result()
        self.dmratio = input_data[0]
        self.bfratio = input_data[1]
        self.re_grow_ratio = input_data[2]
        self.hratio = input_data[3]
        self.moe_fw = input_data[4]
        logger.info(f'{input_data}')

    @staticmethod
    def generate_csv():
        """Define CSV file column names"""
        headers = ['dp', 'tp', 'pp', 'ep', 'vp', 'dmratio', 'bfratio', 'hratio', 'moe_bw', 're_grow_ratio']

        # sample data, modify or add rows as needed
        data = [
            [128, 1, 8, 16, 1, 0.1, 0.2, 0.3, 100, 0.34],
            [128, 1, 8, 8, 1, 0.4, 0.5, 0.6, 200, 0.24]
        ]

        # define CSV file path to save
        csv_file_path = './config/profiling_result.csv'

        # open file and write data
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # write column names
            writer.writerow(headers)

            # write data rows
            for row in data:
                writer.writerow(row)

        logger.info(f"CSV file generated at: {csv_file_path}")

    def parse_profiling_result(self):
        """TODO: to be filled"""
        profiling_data = [0.1, 0.2, 0.3, 100, 2]
        return profiling_data
