# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test base dtensor"""
import os


def run_case(master_port):
    cmd = f"torchrun --nproc-per-node=8 " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"base_dtensor.py"
    ret = os.system(cmd)
    assert ret == 0


def test_base_dtensor():
    '''
    Feature: dtensor dispatch/infer_layout/redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11333
    run_case(master_port)


if __name__ == "__main__":
    test_base_dtensor()
