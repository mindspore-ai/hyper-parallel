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
"""parallel_scatter_update_shell test"""

import os
import shutil
from tests.common.mark_utils import arg_mark


def run_case(file_name, case_name, master_port):
    """Run test case."""
    file_base = os.path.splitext(file_name)[0]
    dir_to_remove = f"./{file_base}/{case_name}"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
    cmd = f"export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"--join=True --log_dir=./{dir_to_remove}/msrun_log pytest -s -v " \
          f"{file_name}::{case_name}"
    ret = os.system(cmd)
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_data_parallel_1():
    """
    Feature: ScatterUpdate operator.
    Description: Test data parallel on dim 1 in python.
    Expectation: Run success.
    """
    file_name = "scatter_update_shard_in_python.py"
    case_name = "test_scatter_update_data_parallel_1"
    master_port = 12290
    run_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_model_parallel_2():
    """
    Feature: ScatterUpdate operator.
    Description: Test model parallel on dim 2 in python.
    Expectation: Run success.
    """
    file_name = "scatter_update_shard_in_python.py"
    case_name = "test_scatter_update_model_parallel_2"
    master_port = 12291
    run_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_hybrid_parallel_3():
    """
    Feature: ScatterUpdate operator.
    Description: Test hybrid parallel on dim 1 and 2 in python.
    Expectation: Run success.
    """
    file_name = "scatter_update_shard_in_python.py"
    case_name = "test_scatter_update_hybrid_parallel_3"
    master_port = 12292
    run_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_multi_dim_indices_4():
    """
    Feature: ScatterUpdate operator.
    Description: Test with 2D indices in python.
    Expectation: Run success.
    """
    file_name = "scatter_update_shard_in_python.py"
    case_name = "test_scatter_update_multi_dim_indices_4"
    master_port = 12293
    run_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_replicate_all_5():
    """
    Feature: ScatterUpdate operator.
    Description: Test full replication in python.
    Expectation: Run success.
    """
    file_name = "scatter_update_shard_in_python.py"
    case_name = "test_scatter_update_replicate_all_5"
    master_port = 12294
    run_case(file_name, case_name, master_port)
