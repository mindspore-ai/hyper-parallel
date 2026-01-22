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
"""test TorchHSDPParamV2"""
from tests.torch.utils import torchrun_case


def test_hsdp_param_v2_fsdp_1d_mesh():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with 1D FSDP mesh
    Expectation: assertion pass.
    """
    master_port = 12343
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_fsdp_1d_mesh"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_hsdp_2d_mesh():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with 2D HSDP mesh (replicate + shard)
    Expectation: assertion pass.
    """
    master_port = 12344
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_hsdp_2d_mesh"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_sharded_state_transitions():
    """
    Feature: TorchHSDPParamV2.
    Description: Test state transitions (sharded -> unsharded -> sharded)
    Expectation: assertion pass.
    """
    master_port = 12345
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_sharded_state_transitions"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_custom_shard_placement():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with custom shard placement function
    Expectation: assertion pass.
    """
    master_port = 12346
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_custom_shard_placement"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_mixed_precision():
    """
    Feature: TorchHSDPParamV2.
    Description: Test TorchHSDPParamV2 with mixed precision policy
    Expectation: assertion pass.
    """
    master_port = 12347
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_mixed_precision"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_all_gather_comm():
    """
    Feature: TorchHSDPParamV2.
    Description: Test param-level all-gather communication
    Expectation: assertion pass.
    """
    master_port = 12348
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_all_gather_comm"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_prefetch_unshard():
    """
    Feature: TorchHSDPParamV2.
    Description: Test async prefetch and unshard workflow
    Expectation: assertion pass.
    """
    master_port = 12349
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_prefetch_unshard"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_unshard_shard_cycle():
    """
    Feature: TorchHSDPParamV2.
    Description: Test complete unshard -> shard cycle with communication
    Expectation: assertion pass.
    """
    master_port = 12350
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_unshard_shard_cycle"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_reduce_scatter_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test reduce-scatter gradient communication
    Expectation: assertion pass.
    """
    master_port = 12351
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_reduce_scatter_grad"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_all_reduce_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test all-reduce gradient communication in HSDP mode
    Expectation: assertion pass.
    """
    master_port = 12352
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_all_reduce_grad"
    torchrun_case(file_name, case_name, master_port)


def test_hsdp_param_v2_accumulate_grad():
    """
    Feature: TorchHSDPParamV2.
    Description: Test gradient accumulation workflow
    Expectation: assertion pass.
    """
    master_port = 12355
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_v2_accumulate_grad"
    torchrun_case(file_name, case_name, master_port)