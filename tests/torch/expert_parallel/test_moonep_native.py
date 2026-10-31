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
"""Lightweight launchers for native MoonEP CPU and representative NPU contracts."""
import importlib
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


NATIVE_CASES = {
    'fp32_inference_then_train': ('native', ('--inference-first', '--retain',), 1),
    'bf16_checkpoint_delayed_retain_cpu': ('native', ('--dtype', 'bfloat16', '--retain', '--delayed', 
                                                      '--checkpoint', 'nonreentrant', '--saved-cpu-hooks',), 1),
    'bf16_reentrant_fsdp_offload': ('native', ('--dtype', 'bfloat16', '--fsdp', '--offload', 
                                               '--checkpoint', 'reentrant', '--master-float32', 
                                               '--saved-cpu-hooks',), 1),
    'async_chunk_capacity_checkpoint': ('native', ('--async-dispatch', '--capacity-factor', '0.5', 
                                                   '--checkpoint', 'reentrant',), 2),
    'gated_bias_2d_optimizer': ('grouped', ('--bias', '--two-dimensional', '--direct-ab', 
                                            '--optimizer-step',), 1),
    'ungated_bf16_direct': ('grouped', ('--ungated', '--direct-ab',), 1),
    'ungated_bias_2d_fp32_oracle': ('grouped', ('--ungated', '--bias', '--two-dimensional', '--dtype', 
                                                'float32',), 1),
    'bias_only_empty_optimizer': ('grouped', ('--ungated', '--bias', '--two-dimensional', '--direct-ab', 
                                              '--optimizer-step', '--empty-receiver', '--freeze', 'base',), 1),
    'frozen_bias_input_gradient': ('grouped', ('--ungated', '--bias', '--two-dimensional', '--direct-ab', 
                                               '--optimizer-step', '--freeze', 'bias',), 1),
    'all_frozen_input_gradient': ('grouped', ('--ungated', '--bias', '--two-dimensional', '--direct-ab', 
                                              '--optimizer-step', '--freeze', 'all',), 1),
    'grouped_bf16_checkpoint_retain_cpu': ('native', ('--dtype', 'bfloat16', '--grouped-gemm', '--retain', 
                                                      '--checkpoint', 'nonreentrant', '--saved-cpu-hooks',), 1),
    'real_v41_factory': ('v41', (), 1),
}


@pytest.mark.parametrize('case_name', tuple(NATIVE_CASES))
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='allcards', essential_mark='unessential')
def test_native_moonep_contract(case_name: str) -> None:
    """
    Feature: Independent native MoonEP contracts.
    Description: Run the real HP dispatcher, expert modules and FP32 owner return.
    Expectation: Precision and lifecycle match ordinary native execution.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name('_test_moonep_native.py')),
        case_name=f'test_native_moonep_case[{case_name}]', num_proc=4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='allcards', essential_mark='unessential')
def test_native_moonep_cpu_transport() -> None:
    """
    Feature: Real CPU transport independent of the NPU worker.
    Description: Run Gloo ownership and both checkpoint modes in two processes.
    Expectation: Source rows, gradients and parameter identities match the oracle.
    """
    worker = importlib.import_module('tests.torch.expert_parallel._test_moonep_native_cpu')
    worker.TestNativeMoonEPTransport().test_real_alltoall_output_gradient_and_checkpoint()
