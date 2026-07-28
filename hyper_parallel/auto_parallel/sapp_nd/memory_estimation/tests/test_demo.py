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
# pylint: skip-file
"""tests"""
import os
import pytest

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2


def test_demo_1():
    """demo script test"""
    # Instantiate evaluator with a model configuration,
    #  log_level=0 removes warning messages
    e = EvaluatorV2("mx_test.yaml", log_level=0)

    # Check all defined node type
    assert set(e.ctx.node_eval.keys()) == set(LayerType)

    # Estimate peak memory (in Megabytes)
    peak_mem = e.estimate_peak(verbose=True)
    assert peak_mem > 0

    # Check whether estimation fits in device's max memory
    assert e.mem_fit(peak_mem)

    # Estimate static memory of a specific pipeline stage (in Megabytes)
    res = e.static_mem_stage(1)
    assert peak_mem > res > 0
    # Estimate dynamic memory of a specific pipeline stage (in Megabytes)
    res = e.dynamic_mem_stage(1)
    assert peak_mem > res > 0
    # Estimate static memory of a specific layer and stage (in Megabytes)
    res = e.static_mem_layer(LayerType.FULL_REC_LAYER, 1)
    assert isinstance(res, int)
    # Estimate dynamic memory of a specific layer and stage (in Megabytes)
    res = e.dynamic_mem_layer(LayerType.FULL_REC_LAYER, 1)
    assert isinstance(res, int)

def test_demo_2():
    """demo script test"""
    e = EvaluatorV2("mx_test.yaml", log_level=0)
    # Retrieve the memory estimation logs of a specific stage (in Megabytes)
    res = e.logs_mem_stage(1)
    # assert isinstance(res, list)
    # assert isinstance(res[0], dict)
    # assert {"func", "occ", "mem"} <= res[0].keys()
    assert isinstance(res, dict)


def test_demo_3():
    """demo script test"""
    e = EvaluatorV2("mx_test.yaml", log_level=0)
    # Fetch memory insights from each pipeline stage
    res = e.estimate_peak_insight()
    assert isinstance(res, list)
    assert isinstance(res[0], dict)
    assert "Node Log" in res[0].keys()

    # PPB Input
    e.estimate_layer_memory()
    # Inspect a specific stage (here is the first one)
    e.estimate_peak(spec_stage_id=0, verbose=True)


def test_demo_4():
    """demo script test"""
    e = EvaluatorV2("mx_test.yaml", log_level=0)
    # Plot
    e.estimate_peak(plot=True)
    assert os.path.exists("plots")
    assert os.path.exists("plots/MemPlot_all_stages.png")
    assert os.path.exists("plots/MemPlot_stage_0.png")
    assert os.path.exists("plots/MemPlot_stage_1.png")


def test_demo_5():
    """demo script test"""
    e = EvaluatorV2("ds_test.yaml", log_level=0)

    # Overwriting context function
    def my_attn_num_param(ccfg, _):
        return 10 * ccfg.h * ccfg.h

    e.set_attn_eval_fun(num_p=my_attn_num_param)

    # Overwriting a training feature
    e.set_passes(swap_os=True)
    assert e.ctx.swap_os

    # Overwriting cost model variables
    def custom(ccfg):
        ccfg.bytes_compute = 1
        ccfg.s = 1024
        ccfg.n_attMM = 5

    e.set_ccfg(custom)
    assert e.ccfg.bytes_compute == 1
    assert e.ccfg.s == 1024
    assert e.ccfg.n_attMM == 5

    # Overwriting strategy
    print(e.get_strategy())
    e.set_strategy(dp=8, tp=8, m=128)
    print(e.get_strategy())

    peak_mem = e.estimate_peak(verbose=True)
    assert peak_mem > 0
    # Inspect ccfg object (cost model variables)
    e.print_ccfg()
    # Inspect ctx object (evaluation variables and functions)
    e.print_ctx()

    # Load a hook class
    # ... when declaring an Evaluator
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hooks.template import Template

    e = EvaluatorV2("ds_test.yaml", log_level=0, hook_cls=Template())
    # ... by using load_hook_cls()
    e.load_hook_cls(Template())
