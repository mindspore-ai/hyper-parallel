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
# pylint: skip-file
"""tests"""
import pytest

from memory_estimation.estimate_v2 import EvaluatorV2


def test_evaluator_instantiation():
    """basic instance case"""
    with pytest.raises(TypeError):
        EvaluatorV2()
    with pytest.raises(AttributeError):
        EvaluatorV2("")
    e = EvaluatorV2("ds_test.yaml")
    assert e.ccfg is not None
    assert e.ctx is not None


def test_invalid_hook_class_definition():
    """define hook class"""
    from memory_estimation.hook_base import MemEvalHook, hook_runner

    MemEvalHook.hook_registry.clear()
    with pytest.raises(TypeError):

        class Test1(MemEvalHook):
            pass

    with pytest.raises(TypeError):

        class Test2(MemEvalHook):
            def run_hooks(e):
                pass

    with pytest.raises(TypeError):

        class Test3(MemEvalHook):
            @hook_runner
            def run_hooks(e):
                pass


def test_parsed_strat():
    """valid parsing"""
    e = EvaluatorV2("ds_test.yaml")
    strategy = e.get_strategy()
    assert strategy["dp"] > 0
    assert strategy["tp"] > 0
    assert strategy["ep"] > 0
    assert strategy["pp"] > 0
    assert strategy["cp"] > 0
    assert strategy["vpp"] > 0
    assert 0 < strategy["op"] <= strategy["dp"] * strategy["tp"]
    assert 0 < strategy["ep"] <= strategy["dp"] * strategy["tp"]
    assert strategy["gbs"] >= strategy["dp"]
    assert strategy["sched"]


def test_parsed_transformer_hyperparameters():
    """valid parsing"""
    e = EvaluatorV2("ds_test.yaml")
    assert e.get_num_layers() > 0
    assert e.ccfg.s > 0
    assert e.ccfg.s_fa > 0
    assert e.ccfg.a > 0
    assert e.ccfg.h > 0
    assert e.ccfg.hff > 0
    assert e.ccfg.n_attMM > 0
    assert e.ccfg.n_ffMM > 0
    assert e.ccfg.bytes_p > 0
    assert e.ccfg.bytes_compute > 0
    assert e.ccfg.bytes_softmax > 0
    assert e.ccfg.bytes_grad > 0
    assert e.ccfg.bytes_os > 0
    assert e.ccfg.bytes_norm > 0
