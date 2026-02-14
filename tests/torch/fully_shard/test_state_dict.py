# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pytest entry — state_dict tests for fully_shard.

Run 2-card tests:
    pytest tests/torch/fully_shard/test_state_dict.py -k "2cards" -v -s

Run 4-card test:
    pytest tests/torch/fully_shard/test_state_dict.py::test_t4_roundtrip_4cards -v -s

Run all:
    pytest tests/torch/fully_shard/test_state_dict.py -v -s
"""
import os
from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_state_dict.py")
_PORT_BASE = 12370


# ---------- 2-card tests (T1–T3) ----------

def test_t1_state_dict_2cards():
    """
    Feature: Test state_dict returns DTensor after fully_shard.
    Description: Create a 2-card fully_shard model and call state_dict(), verify each entry
        is DTensor with correct global shape.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t1_state_dict_2cards", _PORT_BASE, num_proc=2)


def test_t2_load_dtensor_2cards():
    """
    Feature: Test load_state_dict with hyper DTensor.
    Description: Train model_a, save state_dict as DTensor, load into model_b (copy) and
        model_c (assign) on 2 cards, verify forward output matches.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t2_load_dtensor_2cards", _PORT_BASE + 1, num_proc=2)


def test_t3_load_tensor_2cards():
    """
    Feature: Test load_state_dict with plain torch Tensor.
    Description: Train model, extract local shard and global full tensors, load into new
        models via copy and assign on 2 cards, verify forward output matches.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t3_load_tensor_2cards", _PORT_BASE + 2, num_proc=2)


def test_t3b_load_single_npu_checkpoint():
    """
    Feature: Test load single-NPU vanilla checkpoint into fully_shard model.
    Description: Create vanilla non-sharded model, save state_dict as plain Tensor, load
        into 2-card fully_shard model via copy and assign, then continue training.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t3b_load_single_npu_checkpoint", _PORT_BASE + 5, num_proc=2)


# ---------- 4-card test (T4) ----------

def test_t4_roundtrip_4cards():
    """
    Feature: Test full training round-trip on 4 cards.
    Description: Train 3 steps, save state_dict, load with DTensor and Tensor on 4 cards,
        verify forward output matches and continue training.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t4_roundtrip_4cards", _PORT_BASE + 3, num_proc=4)


# ---------- 8-card test (T5) ----------

def test_t5_roundtrip_8cards():
    """
    Feature: Test full training round-trip on 8 cards.
    Description: Train 3 steps, save state_dict, load with DTensor, local shard and global
        full tensor on 8 cards, verify forward output matches and continue training.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t5_roundtrip_8cards", _PORT_BASE + 4, num_proc=8)


# ---------- 8-card tests (T6–T9): get_model_state_dict ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t6_get_model_sd_sharded():
    """
    Feature: Test get_model_state_dict with default options (sharded).
    Description: Call get_model_state_dict with default StateDictOptions on 8 cards,
        verify returned DTensors match model.state_dict() keys and shapes.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t6_get_model_sd_sharded", _PORT_BASE + 6, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t7_get_model_sd_full_cpu():
    """
    Feature: Test get_model_state_dict with full_state_dict=True and cpu_offload=True.
    Description: Call get_model_state_dict on 8 cards, verify rank 0 gets full CPU
        tensors and non-rank0 gets empty dict.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t7_get_model_sd_full_cpu", _PORT_BASE + 7, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t8_get_model_sd_ignore_frozen():
    """
    Feature: Test get_model_state_dict with ignore_frozen_params=True.
    Description: Freeze one parameter, call get_model_state_dict on 8 cards,
        verify frozen param is excluded and non-frozen params are present.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t8_get_model_sd_ignore_frozen", _PORT_BASE + 8, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t9_get_model_sd_sharded_cpu():
    """
    Feature: Test get_model_state_dict with cpu_offload=True (sharded).
    Description: Call get_model_state_dict with cpu_offload=True on 8 cards,
        verify returned DTensors have local shards on CPU.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t9_get_model_sd_sharded_cpu", _PORT_BASE + 9, num_proc=8)


# ---------- 1-card unit test (T10): _to_dtype_if_needed ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_t10_to_dtype_if_needed():
    """
    Feature: Test _to_dtype_if_needed cast and no-op behavior.
    Description: Verify same dtype returns same object, None returns same object,
        and different dtype returns cast tensor with correct dtype.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t10_to_dtype_if_needed", _PORT_BASE + 10, num_proc=1)
