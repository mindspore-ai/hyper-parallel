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
    pytest tests/torch/fully_shard/test_state_dict.py::test_T4_roundtrip_4cards -v -s

Run all:
    pytest tests/torch/fully_shard/test_state_dict.py -v -s
"""
import os
from tests.torch.utils import torchrun_case

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_state_dict.py")
_PORT_BASE = 12370


# ---------- 2-card tests (T1–T3) ----------

def test_T1_state_dict_2cards():
    """T1: state_dict returns DTensor with correct global shape (2-card)."""
    torchrun_case(_FILE, "test_T1_state_dict_2cards", _PORT_BASE, num_proc=2)


def test_T2_load_dtensor_2cards():
    """T2: load_state_dict with hyper DTensor — copy + assign (2-card)."""
    torchrun_case(_FILE, "test_T2_load_dtensor_2cards", _PORT_BASE + 1, num_proc=2)


def test_T3_load_tensor_2cards():
    """T3: load_state_dict with torch Tensor — local + global (2-card)."""
    torchrun_case(_FILE, "test_T3_load_tensor_2cards", _PORT_BASE + 2, num_proc=2)


def test_T3b_load_single_npu_checkpoint():
    """T3b: single-NPU vanilla checkpoint → fully_shard model (2-card)."""
    torchrun_case(_FILE, "test_T3b_load_single_npu_checkpoint", _PORT_BASE + 5, num_proc=2)


# ---------- 4-card test (T4) ----------

def test_T4_roundtrip_4cards():
    """T4: train -> save -> load(DTensor + Tensor) -> train (4-card)."""
    torchrun_case(_FILE, "test_T4_roundtrip_4cards", _PORT_BASE + 3, num_proc=4)


# ---------- 8-card test (T5) ----------

def test_T5_roundtrip_8cards():
    """T5: full training round-trip — DTensor + local + global (8-card)."""
    torchrun_case(_FILE, "test_T5_roundtrip_8cards", _PORT_BASE + 4, num_proc=8)
