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
"""Lightweight launcher for two-rank dynamic joint-graph FSDP on CPU/Gloo."""

import importlib
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_dynamic_fsdp_gloo():
    """
    Feature: Symbolic joint graph training.
    Description: Run variable-shape FSDP with and without communication overlap.
    Expectation: Both ranks match eager gradients and updates with overlap and reshard options.
    """
    worker = str(Path(__file__).with_name("_test_dynamic_shapes.py"))
    torchrun_case(worker, "test_dynamic_fsdp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_dynamic_causal_loss_npu() -> None:
    """
    Feature: Symbolic joint graph training.
    Description: Run symbolic padding and CE loss/gradient checks on one NPU.
    Expectation: Float32 and bfloat16 losses and gradients match eager with one capture.
    """
    importlib.import_module("tests.torch.compile._test_dynamic_shapes_npu").run_dynamic_causal_loss_npu()
