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

"""Framework-free launchers for synchronous stream KV attention tests."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_stream_kv_attention_cp8():
    """Validate compact/replicated KV against independent CPU autograd.

    Feature: Stream KV attention with Ulysses and optional causal balancing.
    Description: Run CP8 workers across head replication and ragged panel shapes.
    Expectation: Output and every input gradient pass both numerical thresholds.
    """
    torchrun_case("tests/torch/context_parallel/_test_stream_kv.py", "test_stream_kv_attention", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_stream_kv_attention_cp6():
    """Validate odd KV groups with real collectives.

    Feature: Stream KV causal balancing with an odd number of KV owners.
    Description: Run six workers including Pu2/Pg3 and zero peer split exchanges.
    Expectation: Output and every input gradient match independent CPU attention.
    """
    torchrun_case("tests/torch/context_parallel/_test_stream_kv.py", "test_stream_kv_attention", num_proc=6)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_stream_kv_component_cp8():
    """Validate actual GatedGQA training and state lifetime.

    Feature: Opt-in stream KV integration with the existing GatedGQA component.
    Description: Run AMP, checkpoint and three optimizer steps with parameter SUM.
    Expectation: Component values/gradients agree with CPU and saved state releases.
    """
    torchrun_case("tests/torch/context_parallel/_test_stream_kv.py", "test_stream_kv_component", num_proc=8)
