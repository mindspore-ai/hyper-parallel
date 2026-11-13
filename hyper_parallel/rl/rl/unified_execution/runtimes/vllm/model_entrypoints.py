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
"""Worker-only compilation registration around the unchanged HP model adapter."""

from vllm.compilation.decorators import support_torch_compile

from rl.unified_execution.runtimes.vllm.model_adapter import (
    HyperParallelQwen3ForCausalLM as SharedQwen3ForCausalLM,
    HyperParallelQwen3MoeForCausalLM as SharedQwen3MoeForCausalLM,
)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0})
class HyperParallelQwen3ForCausalLM(SharedQwen3ForCausalLM):
    """Expose the HP forward to vLLM's compiler without replacing its modules."""


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0})
class HyperParallelQwen3MoeForCausalLM(SharedQwen3MoeForCausalLM):
    """Compile the shared protocol with the existing HP MoE components."""
