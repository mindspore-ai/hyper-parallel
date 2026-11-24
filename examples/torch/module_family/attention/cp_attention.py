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
"""CP member changing only the causal attention operand exchange."""

from __future__ import annotations

from typing import Any

from examples.torch.module_family.attention.attention import AttentionOperands, CausalAttentionModule
from hyper_parallel.distributed.module_binding import ParallelModuleSpec


def _validate_layout(*, module: Any, sharding: Any, context: Any) -> None:
    del module
    if context.axis_sizes["tp"] != 1 or any(placement.is_shard() for axes in sharding.params.values()
                                           for placement in axes.values()):
        raise ValueError("example CP attention requires TP=1 and replicated projection weights")


class CPCausalAttention(CausalAttentionModule):
    """Gather K/V through the resource supplied by the common binder."""

    def bind_parallel(self, binding: Any) -> None:
        """Borrow the existing CP context.

        Args:
            binding: Public parallel resource bundle.
        """
        self.cp_context = binding.cp_context

    def exchange_attention(self, operands: AttentionOperands) -> AttentionOperands:
        """Keep local Q and gather global K/V with their global offset.

        Args:
            operands: Local projected Q/K/V.
        """
        key, partition = self.cp_context.gather_sequence(operands.key)
        value, _ = self.cp_context.gather_sequence(operands.value)
        return AttentionOperands(operands.query, key, value, partition.offset)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare stage signatures and computation-layout support."""
        return ParallelModuleSpec("example.causal_attention.cp", "cp",
                                  {"exchange_attention": ("operands",), "restore_attention": ("output",)},
                                  validate=_validate_layout)
