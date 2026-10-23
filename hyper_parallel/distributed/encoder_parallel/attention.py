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
"""Encoder attention segments and exact execution auditing."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Mapping

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment


@dataclass
class EncoderAttentionContext:
    """One ViT pass: segment boundaries, Ulysses mesh, and call audit."""

    segments: tuple[EncoderSegment, ...]
    cp_mesh: DeviceMesh
    declared_heads: Mapping[str, int]
    expected: frozenset[str]
    seen: dict[str, int] = field(default_factory=dict)

    def record(self, module_fqn: str) -> None:
        """Record one replacement invocation.

        Args:
            module_fqn: Final module path for the attention block.
        """
        self.seen[module_fqn] = self.seen.get(module_fqn, 0) + 1

    def assert_exactly_once(self) -> None:
        """Reject missing or repeated calls in an active ViT pass."""
        if set(self.seen) != self.expected or any(count != 1 for count in self.seen.values()):
            raise RuntimeError(
                "[HP-ENCODER-008] vision attention execution mismatch; "
                f"expected={sorted(self.expected)}; seen={self.seen}"
            )

    def validate_runtime_head_counts(self, *, module_fqn: str, query_heads: int,
                                     key_heads: int, value_heads: int) -> None:
        """Check projected head counts before entering a collective."""
        actual = {"query": query_heads, "key": key_heads, "value": value_heads}
        if actual != self.declared_heads or any(count % self.cp_mesh.size() for count in actual.values()):
            raise ValueError(
                "[HP-ENCODER-005] Ulysses runtime head count mismatch; phase=runtime; "
                f"module_fqn={module_fqn}; declared_head_counts={self.declared_heads}; "
                f"runtime_head_counts={actual}; encoder_cp_size={self.cp_mesh.size()}"
            )


_ACTIVE_CONTEXT: ContextVar[EncoderAttentionContext | None] = ContextVar("encoder_attention_context", default=None)


def current_encoder_attention_context() -> EncoderAttentionContext | None:
    """Return the complete attention state for the current ViT pass."""
    return _ACTIVE_CONTEXT.get()


@contextmanager
def activate_encoder_attention(context: EncoderAttentionContext) -> Iterator[EncoderAttentionContext]:
    """Scope attention work and its exact-call audit to one packed ViT pass.

    Args:
        context: Segments, executor, and expected module paths for this pass.
    """
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)
