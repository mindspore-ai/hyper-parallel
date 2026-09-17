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
"""Render a self-contained attention class from real source with visible boundaries."""

import ast
import importlib.machinery
import re
import sys
import types
import unittest

# The generator inspects source text only; it never evaluates NPU code. The
# components are imported here solely to resolve their source, so a CPU-only
# checkout needs ``import torch_npu`` and a CPU-safe ``torch.npu`` accessor.
import torch  # noqa: E402

if "torch_npu" not in sys.modules:
    try:
        import torch_npu  # noqa: F401  pylint: disable=unused-import
    except ModuleNotFoundError:  # pragma: no cover - CPU-only host
        _torch_npu = types.ModuleType("torch_npu")
        _torch_npu.__spec__ = importlib.machinery.ModuleSpec(name="torch_npu", loader=None)
        sys.modules["torch_npu"] = _torch_npu
if not hasattr(torch, "npu"):

    class _NpuMode:  # pragma: no cover - CPU-only host
        """CPU placeholder so ``torch.npu.is_available()`` reads not-available."""

        def is_available(self):
            return False

        def device_count(self):
            return 0

    torch.npu = _NpuMode()

from hyper_parallel.codegen.inline.attention import render_attention_class  # noqa: E402
from hyper_parallel.components.modules import GQAAttention  # noqa: E402
from hyper_parallel.models.qwen3_moe.adapter.attention import (  # noqa: E402
    run_qwen3_moe_flash_attention,
)


def _forward_body(source: str) -> str:
    tree = ast.parse(source)
    gqa = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GQAAttention")
    forward = next(node for node in gqa.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    return ast.get_source_segment(source, forward)


class TestRenderAttentionClass(unittest.TestCase):
    """Cover generation invariants asserted by preflight and the S3 byte gate."""

    def setUp(self) -> None:
        self.expanded = render_attention_class(
            GQAAttention, interface=run_qwen3_moe_flash_attention
        )
        self.source = "{}\n\n{}".format("\n".join(self.expanded.imports), self.expanded.source)

    def test_generated_source_is_valid_python(self) -> None:
        ast.parse(self.source)

    def test_preserves_fused_qkv_layout(self) -> None:
        self.assertIn("self.linear_qkv = nn.Linear(", self.source)
        self.assertIn("self.qkv_split_sizes =", self.source)
        self.assertNotIn("self.q_proj = nn.Linear(", self.source)
        self.assertNotIn("self.k_proj = nn.Linear(", self.source)
        self.assertNotIn("self.v_proj = nn.Linear(", self.source)
        self.assertIn("InterleaveQKV(", self.source)

    def test_clean_inline_boundary_uses_parallel_state_only(self) -> None:
        body = _forward_body(self.source)
        self.assertIn("ps = get_parallel_state()", body)
        self.assertIn("ps.tp.all_gather", body)
        self.assertIn("ps.tp.reduce_scatter", body)
        self.assertIn("ps.cp_enabled", body)
        # Mirrors check/preflight.py clean-inline gate: no _hyper_* in the forward.
        self.assertIsNone(re.search(r"self\._hyper_(?:inline_boundary|tp|cp_mesh|ep_group)", body))

    def test_kernel_entry_is_inlined_not_imported(self) -> None:
        self.assertIn("run_qwen3_moe_flash_attention(self, query_states", self.source)
        self.assertIn("def run_qwen3_moe_flash_attention(", self.source)
        self.assertIn("def _get_compressed_causal_mask(", self.source)
        self.assertIn("npu_fusion_attention(", self.source)
        # The forward must not reference the instance-level interface (black box).
        self.assertNotIn("self.attention_interface(", _forward_body(self.source))

    def test_forward_keeps_var_kwargs(self) -> None:
        # Real forward bodies forward **kwargs to the kernel; stripping annotations
        # must never remove the parameter itself.
        forward = _forward_body(self.source)
        self.assertIn("def forward(self", forward)
        self.assertIn("**kwargs", forward)


if __name__ == "__main__":
    unittest.main()