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
import builtins
import importlib.machinery
import importlib.util
import re
import symtable
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

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
from hyper_parallel.codegen.inline.framework_spec import replacement_spec_for  # noqa: E402
from hyper_parallel.components.modules import GQAAttention  # noqa: E402
from hyper_parallel.models.qwen3_moe.adapter.attention import (  # noqa: E402
    run_qwen3_moe_flash_attention,
)


def _forward_body(source: str) -> str:
    tree = ast.parse(source)
    gqa = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GQAAttention")
    forward = next(node for node in gqa.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    return ast.get_source_segment(source, forward)


def _undefined_globals(source: str, *, provided: Iterable[str] = ()) -> list[str]:
    """Global names ``source`` reads but neither binds, imports, nor finds builtin.

    ``symtable`` does the scope resolution, so parameters, locals, comprehension
    targets and attribute accesses never register as module-scope reads: only
    names that must resolve outside the emitted module survive. Dunders
    (``__name__``) exist in any module.
    """
    table = symtable.symtable(source, "<emitted attention>", "exec")
    defined = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    undefined: set[str] = set()
    pending = [table]
    while pending:
        current = pending.pop()
        for symbol in current.get_symbols():
            name = symbol.get_name()
            if (
                symbol.is_global()
                and symbol.is_referenced()
                and name not in defined
                and not hasattr(builtins, name)
                and not name.startswith("__")
            ):
                undefined.add(name)
        pending.extend(current.get_children())
    return sorted(undefined - set(provided))


def _spec_installed_names() -> set[str]:
    """Names the replacement spec installs beside the rendered snippets.

    The rendered class and the spec only bind names the artifact itself
    defines; the parallel channel is bound on the instance at install time, so
    the artifact imports no codegen-runtime helper for it. The spec is resolved
    from the real family factory, the same way generation resolves it.
    """
    spec = replacement_spec_for(
        "hyper_parallel.models.qwen3_moe.adapter.replacements."
        "replace_qwen3_moe_flash_attention",
        "qwen3_moe",
        module_type="transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
    )
    return {name for patch in spec.imports for name in patch.names}


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

    def test_fusion_replacement_is_annotated(self) -> None:
        # The fused class stays a genuine class, but a source comment names the
        # module replacement so the generated model file reads the fusion (block
        # 2, requirement 3: 模块替换注释化).
        self.assertIn("# [HYPER INLINE] GQAAttention:", self.source)
        self.assertIn(
            "q_proj/k_proj/v_proj are packed into a single linear_qkv", self.source
        )

    def test_forward_is_native_verbatim_no_inline_orchestration(self) -> None:
        body = _forward_body(self.source)
        # The forward is pure native verbatim: no inline TP or CP orchestration.
        # The emitted boundary form redistributes around it
        # (``hyper_install_boundaries``) and CP is the declared inner wrapper
        # (``hyper_apply_inner_wrapper``), so the artifact must not reference
        # the parallel channel at all.
        self.assertIn("self.attention_interface(", body)
        self.assertNotIn("tp_mesh", body)
        self.assertNotIn("_hyper_tp", body)
        self.assertNotIn("all_gather(hidden_states, dim=1)", body)
        self.assertNotIn("reduce_scatter(attn_output, dim=1)", body)
        self.assertNotIn("flex_cp_allgather(", body)
        self.assertNotIn("_hyper_cp_mesh", body)
        self.assertNotIn("_build_qwen3_moe_flash_attention_cp_interface", self.source)
        self.assertIsNone(re.search(r"self\._hyper_(?:boundary|inline_boundary)", body))

    def test_emitted_artifact_has_no_parallel_state_accessor(self) -> None:
        """The artifact is pure native -- no module-level lookup, no parallel channel.

        Feature: codegen-external-state
        Description: The inlined attention forward is the native
            ``GQAAttention.forward`` verbatim; it takes its parallel behavior
            from the ``attention_interface`` the CP inner wrapper swaps in at
            install time, so no ``get_parallel_state`` accessor (and no ``ps.*``
            reference, no ``_hyper_*`` channel) may survive anywhere in the
            emitted module.
        Expectation: Neither accessor name nor a ``ps.`` reference appears; the
            forward keeps only its native ``self.attention_interface(...)`` call,
            and CP / TP carry no codegen orchestration (no factory, no raw
            dispatch, no ``_hyper_tp`` / ``_hyper_cp_mesh`` reference in the
            rendered class).
        """
        self.assertNotIn("get_parallel_state", self.source)
        self.assertNotIn("get_inline_parallel_state", self.source)
        self.assertIsNone(re.search(r"\bps\.", self.source))
        body = _forward_body(self.source)
        self.assertIn("self.attention_interface(", body)
        self.assertNotIn("_hyper_tp", body)
        self.assertNotIn("tp_mesh", body)
        self.assertNotIn("flex_cp_allgather(", body)
        self.assertNotIn("_build_qwen3_moe_flash_attention_cp_interface", self.source)

    def test_kernel_entry_is_inlined_not_imported(self) -> None:
        self.assertIn("self.attention_interface(", _forward_body(self.source))
        self.assertIn("def run_qwen3_moe_flash_attention(", self.source)
        self.assertIn("def _get_compressed_causal_mask(", self.source)
        self.assertIn("npu_fusion_attention(", self.source)

    def test_forward_keeps_var_kwargs(self) -> None:
        # Real forward bodies forward **kwargs to the kernel; stripping annotations
        # must never remove the parameter itself.
        forward = _forward_body(self.source)
        self.assertIn("def forward(self", forward)
        self.assertIn("**kwargs", forward)

    def test_every_referenced_global_resolves_inside_the_artifact(self) -> None:
        # A copied class body carries its references with it: a module-level
        # helper the renderer never inlines (or an import it never emits) is a
        # NameError in the generated model, so the emitted module -- imports plus
        # source, the way the inline pipeline installs it -- must leave no global
        # name undefined. This is a whole-artifact check, not a list of names that
        # are expected to appear.
        self.assertEqual(
            _undefined_globals(self.source, provided=_spec_installed_names()), []
        )

    def test_module_level_helpers_come_from_the_component_source(self) -> None:
        # rotate_half / _apply_gqa_rope / _projections_can_fuse are copied from
        # the component's own module (the same mechanism _class_methods uses), so
        # the artifact cannot carry a second, drifted copy of any of them; the set
        # of helpers is derived from the copied code, never listed in the renderer.
        self.assertIn("def rotate_half(x):", self.source)
        self.assertIn(
            "def _apply_gqa_rope(query, key, position_embeddings, rotary_interleaved, head_dim):",
            self.source,
        )
        self.assertIn("def _projections_can_fuse(projections):", self.source)
        self.assertIn("return torch.cat((-second, first), dim=-1)", self.source)
        self.assertIn(
            "projection.weight.requires_grad for projection in projections",
            self.source,
        )


#: A stand-in component whose ``forward`` calls a name its module neither defines
#: nor imports -- the shape of the drift this renderer must refuse to emit.  It
#: keeps the local names the injected TP/CP boundary reads (``query_states`` etc.)
#: so the unresolvable helper is the only name left over.
_BROKEN_COMPONENT = '''
from torch import nn


def run_qwen3_moe_flash_attention(attention, query_states, **kwargs):
    """Minimal kernel entry: the call site the renderer rewrites to."""
    del attention, kwargs
    return query_states, None


class MissingHelperAttention(nn.Module):
    """Attention whose forward reads a helper its module never binds."""

    def __init__(self, *, module=None, attention_interface=run_qwen3_moe_flash_attention):
        super().__init__()
        del module
        self.attention_interface = attention_interface

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        actual_seq_len=None,
        **kwargs,
    ):
        del position_embeddings, past_key_values, actual_seq_len
        query_states = key_states = value_states = hidden_states
        attn_output, attn_weights = self.attention_interface(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return _dropped_by_a_refactor(attn_output), attn_weights
'''


class TestRenderAttentionClassRejectsUndefinedNames(unittest.TestCase):
    """An unresolvable reference is a generation error, never an emitted name."""

    def _load_broken_component(self) -> types.ModuleType:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "broken_attention_component.py"
        path.write_text(_BROKEN_COMPONENT, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "codegen_broken_attention_component", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        return module

    def test_unresolvable_reference_fails_generation(self) -> None:
        component = self._load_broken_component()
        with self.assertRaises(ValueError) as raised:
            render_attention_class(
                component.MissingHelperAttention,
                interface=component.run_qwen3_moe_flash_attention,
            )
        message = str(raised.exception)
        self.assertIn("_dropped_by_a_refactor", message)
        self.assertIn("MissingHelperAttention", message)


if __name__ == "__main__":
    unittest.main()
