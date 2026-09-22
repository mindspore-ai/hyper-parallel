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
"""Assemble the generated HF-style modeling file.

The generated file is the original modeling source, verbatim, with two
appended layers:

* a header banner recording where it came from,
* the adapter-driven inline patch — replaced modules and the TP / CP / EP
  strategies lowered into the copied source at their semantic sites.

Boundary forwards are then lowered at the boundary classes.  The artifact
publishes no plan globals: the runtime reads the frozen plan from
``codegen_meta.json`` and shards from it (``runtime.parallelize_from_generated``
falls back to ``_parallelize_inline_from_meta`` when the module defines no
``hyper_parallelize``).  The preflight import checks that the assembled module
remains importable in the training environment.
"""
from __future__ import annotations

import io
import os
import tokenize
from typing import Any

# ---------------------------------------------------------------------------
# source copy
# ---------------------------------------------------------------------------

def copy_original_modeling(source: Any) -> str:
    """Read the original modeling source text from a ``ResolvedSource``-like.

    ``source`` is ``meta.source`` — the ``ResolvedSource.to_dict()`` form with
    a ``file_path`` key.  Accepts either a dict or a ``ResolvedSource`` (the
    dict form is what the manager records, but the helper tolerates the live
    dataclass so it stays usable outside the meta path).
    """
    if isinstance(source, dict):
        path = source.get("file_path")
    else:
        path = getattr(source, "file_path", None)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"codegen emit: original modeling source is not a readable file "
            f"({path!r}); generation must run where the model's modeling file "
            "exists"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def rewrite_relative_imports(text: str, module_name: str) -> str:
    """Rewrite ``from .xxx`` / ``from ...xxx`` imports to absolute imports.

    HF modeling files commonly import siblings with a relative spec, e.g.
    ``from .configuration_qwen3 import Qwen3Config`` or ``from ...activations
    import ACT2FN``.  The generated bundle is loaded by ``spec_from_file_location``
    under a synthetic package (``hyper_parallel_generated.<digest>.<stem>``), so a
    relative spec resolves against that synthetic package instead of the real
    transformers package and fails with ``attempted relative import beyond
    top-level package``.

    This anchors the relative spec on ``module_name`` (the original module's
    real dotted path, e.g. ``transformers.models.qwen3.modeling_qwen3``): N
    leading dots climb N-1 package levels above the module's parent package, then
    append the rest.  ``from ...activations`` → ``from transformers.activations``
    and ``from .configuration_qwen3`` →
    ``from transformers.models.qwen3.configuration_qwen3``.

    ``module_name`` is ``"<remote>"`` for trusted remote code, which has no real
    package to anchor on — the rewrite is a no-op there (import handling for
    remote code keeps its relative imports). A non-dotted or unanchorable ``module_name``
    is likewise left untouched.

    Only the module-spec part of each relative import is rewritten; the symbol
    list, comments, and whitespace are preserved.  It uses ``tokenize`` rather
    than a regex so relative-looking text inside string literals or comments is
    never touched.
    """
    unless_anchor = _relative_anchor(module_name)
    if unless_anchor is None:
        return text

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:
        return text
    if not tokens:
        return text

    edits: list[tuple[int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tk = tokens[i]
        if tk.type != tokenize.NAME or tk.string != "from":
            i += 1
            continue
        # the spec must start with a relative (dots) operator
        j = i + 1
        while j < n and tokens[j].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            j += 1
        if j >= n or tokens[j].type != tokenize.OP or not set(tokens[j].string) <= {"."}:
            i += 1
            continue
        dots_tok = tokens[j]
        dots = len(dots_tok.string)

        # collect the relative module path: NAME ('.' NAME)*
        k = j + 1
        path: list[str] = []
        last_path_tok = None
        while k < n:
            t = tokens[k]
            if t.type == tokenize.NAME and t.string != "import":
                path.append(t.string)
                last_path_tok = t
                k += 1
                if k < n and tokens[k].type == tokenize.OP and tokens[k].string == ".":
                    k += 1  # consume the separator; import follows next round
                else:
                    break
            else:
                break

        # the path must be followed by the ``import`` keyword (allow a comment
        # line in between; a multiline symbol list comes *after* ``import``).
        m = k
        while m < n and tokens[m].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            m += 1
        if m >= n or tokens[m].type != tokenize.NAME or tokens[m].string != "import":
            i += 1
            continue

        abs_spec = _absolute_spec(unless_anchor, dots, path)
        if abs_spec is None:
            i += 1
            continue

        start = dots_tok.start
        end = last_path_tok.end if last_path_tok is not None else dots_tok.end
        edits.append((start, end, abs_spec))
        i = k + 1

    if not edits:
        return text
    return _splice(text, edits)


def _relative_anchor(module_name: str) -> list[str] | None:
    """Return the dotted package segments a relative spec anchors on, or None.

    A real installed-module path (``transformers.models.qwen3.modeling_qwen3``)
    anchors.  The ``"<remote>"`` sentinel (trusted remote code) and any
    non-dotted / placeholder-laden name cannot anchor — the caller leaves such
    imports untouched.
    """
    if not isinstance(module_name, str) or not module_name:
        return None
    if "<" in module_name or ">" in module_name:
        return None
    segs = module_name.split(".")
    if len(segs) < 2 or any(not seg for seg in segs):
        return None
    return segs


def _absolute_spec(anchor: list[str], dots: int, path: list[str]) -> str | None:
    """Compute the absolute import spec for one relative import, or None.

    ``anchor`` is the module's package segments (its path minus the final
    module segment).  ``dots`` leading dots climb ``dots - 1`` package levels
    above the module's parent package, then append ``path``.  Returns None when
    the relative import climbs above the package root (invalid source — leave
    it untouched rather than corrupt it).
    """
    parent = anchor[:-1]
    climb = dots - 1
    if climb > len(parent):
        return None
    base_parts = parent if climb <= 0 else parent[: len(parent) - climb]
    parts = base_parts + path
    return ".".join(part for part in parts if part)


def _splice(text: str, edits: list[tuple[tuple[int, int], tuple[int, int], str]]) -> str:
    """Splice absolute specs back into ``text`` by token (row, col) positions.

    Edits are applied last-first so the earlier offsets stay valid.  A single
    contiguous span is replaced in every case: the relative spec starts at the
    dots token and ends at the last module-path token (or the dots token when
    there is no path, i.e. ``from . import``).
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    line_offsets = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    def offset(row: int, col: int) -> int:
        if 1 <= row <= len(line_offsets):
            return line_offsets[row - 1] + col
        return acc

    positioned = sorted(
        ((offset(*start), offset(*end), new) for start, end, new in edits),
        key=lambda item: item[0],
        reverse=True,
    )
    out = text
    for start, end, new in positioned:
        if start < end <= len(out):
            out = out[:start] + new + out[end:]
    return out


# ---------------------------------------------------------------------------
# file assembly
# ---------------------------------------------------------------------------

def emit_modeling_file(meta: Any) -> str:
    """Assemble the full generated modeling file text.

    Returns the text (not written to disk) so the caller can compute a diff
    first. The header banner carries the generation provenance; the source is
    copied verbatim, rewritten by the adapter-driven inline pipeline, and then
    lowered at the boundary classes.
    """
    text = copy_original_modeling(meta.source)
    # Stale-source sanitize: a trust-remote-code modeling file ships with the
    # checkpoint and reflects the author's transformers, not the installed one.
    # Removing ``is_torch_fx_available`` (a transformers>=5.0 casualty) here
    # keeps the artifact importable without touching the source file. No-op
    # when the source does not import the symbol.
    from hyper_parallel.codegen.source.compat import sanitize_source_compat

    text = sanitize_source_compat(text)
    module_name = (
        meta.source.get("module_name")
        if isinstance(meta.source, dict)
        else getattr(meta.source, "module_name", "")
    )
    text = rewrite_relative_imports(text, module_name)
    text = _apply_inline_modeling(text, meta)
    text = _lower_forward_boundaries(text, meta, module_name)
    return _banner(meta) + text


def _apply_inline_modeling(text: str, meta: Any) -> str:
    """Render the artifact through the adapter-driven inline source pipeline."""

    from hyper_parallel.codegen.inline import render_inline_modeling

    source = getattr(meta, "source", None)
    architecture = source.get("architecture") if isinstance(source, dict) else getattr(source, "architecture", None)
    model_type = getattr(meta, "model_class", None) or architecture
    return render_inline_modeling(text, meta, model_type=model_type)


def _lower_forward_boundaries(text: str, meta: Any, module_name: str) -> str:
    """Sink boundary wrappers into the model's ``forward`` bodies.

    Each boundary class's ``forward`` is rewritten to redistribute through
    the instance-bound compiled plan and run the region's compute
    explicitly. A boundary whose class cannot be resolved or
    whose per-class contract differs across FQNs fails fast — a silently
    dropped redistribution would change the model's numeric behavior.

    Fresh on the unmodified source text, so offsets stay valid for the whole
    call.  The ``local_compute_fn`` / ``inner_wrapper`` members the rewritten
    forward references are bound by the runtime parallelize step, so the
    generated file stays runtime-importable without a mesh.

    Two generation modes exist; exactly one is chosen here per generation:

    - the default :func:`lower_forward_boundaries` (the four parallel forms);
    - the opt-in switch template :func:`lower_forward_boundaries_toggle` (the
      "original forward + per-dimension ``{tp|cp|ep}_enable`` switches" form),
      selected only when ``HP_CODEGEN_TOGGLE_FORWARD`` is set (non-empty).  This
      keeps the default artifact byte-identical while letting a generation opt
      into the switch template without adding a second runtime path.

    ``boundary_classes`` is the FQN -> class-name map frozen at freeze time.  A
    legacy or hand-built plan may have no such map; without
    it the lowerer cannot resolve any boundary FQN to its owning class, so it
    would fail fast on the first boundary instead of degrading to the earlier
    behavior (params sharded by the native Phase A reuse, no forward rewrite).
    When the map is absent, the source remains untouched and runtime dispatch
    handles the boundaries.
    """
    from hyper_parallel.codegen.emit.parallel import (
        lower_forward_boundaries,
        lower_forward_boundaries_toggle,
    )

    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    if not boundary_classes:
        return text
    if os.environ.get("HP_CODEGEN_TOGGLE_FORWARD"):
        return lower_forward_boundaries_toggle(
            text,
            meta,
            boundary_classes=boundary_classes,
            module_name=module_name,
        )
    return lower_forward_boundaries(
        text,
        meta,
        boundary_classes=boundary_classes,
        module_name=module_name,
    )


def _banner(meta: Any) -> str:
    """Render the provenance banner comment."""
    source = meta.source if isinstance(meta.source, dict) else _source_dict(meta.source)
    pd = meta.parallel_dims or {}
    lines = [
        "# Generated by codegen. DO NOT EDIT MANUALLY.",
        "#",
        f"#   config       : {getattr(meta, 'yaml_path', '')}",
        f"#   signature    : {getattr(meta, 'signature', '')}",
        f"#   source       : {source.get('module_name', '')}",
        f"#   source sha256: {source.get('sha256', '')}",
        f"#   architecture : {source.get('architecture', '')}",
        "#   parallel_dims: tp={tp} cp={cp} ep={ep} pp={pp} sp={sp} loss_parallel={lp}".format(
            tp=pd.get("tp_size", 1),
            cp=pd.get("cp_size", 1),
            ep=pd.get("ep_size", 1),
            pp=pd.get("pp_size", 1),
            sp=pd.get("sequence_parallel", False),
            lp=pd.get("loss_parallel", False),
        ),
        "#   injections   : {n} rule(s)".format(n=len(getattr(meta, "injections", []) or [])),
        "#   runtime      : mesh / FSDP2 / PP / AC / compile still handled by runtime",
        "#",
        "# [HYPER PLAN] the frozen sharding plan travels in codegen_meta.json;",
        "# the runtime shards from it.  Regenerate to change.",
        "#",
        "",
    ]
    return "\n".join(lines)


def _source_dict(source: Any) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    return {
        "module_name": getattr(source, "module_name", ""),
        "file_path": getattr(source, "file_path", ""),
        "architecture": getattr(source, "architecture", ""),
        "sha256": getattr(source, "sha256", ""),
    }


__all__ = [
    "copy_original_modeling",
    "emit_modeling_file",
    "rewrite_relative_imports",
]
