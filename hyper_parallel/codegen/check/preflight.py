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
"""Train-time integrity checks over a codegen artifact bundle.

The manager's ``preflight_integrity_check`` is the gate; this module holds
the per-zone checks it composes — signature, output hashes, importability,
meta schema, skipped-override warnings, and parameter-plan coverage.  Each
verify_* function is a standalone unit so the CLI can call them
individually.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def verify_signature(meta: Any, current_signature: str) -> None:
    """Fail fast when the stored signature no longer matches the current one.

    A mismatch means the YAML / source / env changed since generation — the
    bundle is stale and must be regenerated before training.
    """
    if meta.signature != current_signature:
        raise RuntimeError(
            f"codegen preflight: signature mismatch for {meta.yaml_path} "
            f"(stored {meta.signature}, current {current_signature}); regenerate"
        )


def verify_output_hashes(meta: Any, layout: Any) -> None:
    """Fail fast when any recorded output file's sha256 drifted.

    Checks the same file set that ``record_output_hashes`` records — the
    generated modeling file, its diff, ``__init__.py``, and any copied remote
    source siblings — so a post-generation edit surfaces here instead of being
    silently imported.
    The meta file itself is not in that set: a file cannot carry its own
    sha256 (writing the value changes the bytes it hashes), so meta
    integrity is covered by the signature and schema checks instead.
    """
    core_paths = {
        "modeling": layout.modeling_path,
        "diff": layout.diff_path,
        "init": layout.init_path,
    }
    for key in core_paths:
        if key not in meta.outputs:
            raise RuntimeError(f"codegen preflight: missing {key} output hash in meta")

    for sibling in getattr(meta, "remote_siblings", None) or []:
        if sibling not in meta.outputs:
            raise RuntimeError(
                f"codegen preflight: missing output hash for remote sibling {sibling!r}"
            )

    for key, recorded in meta.outputs.items():
        path = core_paths.get(key)
        if path is None:
            if (
                not key
                or os.path.basename(key) != key
                or "/" in key
                or "\\" in key
                or key in {".", ".."}
            ):
                raise RuntimeError(
                    f"codegen preflight: invalid output filename {key!r}"
                )
            path = os.path.join(layout.artifact_dir, key)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"codegen preflight: {key} file missing at {path} "
                f"(recorded sha256 {recorded.get('sha256')})"
            )
        actual = _sha256(path)
        if recorded.get("sha256") != actual:
            raise RuntimeError(
                f"codegen preflight: {key} hash drift at {path} "
                f"(recorded {recorded.get('sha256')}, actual {actual})"
            )


def verify_generated_import(layout: Any) -> None:
    """Fail fast when the generated modeling file cannot be imported.

    Importing the generated module is the strongest single check that the
    bundle is usable: it exercises the module's own imports (runtime, torch,
    transformers) and top-level class definitions.  The module is cached in
    ``sys.modules`` by the loader, so preflight and the later build reuse the
    same import. A minimal placeholder also imports because preflight checks
    bundle structure, while the emit layer owns lowering completeness.
    """
    from hyper_parallel.codegen.loader import import_generated_module

    import_generated_module(layout.artifact_dir)


def verify_meta_required_fields(meta: Any) -> None:
    """Fail fast when the meta record is missing required schema fields.

    Delegates to the meta module's schema validation; a stale or hand-edited
    meta file that lost a required field must not pass preflight.
    """
    from hyper_parallel.codegen.meta import validate_meta_schema

    validate_meta_schema(meta_to_dict(meta))


def warn_skipped_overrides(meta: Any) -> None:
    """Warn (not error) when generation skipped any plan_overrides entries.

    An entry skipped during generation (or a ``when``-gated entry that never
    applied) must be surfaced
    so the user knows the artifact does not contain what the YAML declared.
    """
    skipped = getattr(meta, "skipped_overrides", None) or []
    if not skipped:
        return
    lines = [f"codegen: {len(skipped)} plan_overrides entries were skipped:"]
    for item in skipped:
        match = item.get("match") if isinstance(item, dict) else None
        reason = item.get("reason") if isinstance(item, dict) else item
        lines.append(f"  - match={match!r}: {reason}")
    logger.warning("\n".join(lines))


def verify_param_plan(meta: Any, model: Any) -> None:
    """Verify the frozen param plan covers the model's trainable parameters.

    Two directions, both against ``model`` (F4b semantics at preflight time):

    1. Every key recorded in ``meta.param_plan`` (boundary + relative param
       name, e.g. ``model.layers.0.mlp.experts.gate_up_proj``) must resolve
       to a parameter on ``model``.
    2. Every trainable parameter of ``model`` must be covered by a plan key
       — either exactly, or as a leaf under a grouped key (the planner stacks
       expert params under ``experts.gate_up_proj``, which owns both
       ``.weight`` and ``.bias``).  A parameter is NOT covered by a bare
       ancestor *boundary* (``model.norm`` alone never covers
       ``model.norm.bias``); only an explicit param entry counts.  This is
       deliberately per-boundary rather than ``frozen_sharded_params``-only:
       replicated parameters are still planned (their placement is recorded
       in ``param_plan``) and must be listed there too.

    An *empty* plan fails because generation always freezes a derived plan.
    An all-``Replicate`` topology still records every parameter, so
    ``param_plan == {}`` means derivation failed or was
    skipped — the artifact is incomplete and must not pass.  A model-side
    change that the signature did not catch (e.g. a different checkpoint
    variant) fails fast instead of silently running with a stale sharding
    contract.  Degrades to a warning when ``model`` is not given (preflight
    without a model handle can still check the plan's internal consistency).
    """
    param_plan = getattr(meta, "param_plan", None) or {}
    if not param_plan:
        raise RuntimeError(
            "codegen preflight: param plan is empty — generation did not "
            "freeze a sharding plan (derivation failed or was skipped); "
            "check generation logs and regenerate the artifact"
        )

    if model is None:
        logger.warning(
            "codegen preflight: no model handle — param-plan coverage "
            "skipped (only internal consistency checked)"
        )
        return

    # Every planned parameter key must resolve to a model parameter.
    planned: set[str] = set()
    missing: list[str] = []
    for boundary, entry in param_plan.items():
        params = entry.get("params", {}) if isinstance(entry, dict) else {}
        for name in params:
            fqn = f"{boundary}.{name}" if name else boundary
            planned.add(fqn)
            if _get_parameter(model, fqn) is None:
                missing.append(fqn)
    if missing:
        raise RuntimeError(
            f"codegen preflight: param plan references parameters missing "
            f"from the model: {missing}"
        )

    # Every trainable model parameter must be covered by an explicit plan
    # entry (exact FQN or a leaf under a grouped key).
    uncovered = [
        fqn
        for fqn in _iter_trainable_params(model)
        if not any(fqn == key or fqn.startswith(f"{key}.") for key in planned)
    ]
    if uncovered:
        raise RuntimeError(
            f"codegen preflight: {len(uncovered)} trainable parameters are "
            f"not covered by the frozen plan: {uncovered[:20]}"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def meta_to_dict(meta: Any) -> dict[str, Any]:
    """Serialize meta to dict, or pass through a dict already."""
    if isinstance(meta, dict):
        return meta
    from hyper_parallel.codegen.meta import meta_to_dict as _meta_to_dict

    return _meta_to_dict(meta)


def _iter_trainable_params(model: Any):
    """Yield ``fqn`` for every trainable parameter of ``model`` (or its config)."""
    try:
        import torch
    except ImportError:
        return
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            fqn = f"{module_name}.{param_name}" if module_name else param_name
            if id(param) in seen:
                continue
            seen.add(id(param))
            yield fqn


def _get_parameter(model: Any, fqn: str) -> Any:
    """Look up a parameter by dotted FQN, tolerating an absent attribute."""
    obj = model
    for part in fqn.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj if hasattr(obj, "requires_grad") else None


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "verify_generated_import",
    "verify_meta_required_fields",
    "verify_output_hashes",
    "verify_param_plan",
    "verify_signature",
    "warn_skipped_overrides",
]
