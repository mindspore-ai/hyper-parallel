# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Locate the real modeling source file for a model id.

The generated modeling file is produced from the original ``modeling_*.py``
that the checkpoint actually loads — for native HF models that is the file in
the installed transformers package, for remote-code models it is the file
shipped inside the checkpoint.  Getting the right file matters: the generated
bundle must stay importable in the exact environment the model runs in.

The resolution chain mirrors what HF actually loads (``from_pretrained``):

    pretrained_model_name_or_path -> AutoConfig -> model_type / architectures
                                        -> modeling source file

and prefers the *same* source HF would load.  The deciding input is
``trust_remote_code`` (forwarded from the YAML ``model:`` block): when it is
set and the checkpoint declares ``auto_map``, HF loads the checkpoint's own
modeling file — remote code wins even over an installed transformers module.
Without it, HF uses the installed module (if any), and remote code is only
reached as a fallback.  ``None`` of the resolution helpers import
``transformers`` at module scope: the codegen manager runs before the model is
built, and source resolution must stay functional (degrading) in environments
without torch/transformers.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from hyper_parallel.codegen.hash import file_line_count, sha256_file

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSource:
    """A located modeling source file, ready to be copied into a bundle."""

    module_name: str
    file_path: str
    architecture: Optional[str] = None
    model_type: Optional[str] = None
    sha256: Optional[str] = None
    line_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form for ``CodegenMeta.source``."""
        return {
            "module_name": self.module_name,
            "file_path": self.file_path,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "sha256": self.sha256,
            "line_count": self.line_count,
        }


def resolve_model_source(
    config: Any, model_id: str, *, hf_config: Any = None,
) -> Optional[ResolvedSource]:
    """Resolve the modeling source for a model id, degrading gracefully.

    Resolves ``model_id -> AutoConfig -> architectures/model_type -> modeling
    file``, with fallbacks when the environment cannot do the real lookup:

    1. ``hf_config`` already resolved by the caller -> resolve against it.
    2. ``AutoConfig.from_pretrained(model_id)`` (real HF config, lazy import;
       None on any failure — no transformers / no network / bad id).
    3. ``resolve_source_identity`` — installed module guessed from the
       basename (covers locally-installed model types when AutoConfig is
       unavailable).
    4. Local checkpoint ``config.json`` -> remote code (``SimpleNamespace``
       mirrors ``AutoConfig`` attribute access).

    Returns ``None`` when nothing resolved, so the caller decides whether to
    fail (codegen ``gen`` backend must not run without a source) or degrade
    (``hf`` backend records the unresolved ``SourceSpec`` projection).
    """
    if hf_config is not None:
        return resolve_hf_source(config, hf_config)
    cfg = _load_auto_config(model_id, config)
    if cfg is not None:
        try:
            return resolve_hf_source(config, cfg)
        except FileNotFoundError:
            # AutoConfig resolved but no modeling file exists anywhere (not
            # even remote code in the checkpoint) — keep degrading.
            pass
    source = resolve_source_identity(model_id)
    if source is not None:
        return source
    return _resolve_source_from_yaml(config, model_id)


def resolve_hf_source(config: Any, hf_config: Any) -> ResolvedSource:
    """Resolve the modeling source for a resolved HF config.

    Priority mirrors what the hyper trainer actually loads, which is plain
    transformers ``from_pretrained`` semantics (see auto_model._init_model ->
    _from_pretrained_parent_class, with ``trust_remote_code`` forwarded from the
    YAML ``model:`` block).  The deciding input is ``trust_remote_code``, NOT
    the mere presence of ``auto_map``:

    1. ``trust_remote_code`` AND the checkpoint ships remote code (``auto_map``)
       -> remote code wins, even when an installed module also exists (this is
       what HF loads).  Must resolve to a file, else raise.
    2. Otherwise an installed transformers module for ``model_type`` -> use it
       (this is what HF loads when remote code is not trusted).
    3. Otherwise remote code, if any (a checkpoint whose model_type has no
       installed module; HF would require trust_remote_code to load it, so we
       only reach here to still produce a source for generation).
    """
    arch = _first_architecture(hf_config)
    model_type = getattr(hf_config, "model_type", None)
    has_remote_code = bool(getattr(hf_config, "auto_map", None))
    trust_remote_code = _trust_remote_code(config)
    model_path = _model_path_of(config)

    module_name = _module_for_model_type(model_type)
    installed_path = import_module_file(module_name) if module_name else None

    # 1. Trusted remote code is authoritative — HF loads the checkpoint's own
    #    modeling file over any installed module.
    if has_remote_code and trust_remote_code:
        remote_path = resolve_remote_code_file(model_path, hf_config)
        if remote_path is not None:
            logger.debug("source: resolved to trusted remote code %s", remote_path)
            return _build_resolved_source(remote_path, "<remote>", arch, model_type)
        raise FileNotFoundError(
            f"codegen: trust_remote_code is set and the config for "
            f"{model_path or model_type!r} declares auto_map "
            f"({list(getattr(hf_config, 'auto_map', {}) or {})}), but no "
            "modeling_*.py was found in the checkpoint dir or the loaded "
            "config's module — remote code is the source HF would load"
        )

    # 2. Installed transformers module — what HF loads when remote code is not
    #    trusted (or not present).
    if installed_path is not None:
        logger.debug(
            "source: resolved model_type=%r to installed module %s",
            model_type, installed_path,
        )
        return _build_resolved_source(installed_path, module_name, arch, model_type)

    # 3. Remote code without an installed module (HF would need
    #    trust_remote_code, but we can still locate the file for generation).
    if has_remote_code:
        remote_path = resolve_remote_code_file(model_path, hf_config)
        if remote_path is not None:
            logger.debug(
                "source: resolved to remote code %s (no installed module)",
                remote_path,
            )
            return _build_resolved_source(remote_path, "<remote>", arch, model_type)

    raise FileNotFoundError(
        f"codegen: cannot resolve modeling source for model_type={model_type!r}, "
        f"architecture={arch!r} (no installed transformers module, no remote code)"
    )


def resolve_source_identity(model_id: str) -> Optional[ResolvedSource]:
    """Best-effort source resolution keyed only on a model id.

    Used as a degradation when no ``hf_config`` / ``AutoConfig`` is available.
    Derives the module name from the model id's basename: model ids usually
    carry the repo name the transformers module is keyed on (``DeepSeek-V3``
    -> ``deepseek_v3``).  ``None`` when nothing resolves so the caller falls
    back further (or records the unresolved ``SourceSpec``).
    """
    if not model_id:
        return None
    name = os.path.basename(str(model_id).rstrip("/\\"))
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name).lower()
    module_name = _module_for_model_type(safe)
    if module_name is None:
        return None
    file_path = import_module_file(module_name)
    if file_path is None:
        return None
    return ResolvedSource(
        module_name=module_name,
        file_path=file_path,
        architecture=None,
        model_type=safe,
        sha256=sha256_file(file_path),
        line_count=file_line_count(file_path),
    )


def resolve_transformers_modeling_file(hf_config: Any) -> str:
    """Return the ``modeling_*.py`` path for ``hf_config.model_type``.

    Raises when the model type does not map to a locally installed module.
    """
    model_type = getattr(hf_config, "model_type", None)
    module_name = _module_for_model_type(model_type)
    if module_name is None:
        raise FileNotFoundError(
            f"codegen: no transformers modeling module known for model_type={model_type!r}"
        )
    file_path = import_module_file(module_name)
    if file_path is None:
        raise FileNotFoundError(
            f"codegen: transformers module {module_name} has no __file__"
        )
    return file_path


def resolve_remote_code_file(model_path: str, hf_config: Any) -> Optional[str]:
    """Locate the remote-code modeling file for a checkpoint.

    Remote-code checkpoints carry ``modeling_*.py`` + ``configuration_*.py``
    next to the weights.  Two layouts are handled:

    * A local checkpoint directory — scan it.
    * A Hub id — the files live in the HF cache, not under ``model_path``.  A
      config loaded with ``trust_remote_code`` was constructed from the cached
      ``configuration_*.py``, so its class's ``__module__`` file locates the
      cache dir to scan.

    Within a directory, resolve in order of increasing specificity:

    1. ``auto_map["AutoModel..."]`` value, e.g. ``"modeling_foo.FooForCausalLM"``
       -> ``<dir>/modeling_foo.py``.
    2. The architecture class name, e.g. ``FooForCausalLM`` -> the file whose
       basename contains it.
    3. The single ``modeling_*.py`` when there is exactly one.

    Returns None when no remote code is reachable.
    """
    for directory in _remote_code_dirs(model_path, hf_config):
        found = _scan_remote_code_dir(directory, hf_config)
        if found is not None:
            return found
    return None


def _remote_code_dirs(model_path: str, hf_config: Any) -> list[str]:
    """Directories that may hold this checkpoint's remote code, best first."""
    dirs: list[str] = []
    if model_path and os.path.isdir(model_path):
        dirs.append(model_path)
    # A remote-code config instance is defined in the cached configuration
    # module (``.../modules/transformers_modules/<repo>/configuration_x.py``),
    # which sits next to the cached modeling file.  This is how a Hub id
    # resolves without downloading anything extra.
    module_name = getattr(type(hf_config), "__module__", "") or ""
    if module_name and module_name not in ("builtins", "types"):
        origin = import_module_file(module_name)
        if origin is None:
            module = __import__("sys").modules.get(module_name)
            origin = getattr(module, "__file__", None) if module else None
        if origin:
            cached_dir = os.path.dirname(origin)
            if cached_dir and cached_dir not in dirs and os.path.isdir(cached_dir):
                dirs.append(cached_dir)
    return dirs


def _scan_remote_code_dir(model_path: str, hf_config: Any) -> Optional[str]:
    """Pick this checkpoint's modeling file out of one directory."""
    try:
        candidates = sorted(
            name for name in os.listdir(model_path)
            if name.startswith("modeling_") and name.endswith(".py")
        )
    except OSError:
        return None
    if not candidates:
        return None

    auto_map = getattr(hf_config, "auto_map", None) or {}
    for value in auto_map.values():
        if not isinstance(value, str) or "." not in value:
            continue
        # "modeling_foo.FooForCausalLM" -> "modeling_foo.py".  Hub auto_map
        # values can be "<repo>--modeling_foo.FooForCausalLM"; keep the tail.
        module_ref = value.split("--")[-1]
        module_file = module_ref.split(".", 1)[0] + ".py"
        if module_file in candidates:
            return os.path.join(model_path, module_file)

    arch = _first_architecture(hf_config)
    if arch:
        for name in candidates:
            if arch in name:
                return os.path.join(model_path, name)
    if len(candidates) == 1:
        return os.path.join(model_path, candidates[0])
    return None


def import_module_file(module_name: str) -> Optional[str]:
    """Resolve a Python module's ``__file__`` without importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return None
    if spec is None or spec.origin is None:
        return None
    if not spec.origin.endswith(".py"):
        return None
    return spec.origin


def copy_source_to_artifact(source: ResolvedSource, layout: ArtifactLayout) -> str:
    """Copy the original modeling file into the bundle as ``source_*.py``."""
    from hyper_parallel.codegen.artifact import write_bundle_atomic

    target_name = f"source_{_basename_stem(source.file_path)}.py"
    with open(source.file_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    write_bundle_atomic(layout, {target_name: text})
    return target_name


def _load_auto_config(model_id: str, config: Any) -> Optional[Any]:
    """``AutoConfig.from_pretrained(model_id)`` with lazy import; None on failure.

    ``trust_remote_code`` is forwarded so remote-code checkpoints can load
    their own config classes — mirroring what ``HyperAutoModel.from_pretrained``
    will do at build time.  Any failure (no transformers installed, no network,
    unknown id, remote config raising) returns ``None``; the caller degrades.
    """
    if not model_id:
        return None
    try:
        from transformers import AutoConfig
    except ImportError:
        return None
    kwargs: dict[str, Any] = {}
    if _trust_remote_code(config):
        kwargs["trust_remote_code"] = True
    try:
        return AutoConfig.from_pretrained(model_id, **kwargs)
    except Exception:
        return None


def _resolve_source_from_yaml(config: Any, model_id: str) -> Optional[ResolvedSource]:
    """Best-effort source resolution from a local checkpoint's ``config.json``.

    Used when ``AutoConfig`` and the basename guess both came up empty: for a
    local checkpoint directory, read its ``config.json`` and resolve against
    it.  ``SimpleNamespace`` attribute access mirrors what
    ``AutoConfig.from_pretrained`` produces, without importing transformers.
    Returns ``None`` on any failure so the caller degrades.
    """
    if not model_id or not os.path.isdir(model_id):
        return None
    config_path = os.path.join(model_id, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        hf_config = SimpleNamespace(**raw)
        return resolve_hf_source(config, hf_config)
    except (OSError, ValueError, FileNotFoundError, ImportError):
        return None


def _module_for_model_type(model_type: Optional[str]) -> Optional[str]:
    """Map ``config.model_type`` to a transformers ``modeling_*`` module name."""
    if not model_type:
        return None
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in model_type)
    return f"transformers.models.{safe}.modeling_{safe}"


def _first_architecture(hf_config: Any) -> Optional[str]:
    architectures = getattr(hf_config, "architectures", None) or []
    return architectures[0] if architectures else None


def _model_path_of(config: Any) -> str:
    """Extract the checkpoint dir / HF id from the model target.

    Same key order as ``project_source_spec``: most YAMLs and
    ``HyperAutoModel.from_pretrained`` configure the id under
    ``pretrained_model_name_or_path``, with ``model_name_or_path`` as fallback.
    """
    model = getattr(config, "model", None)
    if model is None:
        return ""
    path = getattr(model, "pretrained_model_name_or_path", None)
    if path is None:
        path = getattr(model, "model_name_or_path", None)
    if path is None:
        path = getattr(model, "_kwargs", {}).get("pretrained_model_name_or_path", "")
    return path or ""


def _trust_remote_code(config: Any) -> bool:
    """Whether this model is loaded with trusted remote code.

    The truth is the YAML's ``model.trust_remote_code`` — it is what the
    trainer forwards into ``AutoConfig.from_pretrained`` / the HF
    ``from_pretrained`` kwargs, and therefore what decides whether HF loads
    the checkpoint's own modeling file or an installed transformers module.
    The ``TrainerConfig`` object itself has no such field, so this must read
    through the model target.
    """
    model = getattr(config, "model", None)
    if model is None:
        return False
    value = getattr(model, "trust_remote_code", None)
    if value is None:
        value = getattr(model, "_kwargs", {}).get("trust_remote_code", False)
    return bool(value)


def _basename_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _relative_imports_of(path: str) -> list[str]:
    """Names a file relatively imports from its own package (one level).

    Mirrors ``transformers.dynamic_module_utils.get_relative_imports`` without
    importing transformers (source resolution must stay dependency-free).  Only
    single-dot relatives are considered: a sibling ``from .name import x`` /
    ``import .name`` resolves within the same package directory, which is what
    a ``<remote>`` modeling source ships.  Multi-dot relatives (``..x``) do not
    resolve against a flat remote code dir and are ignored.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return []
    names = re.findall(r"(?m)^\s*import\s+\.([A-Za-z_][A-Za-z0-9_]*)\s*$", content)
    names += re.findall(r"(?m)^\s*from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import", content)
    return sorted(set(names))


def resolve_remote_siblings(source: ResolvedSource) -> dict[str, str]:
    """Transitive closure of a ``<remote>`` source's package siblings.

    A trust-remote-code modeling file is compiled as the top-level module of a
    synthetic package (loader.import_generated_module), so its single-dot
    relative imports must resolve against *siblings in the same directory*.
    ``source.module_name == "<remote>"`` marks exactly this case; for an
    installed transformers module the source uses absolute imports and this
    returns empty.

    Returns ``{basename: absolute_path}`` for every sibling reachable through
    the source's relative-import closure, e.g. a ``configuration_deepseek.py``
    that ``modeling_deepseek.py`` imports (and any module *it* relatively
    imports).  The caller copies these into the bundle so the artifact is
    self-contained; the loader registers them under the synthetic package.
    """
    if isinstance(source, dict):
        module_name = source.get("module_name")
        file_path = source.get("file_path")
    else:
        module_name = getattr(source, "module_name", None)
        file_path = getattr(source, "file_path", None)
    if module_name != "<remote>" or not file_path:
        return {}
    base_dir = os.path.dirname(os.path.abspath(file_path))
    seen: dict[str, str] = {}
    queue: list[str] = [os.path.abspath(file_path)]
    while queue:
        module_file = queue.pop(0)
        for name in _relative_imports_of(module_file):
            sibling = os.path.join(base_dir, f"{name}.py")
            if not os.path.isfile(sibling):
                continue
            if sibling in seen:
                continue
            seen[sibling] = sibling
            queue.append(sibling)
    return seen


def _build_resolved_source(
    file_path: str, module_name: str, arch: Optional[str], model_type: Optional[str],
) -> ResolvedSource:
    return ResolvedSource(
        module_name=module_name,
        file_path=file_path,
        architecture=arch,
        model_type=model_type,
        sha256=sha256_file(file_path),
        line_count=file_line_count(file_path),
    )


__all__ = [
    "ResolvedSource",
    "copy_source_to_artifact",
    "import_module_file",
    "resolve_hf_source",
    "resolve_model_source",
    "resolve_remote_code_file",
    "resolve_remote_siblings",
    "resolve_source_identity",
    "resolve_transformers_modeling_file",
]
