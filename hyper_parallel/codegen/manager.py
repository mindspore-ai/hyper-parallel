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
"""Codegen manager: decide, generate (rank0), reuse, and preflight.

The manager is the only entry point the trainer / CLI talks to.  It does NOT
do AST patching or source emitting itself — that is delegated to the
``emit`` layer — it decides whether an artifact is stale, who generates it,
and whether it passes preflight before training starts.
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
from typing import Any, Optional

import torch.distributed as dist

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    artifact_failure_path,
    artifact_generation_lock,
    clear_artifact_failure,
    clean_temp_artifacts,
    ensure_artifact_dir,
    publish_artifact_failure,
    resolve_default_artifact_layout,
    resolve_model_artifact_layout,
    resolve_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.emit import emit_bundle
from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.hash import (
    canonical_snapshot_payload,
    sha256_text,
    signature_from_snapshot,
)
from hyper_parallel.codegen.meta import (
    COVERED_DEFAULTS,
    NOT_COVERED_DEFAULT,
    CodegenMeta,
    load_codegen_meta,
)
from hyper_parallel.codegen.snapshot import CodegenSnapshot, CodegenStatus
from hyper_parallel.codegen.source.resolver import resolve_remote_siblings
from hyper_parallel.models.registry import artifact_model_type, family_dir_for_model_type

logger = logging.getLogger(__name__)

CODEGEN_VERSION = "0.1.2"


def ensure_codegen_artifact_from_snapshot(
    snapshot: CodegenSnapshot,
    artifact_dir: Optional[str] = None,
    *,
    timeout_s: float = 600.0,
) -> ArtifactLayout:
    """Generate or reuse a bundle using only frozen native-build data.

    This entry must remain independent of config projection, meta-model
    construction, offline meshes, replacement matching, and planner calls.
    """
    if not isinstance(snapshot, CodegenSnapshot):
        raise TypeError("snapshot must be a CodegenSnapshot")
    if snapshot.status not in (CodegenStatus.DERIVED, CodegenStatus.NOT_REQUIRED):
        raise ValueError(
            "snapshot-driven generation requires status 'derived' or 'not_required', "
            f"got {snapshot.status.value!r}"
        )
    payload = snapshot.to_dict()
    source = payload["source"]
    if not source.get("file_path"):
        raise ValueError("snapshot source must include a modeling file_path")
    model_type = source.get("model_type")
    public_model_type = artifact_model_type(str(model_type)) if model_type else None
    model_name = (
        _model_type_to_modeling_token(str(public_model_type))
        if model_type
        else source.get("architecture")
    )
    if artifact_dir is None and source.get("model_type"):
        layout = resolve_model_artifact_layout(
            family_dir_for_model_type(str(source["model_type"])),
            model_name=str(model_name) if model_name else None,
        )
    else:
        layout = resolve_default_artifact_layout(
            artifact_dir,
            model_name=str(model_name) if model_name else None,
        )
    ensure_artifact_dir(layout)
    is_rank0 = _is_rank0()
    if is_rank0:
        clean_temp_artifacts(layout)

    signature = signature_from_snapshot(snapshot)
    validate_snapshot_signature(signature)
    existing = load_codegen_meta(layout.meta_path)
    if existing is not None and not should_regenerate(existing, signature):
        if is_rank0:
            logger.info(
                "codegen: reuse snapshot artifact %s (signature %s)",
                layout.artifact_dir,
                signature,
            )
        barrier_after_artifact()
        return layout
    if not is_rank0:
        try:
            wait_for_rank0_artifact(layout, signature, timeout_s=timeout_s)
            return layout
        finally:
            barrier_after_artifact()

    try:
        with artifact_generation_lock(layout, timeout_s=timeout_s):
            existing = load_codegen_meta(layout.meta_path)
            if existing is not None and not should_regenerate(existing, signature):
                return layout
            logger.info(
                "codegen: generating snapshot artifact %s (signature %s)",
                layout.artifact_dir,
                signature,
            )
            meta = _build_meta_from_snapshot(snapshot, layout, signature)
            files = emit_bundle(
                layout,
                meta,
                snapshot=snapshot,
            )
            write_bundle_atomic(layout, files, meta)
        return layout
    except Exception as exc:
        publish_artifact_failure(layout, signature, str(exc) or type(exc).__name__)
        raise
    finally:
        barrier_after_artifact()
        clear_artifact_failure(layout, signature)


def _build_meta_from_snapshot(
    snapshot: CodegenSnapshot,
    layout: ArtifactLayout,
    signature: str,
) -> CodegenMeta:
    """Map immutable snapshot fields into the existing emitter metadata."""
    payload = snapshot.to_dict()
    source = dict(payload["source"])
    status = snapshot.status
    frozen_plan = dict(payload["plan"] or {})
    application = dict(payload["plan_application"] or {})
    module_overrides = list(replacement_records_from_snapshot(snapshot))
    covered = dict(COVERED_DEFAULTS)
    is_derived = status is CodegenStatus.DERIVED
    covered["sharding_plan"] = is_derived
    # Bundles are audit/export artifacts. They describe source rewrites but do
    # not take model-construction or replacement ownership from native training.
    covered["module_overrides"] = False
    covered["model_class"] = False
    emission_patches = _snapshot_emission_patches(payload) if is_derived else {}
    return CodegenMeta(
        codegen_version=CODEGEN_VERSION,
        signature=signature,
        yaml_path=layout.yaml_path,
        yaml_sha256=sha256_text(canonical_snapshot_payload(snapshot)),
        source=source,
        parallel_dims=dict(frozen_plan.get("parallel_dims") or {}),
        model_name=(
            _model_type_to_modeling_token(artifact_model_type(str(source["model_type"])))
            if source.get("model_type")
            else source.get("architecture")
        ),
        model_class=source.get("architecture"),
        injections=list(frozen_plan.get("injections") or ()),
        param_plan=dict(frozen_plan.get("param_plan") or {}),
        frozen_sharded_params=list(frozen_plan.get("frozen_sharded_params") or ()),
        tied_pairs=list(frozen_plan.get("tied_pairs") or ()),
        special_handlers=dict(frozen_plan.get("special_handlers") or {}),
        boundary_classes=dict(frozen_plan.get("boundary_classes") or {}),
        boundary_types=dict(frozen_plan.get("boundary_types") or {}),
        nested_components=list(frozen_plan.get("nested_components") or ()),
        mesh_dim_names=list(frozen_plan.get("mesh_dim_names") or ()) or None,
        module_overrides=(module_overrides if is_derived else []),
        emission_patches=emission_patches,
        covered=covered,
        not_covered=(
            list(NOT_COVERED_DEFAULT)
            if is_derived
            else [*NOT_COVERED_DEFAULT, "sharding_plan", "module_overrides", "model_class"]
        ),
        entrypoints={},
        snapshot_schema_version=payload["schema_version"],
        snapshot_status=payload["status"],
        snapshot_reason=payload.get("reason"),
        snapshot_failure=payload.get("failure"),
        snapshot_replacements=dict(payload["replacements"]),
        plan_application=application,
    )


def _snapshot_emission_patches(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate resolved native observations into the legacy emitter DTO."""
    applications = payload["replacements"]["applications"]
    replacements = [
        {
            "module_fqn": item["primary_fqn"],
            "source_type": item["source_type"],
            "replacement_type": item["replacement_type"],
        }
        for item in applications
    ]
    replacement_types = {
        module_fqn: item["replacement_type"]
        for item in applications
        for module_fqn in item.get("module_fqns", (item["primary_fqn"],))
    }
    source = payload["source"]
    plan = payload.get("plan") or {}
    application = payload.get("plan_application") or {}
    injections = application.get("resolved_injections") or plan.get("injections") or ()
    boundary_classes = plan.get("boundary_classes") or {}
    boundary_types = plan.get("boundary_types") or {}
    param_plan = plan.get("param_plan") or {}
    local_computes = []
    inner_wrappers = []
    for injection in injections:
        module_fqn = injection.get("match")
        class_name = boundary_classes.get(module_fqn)
        module_name = source.get("module_name")
        declared_module_type = (
            f"{module_name}:{class_name}" if module_name and class_name else None
        )
        source_module_type = _source_type_identity(source, class_name)
        observed_boundary_type = boundary_types.get(module_fqn)
        if (
            source_module_type is None
            and isinstance(observed_boundary_type, str)
            and not observed_boundary_type.startswith("hyper_parallel.")
            and not observed_boundary_type.startswith("torch.")
        ):
            source_module_type = observed_boundary_type
        bound_forward = _bound_forward_identity(source, module_fqn)
        local_compute = injection.get("local_compute_fn")
        if local_compute is not None:
            if declared_module_type is None:
                raise ValueError(
                    "snapshot local_compute_fn requires a source module and "
                    f"boundary class for {module_fqn!r}"
                )
            if source_module_type is not None or bound_forward is not None:
                module_type = (
                    _imported_type_identity(source, class_name)
                    if bound_forward is not None
                    else source_module_type
                ) or (
                    source_module_type
                    or declared_module_type
                )
                local_computes.append({
                    "module_fqn": module_fqn,
                    "module_type": module_type,
                    "factory": _snapshot_target_path(local_compute),
                    "region_dispatch": (param_plan.get(module_fqn) or {}).get(
                        "region_dispatch"
                    ),
                    "original_forward": (
                        bound_forward or f"{source_module_type}.forward"
                    ),
                })
        inner_wrapper = injection.get("inner_wrapper")
        if inner_wrapper is not None:
            target_type = (
                replacement_types.get(module_fqn)
                or source_module_type
                or declared_module_type
            )
            if target_type is None:
                raise ValueError(
                    "snapshot inner_wrapper requires an observed replacement or "
                    f"boundary class for {module_fqn!r}"
                )
            inner_wrappers.append({
                "module_fqn": module_fqn,
                "target_fqn": injection.get("inner_target") or "self",
                "wrapper": _snapshot_target_path(inner_wrapper),
                "target_type": target_type,
            })
    return {
        "module_replacements": replacements,
        "local_computes": local_computes,
        "inner_wrappers": inner_wrappers,
    }


def _source_class_names(source: dict[str, Any]) -> set[str]:
    """Return classes defined by the snapshot's modeling source file."""
    file_path = source.get("file_path")
    if not file_path:
        return set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def _source_type_identity(source: dict[str, Any], class_name: Optional[str]) -> Optional[str]:
    """Resolve a class to the remote source unit that actually defines it."""
    if not class_name:
        return None
    module_name = source.get("module_name")
    if not isinstance(module_name, str) or not module_name:
        return None
    if class_name in _source_class_names(source):
        return f"{module_name}:{class_name}"

    package, separator, _ = module_name.rpartition(".")
    candidates = []
    for path in resolve_remote_siblings(source).values():
        sibling_source = {"file_path": path}
        if class_name not in _source_class_names(sibling_source):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        sibling_module = f"{package}.{stem}" if separator else stem
        candidates.append(f"{sibling_module}:{class_name}")
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is defined by multiple bundled source units: "
            f"{sorted(candidates)}"
        )
    return candidates[0] if candidates else None


def _bound_forward_identity(
    source: dict[str, Any], module_fqn: Optional[str]
) -> Optional[str]:
    """Resolve a model-owned function installed through ``MethodType``.

    Custom models can import a boundary class but replace an instance forward
    in their constructor.  The installed module-level function, rather than the
    imported class body, is the source unit codegen must rewrite.
    """
    file_path = source.get("file_path")
    module_name = source.get("module_name")
    if not file_path or not module_name or not module_fqn:
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return None

    boundary_name = module_fqn.rsplit(".", 1)[-1]
    candidates = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        call = node.value
        if (
            not isinstance(target, ast.Attribute)
            or target.attr != "forward"
            or not isinstance(target.value, ast.Attribute)
            or target.value.attr != boundary_name
            or not isinstance(call, ast.Call)
            or _ast_name(call.func) != "MethodType"
            or not call.args
            or not isinstance(call.args[0], ast.Name)
        ):
            continue
        candidates.add(call.args[0].id)
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot boundary {module_fqn!r} has multiple MethodType forwards: "
            f"{sorted(candidates)}"
        )
    return f"{module_name}:{next(iter(candidates))}" if candidates else None


def _imported_type_identity(
    source: dict[str, Any], class_name: Optional[str]
) -> Optional[str]:
    """Resolve an imported boundary class from modules referenced by the source."""
    file_path = source.get("file_path")
    if not file_path or not class_name:
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return None

    explicit_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module
        and any(
            alias.name == class_name and (alias.asname or alias.name) == class_name
            for alias in node.names
        )
    }
    if len(explicit_modules) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is imported from multiple modules: "
            f"{sorted(explicit_modules)}"
        )
    if explicit_modules:
        return f"{next(iter(explicit_modules))}:{class_name}"

    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    candidates = []
    for module_name in sorted(modules):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, AttributeError):
            continue
        if hasattr(module, class_name):
            candidates.append(f"{module_name}:{class_name}")
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is exported by multiple imported modules: "
            f"{candidates}"
        )
    return candidates[0] if candidates else None


def _ast_name(node: ast.AST) -> Optional[str]:
    """Return the terminal name of a callable expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _snapshot_target_path(target: Any) -> str:
    """Return an importable ``module:qualname`` from a frozen Target value."""
    if isinstance(target, dict):
        target = target.get("_target_") or target.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError(f"snapshot injection target is not importable: {target!r}")
    if ":" in target:
        return target
    module_name, separator, name = target.rpartition(".")
    if not separator or not module_name or not name:
        raise ValueError(f"snapshot injection target is not importable: {target!r}")
    return f"{module_name}:{name}"


def preflight_integrity_check(
    config: Any,
    yaml_path: str | None = None,
    *,
    artifact_dir: str | None = None,
    hf_config: Any = None,
) -> None:
    """Validate the existing bundle before training starts; fail fast on drift.

    Composes the check-layer per-zone verifiers: snapshot signature, output
    hashes, source syntax, meta schema, boundary-form structure, and skipped
    override warnings. The frozen ``param_plan`` is checked for internal
    consistency without importing generated code.
    """
    from hyper_parallel.codegen.check.preflight import (
        verify_boundary_forms,
        verify_generated_source,
        verify_meta_required_fields,
        verify_output_hashes,
        verify_param_plan,
        verify_snapshot_contract,
        warn_skipped_overrides,
    )

    if hf_config is None:
        hf_config = _resolve_hf_config_for_layout(config)
    layout = _resolve_layout(config, yaml_path, artifact_dir, hf_config=hf_config)
    meta = load_codegen_meta(layout.meta_path)
    if meta is None:
        raise FileNotFoundError(
            f"codegen preflight: missing meta at {layout.meta_path}; run generate first"
        )
    verify_meta_required_fields(meta)
    verify_snapshot_contract(meta)
    if meta.snapshot_schema_version is None:
        raise RuntimeError(
            "codegen preflight: legacy artifacts are no longer supported; "
            "regenerate with 'python -m hyper_parallel.codegen generate'"
        )
    validate_snapshot_signature(meta.signature)
    verify_output_hashes(meta, layout)
    if meta.snapshot_status == CodegenStatus.NOT_REQUIRED.value:
        return
    verify_generated_source(layout)
    verify_boundary_forms(meta, layout)
    verify_param_plan(meta, model=None)
    warn_skipped_overrides(meta)


def should_regenerate(meta: CodegenMeta, signature: str) -> bool:
    """Whether the stored meta no longer matches the current signature."""
    return (
        meta.signature != signature
        or getattr(meta, "codegen_version", None) != CODEGEN_VERSION
    )


def wait_for_rank0_artifact(
    layout: ArtifactLayout,
    current_signature: str,
    *,
    timeout_s: float = 600.0,
) -> None:
    """Non-rank0: poll until rank0 has produced a fully-written artifact.

    ``current_signature`` is this rank's own computed signature.  The wait is
    signature-aware: a parseable meta is only accepted when its signature
    matches, so a stale bundle left on disk from a previous regeneration is
    never mistaken for "rank0 is done" — the poll continues until the new
    bundle's meta (written last inside the staging dir, then swapped in) lands.
    The meta file is written last, so a matching signature also means the
    modeling/diff/init files are already in place.
    """
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failure_path = artifact_failure_path(layout, current_signature)
        if os.path.isfile(failure_path):
            with open(failure_path, "r", encoding="utf-8") as handle:
                reason = handle.read().strip()
            raise RuntimeError(
                "codegen: rank0 artifact generation failed"
                + (f": {reason}" if reason else "")
            )
        meta = load_codegen_meta(layout.meta_path)
        if meta is not None and not should_regenerate(meta, current_signature):
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"codegen: timed out waiting for rank0 artifact at {layout.artifact_dir}"
    )


def barrier_after_artifact() -> None:
    """Rejoin all ranks after the artifact is ready.

    Every rank must call it exactly once on the way out of
    ``ensure_codegen_artifact_from_snapshot``, on every path: rank0 after writing the
    bundle, each other rank right after ``wait_for_rank0_artifact`` returns,
    and every rank that takes the reuse path (the reuse decision is a per-rank
    disk read, so ranks can disagree while rank0 swaps in a fresh bundle — a
    rank that skips the barrier deadlocks its peers inside theirs).  It is the
    collective that lets all ranks leave the codegen entry together before
    training starts.  The half-written-bundle race it was written to guard
    against is already closed by ``wait_for_rank0_artifact`` (signature aware,
    meta written last); the barrier is purely a rendezvous point, so it must
    be reached by every rank.

    Uses ``torch.distributed`` directly (the process group was initialized in
    ``trainer._setup`` before the codegen manager runs); a no-op without a
    live process group.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def validate_snapshot_signature(signature: str) -> None:
    """Fail every rank together when snapshot artifact identities diverge.

    File polling cannot coordinate rank-local signatures: each non-owner would
    otherwise wait for an artifact rank 0 can never publish under its identity.
    The collective runs only for snapshot sidecar generation and completes
    before any rank enters artifact reuse, generation, or polling.

    Args:
        signature: This rank's canonical snapshot signature.

    Raises:
        RuntimeError: If initialized ranks computed different signatures.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    signatures: list[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    if any(item != signatures[0] for item in signatures[1:]):
        raise RuntimeError(
            "codegen: snapshot signature mismatch across ranks: "
            + ", ".join(
                f"rank{rank}={item}" for rank, item in enumerate(signatures)
            )
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_rank0() -> bool:
    """True when this process is global rank 0 (or is a single process).

    Training config is parsed before the process group is initialized, so a
    torchrun worker must use its ``RANK`` environment value. Other callers use
    the initialized process group when available and otherwise act as rank 0.
    """
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError as exc:
            raise ValueError(f"codegen: invalid RANK value {rank!r}") from exc
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _model_name(config: Any, hf_config: Any = None) -> Optional[str]:
    """Resolve the model name from an HF config or the YAML model target."""
    model_type = getattr(hf_config, "model_type", None)
    if model_type:
        return _model_type_to_modeling_token(str(model_type))
    target = getattr(config, "model", None)
    path = getattr(target, "_target_", None)
    if isinstance(path, str):
        name = path.rsplit(".", 1)[-1]
        if name.startswith(("AutoModel", "Pretrained", "from_")):
            return None
        return name
    return getattr(config, "model_name", None)


def _model_type(config: Any, hf_config: Any = None) -> Optional[str]:
    """The HF ``model_type`` of the target model, for the model-anchored layout.

    Shares the best-effort config lookup with :func:`_model_name` (including
    the ``from_config`` ``config_path`` spelling), so the CLI generation and
    the trainer anchor the bundle to the same family directory. The raw HF
    spelling is normalized through the registry family aliases (K2.6 reports
    ``kimi_k25``; the family directory is ``kimi_k26``).
    """
    if hf_config is None:
        hf_config = _resolve_hf_config_for_layout(config)
    model_type = getattr(hf_config, "model_type", None)
    if model_type:
        return family_dir_for_model_type(str(model_type))
    return _model_family_from_target(config)


def _model_family_from_target(config: Any) -> Optional[str]:
    """Infer a registered model family from a custom builder target path."""
    model = getattr(config, "model", None)
    target_path = getattr(model, "_target_path", None)
    target = getattr(model, "_target_", None)
    if not isinstance(target_path, str) and isinstance(target, str):
        target_path = target
    if not isinstance(target_path, str) and callable(target):
        module_name = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if module_name and qualname:
            target_path = f"{module_name}.{qualname}"
    if not isinstance(target_path, str):
        return None
    prefix = "hyper_parallel.models."
    if not target_path.startswith(prefix):
        return None
    family, separator, _ = target_path[len(prefix):].partition(".")
    if not separator or not family:
        return None
    return family_dir_for_model_type(family)


def _model_type_to_modeling_token(model_type: str) -> str:
    """Convert HF ``model_type`` into the generated modeling file token."""
    parts = [part for part in model_type.replace("-", "_").split("_") if part]
    if not parts:
        return "model"
    return "_".join(part[:1].upper() + part[1:] for part in parts)


def _model_class(config: Any, hf_config: Any = None) -> Optional[str]:
    """The entry model class name the override rewrite targets in ``__init__``.

    The HF config's ``architectures`` list supplies the emitted entry class
    name (for example ``AnthropicV3ForCausalLM``).

    When neither ``hf_config`` nor ``config`` exposes ``architectures``, the
    layout resolver loads the model config lazily.
    """
    for candidate in (
        getattr(hf_config, "architectures", None),
        getattr(config, "architectures", None),
    ):
        if isinstance(candidate, (list, tuple)) and candidate:
            return str(candidate[0])

    model = getattr(config, "model", None)
    path = getattr(model, "pretrained_model_name_or_path", None)
    if path is None:
        path = getattr(model, "model_name_or_path", None)
    if path is None:
        return None
    try:
        from hyper_parallel.models._transformers.config_resolver import get_hf_config
    except ImportError:
        return None
    try:
        arch = getattr(get_hf_config(path), "architectures", None)
    except Exception:
        return None
    if isinstance(arch, (list, tuple)) and arch:
        return str(arch[0])
    return None


def _resolve_layout(
    config: Any,
    yaml_path: str | None,
    artifact_dir: str | None,
    *,
    hf_config: Any = None,
) -> ArtifactLayout:
    """Resolve the artifact layout for one codegen run.

    Priority: an explicit ``artifact_dir``, then the YAML's
    ``model.codegen_artifact_dir`` override, then the model-anchored default
    (``hyper_parallel/models/<model_type>/generated`` — one bundle per model
    type; a missing ``models/<model_type>/`` directory warns and is created
    on demand), then the YAML-anchored layout (``<yaml_dir>/generated``) for
    configs that carry no model identity, and finally ``cwd/generated``.
    """
    model_name = _model_name(config, hf_config)
    if artifact_dir is not None:
        return resolve_default_artifact_layout(artifact_dir, model_name=model_name)
    model = getattr(config, "model", None)
    configured_dir = getattr(model, "codegen_artifact_dir", None)
    if configured_dir:
        return resolve_default_artifact_layout(str(configured_dir), model_name=model_name)
    model_type = _model_type(config, hf_config)
    if model_type:
        source_yaml = yaml_path or getattr(config, "_yaml_path", None)
        # A custom builder may not expose an HF config until native build.
        # Reuse the published snapshot's filename identity instead of the
        # builder function name so check resolves exactly the generate layout.
        probe = resolve_model_artifact_layout(str(model_type), model_name=None)
        published = load_codegen_meta(probe.meta_path)
        if published is not None and published.model_name:
            model_name = published.model_name
        return resolve_model_artifact_layout(
            str(model_type),
            model_name=model_name,
            yaml_path=str(source_yaml) if source_yaml else None,
        )
    if yaml_path is not None:
        return resolve_artifact_layout(str(yaml_path), model_name=model_name)
    configured_yaml_path = getattr(config, "_yaml_path", None)
    if configured_yaml_path:
        return resolve_artifact_layout(str(configured_yaml_path), model_name=model_name)
    return resolve_default_artifact_layout(model_name=model_name)


def resolve_layout_for_config(
    config: Any,
    yaml_path: str | None = None,
    *,
    artifact_dir: str | None = None,
    hf_config: Any = None,
) -> ArtifactLayout:
    """Public layout resolution shared by the CLI and the drift check."""
    return _resolve_layout(config, yaml_path, artifact_dir, hf_config=hf_config)


def _resolve_hf_config_for_layout(config: Any) -> Any:
    """Best-effort HF config lookup used only for artifact naming/layout.

    Covers both model spellings: ``from_pretrained`` (``pretrained_model_name_or_path``)
    and ``from_config`` (``config_path``), so the CLI chain anchors bundles
    exactly like the trainer chain.
    """
    model = getattr(config, "model", None)
    if model is None:
        return None
    path = getattr(model, "pretrained_model_name_or_path", None)
    if path is None:
        path = getattr(model, "model_name_or_path", None)
    if path is None:
        path = getattr(model, "config_path", None)
    if path is None:
        return None
    kwargs = _hf_config_kwargs(model)
    try:
        from hyper_parallel.models._transformers.config_resolver import get_hf_config
    except ImportError:
        return None
    try:
        return get_hf_config(
            path,
            getattr(model, "attn_implementation", "sdpa"),
            getattr(model, "torch_dtype", "auto"),
            **kwargs,
        )
    except Exception:
        return None


def _hf_config_kwargs(model: Any) -> dict[str, Any]:
    """Collect model target kwargs that affect ``AutoConfig.from_pretrained``."""
    names = (
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "token",
        "trust_remote_code",
        "use_auth_token",
    )
    kwargs = {
        name: getattr(model, name)
        for name in names
        if getattr(model, name, None) is not None
    }
    overrides = getattr(model, "config_overrides", None)
    if isinstance(overrides, dict):
        kwargs.update(overrides)
    return kwargs


__all__ = [
    "CODEGEN_VERSION",
    "barrier_after_artifact",
    "ensure_codegen_artifact_from_snapshot",
    "preflight_integrity_check",
    "should_regenerate",
    "wait_for_rank0_artifact",
]
