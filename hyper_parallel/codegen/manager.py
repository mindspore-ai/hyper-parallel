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

import hashlib
from dataclasses import asdict
from fnmatch import fnmatchcase
import logging
import os
from typing import Any, Optional

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    clean_temp_artifacts,
    ensure_artifact_dir,
    resolve_default_artifact_layout,
    resolve_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.emit import emit_bundle
from hyper_parallel.codegen.hash import canonical_json, signature_from_spec
from hyper_parallel.codegen.inline.specs import get_inline_spec_bundle
from hyper_parallel.codegen.meta import (
    CodegenMeta,
    load_codegen_meta,
)

logger = logging.getLogger(__name__)

CODEGEN_VERSION = "0.1.1"


def ensure_codegen_artifact(
    config: Any,
    yaml_path: str | None = None,
    *,
    artifact_dir: str | None = None,
    emit_fn=emit_bundle,
    hf_config: Any = None,
) -> Optional[ArtifactLayout]:
    """Ensure an up-to-date artifact bundle.

    Returns the layout when codegen is active, ``None`` otherwise.  If the
    signature hits an existing bundle it is reused; otherwise ``emit_fn`` is
    called to produce the bundle. ``emit_fn`` defaults to ``emit_bundle``;
    pass ``emit_fn=None`` explicitly to create a minimal diagnostic bundle
    without invoking source emission.

    ``hf_config`` is the resolved HuggingFace config of the model; when given,
    it is folded into the signature and meta ``source`` (so a transformers
    modeling-file change regenerates).  When omitted, the source identity is
    resolved from the model id alone.
    """
    if not getattr(config, "codegen", False):
        return None

    if hf_config is None:
        hf_config = _resolve_hf_config_for_layout(config)
    layout = _resolve_layout(config, yaml_path, artifact_dir, hf_config=hf_config)
    ensure_artifact_dir(layout)
    is_rank0 = _is_rank0()
    if is_rank0:
        clean_temp_artifacts(layout)

    signature = _compute_signature(config, layout, hf_config=hf_config)
    existing = load_codegen_meta(layout.meta_path)

    if existing is not None and not should_regenerate(existing, signature):
        if is_rank0:
            logger.info(
                "codegen: reuse artifact %s (signature %s)",
                layout.artifact_dir,
                signature,
            )
        return layout

    if not is_rank0:
        wait_for_rank0_artifact(layout, signature)
        barrier_after_artifact()
        return layout

    # rank0 generates.
    logger.info(
        "codegen: generating artifact %s (signature %s)", layout.artifact_dir, signature
    )
    meta = _build_meta(config, layout, signature, hf_config=hf_config)
    _fill_plan_fields(meta, config, layout)
    _record_inline_declarations(meta)
    if emit_fn is not None:
        files = emit_fn(layout, meta)
    else:
        files = _write_placeholder_bundle(layout, meta)
    write_bundle_atomic(layout, files, meta)
    barrier_after_artifact()
    return layout


def preflight_integrity_check(
    config: Any,
    yaml_path: str | None = None,
    *,
    artifact_dir: str | None = None,
    hf_config: Any = None,
) -> None:
    """Validate the existing bundle before training starts; fail fast on drift.

    Composes the check-layer per-zone verifiers: signature, output hashes,
    importability, meta schema, boundary-form structure, and a warning for
    skipped overrides.  The frozen ``param_plan`` is verified for internal
    consistency (an empty plan fails — generation always freezes a derived
    plan); full model-side coverage against the live parameter tree runs
    later in :func:`init_generated_model`, where a model handle exists.
    """
    if not getattr(config, "codegen", False):
        return
    from hyper_parallel.codegen.check.preflight import (
        verify_boundary_forms,
        verify_generated_import,
        verify_meta_required_fields,
        verify_output_hashes,
        verify_param_plan,
        verify_signature,
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
    signature = _compute_signature(config, layout, hf_config=hf_config)
    verify_signature(meta, signature)
    verify_output_hashes(meta, layout)
    verify_generated_import(layout)
    verify_boundary_forms(meta, layout)
    verify_param_plan(meta, model=None)
    warn_skipped_overrides(meta)


def should_regenerate(meta: CodegenMeta, signature: str) -> bool:
    """Whether the stored meta no longer matches the current signature."""
    return meta.signature != signature


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
        meta = load_codegen_meta(layout.meta_path)
        if meta is not None and meta.signature == current_signature:
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"codegen: timed out waiting for rank0 artifact at {layout.artifact_dir}"
    )


def barrier_after_artifact() -> None:
    """Rejoin all ranks after the artifact is ready.

    Rank0 calls it after writing the bundle; every other rank must call it too,
    right after ``wait_for_rank0_artifact`` returns.  It is the collective that
    lets all ranks leave the codegen entry together before training starts —
    calling it on rank0 only would deadlock rank0 (it waits for the peers, who
    have already returned).  The half-written-bundle race it was written to
    guard against is already closed by ``wait_for_rank0_artifact`` (signature
    aware, meta written last); the barrier is purely a rendezvous point, so it
    must be reached by every rank.

    Uses ``torch.distributed`` directly (the process group was initialized in
    ``trainer._setup`` before the codegen manager runs); a no-op without a
    live process group.
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
    except Exception:  # pragma: no cover - best-effort without a process group
        pass


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
    try:
        from hyper_parallel.trainer.runtime.distributed import (
            get_global_rank_safe,
        )

        return get_global_rank_safe() == 0
    except Exception:
        return True


def _model_name(config: Any, hf_config: Any = None) -> Optional[str]:
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


def _model_type_to_modeling_token(model_type: str) -> str:
    """Convert HF ``model_type`` into the generated modeling file token."""
    parts = [part for part in model_type.replace("-", "_").split("_") if part]
    if not parts:
        return "model"
    return "_".join(part[:1].upper() + part[1:] for part in parts)


def _model_class(config: Any, hf_config: Any = None) -> Optional[str]:
    """The entry model class name the override rewrite targets in ``__init__``.

    The HF config's ``architectures`` list is the lookup key the loader uses to
    pick the class from the generated file, so it is the architecture's class
    name (e.g. ``AnthropicV3ForCausalLM``).  Falls back to the loaded config's
    own attribute when ``hf_config`` was not passed in.

    ``ensure_codegen_artifact`` is reached from config preparation and the CLI,
    neither of which has a resolved ``hf_config`` in hand (the model resolves
    it only inside ``from_pretrained``, after generation).  So when
    neither ``hf_config`` nor ``config`` exposes ``architectures``, resolve the
    model path with the same rule ``spec/project`` uses and load the config —
    lazily, to keep this module importable without ``transformers``.
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
    """Resolve either YAML-anchored or model-construction artifact layout."""
    model_name = _model_name(config, hf_config)
    if artifact_dir is not None:
        return resolve_default_artifact_layout(artifact_dir, model_name=model_name)
    model = getattr(config, "model", None)
    configured_dir = getattr(model, "codegen_artifact_dir", None)
    if configured_dir:
        return resolve_default_artifact_layout(str(configured_dir), model_name=model_name)
    if yaml_path is not None:
        return resolve_artifact_layout(str(yaml_path), model_name=model_name)
    configured_yaml_path = getattr(config, "_yaml_path", None)
    if configured_yaml_path:
        return resolve_artifact_layout(str(configured_yaml_path), model_name=model_name)
    return resolve_default_artifact_layout(model_name=model_name)


def _resolve_hf_config_for_layout(config: Any) -> Any:
    """Best-effort HF config lookup used only for artifact naming/layout."""
    model = getattr(config, "model", None)
    if model is None:
        return None
    path = getattr(model, "pretrained_model_name_or_path", None)
    if path is None:
        path = getattr(model, "model_name_or_path", None)
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


def _compute_signature(
    config: Any,
    layout: ArtifactLayout,
    *,
    hf_config: Any = None,
) -> str:
    """Canonical signature over the projected spec, parallel dims, overrides,
    codegen switches, and the resolved modeling source.

    The full ``project_codegen_spec`` projection ensures the digest
    covers the same fields the plan/emit layers consume.  The YAML file bytes
    are deliberately excluded (see ``_project_spec``).
    """
    spec = _project_spec(config, layout, hf_config=hf_config)
    return signature_from_spec(spec)


def _project_spec(
    config: Any,
    layout: ArtifactLayout,
    *,
    hf_config: Any = None,
) -> dict:
    """CodegenSpec canonical dict used for the signature.

    The payload covers the projected CodegenSpec plus the resolved modeling
    source (installed transformers module or remote code), ``sha256`` included
    — a change to the HF modeling file changes the signature and regenerates
    the bundle.  The YAML file bytes are intentionally NOT hashed: the spec
    projection already captures every codegen-affecting field, so an unrelated
    YAML edit (comment, logging, optimizer tweak) must not invalidate a
    bundle.
    """
    from hyper_parallel.codegen.spec.project import project_codegen_spec
    from hyper_parallel.codegen.source.resolver import resolve_model_source

    payload = project_codegen_spec(config, layout=layout).to_dict()
    model_id = payload.get("source", {}).get("model_name_or_path") or ""
    is_gen = _gen_backend(config)

    source = resolve_model_source(config, model_id, hf_config=hf_config)
    if source is not None:
        # Replace the SourceSpec projection with the resolved file identity;
        # the sha256 here is what makes a modeling-file change regenerate.
        # ``transformers_version`` is a SourceSpec field the pure projection
        # leaves empty; record the actual env version whenever we resolved a
        # real file (an environment upgrade that changes the modeling file's
        # behavior must regenerate, not silently reuse).
        payload["source"] = source.to_dict()
        payload["source"]["transformers_version"] = _transformers_version()
    elif is_gen:
        # codegen=True resolves to the ``gen`` backend, whose whole point is
        # producing an artifact FROM the original modeling file.  A bundle
        # generated from a guessed/unresolved source would be silently wrong,
        # so fail fast instead of degrading — the user must fix the model id,
        # install transformers, or make the checkpoint reachable.
        raise RuntimeError(
            f"codegen: cannot resolve modeling source for {model_id!r} — no "
            "AutoConfig, no installed transformers module, no remote code.  "
            "The gen backend generates the artifact from the original "
            "modeling file; fix the model id, install transformers, or make "
            "the checkpoint reachable"
        )

    payload["codegen_implementation"] = {
        "version": CODEGEN_VERSION,
        "sha256": _codegen_implementation_digest(),
    }
    identity = payload.get("source", {}).get("architecture")
    bundle = get_inline_spec_bundle(identity) if identity else None
    # Adapter declarations live outside the codegen implementation tree;
    # their templates and mappings must also invalidate cached artifacts.
    payload["inline_codegen"] = {"specs": asdict(bundle) if bundle is not None else None}

    return payload


def _codegen_implementation_digest(package_dir: Optional[str] = None) -> str:
    """Hash the Python implementation that can affect generated bundles.

    The relative path is part of the digest so moving content between modules
    is treated as an implementation change. Non-Python files and bytecode
    caches are excluded because they do not define codegen semantics.
    """
    root = os.path.abspath(package_dir or os.path.dirname(__file__))
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with open(path, "r", encoding="utf-8", newline=None) as handle:
                digest.update(handle.read().encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _build_meta(
    config: Any,
    layout: ArtifactLayout,
    signature: str,
    *,
    hf_config: Any = None,
) -> CodegenMeta:
    """Build the meta record from the same projection the signature used.

    ``meta.source`` is taken verbatim from the signature payload's ``source``
    (resolved ``ResolvedSource`` dict, or the ``SourceSpec`` projection when
    nothing resolved) so the recorded identity always matches what the
    signature covers.
    """
    spec = _project_spec(config, layout, hf_config=hf_config)
    from hyper_parallel.codegen.meta import (
        COVERED_DEFAULTS,
        NOT_COVERED_DEFAULT,
    )

    return CodegenMeta(
        codegen_version=CODEGEN_VERSION,
        signature=signature,
        yaml_path=layout.yaml_path,
        yaml_sha256=_config_sha256(spec, layout),
        source=dict(spec.get("source") or {}),
        parallel_dims=spec.get("parallel_dims", {}),
        model_name=_model_name(config, hf_config),
        model_class=_model_class(config, hf_config),
        # COVERED_DEFAULTS is the starting copy; _fill_plan_fields sets
        # covered.sharding_plan True once the plan is frozen, so a real
        # artifact short-circuits the planner and the runtime shards it.  The
        # entrypoint name is the contract the runtime dispatches on; an inline
        # artifact that defines no such function is parallelized from meta by
        # ``runtime._parallelize_inline_from_meta``.
        covered=dict(COVERED_DEFAULTS),
        not_covered=list(NOT_COVERED_DEFAULT),
        entrypoints={"parallelize": "hyper_parallelize"},
    )


def _fill_plan_fields(meta: CodegenMeta, config: Any, layout: ArtifactLayout) -> None:
    """Derive + freeze the sharding plan into ``meta``'s plan slots.

    Generation-time pipeline: build the meta-device model, run the real
    ``ShardingPlanner`` over the offline mesh, and freeze the result into the
    ``param_plan`` / ``frozen_sharded_params`` / ``injections`` /
    ``tied_pairs`` / ``mesh_dim_names`` fields.  Derivation failures are NOT
    degraded: a None plan or an empty frozen plan would produce a plan-less
    artifact that still passes preflight (nothing verifies an empty
    ``param_plan``), so this stage raises and generation fails instead of
    silently continuing.
    """
    from hyper_parallel.codegen.plan.derive import (
        build_meta_model,
        build_plan_overrides,
        derive_sharding_plan,
    )
    from hyper_parallel.codegen.plan.freeze import freeze_plan
    from hyper_parallel.codegen.spec.project import project_codegen_spec

    spec = project_codegen_spec(config, layout=layout)
    model = build_meta_model(spec, config_overrides=_config_overrides(config))
    if model is None:
        raise RuntimeError(
            "codegen: cannot derive a sharding plan — the spec carries no "
            "model identity to build the meta model from; fix the model "
            "source in the YAML"
        )
    replace_meta = _apply_generation_replacements(meta, config, model, spec)
    plan_overrides = build_plan_overrides(config, spec)
    # S3: a plan override whose matched MoE boundary carries no explicit
    # ``local_compute_fn`` has its EP compute factory inferred from structure
    # (never by model name) and written into the spec so it lands in
    # ``meta.injections``. An explicit ``_target_`` stays the opt-out escape
    # hatch and is never overwritten.
    _fill_inferred_ep_targets(model, plan_overrides)
    plan = derive_sharding_plan(
        model,
        spec,
        plan_overrides=plan_overrides,
    )
    frozen = freeze_plan(plan, model)
    if not frozen.param_plan:
        raise RuntimeError(
            "codegen: plan derivation produced an empty param plan for "
            f"{spec.source.model_name_or_path!r} — the artifact would carry "
            "no sharding contract and pass preflight; check the generation "
            "logs for the planner failure and regenerate"
        )
    meta.param_plan = frozen.param_plan
    meta.frozen_sharded_params = frozen.frozen_sharded_params
    meta.injections = frozen.injections
    meta.tied_pairs = frozen.tied_pairs
    meta.special_handlers = frozen.special_handlers
    meta.boundary_classes = frozen.boundary_classes
    meta.mesh_dim_names = list(frozen.mesh_dim_names) or None

    meta.module_overrides = replace_meta
    # The inline pipeline lowers the matched replacements into the generated
    # file, so the artifact owns ``replace_module``.  Declare coverage so the
    # trainer's HF replacement step is skipped on the gen path — but only when
    # something was actually matched.  ``covered`` is always a
    # COVERED_DEFAULTS copy, so the key is present either way.
    meta.covered["module_overrides"] = bool(replace_meta)

    # A non-empty frozen plan is the artifact's full parallel logic; the
    # runtime shards from ``codegen_meta.json``.  ``covered`` is always a
    # COVERED_DEFAULTS copy, so the key is present either way.
    meta.covered["sharding_plan"] = True


def _record_inline_declarations(meta: CodegenMeta) -> None:
    """Copy the adapter's inline declarations into ``meta``.

    The render spec is the single declaration site for the facts the *runtime*
    also needs — today the class names whose forwards are fully inlined and
    take their parallel state externally.  Recording them in ``meta`` (the file
    the runtime loads at training time) is what keeps the runtime from carrying
    its own copy that drifts when a family adds a model.
    """
    identity = (meta.source or {}).get("architecture")
    bundle = get_inline_spec_bundle(identity) if identity else None
    if bundle is None:
        return
    meta.external_state_classes = sorted(bundle.external_state_classes)


def _apply_generation_replacements(
    meta: CodegenMeta,
    config: Any,
    model: Any,
    spec: Any,
) -> list[dict[str, Any]]:
    """Apply YAML ``replace_module`` actions to the generation meta model.

    Generated model ``__init__`` applies module replacements before runtime
    sharding.  Generation must mirror that order: compile the raw YAML entries
    against the original meta model to record serializable FQNs, apply the same
    replacements to the meta model, then derive the sharding plan from the
    replaced parameter tree.  Otherwise the frozen plan may reference
    pre-replacement parameters (for example ``q_proj``/``k_proj``/``v_proj``)
    that no longer exist after a generated attention module installs fused
    ``qkv_proj`` weights.
    """
    raw_entries = getattr(config, "plan_overrides", None) or []
    if not raw_entries:
        return []

    from hyper_parallel.models.replacement import (
        _apply_module_replacement_actions,
    )
    from hyper_parallel.codegen.emit.replacement import (
        compile_overrides_for_meta,
    )
    from hyper_parallel.trainer.config import (
        entries_to_module_replacements,
    )

    specs = entries_to_module_replacements(raw_entries)
    factory_paths = [
        _target_path(getattr(entry, "replace_module", None))
        for entry in raw_entries
        if getattr(entry, "replace_module", None) is not None
    ]
    records, skipped = compile_overrides_for_meta(
        model,
        specs,
        factory_paths=factory_paths,
    )
    replace_meta = list(records)
    for item in skipped:
        meta.skipped_overrides.append(item)
        logger.warning(
            "codegen: plan_overrides replace_module matched no module "
            "and was skipped: %r",
            item.get("match"),
        )
    if replace_meta:
        model, _ = _apply_module_replacement_actions(
            model,
            specs,
            context={
                "low_precision": None,
                "tp": spec.parallel_dims.tp_size > 1,
                "cp": spec.parallel_dims.cp_size > 1,
                "ep": spec.parallel_dims.ep_size > 1,
                "pp": spec.parallel_dims.pp_size > 1,
            },
            capture_checkpoint_metadata=False,
        )
    return replace_meta


def _config_overrides(config: Any) -> dict[str, Any]:
    """Return config-only kwargs passed to AutoConfig for generated models."""
    model = getattr(config, "model", None)
    overrides = getattr(model, "config_overrides", None) or {}
    if not isinstance(overrides, dict):
        raise TypeError(
            "codegen: model.config_overrides must be a mapping of "
            f"AutoConfig keyword arguments, got {type(overrides).__name__}"
        )
    return dict(overrides)


def _target_path(target: Any) -> Optional[str]:
    """Return the serialized target path for a YAML Target-like object."""
    if target is None or not hasattr(target, "to_dict"):
        return None
    data = target.to_dict()
    return data.get("_target_") if isinstance(data, dict) else None


#: Structure fingerprint -> model-family MoE EP compute factory.
#:
#: S3 maps a recognized ``MoeStructure`` (router / shared-expert branch /
#: expert-storage) to the factory that would have been wired through a YAML
#: ``local_compute_fn._target_``. This is deliberately per-family and named,
#: NOT keyed by model identity: the fingerprint is the *only* discriminator,
#: so a model with the same structure is covered without new registry work.
#: The table is transitional — S4+ replaces the factory path with a
#: self-contained archetype record; the path is today's meta transport.
_EP_FACTORY_BY_STRUCTURE = {
    ("topk_router_module", "none", "batched_parameters"):
        "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
        "qwen3moe_ep_compute_fn",
    ("topk_router_module", "additive", "batched_parameters"):
        "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn",
}


def _fill_inferred_ep_targets(model: Any, plan_overrides: dict[str, Any]) -> None:
    """S3: fill a missing ``local_compute_fn`` on MoE EP overrides.

    Every desugared plan override whose ``match`` resolves to a MoE boundary
    but which declares no explicit ``local_compute_fn`` gets its EP compute
    factory inferred from structure (``detect_moe_structure``) instead of a
    YAML ``_target_``. Entries that already carry a ``local_compute_fn`` are
    left untouched — the explicit ``_target_`` remains the escape hatch for
    structures detection cannot cover. Ordinary (non-MoE) overrides are
    untouched. This is a no-op on a struct construct; the ``plan_overrides``
    dict's spec values are mutated in place so ``freeze_injections`` records
    the inferred factory in ``meta.injections``.
    """
    for match, spec in plan_overrides.items():
        if getattr(spec, "local_compute_fn", None) is not None:
            continue
        inferred = _infer_ep_compute_for_match(model, match)
        if inferred is not None:
            spec.local_compute_fn = inferred


def _infer_ep_compute_for_match(
    model: Any, match: str
) -> Optional["Any"]:
    """Return the EP compute ``_target_`` covering every MoE block ``match`` hits.

    Resolves ``match`` (an FQN or FQN glob, ``fnmatchcase``) against the meta
    model's module tree. Returns a ``Target`` for the structure-inferred EP
    factory when at least one matched module is a MoE boundary; ``None`` when
    ``match`` touches no MoE boundary (an ordinary sharding override needs no
    compute injection). An unrecognized MoE structure raises
    ``UnsupportedModuleStructure`` so the artifact never silently picks a wrong
    strategy — the escape-hatch text points the user at writing an explicit
    ``_target_``.
    """
    candidate = None
    for fqn, module in model.named_modules():
        if not fnmatchcase(fqn, match):
            continue
        if not _looks_like_moe_boundary(module):
            continue
        target = _import_ep_target_for(module, match)
        if candidate is None:
            candidate = target
        elif candidate != target:
            from hyper_parallel.distributed.expert_parallel.structure import (  # pylint: disable=C0415
                UnsupportedModuleStructure,
            )

            raise UnsupportedModuleStructure(
                f"{match!r} matched multiple MoE boundaries with different "
                "EP compute factories; declare an explicit "
                "local_compute_fn._target_ in the YAML"
            )
    return candidate


def _looks_like_moe_boundary(module: Any) -> bool:
    """Cheap gate: does the module carry an expert plus a router gate?"""
    return hasattr(module, "experts") and (
        hasattr(module, "gate") or hasattr(module, "router")
    )


def _import_ep_target_for(module: Any, match: str) -> "Any":
    """Import the delayed target bound to the structure-inferred factory."""
    from hyper_parallel.distributed.expert_parallel.structure import (  # pylint: disable=C0415
        UnsupportedModuleStructure,
        detect_moe_structure,
    )

    structure = detect_moe_structure(module)
    fingerprint = (
        structure.router,
        structure.shared_experts,
        structure.expert_storage,
    )
    factory_path = _EP_FACTORY_BY_STRUCTURE.get(fingerprint)
    if factory_path is None:
        raise UnsupportedModuleStructure(
            f"{match!r} matched {type(module).__name__} with structure "
            f"{fingerprint} that has no EP compute factory. Declare an "
            "explicit local_compute_fn._target_ to opt out of structure "
            "detection."
        )
    return _build_ep_target(factory_path)


def _build_ep_target(factory_path: str) -> "Any":
    """Build a delayed ``Target`` for the EP factory at ``factory_path``.

    ``local_compute_fn`` is a *factory* Target — the framework calls
    ``build()`` (which resolves the import) only at apply time, never during
    generation. So the callable here resolves lazily; generation serializes
    only the ``_target_`` path into meta, and the runtime re-resolves it from
    that path. Deferring the import keeps codegen free of a hard dependency on
    any concrete model package.
    """
    from hyper_parallel.trainer.config.target import Target  # pylint: disable=C0415

    def _factory(**kwargs: Any) -> Any:
        import importlib  # pylint: disable=C0415

        module_path, _, attr = factory_path.rpartition(".")
        fn = getattr(importlib.import_module(module_path), attr)
        return fn(**kwargs)

    return Target(_factory, target_path=factory_path)


def _gen_backend(config: Any) -> bool:
    """Whether this config resolves to the ``gen`` backend.

    ``codegen=True`` implies ``gen`` (see modeling_backend.resolve: precedence
    is force_hf -> explicit modeling_backend -> codegen).  So this is True for
    any codegen run that has not been explicitly forced/requested to a
    different backend — and those are exactly the runs that must fail fast
    when the modeling source cannot be resolved.
    """
    if not getattr(config, "codegen", False):
        return False
    if getattr(config, "modeling_backend", None) is not None:
        return getattr(config, "modeling_backend", None) == "gen"
    return True


def _transformers_version() -> str:
    """Installed transformers version, or ``""`` when not importable.

    Folded into the signature/meta so an environment upgrade that changes the
    modeling file's behavior regenerates instead of silently reusing.
    """
    try:
        import transformers

        return getattr(transformers, "__version__", "")
    except ImportError:
        return ""


def _write_placeholder_bundle(
    layout: ArtifactLayout, meta: CodegenMeta
) -> dict[str, str]:
    """Render a minimal importable modeling stub for an explicit fallback.

    Returns the ``filename -> content`` dict (no disk write); the caller stages
    and atomically swaps the bundle. The metadata marks this diagnostic bundle
    as a placeholder so preflight can report the explicit fallback.
    """
    stub = (
        "# Placeholder generated modeling file.\n"
        "# codegen_meta.json marks this fallback so preflight can detect it.\n"
        "from transformers import AutoModelForCausalLM  # noqa: F401\n"
    )
    meta.skipped_overrides.append(
        {"reason": "placeholder", "detail": "source emission explicitly disabled"}
    )
    return {
        os.path.basename(layout.modeling_path): stub,
        os.path.basename(layout.diff_path): "",
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(spec: dict, layout: ArtifactLayout) -> str:
    """Return YAML file hash when available, otherwise hash the projected spec."""
    if layout.yaml_path:
        return _sha256(layout.yaml_path)
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


__all__ = [
    "CODEGEN_VERSION",
    "barrier_after_artifact",
    "ensure_codegen_artifact",
    "preflight_integrity_check",
    "should_regenerate",
    "wait_for_rank0_artifact",
]
