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
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    clean_temp_artifacts,
    ensure_artifact_dir,
    resolve_default_artifact_layout,
    resolve_model_artifact_layout,
    resolve_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.emit import emit_bundle
from hyper_parallel.codegen.hash import canonical_json, sha256_file, signature_from_spec
from hyper_parallel.codegen.inline.framework_spec import declaration_summary
from hyper_parallel.codegen.meta import (
    CodegenMeta,
    load_codegen_meta,
)
from hyper_parallel.distributed._builder.applier import _apply_phase_c
from hyper_parallel.distributed.recipe_spec import _normalize_out_fields

logger = logging.getLogger(__name__)

CODEGEN_VERSION = "0.1.1"


@dataclass(frozen=True)
class CodegenBuildInput:
    """Builder-owned model, plan, and patch decisions used by emitters."""

    source_model: Any
    transformed_model: Any
    sharding_plan: Any
    patch_plan: Any
    module_overrides: tuple[dict[str, Any], ...]


def ensure_codegen_artifact(
    config: Any,
    yaml_path: str | None = None,
    *,
    artifact_dir: str | None = None,
    hf_config: Any = None,
) -> Optional[ArtifactLayout]:
    """Ensure an up-to-date artifact bundle.

    Returns the layout when codegen is active, ``None`` otherwise.  If the
    signature hits an existing bundle it is reused; otherwise ``emit_bundle``
    is called to produce the bundle.

    ``hf_config`` is the resolved HuggingFace config of the model; when given,
    it is folded into the signature and meta ``source`` (so a transformers
    modeling-file change regenerates).  When omitted, the source identity is
    resolved from the model id alone.

    Rank rendezvous: every path out of this function — reuse, wait, and
    generate — must cross exactly one ``barrier_after_artifact`` call.  The
    reuse decision is a per-rank disk read, so ranks can disagree while rank0
    swaps in a fresh bundle (a late-starting rank reads the new meta and takes
    the reuse path while its peers hold the stale one and take the wait path);
    a rank that skips the barrier on reuse deadlocks the whole run.
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
        # Ranks can disagree on reuse while rank0 swaps the bundle in; this
        # path must cross the barrier too, or the peers deadlock inside theirs.
        barrier_after_artifact()
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
    patch_plan = _fill_plan_fields(meta, config, layout, hf_config=hf_config)
    files = emit_bundle(layout, meta, patch_plan=patch_plan)
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

    Every rank must call it exactly once on the way out of
    ``ensure_codegen_artifact``, on every path: rank0 after writing the
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
    if not model_type:
        return None
    from hyper_parallel.models.registry import family_dir_for_model_type

    return family_dir_for_model_type(str(model_type))


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
        resolved = source.to_dict()
        # ``SourceSpec.to_dict`` keeps only architecture / model_type / sha256 —
        # neither the checkpoint identity nor its structural config.  Without
        # them, two checkpoints that share a modeling file but differ in shape
        # (e.g. the same architecture at different layer counts) would produce
        # the same signature and silently reuse a frozen per-layer plan written
        # for the other shape.  Fold the model id and a structural-config digest
        # back in so a shape change invalidates the bundle.
        resolved["model_id"] = model_id
        if hf_config is not None:
            resolved["structure"] = _structure_summary(hf_config)
        payload["source"] = resolved
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
    # The declarations a target resolves to are derived from structure (the
    # family's runtime factories + the framework archetype table), which live
    # outside the codegen implementation tree; summarizing them here folds
    # their content into the signature so a change to a factory's component or
    # a strategy's body still regenerates an artifact.
    payload["inline_declarations"] = {
        "specs": declaration_summary(identity, payload.get("overrides")),
    }

    return payload


_STRUCTURE_FIELDS = (
    # Architectural (shape-defining) config fields.  A change in any of these
    # — most importantly the layer/experts counts a per-layer sharding plan is
    # expanded against — must invalidate a generated bundle even when the
    # underlying modeling file is unchanged.
    "num_hidden_layers",
    "num_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "first_k_dense_replace",
    "moe_intermediate_size",
    "n_group",
    "topk_group",
    "num_experts_per_tok",
    "top_k",
)


def _structure_summary(hf_config: Any) -> dict[str, Any]:
    """A stable digest of the config fields that define the model's structure."""
    return {
        name: getattr(hf_config, name)
        for name in _STRUCTURE_FIELDS
        if getattr(hf_config, name, None) is not None
    }


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
        entrypoints={"parallelize": "hp_parallelize"},
    )


def _fill_plan_fields(
    meta: CodegenMeta,
    config: Any,
    layout: ArtifactLayout,
    *,
    hf_config: Any = None,
) -> Any:
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
    from hyper_parallel.codegen.plan.derive import build_meta_model
    from hyper_parallel.codegen.plan.freeze import freeze_plan
    from hyper_parallel.codegen.patch_plan import (
        PatchEmissionPlan,
        serialize_patch_emission_plan,
    )
    from hyper_parallel.codegen.source.resolver import _trust_remote_code
    from hyper_parallel.codegen.spec.project import project_codegen_spec

    spec = project_codegen_spec(config, layout=layout)
    model = build_meta_model(
        spec,
        config_overrides=_config_overrides(config),
        hf_config=hf_config,
        trust_remote_code=_trust_remote_code(config),
    )
    if model is None:
        raise RuntimeError(
            "codegen: cannot derive a sharding plan — the spec carries no "
            "model identity to build the meta model from; fix the model "
            "source in the YAML"
        )
    build_input = _prepare_codegen_build(model, config, spec)
    frozen = freeze_plan(build_input.sharding_plan, build_input.transformed_model)
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

    meta.emission_patches = serialize_patch_emission_plan(build_input.patch_plan)
    from hyper_parallel.models.registry import (  # pylint: disable=import-outside-toplevel
        _resolve_custom_model_cls,
    )

    runtime_owned_custom_model = _resolve_custom_model_cls(meta.model_class or "") is not None
    if not runtime_owned_custom_model:
        # HF-only models still rely on inline source emission for state bound
        # by their replacement/forward patches (for example Qwen3-MoE's
        # ``ep_enable``). Preserve that established generated-source contract.
        meta.module_overrides = list(build_input.module_overrides)
        meta.covered["module_overrides"] = bool(build_input.module_overrides)
        meta.covered["model_class"] = True
        meta.covered["sharding_plan"] = True
        return build_input.patch_plan

    # ``emission_patches`` retains the observed replacements for drift audit.
    # ``module_overrides`` is emitter ownership and must stay empty when the
    # live Trainer owns replacement application.
    meta.module_overrides = []
    # Replacements on the temporary model are planning observations, not
    # generated-source ownership. Trainer must apply them to the live model so
    # native and codegen modes share the replacement/FSDP/optimizer lifecycle.
    meta.covered["module_overrides"] = False
    # The emitted file is an audit copy when it owns no source patches.  For
    # registered custom models the live Trainer must instantiate the native
    # class as well as apply the native replacements; otherwise Python sees a
    # second, synthetic class hierarchy even though the source text is equal.
    meta.covered["model_class"] = False

    # A non-empty frozen plan is the artifact's full parallel logic; the
    # runtime shards from ``codegen_meta.json``.  ``covered`` is always a
    # COVERED_DEFAULTS copy, so the key is present either way.
    meta.covered["sharding_plan"] = True
    # Phase C is also runtime-owned.  Emitting its observed forward patches
    # creates a second implementation that the runtime later tries to undo
    # before applying the native boundary wrappers.  That extra bind/restore
    # cycle changes DeepSeek-V4.1's backward graph even when its first forward
    # losses match.  Keep the observations only in metadata for drift audit;
    # the generated modeling source must remain structurally native here.
    return PatchEmissionPlan()


def _prepare_codegen_build(model: Any, config: Any, spec: Any) -> CodegenBuildInput:
    """Run the shared pre-runtime builder against an offline topology."""
    # Lazy imports preserve the manager's import-light CLI boundary.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.codegen.plan.derive import build_plan_overrides
    from hyper_parallel.codegen.plan.offline_mesh import build_offline_mesh
    from hyper_parallel.codegen.plan.offline_mesh import OfflineMesh
    from hyper_parallel.codegen.patch_plan import (
        build_patch_emission_plan,
        snapshot_modules,
    )
    from hyper_parallel.distributed._builder.planner import ShardingPlanner
    from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
    from hyper_parallel.models._transformers.model_builder import (
        _apply_materialization_adapter,
        _maybe_apply_module_replacements,
        get_model_conversion_mapping,
    )
    from hyper_parallel.trainer.config import entries_to_module_replacements
    # pylint: enable=import-outside-toplevel

    dims = spec.parallel_dims
    mesh = MeshContext(
        device_mesh=build_offline_mesh(dims),
        dp_size=dims.dp_size,
        tp_size=dims.tp_size,
        cp_size=dims.cp_size,
        pp_size=dims.pp_size,
        ep_size=dims.ep_size,
        sequence_parallel=dims.sequence_parallel,
        loss_parallel=dims.loss_parallel,
    )
    replacement_specs = entries_to_module_replacements(
        getattr(config, "plan_overrides", None) or []
    )
    setup = DistributedSetup(
        mesh_context=mesh,
        plan_overrides=build_plan_overrides(config, spec),
        module_replacements=replacement_specs,
    )
    planner = ShardingPlanner(plan_overrides=setup.plan_overrides)
    source_model = model
    before_builder = snapshot_modules(model)
    model, _ = _maybe_apply_module_replacements(
        model,
        None,
        setup,
        None,
        False,
        get_model_conversion_mapping(model),
    )
    _apply_materialization_adapter(model)
    sharding_plan = planner.plan(
        model,
        mesh.device_mesh,
        tp_size=mesh.tp_size,
        cp_size=mesh.cp_size,
        ep_size=mesh.ep_size,
        sequence_parallel=mesh.sequence_parallel,
        loss_parallel=mesh.loss_parallel,
    )
    before_phase_c = snapshot_modules(model)
    expert_mesh = (
        OfflineMesh(("ep",), (dims.ep_size,))
        if dims.ep_size > 1
        else None
    )
    _apply_recorded_phase_c(
        model, sharding_plan, mesh.device_mesh, expert_mesh=expert_mesh,
    )
    after_phase_c = snapshot_modules(model)
    patch_plan = build_patch_emission_plan(
        model,
        sharding_plan,
        replacement_specs,
        before_builder,
        before_phase_c,
        after_phase_c,
    )
    module_overrides = _replacement_records_for_meta(
        config,
        replacement_specs,
        patch_plan.module_replacements,
    )
    return CodegenBuildInput(
        source_model=source_model,
        transformed_model=model,
        sharding_plan=sharding_plan,
        patch_plan=patch_plan,
        module_overrides=module_overrides,
    )


def _apply_recorded_phase_c(
    model: Any, plan: Any, mesh: Any, *, expert_mesh: Any = None,
) -> Any:
    """Run trainer Phase C at the codegen recording cutoff."""
    for module_spec in plan.modules.values():
        _normalize_out_fields(module_spec)
    _apply_phase_c(model, plan, mesh, True, expert_mesh=expert_mesh)
    return model


def _replacement_records_for_meta(
    config: Any,
    specs: Any,
    records: Any,
) -> tuple[dict[str, Any], ...]:
    """Serialize successful builder replacement events for existing emitters."""
    entries = [
        entry
        for entry in (getattr(config, "plan_overrides", None) or [])
        if getattr(entry, "replace_module", None) is not None
        and getattr(entry, "when", None) in (None,)
    ]
    if len(entries) != len(specs):
        raise RuntimeError(
            "codegen: normalized replacement specs cannot be paired with their YAML declarations"
        )
    metadata_by_factory = {
        id(rule.factory): (rule, entry)
        for rule, entry in zip(specs, entries)
    }
    serialized = []
    for record in records:
        rule_entry = metadata_by_factory.get(id(record.factory))
        if rule_entry is None:
            raise RuntimeError(
                f"codegen: builder replacement at {record.module_fqn!r} has no YAML declaration"
            )
        rule, entry = rule_entry
        target = entry.replace_module
        factory_path = _target_path(target)
        if factory_path is None:
            raise TypeError(
                f"codegen: replacement factory for {rule.match!r} has no serializable import path"
            )
        item = {
            "match": list(rule.match),
            "fqn": record.module_fqn,
            "fqns": [record.module_fqn],
            "module_type": f"{record.source_type.__module__}.{record.source_type.__qualname__}",
            "factory": factory_path,
            "exact_type": bool(rule.exact_type),
        }
        target_config = {
            name: value
            for name, value in target.to_dict().items()
            if name != "_target_"
        }
        if target_config:
            item["target_config"] = target_config
        serialized.append(item)
    return tuple(serialized)


def _apply_generation_replacements(
    _meta: CodegenMeta,
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
    factory_paths = []
    target_configs = []
    for entry in raw_entries:
        replace_module = getattr(entry, "replace_module", None)
        if replace_module is None:
            continue
        factory_paths.append(_target_path(replace_module))
        # Capture the YAML Target's static args so the runtime can rebuild the
        # same pre-bound factory (see ``_target_replacement_factory``) instead
        # of importing the bare ``@module_replacement`` fn — otherwise a
        # configured arg would fall back to its default or fail to build.
        target_configs.append(
            {
                name: value
                for name, value in replace_module.to_dict().items()
                if name != "_target_"
            }
        )
    records = compile_overrides_for_meta(
        model,
        specs,
        factory_paths=factory_paths,
        configs_by_spec={
            id(spec): config for spec, config in zip(specs, target_configs)
        },
    )
    # ``compile_overrides_for_meta`` already raised on any no-match rule (a
    # configuration error); every returned record here matched a live module.
    replace_meta = list(records)
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


_STRUCTURAL_CONFIG_KEYS = ("num_hidden_layers", "num_layers", "use_cache")


def _config_overrides(config: Any) -> dict[str, Any]:
    """Return config-only kwargs passed to AutoConfig for generated models.

    ``from_pretrained`` configs carry these explicitly under
    ``model.config_overrides``.  A ``from_config`` builder instead declares its
    structure as model-target kwargs (e.g. ``num_hidden_layers``), so when
    ``config_overrides`` is absent the recognized structural keys are folded in
    — otherwise the frozen plan would be derived at the source checkpoint's
    shape and disagree with the (cropped) model the builder constructs.
    """
    model = getattr(config, "model", None)
    overrides = getattr(model, "config_overrides", None)
    if overrides is None:
        overrides = {
            key: getattr(model, key)
            for key in _STRUCTURAL_CONFIG_KEYS
            if getattr(model, key, None) is not None
        }
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


def _gen_backend(config: Any) -> bool:
    """Whether the codegen pipeline is active for this config.

    ``codegen=True`` drives artifact generation regardless of the modeling
    backend (a ``force_hf`` run still generates), so this is simply the
    codegen switch — and those are exactly the runs that must fail fast
    when the modeling source cannot be resolved.
    """
    return bool(getattr(config, "codegen", False))


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


def _config_sha256(spec: dict, layout: ArtifactLayout) -> str:
    """Return YAML file hash when available, otherwise hash the projected spec."""
    if layout.yaml_path:
        return sha256_file(layout.yaml_path)
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


__all__ = [
    "CODEGEN_VERSION",
    "barrier_after_artifact",
    "ensure_codegen_artifact",
    "preflight_integrity_check",
    "should_regenerate",
    "wait_for_rank0_artifact",
]
