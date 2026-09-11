# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Import a generated modeling file and resolve its model class.

The bundle at ``<yaml_dir>/generated/`` is not on ``sys.path`` and
is not a normal package, so it is loaded by file location.  The loaded module
is registered in ``sys.modules`` under a stable synthetic name, which is what
lets ``dataclasses``, ``pickle``, and (importantly) ``torch.load`` resolve
classes defined inside it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Synthetic package prefix for generated modules.  Keyed by artifact dir so two
# YAMLs generating for the same model do not collide in ``sys.modules``.
_GENERATED_MODULE_PREFIX = "hyper_parallel_generated"


def _module_name_for(artifact_dir: str, stem: str) -> str:
    """Build a stable, collision-free ``sys.modules`` name for one bundle.

    The basename alone is not unique — two YAMLs in different directories both
    produce ``generated/train/``, which would map to the same synthetic module
    and make the second import silently return the first bundle's classes. The
    absolute path is hashed instead, so the name is unique per bundle and still
    stable across runs (a prerequisite for ``torch.load``/``pickle`` to resolve
    classes defined in the generated file).
    """
    abs_dir = os.path.abspath(os.path.normpath(artifact_dir))
    digest = hashlib.sha256(abs_dir.encode("utf-8")).hexdigest()[:12]
    marker = os.path.basename(abs_dir)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in marker)
    return f"{_GENERATED_MODULE_PREFIX}.{safe}_{digest}.{stem}"


def find_generated_modeling_file(artifact_dir: str) -> str:
    """Locate the single ``modeling_*_gen_npu.py`` in the bundle.

    Raises when absent or ambiguous rather than picking one — a bundle with two
    modeling files means generation left something stale behind, and silently
    choosing would produce a model that does not match the meta.
    """
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(
            f"codegen: artifact dir {artifact_dir} does not exist; "
            "run generation before building with modeling_backend='gen'"
        )
    candidates = sorted(
        name for name in os.listdir(artifact_dir)
        if name.startswith("modeling_") and name.endswith("_gen_npu.py")
    )
    if not candidates:
        raise FileNotFoundError(
            f"codegen: no modeling_*_gen_npu.py in {artifact_dir}; "
            "the bundle is incomplete"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"codegen: {len(candidates)} generated modeling files in "
            f"{artifact_dir} ({candidates}); remove the stale bundle and "
            "regenerate"
        )
    return os.path.join(artifact_dir, candidates[0])


def import_generated_module(artifact_dir: str) -> ModuleType:
    """Import the generated modeling module from ``artifact_dir``.

    Repeat calls for the same bundle return the cached module.
    """
    modeling_path = find_generated_modeling_file(artifact_dir)
    stem = os.path.splitext(os.path.basename(modeling_path))[0]
    module_name = _module_name_for(artifact_dir, stem)

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    # A ``<remote>`` source uses
    # single-dot relative imports (``from .configuration_deepseek import ...``)
    # that resolve against its own package.  spec_from_file_location compiles
    # the generated file as the synthetic top-level module
    # ``hyper_parallel_generated.<digest>.<stem>`` — setting its ``__package__``
    # to the synthetic *package* (hyper_parallel_generated.<digest>) and
    # registering that package + its sibling modules before exec makes those
    # relative imports resolve inside the bundle.  Both are keyed on the same
    # path-derived name as the module, so they stay collision-free per bundle
    # and the registered modules are reused on the next import.
    _register_synthetic_package(module_name, artifact_dir)

    spec = importlib.util.spec_from_file_location(module_name, modeling_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"codegen: cannot build an import spec for {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _package_name_for(module_name)
    # Registered before exec so a self-referential import inside the generated
    # file resolves to this partially-initialized module rather than looping.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    logger.info("codegen: imported generated modeling file %s", modeling_path)
    return module


def _package_name_for(module_name: str) -> str:
    """The synthetic package a generated module lives in (minus the stem).

    ``hyper_parallel_generated.<safe>_<digest>.<stem>`` -> ``...<safe>_<digest>``.
    This is the package the generated file's relative imports resolve against,
    and it is stable per bundle (the digest is the artifact dir's path hash).
    """
    return module_name.rsplit(".", 1)[0]


def _register_synthetic_package(module_name: str, artifact_dir: str) -> None:
    """Register the synthetic package + its sibling modules in ``sys.modules``.

    A plain ``ModuleType`` package with ``__path__`` set to the bundle dir lets
    ``from .name import x`` resolve against the bundle's own files without any
    ``sys.path`` mutation.  Only the package root is created; sibling modules
    are loaded on demand by the import machinery (exec'd here) and registered
    under the synthetic package name so ``torch.load``/``pickle`` resolve
    classes defined in them the same way they resolve the generated module.
    """
    package_name = _package_name_for(module_name)
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [os.path.abspath(artifact_dir)]
        package.__file__ = "<codegen-synthetic-package>"
        sys.modules[package_name] = package
    elif not hasattr(package, "__path__"):
        package.__path__ = [os.path.abspath(artifact_dir)]

    # Pre-register sibling .py files (the ``<remote>`` relative-import
    # closure, copied into the bundle by ``emit_bundle``) so their own
    # cross-references resolve inside the same package.  The generated
    # modeling file is the one being imported — skip it.
    prefix = package_name + "."
    modeling_name = find_generated_modeling_file(artifact_dir)
    for name in sorted(os.listdir(artifact_dir)):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        if name == os.path.basename(modeling_name):
            continue
        sibling_name = prefix + os.path.splitext(name)[0]
        if sibling_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            sibling_name, os.path.join(artifact_dir, name),
        )
        if spec is None or spec.loader is None:
            continue
        sibling = importlib.util.module_from_spec(spec)
        sibling.__package__ = package_name
        sys.modules[sibling_name] = sibling
        try:
            spec.loader.exec_module(sibling)
        except Exception:
            sys.modules.pop(sibling_name, None)
            raise


def resolve_generated_model_class(module: ModuleType, hf_config: Any) -> type:
    """Find the model class in ``module`` matching the config's architecture.

    The generated file keeps the original class names, so the architecture name
    from the HF config is the lookup key.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        raise ValueError(
            "codegen: hf_config has no 'architectures'; cannot pick a model "
            f"class from {module.__name__}"
        )
    arch_name = architectures[0]
    model_cls = getattr(module, arch_name, None)
    if model_cls is None:
        available = sorted(
            name for name, value in vars(module).items() if isinstance(value, type)
        )
        raise AttributeError(
            f"codegen: generated module {module.__name__} has no class "
            f"{arch_name!r}; defined classes: {available}"
        )
    if not isinstance(model_cls, type):
        raise TypeError(
            f"codegen: {module.__name__}.{arch_name} is not a class"
        )
    return model_cls


def _register_generated_model_conversions(model: Any) -> None:
    """Let the generated model share the library's native weight conversions.

    The generated modeling file is loaded as custom code (its module name does
    not start with ``transformers.``), so ``PreTrainedModel.is_custom_code()``
    is True.  ``get_model_conversion_mapping`` skips custom-code submodules
    unless their class name or ``model_type`` is in
    ``transformers.conversion_mapping.USER_REGISTERED_MAPPINGS``.  The generated
    model only patches the parallel-strategy layer — its weight layout is the
    native one — so it *should* use the same checkpoint conversions as the
    installed model (e.g. the qwen2_moe / qwen3_moe fusion of per-expert
    ``gate_proj``/``up_proj``/``down_proj`` into ``experts.gate_up_proj``/
    ``experts.down_proj``).  Registering the identifiers here marks the model
    as user-registered so those library conversions apply instead of being
    silently skipped, which would leave every fused expert target unmatched
    during checkpoint loading.

    Only marks identifiers already covered by the library mapping; an unknown
    ``model_type`` gets no entry and registration is a no-op.  The lookup is by
    ``model_type`` (class-name lookup falls back to it), so registering the
    class name is not required for the root model.
    """
    from transformers.conversion_mapping import USER_REGISTERED_MAPPINGS

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type:
        # Guard against API drift across transformers versions: if the set is
        # renamed or removed the whole registration is skipped rather than
        # breaking the generated-model build.
        if getattr(USER_REGISTERED_MAPPINGS, "add", None):
            USER_REGISTERED_MAPPINGS.add(model_type)
            logger.info(
                "codegen: registered generated model_type %r for checkpoint "
                "conversion mapping", model_type,
            )


def init_generated_model(
    artifact_dir: str,
    pretrained_model_name_or_path: Optional[str],
    hf_config: Any,
    *model_args: Any,
    torch_dtype: Any = "auto",
    **kwargs: Any,
) -> Any:
    """Build the model from the generated modeling file.

    Mirrors the custom-model path in ``model_init``: ``from_pretrained`` when a
    checkpoint path is given, ``from_config`` otherwise.  After the model is
    built, its identifiers are registered for the library weight conversions
    (``_register_generated_model_conversions``) and the frozen param plan is
    verified against the *live* parameter tree (``verify_param_plan``) — this is
    the check that catches a stale bundle: generation-time failures that degrade
    to an empty plan, or a model whose parameter tree drifted from what the plan
    records, fail here instead of training with a stale sharding contract.
    """
    module = import_generated_module(artifact_dir)
    model_cls = resolve_generated_model_class(module, hf_config)

    if pretrained_model_name_or_path is not None:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    else:
        # transformers >= 5.14 removed ``PreTrainedModel.from_config`` in
        # favour of ``_from_config`` (a classmethod on the concrete model
        # class).  The generated model class only inherits ``PreTrainedModel``,
        # so ``from_config`` no longer exists on it.  ``_from_config`` is the
        # canonical entry in both 5.14.x and the pinned 5.15.0; the AutoModel
        # family's ``from_config`` (which the native path borrows) is just a
        # resolver that ultimately calls ``_from_config`` anyway.
        model = model_cls._from_config(hf_config, *model_args, **kwargs)

    _register_generated_model_conversions(model)

    from hyper_parallel.codegen.check.preflight import verify_param_plan
    from hyper_parallel.codegen.meta import load_codegen_meta

    meta = load_codegen_meta(os.path.join(artifact_dir, "codegen_meta.json"))
    if meta is None:
        raise FileNotFoundError(
            f"codegen: missing meta at {os.path.join(artifact_dir, 'codegen_meta.json')}; "
            "the bundle is incomplete — regenerate the artifact"
        )
    verify_param_plan(meta, model)
    return model


__all__ = [
    "find_generated_modeling_file",
    "import_generated_module",
    "init_generated_model",
    "resolve_generated_model_class",
]
