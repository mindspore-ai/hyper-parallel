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

    Args:
        artifact_dir (str): Directory holding the generated bundle.

    Returns:
        str: Absolute path of the generated modeling file.
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

    Args:
        artifact_dir (str): Directory holding the generated bundle.

    Returns:
        ModuleType: The imported (and cached) generated modeling module.
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

    Args:
        module (ModuleType): The imported generated modeling module.
        hf_config (Any): HF config whose ``architectures`` names the class.

    Returns:
        type: The model class to instantiate.
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
    from transformers.conversion_mapping import (  # pylint: disable=import-outside-toplevel
        USER_REGISTERED_MAPPINGS,
    )

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


def _load_generated_checkpoint(
    model_cls: type,
    pretrained_path: str,
    hf_config: Any,
    model_args: tuple,
    torch_dtype: Any,
    kwargs: dict,
) -> Any:
    """Construct a generated model and load its weights with layout conversions.

    The generated modeling file embeds the final (post-replacement) modules
    directly -- notably ``GroupedExperts`` (expert weights stored transposed
    relative to the checkpoint) and the fused ``GQAAttention`` (``linear_qkv``
    built from ``q_proj``/``k_proj``/``v_proj``).  The native path records these
    conversions through ``apply_module_replacements``; no replacement runs for
    the generated artifact (the fused modules are literals in the generated
    code), so synthesise the same wiring here: collect every module's
    ``make_transforms()`` (scoped to its FQN) and publish the *pre-replacement*
    parameter shapes, then let ``CheckpointManager`` apply the conversions
    through the replacement-conversion route.
    """
    from hyper_parallel.models._transformers.checkpoint_loader import (  # pylint: disable=import-outside-toplevel
        CheckpointManager,
    )

    model = model_cls._from_config(hf_config, *model_args, **kwargs)
    _register_generated_model_conversions(model)

    transforms: list[Any] = []
    for module_fqn, module in model.named_modules():
        make_transforms = getattr(module, "make_transforms", None)
        if make_transforms is None:
            continue
        for transform in make_transforms():
            transform.scope_prefix = module_fqn
            transforms.append(transform)
    if transforms:
        model._hp_checkpoint_source_shapes = _source_model_shapes(hf_config)  # pylint: disable=protected-access
        model._hp_replacement_weight_conversions = transforms  # pylint: disable=protected-access

    # from_pretrained materializes meta tensors internally; _from_config
    # does not. Materialize the meta model before loading so checkpoint
    # copy targets are real tensors (init_device: meta in the training
    # config). Downstream infra moves it to the accelerator afterwards.
    if any(p.is_meta for p in model.parameters()):
        model.to_empty(device="cpu")

    CheckpointManager(model).load_checkpoint(pretrained_path, strict=False)
    if torch_dtype not in (None, "auto"):
        # ``torch_dtype`` may be the raw HF string (e.g. "bfloat16"); the native
        # path resolves it before calling ``model.to(dtype=...)``.  Passing the
        # bare string makes torch_npu's ``_parse_to`` treat it as a device name
        # and raise "Invalid device string", so resolve it through the platform
        # abstraction instead of importing the backend here.
        from hyper_parallel.platform import get_platform  # pylint: disable=import-outside-toplevel

        dtype = torch_dtype if "." in torch_dtype else f"torch.{torch_dtype}"
        _cast_parameters_to(model, get_platform().str_to_dtype(dtype))
    return model


def _cast_parameters_to(model: Any, dtype: Any) -> None:
    """Cast the model's parameters to ``dtype`` the way the native path does.

    ``model.to(dtype)`` also down-casts floating-point *buffers*, but the native
    ``from_pretrained(dtype=...)`` path only casts parameters: buffers such as
    the RoPE ``inv_freq`` table keep the precision they were computed with.
    Casting them here rounds the table to bf16, which changes every cos/sin
    value and makes the generated model diverge numerically from the trained
    one.  Modules listed in ``_keep_in_fp32_modules`` stay in fp32 on both
    paths, so they are skipped as well.

    Args:
        model: Model whose parameters are cast in place.
        dtype: Target ``torch`` dtype.
    """

    keep_fp32 = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    for name, parameter in model.named_parameters():
        if keep_fp32 and any(module_name in name for module_name in keep_fp32):
            continue
        parameter.data = parameter.data.to(dtype)


def _source_model_shapes(hf_config: Any) -> dict[str, tuple[int, ...]]:
    """Capture the pre-replacement parameter shapes for the conversion route.

    The native path records ``_named_tensor_shapes(model)`` *before* swapping in
    the fused modules, so its converters are routed by the original parameter
    names and layouts.  The generated artifact has the fused modules baked in
    and no longer owns the converter source parameters (``q_proj``/``k_proj``/
    ``v_proj`` exist only as converter source patterns), so rebuild the original
    parameter tree from the config on the meta device -- no storage is
    allocated -- and publish its shapes instead.
    """
    import transformers  # pylint: disable=import-outside-toplevel
    from hyper_parallel.models.replacement import (  # pylint: disable=import-outside-toplevel
        _named_tensor_shapes,
    )
    from hyper_parallel.platform import get_platform  # pylint: disable=import-outside-toplevel

    platform = get_platform()
    with platform.init_on_device(platform.meta_device):
        source = transformers.AutoModelForCausalLM.from_config(hf_config)
    return _named_tensor_shapes(source)


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

    Args:
        artifact_dir (str): Directory holding the generated bundle.
        pretrained_model_name_or_path (Optional[str]): Checkpoint to load, or
            None to build from the config only.
        hf_config (Any): HF config of the model being generated.
        *model_args (Any): Extra positional args for the constructor.
        torch_dtype (Any): Target dtype, an HF-style string, or "auto".
        **kwargs (Any): Extra keyword args for the constructor.

    Returns:
        Any: The instantiated model with converted weights loaded.
    """
    module = import_generated_module(artifact_dir)
    model_cls = resolve_generated_model_class(module, hf_config)

    if pretrained_model_name_or_path is not None:
        model = _load_generated_checkpoint(
            model_cls,
            pretrained_model_name_or_path,
            hf_config,
            model_args,
            torch_dtype,
            kwargs,
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

    from hyper_parallel.codegen.check.preflight import (  # pylint: disable=import-outside-toplevel
        verify_param_plan,
    )
    from hyper_parallel.codegen.meta import (  # pylint: disable=import-outside-toplevel
        load_codegen_meta,
    )

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
