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
"""Shared Transformers-config resolution for the Auto Parallel cost model.

Both the SAPP-ND parser (``CostModelParserHyperV2``) and the config-adapter
loader need the same model dimensions out of an AutoModels ``model`` section.
This module owns that translation exactly once, so the two callers cannot
drift apart on field names, aliases, or fallback behaviour.

It is also the single place that imports ``transformers``.  The import is
function-local because ``transformers`` is not a hard dependency of
``hyper_parallel`` (``requirements.txt`` only pins numpy) and the non-Hyper
cost-model backends must keep working without it.
"""
import copy
import dataclasses
import hashlib
import importlib
import json
import logging
import os
import sys
from typing import Any, Dict, Mapping, Optional, Tuple

from hyper_parallel.auto_parallel._layer_census import (
    CensusUnavailable,
    census_activations,
    census_output_activations,
)
from hyper_parallel.auto_parallel._layer_stack import spec_layer_stack, tower_layer_stack
from hyper_parallel.auto_parallel._model_spec import ModelSpec
from hyper_parallel.auto_parallel._op_profiles import (
    infer_arch,
    infer_qk_norm,
    infer_qkv_bias,
    infer_shared_expert_gate,
    resolve_ops,
)
from hyper_parallel.auto_parallel._spec_census import census_model_spec

logger = logging.getLogger(__name__)

# AutoModels root sections that identify the current Trainer schema.
_AUTO_MODELS_SECTIONS = ("training", "accelerator", "fsdp_config")

# ``model`` keys forwarded verbatim to ``AutoConfig.from_pretrained``.
_CONFIG_KWARG_NAMES = (
    "cache_dir", "local_files_only", "revision", "subfolder",
    "token", "trust_remote_code",
)

# Where the model directory is named. A from_config recipe builds the model
# through a factory (``_target_`` plus ``config_path``) and never loads a
# checkpoint, so it carries no pretrained path.
_MODEL_PATH_ALIASES = ("pretrained_model_name_or_path", "config_path")

# The config a recipe's own factory builds, for the factories whose model
# ``AutoConfig`` cannot resolve: the path beside ``_target_`` is then a
# directory Transformers refuses, and the shape the run really trains is the
# one this function returns.  DeepSeek-V4.1 is the case that forced it:
# Transformers 5.14.1 has no ``deepseek_v41``, and the factory's arguments are
# a recipe for deriving the shape (a divisor and an expert count) rather than
# the canonical fields :func:`_explicit_overrides` can harvest, so nothing
# short of calling it knows what the run builds.  A factory is consulted
# BEFORE ``AutoConfig``, because where both answer they answer differently and
# the factory is the one that says what the run trains: V4.1's crops the
# released model eightfold, and the released config would price a model nobody
# runs.  One line a family, and the entry is the declaration that the two are
# a pair.
_CONFIG_FACTORIES = {
    "hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model."
    "build_cropped_deepseek_v41":
        "hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model."
        "build_deepseek_v41_validation_config",
}

# Canonical text-tower fields, with the Transformers aliases that carry them.
# First populated alias wins, so Hyper-internal names keep priority over the
# Hugging Face spelling when a config happens to define both.
_TEXT_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("num_hidden_layers",),
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "head_dim": ("head_dim",),
    "intermediate_size": ("intermediate_size",),
    "vocab_size": ("vocab_size",),
    "max_position_embeddings": ("max_position_embeddings",),
    "num_experts": ("num_experts", "n_routed_experts"),
    "num_experts_per_tok": ("num_experts_per_tok",),
    "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
    "moe_intermediate_size": ("moe_intermediate_size",),
    "shared_expert_intermediate_size": ("shared_expert_intermediate_size",),
    "first_k_dense_replace": ("first_k_dense_replace",),
    "mtp_depth": ("mtp_depth", "num_nextn_predict_layers", "mtp_num_hidden_layers"),
    "multiple_of": ("multiple_of",),
    "ffn_dim_multiplier": ("ffn_dim_multiplier",),
    "kv_lora_rank": ("kv_lora_rank",),
    "q_lora_rank": ("q_lora_rank",),
    "qk_rope_head_dim": ("qk_rope_head_dim",),
    "qk_nope_head_dim": ("qk_nope_head_dim",),
    "v_head_dim": ("v_head_dim",),
    "attn_output_gate": ("attn_output_gate",),
    "tie_word_embeddings": ("tie_word_embeddings",),
    # Llama's attention_bias biases all four projections; Qwen2's configs
    # state none and bias the query, key and value projections alone.
    "qkv_bias": ("qkv_bias", "attention_bias"),
    "o_bias": ("o_bias", "attention_bias"),
    "mlp_bias": ("mlp_bias",),
    # Stated by a few families; Qwen3's config does not state its own.
    "qk_norm": ("qk_norm", "use_qk_norm", "qk_layernorm"),
    # A hybrid stack states its layers; without this every layer is costed
    # as full attention, which is quadratic in the sequence length.
    "layer_types": ("layer_types",),
    "linear_num_key_heads": ("linear_num_key_heads",),
    "linear_key_head_dim": ("linear_key_head_dim",),
    "linear_num_value_heads": ("linear_num_value_heads",),
    "linear_value_head_dim": ("linear_value_head_dim",),
    "linear_conv_kernel_dim": ("linear_conv_kernel_dim",),
    # The dtype the checkpoint's weights are saved in, which the runtime
    # loads them in under its default torch_dtype of auto; newer
    # Transformers name it dtype.
    "torch_dtype": ("dtype", "torch_dtype"),
}

# Vision towers use their own spelling for the shared concepts.
_VISION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("depth", "num_hidden_layers"),
    "num_attention_heads": ("num_heads", "num_attention_heads"),
    "intermediate_size": ("intermediate_size",),
    "out_hidden_size": ("out_hidden_size",),
    "patch_size": ("patch_size",),
    "spatial_merge_size": ("spatial_merge_size",),
    "num_position_embeddings": ("num_position_embeddings",),
}

# Fallback when a vision tower declares no positional-embedding grid.
_DEFAULT_VISUAL_SEQ_LEN = 1024

# Keys of ``model.config_overrides`` that state the run, not the model.  The
# model spec never sees them; the Hyper parser reads them into its ExecSpec.
EXEC_OVERRIDE_KEYS = (
    "offset", "full_rec", "sel_rec", "capacity_factor", "cap_fact", "use_gmm", "gmm", "seq_length",
)


def _declares_pretrained_path(mapping: Any) -> bool:
    """Return whether the ``model`` section names a Transformers checkpoint.

    This is the one key only this schema has: the older HyperParallel yamls
    name a checkpoint as ``model.weights_path``. Recognising it means a
    config needs no empty ``training``/``accelerator``/``fsdp_config``
    section just to be identified.
    """
    if isinstance(mapping, Mapping):
        model = mapping.get("model")
    else:
        model = getattr(mapping, "model", None)
    if model is None:
        return False
    if isinstance(model, Mapping):
        return bool(model.get("pretrained_model_name_or_path"))
    return bool(getattr(model, "pretrained_model_name_or_path", None))


def is_auto_models_schema(mapping: Any) -> bool:
    """Return whether *mapping* looks like an AutoModels Trainer config.

    Accepts a plain mapping or any object exposing the root sections as
    attributes, so the SAPP-ND ``Config`` tree and a parsed YAML dict can be
    tested with the same call.
    """
    if _declares_pretrained_path(mapping):
        return True
    if isinstance(mapping, Mapping):
        return any(name in mapping for name in _AUTO_MODELS_SECTIONS)
    holder = getattr(mapping, "__dict__", None)
    if isinstance(holder, dict):
        return any(name in holder for name in _AUTO_MODELS_SECTIONS)
    return False


def _first_attr(config: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first populated attribute of *config* among *names*."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _text_tower(model_config: Any) -> Any:
    """Return the language-model sub-config of a composite config.

    Multimodal Transformers configs (``Qwen3VLMoeConfig`` and friends) keep
    the language model under ``text_config`` and expose none of its fields at
    the top level, so reading ``hidden_size`` directly raises AttributeError.
    """
    return getattr(model_config, "text_config", None) or model_config


def _spec_from_aliases(config: Any, aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """Collect the canonical fields declared by *config*."""
    spec = {}
    for canonical, names in aliases.items():
        value = _first_attr(config, names)
        if value is not None:
            spec[canonical] = value
    return spec


def _derive_shared_experts(spec: Dict[str, Any]) -> None:
    """Infer the shared-expert count from its total feed-forward width.

    Qwen2-MoE style configs declare one wide shared expert instead of a
    count.  The cost model only ever uses ``n_shared_exp * hff_exp``, so
    encoding that width as several narrow experts is width-equivalent.
    """
    if spec.get("num_shared_experts"):
        return
    moe_inter = int(spec.get("moe_intermediate_size", 0) or 0)
    shared_inter = int(spec.get("shared_expert_intermediate_size", 0) or 0)
    if moe_inter and shared_inter:
        spec["num_shared_experts"] = max(1, shared_inter // moe_inter)


def _derive_dense_ffn_width(spec: Dict[str, Any]) -> None:
    """Fill the dense feed-forward width for an all-MoE config.

    A model whose every layer is MoE (Qwen3.5-MoE) declares no
    ``intermediate_size``, but the cost model prices the shared expert at
    that width, so leaving it absent drops the shared expert entirely.
    The shared expert's own width is the value meant there.
    """
    if spec.get("intermediate_size"):
        return
    width = spec.get("shared_expert_intermediate_size") or spec.get("moe_intermediate_size")
    if width:
        spec["intermediate_size"] = int(width)


def _visual_seq_len(vision_spec: Dict[str, Any], override: Optional[int]) -> int:
    """Resolve the encoder sequence length in merged visual tokens.

    The true count depends on the image resolution served by the dataset,
    which no configuration we read carries.  The positional-embedding grid
    divided by the spatial merge is the tightest bound available from the
    model alone, so it is the default and *override* exists to correct it.
    """
    if override:
        logger.info("visual sequence length taken from context.visual_seq_len: %d", override)
        return int(override)
    grid = int(vision_spec.get("num_position_embeddings", 0) or 0)
    merge = max(1, int(vision_spec.get("spatial_merge_size", 1) or 1))
    if not grid:
        logger.warning(
            "vision tower declares no num_position_embeddings; assuming %d visual tokens. "
            "Set context.visual_seq_len to the value your dataset produces.",
            _DEFAULT_VISUAL_SEQ_LEN,
        )
        return _DEFAULT_VISUAL_SEQ_LEN
    derived = max(1, grid // (merge * merge))
    logger.info(
        "visual sequence length derived from the vision config: %d "
        "(num_position_embeddings=%d, spatial_merge_size=%d). "
        "Set context.visual_seq_len to override.",
        derived, grid, merge,
    )
    return derived


def _get_hf_config(model_raw: Mapping[str, Any]) -> Any:
    """Call the AutoModels Transformers entry point for *model_raw*."""
    # Transformers is an optional dependency; import only when a config
    # actually needs it so the other cost-model backends stay importable.
    from hyper_parallel.models._transformers.config_resolver import get_hf_config  # pylint: disable=C0415

    config_kwargs = {
        name: model_raw[name]
        for name in _CONFIG_KWARG_NAMES
        if model_raw.get(name) is not None
    }
    return get_hf_config(
        str(_model_path(model_raw)),
        str(model_raw.get("attn_implementation", "sdpa")),
        model_raw.get("torch_dtype", "auto"),
        **config_kwargs,
    )


def _factory_config(model_raw: Mapping[str, Any]) -> Any:
    """The config the recipe's own factory builds, or None where it names none.

    The model section's keys are offered to the factory by name, filtered to
    the ones it accepts, so a recipe states its shape once and the cost model
    reads the same statement the trainer does.  Anything the factory cannot
    take, such as a dtype or a placement switch, is left out rather than
    refused.

    Returns:
        A Transformers config, or None where ``_target_`` names no factory
        this knows.

    Raises:
        ValueError: If the factory is known but cannot build its config, since
            a recipe that names one and then fails is a broken recipe rather
            than a model to price by formula.
    """
    import inspect  # pylint: disable=C0415

    target = str(model_raw.get("_target_") or "")
    path = _CONFIG_FACTORIES.get(target)
    if path is None:
        return None
    module_path, _, name = path.rpartition(".")
    try:
        factory = getattr(importlib.import_module(module_path), name)
        accepted = inspect.signature(factory).parameters
        offered = {key: value for key, value in model_raw.items() if key in accepted}
        logger.info("building the config %s states, through %s", target.rpartition(".")[2], name)
        return factory(**offered)
    except (ImportError, AttributeError, TypeError, OSError, KeyError, ValueError) as exc:
        raise ValueError(
            f"model._target_ names {target}, whose config factory {name} could not build its "
            f"config ({exc}); the cost model prices what that factory builds, so fix the recipe "
            f"rather than let the formulas price a different model"
        ) from exc


def _model_path(model_raw: Mapping[str, Any]) -> Optional[str]:
    """Return the model directory the section names, under any accepted key."""
    for alias in _MODEL_PATH_ALIASES:
        value = model_raw.get(alias)
        if value:
            return str(value)
    return None


def _explicit_overrides(model_raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model fields the model section states itself, as a plain dict.

    ``config_overrides`` is the declared form, less its run keys
    (:data:`EXEC_OVERRIDE_KEYS`). A from_config recipe instead passes the shape
    it builds as factory arguments beside ``_target_``, such as a
    ``num_hidden_layers`` that crops the released model; those name canonical
    fields, so they are overrides too, and the released config must not win
    over them. ``config_overrides`` keeps priority where both are present. A
    value of ``auto``, which ``torch_dtype`` takes to load the checkpoint's
    own, states nothing over the checkpoint.
    """
    overrides = model_raw.get("config_overrides")
    explicit = {}
    if isinstance(overrides, Mapping):
        explicit = {key: value for key, value in overrides.items() if key not in EXEC_OVERRIDE_KEYS}
    for field in _TEXT_FIELD_ALIASES:
        value = model_raw.get(field)
        if value is not None and value != "auto":
            explicit.setdefault(field, value)
    return explicit


def exec_overrides(model_raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the run keys of the ``config_overrides`` fallback.

    Args:
        model_raw: The ``model`` section, as a plain mapping.

    Returns:
        The entries whose keys are in :data:`EXEC_OVERRIDE_KEYS`.
    """
    overrides = model_raw.get("config_overrides")
    if not isinstance(overrides, Mapping):
        return {}
    return {key: value for key, value in overrides.items() if key in EXEC_OVERRIDE_KEYS}


def _validated(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Check *spec* against the model IR before any consumer reads it.

    Parsing and re-serialising is what makes the schema load-bearing rather
    than advisory: a field the model declares but the IR does not know would
    be dropped here, and an incoherent one raises by name.  The op profile
    and the layer stack are settled here too, so the serialised spec states
    what it will be priced with and which kind each layer is, even when the
    family was only inferred from the model name and the stack from
    ``layer_types`` or ``first_k_dense_replace``; and so is whether its
    attention normalizes queries and keys, which a Transformers config does
    not state, and the biases and norms its layers hold: a Transformers
    decoder layer holds two RMSNorms, no bias unless its config states one
    or its family has one, and a shared expert gated where its family
    gates it.  ``layer_types`` is
    consumed: ``layers`` says the same, checked against the profile.  A
    vision tower states its stack likewise.
    """
    typed = ModelSpec.from_dict(spec)
    typed = dataclasses.replace(
        typed, arch=typed.arch or infer_arch(typed.name),
        qk_norm=typed.qk_norm if typed.qk_norm is not None else infer_qk_norm(typed.name),
        qkv_bias=typed.qkv_bias if typed.qkv_bias is not None else infer_qkv_bias(typed.name),
        o_bias=bool(typed.o_bias), mlp_bias=bool(typed.mlp_bias), norm_bias=bool(typed.norm_bias),
        layer_norms=typed.layer_norms or 2,
        shared_expert_gate=(typed.shared_expert_gate if typed.shared_expert_gate is not None
                            else infer_shared_expert_gate(typed.name)),
    )
    resolve_ops(typed.arch, typed.ops)
    layers = spec_layer_stack(typed).to_layers()
    vision = typed.vision
    if vision is not None:
        vision = dataclasses.replace(vision, layers=tower_layer_stack(vision).to_layers())
    return dataclasses.replace(typed, layers=layers, vision=vision, layer_types=None).to_dict()
# Canonical fields that say how a model is held, not what it is: a section
# stating only these describes no model.
_HOLDING_FIELDS = frozenset({"torch_dtype"})


def _describes_a_model(explicit: Mapping[str, Any]) -> bool:
    """Whether the fields a section states describe a model, beyond how it is held."""
    return any(key not in _HOLDING_FIELDS for key in explicit)


def _dtype_name(value: Any) -> Any:
    """A dtype as its plain name, ``bfloat16`` for ``torch.bfloat16``; anything else as it is."""
    return str(value).replace("torch.", "") if value is not None else None


# The censuses this process has run, by config, layer stack and length: a
# harness pricing several runs of one model runs one.
_CENSUSES: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

# Where a census is kept between runs, which it then costs nothing:
# ND_CENSUS_CACHE names another folder, or is "off" to keep none.
_CENSUS_FOLDER = os.path.join(os.path.expanduser("~"), ".cache", "hyper_parallel", "census")
# The package whose files a kept census is checked against.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The search's own code, which runs after a census and hands it only what its
# key holds: a change there leaves the kept censuses valid.
_NOT_CENSUS_CODE = (
    "hyper_parallel.auto_parallel.sapp_nd",
    "hyper_parallel.auto_parallel.sapp_ppb",
    "hyper_parallel.auto_parallel.config_adapter",
)


def _census_file(key: Tuple[Any, ...]) -> Optional[str]:
    """The file a census of *key* is kept in, named by the key and the torch and Transformers it runs on.

    ``None`` where ``ND_CENSUS_CACHE`` is ``off``.
    """
    folder = os.environ.get("ND_CENSUS_CACHE") or _CENSUS_FOLDER
    if folder.lower() == "off":
        return None
    versions = [str(getattr(sys.modules.get(name), "__version__", None)) for name in ("torch", "transformers")]
    name = hashlib.sha256(json.dumps([repr(key), *versions]).encode("utf-8")).hexdigest()
    return os.path.join(folder, name[:32] + ".json")


def _source_digest(path: str) -> str:
    """The SHA-256 of the file at *path*, its line endings read as Python reads them.

    Two checkouts of one commit can hold a file with Windows line endings in
    one and Unix ones in the other, which is the same code.
    """
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def _census_sources() -> Dict[str, str]:
    """The digest of each file of the package this process has loaded, the search's left out, by its path.

    A census runs on them: its own code, the kernels' contracts, the
    checkpoint policy and the modules a run's replacements build.
    """
    sources = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.split(".")[0] == "hyper_parallel" and not name.startswith(_NOT_CENSUS_CODE):
            sources[os.path.relpath(path, _PACKAGE_ROOT).replace(os.sep, "/")] = _source_digest(path)
    return sources


def _kept_census(path: str) -> Optional[Dict[str, Any]]:
    """The census kept in *path*, or ``None`` where there is none or a file it ran on has changed since."""
    try:
        with open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        if all(_source_digest(os.path.join(_PACKAGE_ROOT, source)) == digest
               for source, digest in kept["sources"].items()):
            return kept["census"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def _keep_census(path: str, census: Dict[str, Any]) -> None:
    """Keep *census* in *path* with the digest of each file it ran on; failing costs the next run a census."""
    partial = f"{path}.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump({"sources": _census_sources(), "census": census}, handle)
        os.replace(partial, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("the census could not be kept in %s: %s", path, exc)


def _census(text_config: Any, layers: Any, seq_length: int, replacements: Tuple[Any, ...] = ()) -> Dict[str, Any]:
    """The spec's census of each layer kind of *layers* and of the output layer, once per config and length.

    A census is kept on disk (:func:`_census_file`) and read back by a later
    run of the same model, length and replacements, on the same torch and
    Transformers, while no file of the package it ran on has changed.
    """
    key = (
        text_config.to_json_string(),
        tuple((group["kind"], int(group["count"]), bool(group.get("mtp"))) for group in layers),
        int(seq_length),
        tuple((f"{spec.factory.__module__}.{spec.factory.__qualname__}",
               f"{spec.module_type.__module__}.{spec.module_type.__qualname__}",
               tuple(spec.match), bool(spec.exact_type)) for spec in replacements),
    )
    if key not in _CENSUSES:
        path = _census_file(key)
        kept = _kept_census(path) if path is not None else None
        if kept is not None:
            logger.info("census of each layer kind and of the output layer at %d tokens read from %s",
                        seq_length, path)
            _CENSUSES[key] = kept
            return copy.deepcopy(kept)
        logger.info("census of each layer kind and of the output layer at %d tokens%s", seq_length,
                    f", {len(replacements)} module replacements installed" if replacements else "")
        try:
            kinds = census_activations(text_config, layers, seq_length, replacements)
            _CENSUSES[key] = {
                "activations": {kind: record.to_dict() for kind, record in kinds.items()},
                "output_activations": census_output_activations(text_config, seq_length, replacements).to_dict(),
            }
        except CensusUnavailable as exc:
            # The formulas price the layers, as they do without a census: a
            # census of Transformers' own modules would price a layer the run
            # does not build (F53).
            logger.warning("%s", _NOT_A_BOUND % exc)
            _CENSUSES[key] = {}
        else:
            if path is not None:
                _keep_census(path, _CENSUSES[key])
    return copy.deepcopy(_CENSUSES[key])


# What a run is told when a census it asked for did not happen.  Measured
# without one, the formulas read 37 to 55% below Qwen3.5's peak and 51.1%
# below the DeepSeek-V4.1 crop's, so a memory a search filters on is an
# estimate and not an upper bound, and a strategy it calls feasible may not
# be.  Said once, in the run's own log, where the number is read.
_NOT_A_BOUND = (
    "no census of the layers: %s. The formulas price them instead, and without a census they have "
    "measured 37 to 55%% below a real peak, so this run's memory is an estimate and NOT a bound: a "
    "strategy it reports as fitting may still run out of memory"
)


def _no_census(census_seq_len: int, explicit: Mapping[str, Any]) -> None:
    """Warn that a census asked for cannot run, unless the overrides state its records.

    A census builds its layers from the checkpoint's config; a search hands
    ND a spec whose overrides state the records its reader measured.
    """
    if census_seq_len and not explicit.get("activations"):
        logger.warning("%s", _NOT_A_BOUND % "it needs the checkpoint's Transformers config")


def _stated_spec(model_raw: Mapping[str, Any], explicit: Dict[str, Any], census_seq_len: int) -> Dict[str, Any]:
    """The spec of a model its section states by hand, derived and settled as a resolved one is.

    A model no Transformers config describes, every model Transformers
    cannot build, is stated field by field, and it takes the derivations a
    resolved spec does: a shared expert stated by its width counts as that
    many experts, and an all-MoE model's dense width is its shared
    expert's.  Without them a hand-described all-MoE model kept no
    ``intermediate_size``, the width its shared expert is priced at, and
    its shared expert was dropped (I1).
    """
    explicit.setdefault("name", model_raw.get("name", "custom"))
    _derive_shared_experts(explicit)
    _derive_dense_ffn_width(explicit)
    _no_census(census_seq_len, explicit)
    return _validated(explicit)


def resolve_hf_model_spec(
    model_raw: Mapping[str, Any],
    visual_seq_len: Optional[int] = None,
    census_seq_len: int = 0,
    census_spec: bool = False,
    replacements: Tuple[Any, ...] = (),
) -> Dict[str, Any]:
    """Return canonical cost-model fields for a Trainer ``model`` section.

    Resolves ``model.pretrained_model_name_or_path`` through the same
    Transformers path AutoModels uses.  A composite (vision-language) config
    contributes its language tower under the canonical keys and its vision
    tower under ``"vision"``; a plain config leaves ``"vision"`` absent.

    ``model.config_overrides`` stays supported for standalone cost-model
    search files, and doubles as the fallback when the Transformers config
    cannot be reached (offline node, unreachable repository).  Explicit
    overrides always win over resolved values.  Its run keys,
    :data:`EXEC_OVERRIDE_KEYS`, are not the model's: :func:`exec_overrides`
    returns them.

    Args:
        model_raw: The ``model`` section, as a plain mapping.
        visual_seq_len: Optional override for the encoder sequence length.
        census_seq_len: The tokens to run a census of each layer kind of
            the language model at, and of its output layer
            (:mod:`hyper_parallel.auto_parallel._layer_census`), which the
            spec states as ``"activations"`` and ``"output_activations"``;
            0 runs none.  The
            census builds its layers from the checkpoint's config: a spec
            from ``config_overrides`` alone gets none, and an override of a
            model field does not reach it.
        census_spec: Whether the language model's fields, its layer stack
            and its op counts are the census's, measured on the layers
            Transformers builds of the checkpoint
            (:func:`~hyper_parallel.auto_parallel._spec_census.census_model_spec`),
            rather than read from its config's fields.  A spec from
            ``config_overrides`` alone is the overrides' either way.
        replacements: The module replacements the run installs
            (:func:`~hyper_parallel.auto_parallel._layer_census.replacement_specs`),
            which the census runs its layers with, so it measures what the
            trainer saves.

    Returns:
        A dict of canonical model fields, always carrying ``"name"``.

    Raises:
        ValueError: If neither a pretrained path nor overrides can supply
            the model dimensions.
    """
    explicit = _explicit_overrides(model_raw)
    model_path = _model_path(model_raw)

    if not model_path:
        if _describes_a_model(explicit):
            return _stated_spec(model_raw, explicit, census_seq_len)
        raise ValueError(
            "AutoModels train.yaml requires model.pretrained_model_name_or_path, "
            "model.config_path or model.config_overrides for Auto Parallel search"
        )

    # A recipe that builds its model through a factory is asked first: the
    # path beside _target_ is a directory that factory reads, not a checkpoint
    # to price, and where the two disagree the factory is what the run trains.
    model_config = _factory_config(model_raw)
    try:
        model_config = model_config if model_config is not None else _get_hf_config(model_raw)
    except (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        if _describes_a_model(explicit):
            logger.warning(
                "Transformers config resolution failed (%s); "
                "falling back to model.config_overrides", exc,
            )
            return _stated_spec(model_raw, explicit, census_seq_len)
        raise ValueError(
            f"cannot resolve model.pretrained_model_name_or_path '{model_path}'; "
            "install transformers, set model.config_overrides, or make the config "
            "available offline (warm the HF_HOME cache, or pass local_files_only)"
        ) from exc

    name = str(getattr(model_config, "model_type", None) or model_path)
    if census_spec:
        spec = census_model_spec(model_config, explicit.get("arch"), name)
    else:
        spec = _spec_from_aliases(_text_tower(model_config), _TEXT_FIELD_ALIASES)
        _derive_shared_experts(spec)
        _derive_dense_ffn_width(spec)
        spec["name"] = name
    # The dtype the checkpoint is saved in, by name, which the census's
    # fields do not state (I12).
    dtype = _first_attr(_text_tower(model_config), _TEXT_FIELD_ALIASES["torch_dtype"])
    if dtype is not None:
        spec["torch_dtype"] = _dtype_name(dtype)

    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        vision_spec = _spec_from_aliases(vision_config, _VISION_FIELD_ALIASES)
        vision_spec["name"] = f"{spec['name']}_vision"
        vision_spec["max_position_embeddings"] = _visual_seq_len(vision_spec, visual_seq_len)
        spec["vision"] = vision_spec

    spec.update(explicit)
    resolved = _validated(spec)
    if census_seq_len:
        resolved.update(_census(_text_tower(model_config), resolved["layers"], census_seq_len, replacements))
        resolved = ModelSpec.from_dict(resolved).to_dict()
    return resolved
