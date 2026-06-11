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
"""Training configuration schema — strict three-tier (model/data/train).

Top-level keys are exactly ``model``, ``data``, ``train`` and nothing else.
arguments schema.

YAML shape::

    model:
      name: qwen3_5
      weights_path: /path/to/weights
    data:
      type: hf_datasets
      train_path: /path/to/data
    train:
      max_steps: 100
      micro_batch_size: 1
      global_batch_size: 8
      seed: 42
      init_device: meta
      optimizer:
        type: adamw
        lr: 1.0e-4
      accelerator:
        dp_shard: 8
        tp: 1
      mixed_precision:
        enabled: true
        param_dtype: bfloat16
      checkpoint:
        output_dir: outputs/run1
      ...
"""
import argparse
import difflib
import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_args, get_origin

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BOOL_TRUE_STRINGS = frozenset(("true", "yes", "y", "on", "1", "t"))
_BOOL_FALSE_STRINGS = frozenset(("false", "no", "n", "off", "0", "f"))

# ============================================================================
# model:
# ============================================================================


@dataclass
class ModelConfig:
    """``model.*`` — model identity, weights, and architecture overrides.

    Only universal transformer fields are typed here. Anything model-
    specific (mRoPE section split, MoE expert geometry, linear-attention
    head counts, ``layer_types`` ...) goes through ``config_overrides`` —
    a free-form ``dict`` that is merged into the underlying model
    constructor by the model's ``build_model_fn``.
    """
    name: str = "qwen3_5"
    weights_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    freeze_modules: Optional[list] = None
    tp_plan: Optional[dict] = None
    cp_modules: Optional[list] = None
    ep_modules: Optional[list] = None
    # Universal transformer architecture overrides.
    num_hidden_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    # Free-form per-model overrides handed to ``build_model_fn``.
    config_overrides: Optional[dict] = None

# ============================================================================
# data:
# ============================================================================


@dataclass
class DataConfig:
    """``data.*`` — dataset, tokenizer/processor, sampler, batch shape.

    - ``streaming``: only ``False`` supported today.
    - ``num_workers``: keep ≥ 2 for real datasets.
    - ``shuffle``: when ``False``, sampler reads samples in dataset order.
    """
    type: str = "dummy"
    train_path: Optional[str] = None
    subset: Optional[str] = None
    max_seq_len: int = 2048
    text_key: str = "text"
    train_size: Optional[int] = None
    # multimodal / VL (synthetic vl_dummy path)
    template: str = "empty"
    image_key: str = "image"
    messages_key: str = "messages"
    image_token_id: int = 151655
    vl_grid_t: int = 2
    vl_grid_h: int = 2
    vl_grid_w: int = 2
    # loader perf
    streaming: bool = False
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = True
    shuffle: bool = True

# ============================================================================
# train.* — sub-configs
# ============================================================================


@dataclass
class AcceleratorConfig:
    """``train.accelerator.*`` — parallelism topology.

    Two ways to express data parallelism:

    - **Legacy single field** (back-compat): ``dp`` only — maps to
      ``dp_shard`` for FSDP.
    - ** split**: ``dp_replicate`` (HSDP outer) and
      ``dp_shard`` (FSDP inner). Pass ``dp_shard=-1`` to auto-fill from
      ``world_size / (dp_replicate * cp * tp * pp)``.

    For MoE: ``etp`` controls expert TP. Must equal ``tp`` or ``1``
    ( rule).
    """
    dp: Optional[int] = None
    dp_replicate: int = 1
    dp_shard: Optional[int] = None
    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    etp: int = 1
    zero_stage: int = 0
    reshard_after_forward: bool = True
    async_cp: bool = False
    ulysses_degree: Optional[int] = None
    # Bucketed reduce-scatter: single fused RS per FSDP unit, stable fp32
    # reduction order across runs.
    comm_fusion: bool = True


@dataclass
class MixedPrecisionConfig:
    """``train.mixed_precision.*`` — FSDP2 mp_policy fields.

    ``output_dtype`` controls forward-output dtype at FSDP wrap boundaries.
    Set this to ``"bfloat16"`` to keep the cross-FSDP-unit forward output
    in bf16 (matches the typical "uniform low-precision forward" mp_policy
    setup); leave ``None`` to inherit from ``param_dtype``.
    """
    enabled: bool = False
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    output_dtype: Optional[str] = None


@dataclass
class GradientCheckpointingConfig:
    """``train.gradient_checkpointing.*`` — activation recomputation.

    . Modes: ``"off"``, ``"full"``, or ``"selective"``.
    """
    activation_checkpoint: str = "off"


@dataclass
class OptimizerConfig:
    """``train.optimizer.*`` — optimizer + LR schedule + grad clip.

    ``loss_aggregation``: how the per-micro-batch loss is scaled before
    backward. ``"token_weighted"`` divides the summed loss by the global
    valid-token count; ``"rank_average"`` averages per-rank micro-batch
    means and is preferred when batches have variable valid-token counts
    across ranks.
    """
    type: str = "adamw"
    lr: float = 1e-4
    lr_min: float = 1e-5
    lr_decay_style: str = "cosine"
    lr_warmup_ratio: float = 0.1
    loss_aggregation: str = "token_weighted"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bsz_warmup_ratio: float = 0.0
    eps: float = 1e-8
    betas: tuple = (0.9, 0.999)


@dataclass
class CheckpointConfig:
    """``train.checkpoint.*`` — DCP save / load + HF export."""
    output_dir: str = "outputs"
    save_steps: int = 500
    save_hf_weights: bool = True
    load_path: Optional[str] = None
    save_async: bool = False


@dataclass
class LoggingConfig:
    """``train.logging.*`` — console / metric output (consumed by LoggingCallback)."""
    report_to: str = "none"
    report_global_loss: bool = False
    log_steps: int = 10
    report_throughput: bool = True
    model_flops_per_token: Optional[int] = None
    peak_tflops: Optional[float] = None  # e.g. 312.0 for A100 bf16


@dataclass
class TensorBoardConfig:
    """``train.tensorboard.*`` — TB SummaryWriter on rank 0."""
    enabled: bool = False
    output_dir: str = "tb_traces"
    log_steps: int = 1


@dataclass
class WandbConfig:
    """``train.wandb.*`` — W&B run logging on rank 0."""
    enabled: bool = False
    project: str = "hyper-parallel"
    run_name: Optional[str] = None
    log_steps: int = 1


@dataclass
class ProfileConfig:
    """``train.profile.*`` — torch.profiler schedule ().

    Schedule semantics: wait → warmup → active."""
    enabled: bool = False
    output_dir: str = "profiler_traces"
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3


@dataclass
class MemoryMonitorConfig:
    """``train.memory_monitor.*`` — periodic device-memory snapshot."""
    enabled: bool = False
    log_steps: int = 50
    reset_peak_each_step: bool = False


@dataclass
class MoEMonitorConfig:
    """``train.moe_monitor.*`` — MoE routing / load-balance monitor.

    When enabled, :class:`~hyper_parallel.core.moe_utils.MoEMonitorCallback`
    automatically syncs ``tokens_per_expert`` across distributed ranks and
    updates ``expert_bias`` after each optimizer step.  The mean ``aux_loss``
    across MoE layers is exposed via ``last_mean_aux_loss`` so that
    :class:`LoggingCallback` can print it alongside the main training loss.

    DP/TP+SP/CP group information is automatically obtained from the trainer's
    device mesh — no manual group configuration needed.

    Args:
        enabled: Whether to activate the MoE monitor callback.
        lr: Step size for expert bias updates. Defaults to ``1e-3``.
        num_recomputations: Number of forward executions per optimizer step.
            Default ``1``. Set to ``2`` when activation checkpoint is enabled.
    """
    enabled: bool = False
    lr: float = 1e-3
    num_recomputations: int = 1


@dataclass
class EvalConfig:
    """``train.eval.*`` — eval cadence + dataset."""
    eval_steps: int = 0
    eval_dataset: Optional[str] = None


@dataclass
class DebugConfig:
    """``train.debug.*`` — reproducibility and numerical-stability knobs.

    All flags here are production-safe; they tune determinism (CI / paper
    reproducibility), guard against numerical blow-ups, and bound memory
    growth in long runs.
    """
    deterministic: bool = False
    deterministic_warn_only: bool = False
    check_nan_inf: bool = False
    gc_steps: int = 0

# ============================================================================
# train: (top of the train section, holds the sub-configs)
# ============================================================================


@dataclass
class TrainConfig:
    """``train.*`` — full training-section config.

    Flat fields cover the basic loop knobs (steps, batch shape, init device,
    seed, comm backend); nested sub-configs cover everything else.
    """
    # Loop shape
    max_steps: int = 100
    num_train_epochs: int = 1
    global_batch_size: int = 8
    micro_batch_size: int = 1
    seed: int = 42

    # Runtime / device
    backend: str = "torch"
    init_device: str = "meta"
    comm_backend: Optional[str] = None
    local_rank: int = 0  # set from LOCAL_RANK env by parser

    # Sub-configs
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    memory_monitor: MemoryMonitorConfig = field(default_factory=MemoryMonitorConfig)
    moe_monitor: MoEMonitorConfig = field(default_factory=MoEMonitorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

# ============================================================================
# Top-level: model / data / train (and only these three)
# ============================================================================


@dataclass
class HyperTrainerConfig:
    """Top-level config — strict three-tier ().

    Allowed top-level keys: ``model``, ``data``, ``train``. Anything else in
    the YAML is rejected by the parser with a typo-suggestion message.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Computed (no user input)
    train_steps: int = 0

    def __post_init__(self):
        self.train_steps = self.train.max_steps

# ==============================================================================
# CLI / YAML parser
# ==============================================================================
# Configuration parser: YAML file + CLI dot-path overrides.
#
# Supports:
# - Unknown YAML/CLI keys emit a warning with difflib closest-match suggestions.
# - Bool fields accept string aliases: ``true/yes/y/on/1/t`` -> ``True``,
#   ``false/no/n/off/0/f`` -> ``False``. Only applied when the dataclass
#   field type resolves to ``bool`` or ``Optional[bool]`` to avoid ambiguity.


def _string_to_bool(value: Any) -> bool:
    """Convert common string representations of booleans to ``bool``.

    Accepts: ``true/yes/y/on/1/t`` → ``True``,
    ``false/no/n/off/0/f`` → ``False``.

    Args:
        value: A string or bool value.

    Returns:
        The corresponding ``bool``.

    Raises:
        ValueError: When the string cannot be mapped to a bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in _BOOL_TRUE_STRINGS:
            return True
        if lower in _BOOL_FALSE_STRINGS:
            return False
    raise ValueError(
        f"Cannot convert {value!r} to bool. "
        "Expected one of: true/false/yes/no/y/n/on/off/1/0/t/f"
    )


def _resolve_field_type(cls: Type, dot_path: str) -> Optional[Type]:
    """Walk a dataclass hierarchy to find the resolved type of a dot-path field.

    Args:
        cls: Root dataclass class.
        dot_path: Dot-separated field path, e.g. ``"debug.deterministic"``.

    Returns:
        The resolved Python type, or ``None`` if the path cannot be resolved.
    """
    parts = dot_path.split(".")
    current_cls = cls
    for part in parts:
        if not is_dataclass(current_cls):
            return None
        found = None
        for f in fields(current_cls):
            if f.name == part:
                found = f
                break
        if found is None:
            return None
        field_type = found.type
        # Unwrap Optional[X] → X
        origin = get_origin(field_type)
        if origin is Union:
            unwrapped = [a for a in get_args(field_type) if a is not type(None)]
            field_type = unwrapped[0] if len(unwrapped) == 1 else field_type
        current_cls = field_type
    return current_cls


def _coerce_cli_value(raw: str, dot_path: str, root_class: Type) -> Any:
    """Parse a CLI string value, coercing to the correct type for the field.

    Bool fields accept an extended string set.  For all other
    fields the existing int → float → str heuristic is used.

    Args:
        raw: Raw string from the CLI.
        dot_path: Dot-separated field path used for type lookup.
        root_class: Root dataclass class for type resolution.

    Returns:
        Coerced value.
    """
    field_type = _resolve_field_type(root_class, dot_path)
    if field_type is bool:
        try:
            return _string_to_bool(raw)
        except ValueError:
            pass  # fall through to heuristic below
    # Existing heuristic: int → float → bool-string → str
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update source dict with overrides dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            _deep_update(source[key], value)
        else:
            source[key] = value
    return source

_ALLOWED_TOP_LEVEL_KEYS = frozenset({"model", "data", "train"})


def _validate_top_level(config: Dict[str, Any]) -> None:
    """Reject any top-level key other than ``model`` / ``data`` / ``train``.

    Strict three-tier YAML shape. Any flat-style legacy key
    (``parallel``, ``optim``, ``mixed_precision``, ``runtime``, ``debug`` ...)
    must be moved under ``train.*`` — see schema.py for the canonical layout.

    Raises:
        ValueError: With migration hints when forbidden top-level keys are
            present in the YAML.
    """
    forbidden = sorted(set(config) - _ALLOWED_TOP_LEVEL_KEYS)
    if not forbidden:
        return

    legacy_to_train_path = {
        "parallel": "train.accelerator",
        "optim": "train.optimizer",
        "mixed_precision": "train.mixed_precision",
        "memory": "train.gradient_checkpointing",
        "checkpoint": "train.checkpoint",
        "logging": "train.logging",
        "tensorboard": "train.tensorboard",
        "wandb": "train.wandb",
        "profiler": "train.profile",
        "memory_monitor": "train.memory_monitor",
        "moe_monitor": "train.moe_monitor",
        "eval": "train.eval",
        "runtime": "train (flatten init_device / backend / comm_backend)",
        "debug": "train.debug",
        "seed": "train.seed",
    }
    hints = []
    for key in forbidden:
        new_path = legacy_to_train_path.get(key)
        if new_path:
            hints.append(f"  - top-level '{key}:' → move under {new_path}")
        else:
            hints.append(f"  - top-level '{key}:' is not allowed")
    raise ValueError(
        "Forbidden top-level YAML keys: %s. The schema is strict three-tier "
        "(model / data / train) — see config/schema.py. Migrate as follows:\n%s"
        % (forbidden, "\n".join(hints))
    )


def _instantiate_recursive(cls: Type[T], config_dict: Dict[str, Any]) -> T:
    """Recursively convert a dict into nested dataclass instances.

    Unknown keys in ``config_dict`` that have no corresponding field on
    ``cls`` emit a ``logger.warning`` with a closest-match suggestion from
    ``difflib``, helping users catch typos in YAML configs.
    """
    if not is_dataclass(cls):
        return config_dict

    known = {f.name for f in fields(cls)}
    unknown = set(config_dict) - known
    for name in sorted(unknown):
        matches = difflib.get_close_matches(name, known, n=1)
        suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
        logger.warning(
            "Unknown config key '%s' for %s ignored.%s",
            name, cls.__name__, suggestion,
        )

    field_values = {}
    for field_info in fields(cls):
        if field_info.name not in config_dict:
            continue
        raw_value = config_dict[field_info.name]
        field_type = field_info.type

        # Unwrap Optional[X] → X
        origin = get_origin(field_type)
        if origin is Union:
            unwrapped = [a for a in get_args(field_type) if a is not type(None)]
            if len(unwrapped) == 1:
                field_type = unwrapped[0]

        if is_dataclass(field_type) and isinstance(raw_value, dict):
            field_values[field_info.name] = _instantiate_recursive(field_type, raw_value)
        elif field_type is bool and isinstance(raw_value, str):
            field_values[field_info.name] = _string_to_bool(raw_value)
        else:
            field_values[field_info.name] = raw_value

    return cls(**field_values)


def parse_args(root_class: Type[T]) -> T:
    """Parse training config from YAML file + CLI overrides.

    Usage::

        args = parse_args(HyperTrainerConfig)

    The first positional argument is the YAML config file path.
    CLI arguments use dot-path notation under the strict three-tier schema:
    ``--train.accelerator.dp_shard=8 --train.optimizer.lr=3e-4``

    Bool fields accept extended string aliases (``yes/no/on/off/y/n/t/f/1/0``).
    Unknown YAML keys emit a warning with a closest-match suggestion.

    Args:
        root_class: The root config dataclass type.

    Returns:
        An instance of root_class populated from YAML + CLI.
    """
    parser = argparse.ArgumentParser(description="HyperParallel Trainer")
    parser.add_argument("config_file", nargs="?", help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    # Load YAML
    final_config: dict = {}
    if args.config_file:
        if not os.path.isfile(args.config_file):
            logger.warning(
                "Config file not found: %s (cwd=%s). Using all defaults.",
                args.config_file, os.getcwd(),
            )
        else:
            with open(args.config_file, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    final_config = yaml_config

    # Parse CLI dot-path overrides: --train.accelerator.dp=8 → nested dict
    cli_config: dict = {}
    for item in remaining:
        if item.startswith("--") and "=" in item:
            dot_key, raw_value = item[2:].split("=", 1)
            coerced = _coerce_cli_value(raw_value, dot_key, root_class)
            keys = dot_key.split(".")
            current = cli_config
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = coerced

    # CLI overrides YAML
    final_config = _deep_update(final_config, cli_config)

    # Strict three-tier validation — only model / data / train allowed.
    _validate_top_level(final_config)

    # local_rank from environment (torchrun sets it).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    final_config.setdefault("train", {})["local_rank"] = local_rank

    return _instantiate_recursive(root_class, final_config)
