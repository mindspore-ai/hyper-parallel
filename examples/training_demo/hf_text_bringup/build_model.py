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
"""Generic HF-native ``from_config`` builder for the text-training demo.

Works for any architecture the installed ``transformers`` can construct:
the Trainer passes its injected build options as keyword arguments, the
function reads only the Hugging Face configuration from ``config_path``
(no checkpoint tensor is loaded), and construction goes through
``HyperAutoModelForCausalLM`` so a registered family adapter applies when
one exists -- an unregistered HF-native family builds exactly the same way
with no adapter at all. The released depth is kept by default;
``num_hidden_layers`` crops it for fast debug runs, and
``expected_model_type`` optionally pins the directory to the family the
recipe was written for.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import CompileConfig


@dataclass(frozen=True)
class HfCausalLmOptions:
    """Model construction options for the generic text-training example.

    Args:
        expected_model_type: When set, fail fast unless the directory's
            ``config.json`` carries this ``model_type``.
        num_hidden_layers: Decoder layers to keep; ``None`` keeps the
            released depth.
        load_pretrained: Load the released checkpoint weights instead of
            random initialization. Each rank materializes the full model on
            the host before sharding, so set ``training.init_device: cpu``
            and budget host memory accordingly; required for cross-layout
            numerical-equivalence runs, where every layout must start from
            identical weights.
        local_files_only: Disable implicit Hub downloads when true.
        torch_dtype: Model parameter dtype accepted by HyperAutoModel.
        attn_implementation: Hugging Face attention implementation name.
            ``eager`` is the bring-up default until a fused path is pinned
            value-for-value against it on device.
        validate_placement: Enable HyperParallel placement validation.
        distributed_setup: Trainer-provided distributed topology.
        peft_config: Optional Trainer-provided PEFT configuration.
        compile_config: Optional Trainer-provided compile configuration.
        activation_checkpoint: Activation checkpoint mode.
        activation_checkpoint_selection: Trainer-injected selection of
            checkpoint-safe submodules; forwarded to the build pipeline.
        swap_inputs: Trainer-injected activation-swap input flag.
        allow_cache_entry_mutation: Trainer-injected selective-recompute
            cache-mutation permission (``ActivationCheckpointConfig`` field).
        model_init_dtype: Trainer-injected parameter initialization dtype.
        activation_swap: Activation swap mode.
        codegen: Select the codegen ``gen`` backend for this build.
        codegen_config: Trainer config carrying the codegen settings.
    """

    expected_model_type: str | None = None
    num_hidden_layers: int | None = None
    load_pretrained: bool = False
    local_files_only: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    validate_placement: bool = False
    distributed_setup: DistributedSetup | None = None
    peft_config: Any | None = None
    compile_config: CompileConfig | dict[str, Any] | None = None
    activation_checkpoint: str | None = None
    activation_checkpoint_selection: Any | None = None
    swap_inputs: bool = False
    allow_cache_entry_mutation: bool = False
    model_init_dtype: str | None = None
    activation_swap: str = "none"
    codegen: bool = False
    codegen_config: Any | None = None


def _resolve_build_options(
        legacy_args: tuple[Any, ...],
        legacy_kwargs: dict[str, Any],
        options: HfCausalLmOptions | None,
) -> HfCausalLmOptions:
    """Bind the original build arguments while supporting grouped options."""
    option_names = tuple(field.name for field in fields(HfCausalLmOptions))
    if len(legacy_args) > len(option_names):
        raise TypeError(
            "build_hf_causal_lm() takes from 1 to "
            f"{len(option_names) + 1} positional arguments "
            f"but {len(legacy_args) + 1} were given"
        )

    positional_names = option_names[:len(legacy_args)]
    duplicated_name = next(
        (name for name in positional_names if name in legacy_kwargs),
        None,
    )
    if duplicated_name is not None:
        raise TypeError(
            "build_hf_causal_lm() got multiple values for argument "
            f"'{duplicated_name}'"
        )

    unexpected_name = next(
        (name for name in legacy_kwargs if name not in option_names),
        None,
    )
    if unexpected_name is not None:
        raise TypeError(
            "build_hf_causal_lm() got an unexpected keyword argument "
            f"'{unexpected_name}'"
        )

    option_updates = dict(zip(positional_names, legacy_args))
    option_updates.update(legacy_kwargs)
    return replace(options or HfCausalLmOptions(), **option_updates)


def _selection_requests_regions(selection: Any) -> bool:
    """Say whether a recompute selection actually asks for anything.

    The Trainer injects its selection object on *every* build, so the object
    being present means nothing: its default form (``source="default"`` with
    no layer selector) is "nothing requested". Treating the object as a
    request made ``load_pretrained`` unreachable through the Trainer, which
    is how this function came to exist. A plain container is judged by
    emptiness instead, so both call shapes work.
    """
    if selection is None:
        return False
    if hasattr(selection, "source"):
        return (getattr(selection, "source") != "default"
                or getattr(selection, "layer_count", None) is not None
                or getattr(selection, "layer_indices", None) is not None)
    return bool(selection)


def build_hf_causal_lm(
        config_path: str,
        *legacy_args: Any,
        options: HfCausalLmOptions | None = None,
        **legacy_kwargs: Any,
) -> PreTrainedModel:
    """Create any HF-native causal-LM with random weights from its config.

    The function intentionally calls ``from_config`` instead of
    ``from_pretrained``: it reads only the Hugging Face configuration from
    ``config_path`` and no checkpoint tensor is loaded, matching the
    repository's smoke and acceptance convention.

    Args:
        config_path: Local Hugging Face model directory.
        *legacy_args: Existing positional build options in their original order.
        options: Grouped model construction options.
        **legacy_kwargs: Existing keyword build options, which override
            ``options``.

    Returns:
        A parallelized, randomly initialized model.
    """
    options = _resolve_build_options(legacy_args, legacy_kwargs, options)

    config = AutoConfig.from_pretrained(
        config_path,
        local_files_only=options.local_files_only,
        trust_remote_code=False,
    )
    if options.expected_model_type is not None and (
            getattr(config, "model_type", None) != options.expected_model_type):
        raise ValueError(
            f"config_path must carry model_type={options.expected_model_type!r}; "
            f"got {getattr(config, 'model_type', None)!r}"
        )

    if options.num_hidden_layers is not None:
        if options.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        config.num_hidden_layers = options.num_hidden_layers
    config.use_cache = False

    if options.load_pretrained:
        if _selection_requests_regions(options.activation_checkpoint_selection):
            raise ValueError(
                "activation_checkpoint_selection is not supported together "
                "with load_pretrained yet")
        overrides = {"use_cache": False}
        if options.num_hidden_layers is not None:
            overrides["num_hidden_layers"] = options.num_hidden_layers
        return HyperAutoModelForCausalLM.from_pretrained(
            config_path,
            local_files_only=options.local_files_only,
            config_overrides=overrides,
            distributed_setup=options.distributed_setup,
            peft_config=options.peft_config,
            torch_dtype=options.torch_dtype,
            attn_implementation=options.attn_implementation,
            validate_placement=options.validate_placement,
            compile_config=options.compile_config,
            activation_checkpoint=options.activation_checkpoint,
            swap_inputs=options.swap_inputs,
            allow_cache_entry_mutation=options.allow_cache_entry_mutation,
            activation_swap=options.activation_swap,
            model_init_dtype=options.model_init_dtype,
            codegen=options.codegen,
            codegen_config=options.codegen_config,
        )

    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=options.distributed_setup,
        peft_config=options.peft_config,
        torch_dtype=options.torch_dtype,
        attn_implementation=options.attn_implementation,
        validate_placement=options.validate_placement,
        compile_config=options.compile_config,
        activation_checkpoint=options.activation_checkpoint,
        activation_checkpoint_selection=options.activation_checkpoint_selection,
        swap_inputs=options.swap_inputs,
        allow_cache_entry_mutation=options.allow_cache_entry_mutation,
        activation_swap=options.activation_swap,
        model_init_dtype=options.model_init_dtype,
        codegen=options.codegen,
        codegen_config=options.codegen_config,
    )
