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
"""Hugging Face pretrained-weight backend for :class:`CheckpointerBase`.

Unlike :class:`~hyper_parallel.components.checkpoint.dcp_checkpointer.DistributedCheckpointer`,
which round-trips its own format, this backend reads a foreign one: a Transformers checkpoint whose
tensor names and layouts belong to the released model, not to the finalized distributed one. Keys are
therefore renamed and tensors converted on the way in, which is what
``load_groups.py`` exists for.

The payload this backend fills is ``{"model": <nn.Module>}`` --- a live module rather than a state
dict, because loading writes into the distributed parameters in place and must not replace them. It
adds a ``load_report`` entry naming what was loaded, which
``models/_transformers/checkpoint_loader.py::_finalize_model_loading`` consumes to initialize
whatever the checkpoint did not cover.

Only :meth:`HuggingFaceCheckpointer.load` is implemented; writing a Transformers-format checkpoint
back out still lives on the model.
"""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.checkpoint.base import CheckpointerBase
from hyper_parallel.components.checkpoint.huggingface_load_planner import load_hf_checkpoint
from hyper_parallel.components.checkpoint.load_groups import (
    CheckpointIndex,
    LoadGroup,
    LoadReport,
    SourceModelView,
    alias_names_by_target,
    base_weights_mapping,
    build_load_groups,
    build_load_targets,
    build_replacement_routes,
    convert_group,
    copy_into_target,
    make_tensor_loader,
    resolve_checkpoint_index,
    validate_load_result,
)
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    get_model_conversion_mapping,
)

logger = logging.getLogger(__name__)

# Picks the pretrained loader when HuggingFaceCheckpointer is not told which one to use.
HF_LOADER_ENV = "HYPER_PARALLEL_HF_LOADER"
_HF_LOADERS = ("legacy", "dcp")

# Keys of the payload :meth:`HuggingFaceCheckpointer.load` reads and fills. ``model`` holds the
# finalized module to load into; ``load_report`` is written back for the caller that finalizes it.
_MODEL_KEY = "model"
_LOAD_REPORT_KEY = "load_report"

WeightsMapping = List[Union[WeightRenaming, WeightConverter]]


def resolve_hf_loader(loader: Optional[str] = None) -> str:
    """
    Name the pretrained loader to use.

    Args:
        loader (str | None): ``"legacy"`` reads whole checkpoint tensors on every rank and shards them in
            memory. ``"dcp"`` plans the same conversions as distributed checkpoint reads, so that each
            rank reads only the regions of the checkpoint its shards need. Default None, which reads
            ``HYPER_PARALLEL_HF_LOADER`` and falls back to ``"legacy"``.

    Returns:
        str: ``"legacy"`` or ``"dcp"``.

    Raises:
        ValueError: If the loader named is neither.
    """
    choice = (loader or os.environ.get(HF_LOADER_ENV) or "legacy").strip().lower()
    if choice not in _HF_LOADERS:
        raise ValueError(
            f"Unknown pretrained loader {choice!r}; expected one of {', '.join(_HF_LOADERS)} "
            f"(set through the loader argument or {HF_LOADER_ENV})"
        )
    return choice


class HuggingFaceCheckpointer(CheckpointerBase):
    """Load complete Hugging Face weights into one finalized model.

    Both loaders selected by ``loader`` reach the same values under the same names --- they differ
    only in how much of the checkpoint each rank reads:

    * ``"legacy"`` --- every rank reads whole checkpoint tensors, converts them in host memory and
      keeps the slice its shards need.
    * ``"dcp"`` --- the conversions are compiled into checkpoint coordinates first, so each rank
      reads only the regions its own shards cover.

    :meth:`save` is not implemented: exporting a Transformers-format checkpoint stays on the model's
    own ``save_pretrained``.
    """

    def __init__(
        self,
        *,
        loader: Optional[str] = None,
        weights_mapping: Optional[WeightsMapping] = None,
    ) -> None:
        """Select the loader and the conversion rules :meth:`load` applies.

        Args:
            loader: Which loader reads the weights; see :func:`resolve_hf_loader`. Resolved here
                rather than per load, so an unusable value fails before any checkpoint is opened.
            weights_mapping: Rules renaming and converting checkpoint tensors. Default None, for the
                model's Transformers conversion mapping, resolved once the model is known.
        """
        self.loader = resolve_hf_loader(loader)
        self.weights_mapping = weights_mapping

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        global_step: int,
        save_async: bool = False,
    ) -> None:
        """Reject saving; see the class docstring."""
        raise NotImplementedError(
            "HuggingFaceCheckpointer only loads pretrained weights. Write a resumable checkpoint "
            "with DistributedCheckpointer, or export Transformers format through the model's own "
            "save_pretrained()."
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        strict_model: bool = True,
        extra_state_skeleton: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load the pretrained weights at ``path`` into ``state["model"]``.

        Args:
            path: A safetensors file, a checkpoint directory or a Hub repository id.
            state: Payload carrying ``model``, the finalized module to load into. Loading writes into
                its distributed parameters in place, so this is a live module and not a state dict.
            strict_model: Raise if a model tensor is left unloaded.
            extra_state_skeleton: Unsupported --- a pretrained checkpoint holds no training progress.

        Returns:
            The same ``state`` object, with ``load_report`` set to the :class:`LoadReport` naming what
            was loaded, what is missing and what the checkpoint holds beyond the model.

        Raises:
            ValueError: If ``path`` is empty or ``extra_state_skeleton`` is requested.
            TypeError: If ``state["model"]`` is not a module.
        """
        if extra_state_skeleton is not None:
            raise ValueError(
                "A pretrained Hugging Face checkpoint carries no training state; restore progress "
                "from a resumable checkpoint through DistributedCheckpointer instead."
            )
        if not path:
            raise ValueError("pretrained_path must be provided when load_base_model=True")
        model = state.get(_MODEL_KEY)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"HuggingFaceCheckpointer.load expects state[{_MODEL_KEY!r}] to be the finalized "
                f"nn.Module to load into, got {type(model).__name__}"
            )

        state[_LOAD_REPORT_KEY] = self._load_model(model, path, strict=strict_model)
        return state

    def _load_model(self, model: nn.Module, pretrained_path: str, *, strict: bool) -> LoadReport:
        """Dispatch one model to the configured loader and the conversions it needs."""
        if self.loader == "dcp":
            return load_hf_checkpoint(
                model, pretrained_path, weights_mapping=self.weights_mapping, strict=strict
            )

        weights_mapping = self.weights_mapping
        if weights_mapping is None:
            weights_mapping = get_model_conversion_mapping(
                model,
                key_mapping=None,
                hf_quantizer=None,
            )

        checkpoint_index = resolve_checkpoint_index(pretrained_path)
        targets = build_load_targets(model)
        replacement_mapping = getattr(
            model,
            "_hp_replacement_weight_conversions",
            None,
        )
        source_shapes = getattr(
            model,
            "_hp_checkpoint_source_shapes",
            None,
        )
        if replacement_mapping and source_shapes:
            return self._load_with_replacement_conversions(
                model,
                checkpoint_index,
                targets,
                weights_mapping,
                replacement_mapping,
                source_shapes,
                pretrained_path,
                strict,
            )
        return self._load_base_conversions(
            model,
            checkpoint_index,
            targets,
            weights_mapping,
            pretrained_path,
            strict,
        )

    def _load_base_conversions(
        self,
        model: nn.Module,
        checkpoint_index: CheckpointIndex,
        targets: Dict[str, torch.Tensor],
        weights_mapping: WeightsMapping,
        pretrained_path: str,
        strict: bool,
    ) -> LoadReport:
        """Load a model whose tensors the Transformers conversion rules reach directly."""
        groups, unexpected_keys, weight_mapping = build_load_groups(
            model,
            checkpoint_index.keys(),
            targets,
            weights_mapping=weights_mapping,
            make_loader=partial(make_tensor_loader, checkpoint_index),
        )
        aliases_by_target = alias_names_by_target(targets)
        loaded_keys = set()
        loaded_target_ids = set()

        for group in groups:
            converted = self._convert_group(group, model)
            for target_name, tensor in converted.items():
                target = targets.get(target_name)
                if target is None:
                    unexpected_keys += (target_name,)
                    continue
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                target_id = id(target)
                if target_id not in loaded_target_ids:
                    copy_into_target(target_name, tensor, target)
                    loaded_target_ids.add(target_id)
                loaded_keys.update(aliases_by_target[target_id])

        missing_keys = tuple(sorted(set(targets) - loaded_keys, key=dot_natural_key))
        unexpected_keys = tuple(sorted(set(unexpected_keys), key=dot_natural_key))
        self._validate_load_result(missing_keys, unexpected_keys, strict)
        used_conversions = [transform for transform in weight_mapping if transform.was_used()]
        model._weight_conversions = used_conversions  # pylint: disable=W0212
        report = LoadReport(
            loaded_keys=tuple(sorted(loaded_keys, key=dot_natural_key)),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        logger.info(
            "Loaded %d model tensors from %s",
            len(report.loaded_keys),
            pretrained_path,
        )
        return report

    def _load_with_replacement_conversions(
        self,
        model: nn.Module,
        checkpoint_index: CheckpointIndex,
        targets: Dict[str, torch.Tensor],
        weights_mapping: WeightsMapping,
        replacement_mapping: WeightsMapping,
        source_shapes: Dict[str, tuple],
        pretrained_path: str,
        strict: bool,
    ) -> LoadReport:
        """Normalize original weights before applying replacement conversions."""
        base_mapping = base_weights_mapping(weights_mapping, replacement_mapping)
        source_model = SourceModelView(model, source_shapes)
        base_groups, unexpected_keys, _ = build_load_groups(
            source_model,
            checkpoint_index.keys(),
            source_model.targets,
            weights_mapping=base_mapping,
            make_loader=partial(make_tensor_loader, checkpoint_index),
        )
        routes = build_replacement_routes(
            model,
            tuple(source_shapes),
            targets,
            replacement_mapping,
        )
        aliases_by_target = alias_names_by_target(targets)
        loaded_keys = set()
        loaded_target_ids = set()
        used_replacements = []

        def copy_converted(converted: Dict[str, torch.Tensor]) -> None:
            """Copy converted tensors into their finalized model targets."""
            nonlocal unexpected_keys
            for target_name, tensor in converted.items():
                target = targets.get(target_name)
                if target is None:
                    unexpected_keys += (target_name,)
                    continue
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                target_id = id(target)
                if target_id not in loaded_target_ids:
                    copy_into_target(target_name, tensor, target)
                    loaded_target_ids.add(target_id)
                loaded_keys.update(aliases_by_target[target_id])

        for base_group in base_groups:
            normalized = self._convert_group(base_group, source_model)
            for source_name, tensor in normalized.items():
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                route = routes.get(source_name)
                if route is None:
                    copy_converted({source_name: tensor})
                    continue
                state, target_name, source_pattern = route
                state.group.transform.add_tensor(
                    target_name,
                    source_name,
                    source_pattern,
                    lambda value=tensor: value,
                )
                state.received[source_pattern] += 1
                if not state.completed and state.received == state.expected:
                    copy_converted(self._convert_group(state.group, model))
                    state.completed = True
                    used_replacements.append(state.group.transform)

        missing_keys = tuple(sorted(set(targets) - loaded_keys, key=dot_natural_key))
        unexpected_keys = tuple(sorted(set(unexpected_keys), key=dot_natural_key))
        self._validate_load_result(missing_keys, unexpected_keys, strict)
        used_base = [transform for transform in base_mapping if transform.was_used()]
        model._hp_used_base_weight_conversions = used_base  # pylint: disable=protected-access
        model._hp_used_replacement_weight_conversions = (  # pylint: disable=protected-access
            used_replacements
        )
        model._weight_conversions = used_base + used_replacements  # pylint: disable=protected-access
        report = LoadReport(
            loaded_keys=tuple(sorted(loaded_keys, key=dot_natural_key)),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        logger.info("Loaded %d model tensors from %s", len(report.loaded_keys), pretrained_path)
        return report

    @staticmethod
    def _convert_group(
        group: LoadGroup,
        model: Union[nn.Module, SourceModelView],
    ) -> Dict[str, torch.Tensor]:
        """Convert all checkpoint tensors belonging to one load group."""
        return convert_group(group, model)

    @staticmethod
    def _validate_load_result(
        missing_keys: tuple,
        unexpected_keys: tuple,
        strict: bool,
    ) -> None:
        """Validate missing keys and report ignored checkpoint tensors."""
        validate_load_result(missing_keys, unexpected_keys, strict)


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: Union[str, Path],
    *,
    strict: bool = True,
) -> LoadReport:
    """Load pretrained weights into ``model`` and return the report, without a payload dict."""
    state = {_MODEL_KEY: model}
    HuggingFaceCheckpointer().load(str(pretrained_path), state, strict_model=strict)
    return state[_LOAD_REPORT_KEY]


__all__ = [
    "HF_LOADER_ENV",
    "HuggingFaceCheckpointer",
    "load_pretrained_weights",
    "resolve_hf_loader",
]
