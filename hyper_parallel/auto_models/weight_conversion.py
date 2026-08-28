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

# Adapted from:
# https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/core_model_loading.py
# https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/conversion_mapping.py
# ============================================================================
"""Transformers 4.57-and-later weight-conversion compatibility."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable

import torch

try:
    from transformers.conversion_mapping import (
        get_model_conversion_mapping as _hf_get_model_conversion_mapping,
    )
    from transformers.core_model_loading import (
        ConversionOps as _HFConversionOps,
        Transpose as _HFTranspose,
        WeightConverter as _HFWeightConverter,
        WeightRenaming as _HFWeightRenaming,
    )
except ImportError:
    _hf_get_model_conversion_mapping = None
    _HFConversionOps = None
    _HFTranspose = None
    _HFWeightConverter = None
    _HFWeightRenaming = None

try:
    from transformers.core_model_loading import (
        revert_weight_conversion as _hf_revert_weight_conversion,
    )
except ImportError:
    _hf_revert_weight_conversion = None


def _has_scoped_weight_transforms() -> bool:
    """Return whether the installed Transformers exposes scoped transforms."""
    if _HFWeightConverter is None:
        return False
    return any(
        "scope_prefix" in getattr(base, "__slots__", ())
        for base in _HFWeightConverter.__mro__
    )


if _has_scoped_weight_transforms() and _hf_revert_weight_conversion is not None:
    ConversionOps = _HFConversionOps
    Transpose = _HFTranspose
    WeightConverter = _HFWeightConverter
    WeightRenaming = _HFWeightRenaming
    from transformers.core_model_loading import dot_natural_key, rename_source_key

    get_model_conversion_mapping = _hf_get_model_conversion_mapping
    revert_weight_conversion = _hf_revert_weight_conversion
else:
    _WeightConverterBase = _HFWeightConverter or object
    _WeightRenamingBase = _HFWeightRenaming or object

    def _process_target_pattern(pattern: str) -> tuple[str, str | None]:
        """Normalize one target pattern for reversible conversion."""
        pattern = pattern.removeprefix("^").removesuffix("$")
        pattern = re.sub(r"\(\?.+?\)?\)", "", pattern)
        pattern = pattern.replace(r"\.", ".")
        capturing_group = re.search(r"\(.+?\)", pattern)
        if capturing_group is None:
            return pattern, None
        captured = capturing_group.group(0)
        return pattern.replace(captured, r"\1", 1), captured


    def _process_source_pattern(source_pattern: str, target_pattern: str) -> str:
        """Preserve source anchors when building a reverse transform."""
        if target_pattern.startswith("^") and not source_pattern.startswith("^"):
            source_pattern = f"^{source_pattern}"
        if target_pattern.endswith("$") and not source_pattern.endswith("$"):
            source_pattern = f"{source_pattern}$"
        return source_pattern


    class ConversionOps(ABC):
        """Base operation matching the Transformers 5.x conversion contract."""

        @abstractmethod
        def convert(
            self,
            input_dict: dict[str, Any],
            source_patterns: list[str],
            target_patterns: list[str],
            **kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            """Convert collected checkpoint tensors."""

        @property
        def reverse_op(self) -> "ConversionOps":
            """Return the inverse operation."""
            raise NotImplementedError


    class Transpose(ConversionOps):
        """Transpose the first two configured dimensions."""

        def __init__(self, dim0: int = 0, dim1: int = 1, check_dims: bool = False):
            self.dim0 = dim0
            self.dim1 = dim1
            self.check_dims = check_dims

        @torch.no_grad()
        def convert(
            self,
            input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
            source_patterns: list[str],
            target_patterns: list[str],
            **kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            if len(input_dict) != 1:
                raise ValueError("Transpose requires exactly one collected source")
            if len(target_patterns) > 1:
                if len(source_patterns) != 1:
                    raise ValueError("Transpose cannot select an intermediate pattern")
                target_pattern = source_patterns[0]
            else:
                target_pattern = target_patterns[0]
            value = next(iter(input_dict.values()))
            tensor = value[0] if isinstance(value, list) else value
            if self.check_dims:
                model = kwargs["model"]
                expected = model.get_parameter(kwargs["full_layer_name"]).shape
                if tensor.shape == expected:
                    return {target_pattern: tensor}
            return {
                target_pattern: torch.transpose(
                    tensor, dim0=self.dim0, dim1=self.dim1
                ).contiguous()
            }

        @property
        def reverse_op(self) -> "ConversionOps":
            return Transpose(self.dim0, self.dim1, self.check_dims)


    class _WeightTransform:
        """Minimal Transformers 5.x weight-transform contract for 4.57."""

        def __init__(
            self,
            source_patterns: str | list[str],
            target_patterns: str | list[str],
        ) -> None:
            self.source_patterns = (
                [source_patterns] if isinstance(source_patterns, str) else list(source_patterns)
            )
            self.target_patterns = (
                [target_patterns] if isinstance(target_patterns, str) else list(target_patterns)
            )
            self._original_source_patterns = self.source_patterns.copy()
            self._original_target_patterns = self.target_patterns.copy()
            captured_groups = []
            for index, pattern in enumerate(self.target_patterns):
                self.target_patterns[index], captured = _process_target_pattern(pattern)
                if captured is not None:
                    captured_groups.append(captured)
            unique_captured_groups = set(captured_groups)
            if len(unique_captured_groups) > 1:
                raise ValueError(
                    "target_patterns must use at most one unique capturing group"
                )
            captured_group = (
                unique_captured_groups.pop() if unique_captured_groups else None
            )
            for index, pattern in enumerate(self.source_patterns):
                if r"\1" in pattern:
                    if captured_group is None:
                        raise ValueError(
                            "source pattern contains a backreference without a "
                            "target capturing group"
                        )
                    pattern = pattern.replace(r"\1", captured_group, 1)
                if len(self.source_patterns) == len(self.target_patterns):
                    pattern = _process_source_pattern(
                        pattern, self._original_target_patterns[index]
                    )
                self.source_patterns[index] = pattern
            self.collected_tensors: dict[
                str, list[torch.Tensor | Callable[[], torch.Tensor]]
            ] = defaultdict(list)
            self.scope_prefix: str | None = None
            self.base_model_prefix: str | None = None
            self._was_used = False
            branches = []
            for index, pattern in enumerate(self.source_patterns):
                pattern = pattern.replace(".*.", r"\..*\.")
                branches.append(f"(?P<g{index}>{pattern})")
            self._compiled_sources = re.compile("|".join(branches))

        def _key_to_match(self, source_key: str) -> tuple[str | None, str] | None:
            if self.scope_prefix is None:
                return None, source_key
            scope = f"{self.scope_prefix}." if self.scope_prefix else ""
            base = f"{self.base_model_prefix}." if self.base_model_prefix else ""
            for prefix in (base + scope, scope):
                if source_key.startswith(prefix):
                    return prefix, source_key.removeprefix(prefix)
            return None

        def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
            """Return the renamed key and the source pattern that matched it."""
            scoped_key = self._key_to_match(source_key)
            if scoped_key is None:
                return source_key, None
            prefix, key_to_match = scoped_key
            match = self._compiled_sources.search(key_to_match)
            if match is None:
                return source_key, None
            group_name = next(
                name for name, value in match.groupdict().items() if value is not None
            )
            pattern = self.source_patterns[int(group_name[1:])]
            replacement = self.target_patterns[0]
            if r"\1" in replacement:
                group_index = self._compiled_sources.groupindex[group_name] + 1
                replacement = replacement.replace(
                    r"\1", match.group(group_index), 1
                )
            self._was_used = True
            renamed = key_to_match.replace(match.group(0), replacement, 1)
            return f"{prefix or ''}{renamed}", pattern

        def add_tensor(
            self,
            target_key: str,
            source_key: str,
            source_pattern: str,
            loader: Callable[[], torch.Tensor],
        ) -> None:
            del target_key, source_key
            self.collected_tensors[source_pattern].append(loader)

        def _materialize_tensors(self) -> dict[str, list[torch.Tensor]]:
            materialized = {}
            for pattern, values in list(self.collected_tensors.items()):
                materialized[pattern] = [
                    value() if callable(value) else value for value in values
                ]
            self.collected_tensors.clear()
            return materialized

        def was_used(self) -> bool:
            """Return whether this transform matched a checkpoint key."""
            return self._was_used

        def reverse_transform(self) -> "_WeightTransform":
            """Build the transform used to restore original checkpoint names."""
            kwargs = {}
            if hasattr(self, "operations"):
                kwargs["operations"] = [
                    operation.reverse_op for operation in self.operations[::-1]
                ]
            transform = self.__class__(
                source_patterns=self._original_target_patterns,
                target_patterns=self._original_source_patterns,
                **kwargs,
            )
            transform.scope_prefix = self.scope_prefix
            transform.base_model_prefix = self.base_model_prefix
            return transform


    class WeightRenaming(_WeightTransform, _WeightRenamingBase):
        """Rename one checkpoint pattern without changing its tensor."""

        def convert(
            self,
            layer_name: str,
            model: Any = None,
            config: Any = None,
            hf_quantizer: Any = None,
            loading_info: Any = None,
        ) -> dict[str, torch.Tensor]:
            """Materialize and rename the collected checkpoint tensor."""
            del model, config, hf_quantizer, loading_info
            collected = self._materialize_tensors()
            tensor = collected[self.source_patterns[0]]
            target = self.target_patterns[0]
            if target not in layer_name:
                target = layer_name
            return {target: tensor}


    class WeightConverter(_WeightTransform, _WeightConverterBase):
        """Apply conversion operations to one checkpoint tensor group."""

        def __init__(
            self,
            source_patterns: str | list[str],
            target_patterns: str | list[str],
            operations: list[ConversionOps],
        ) -> None:
            super().__init__(source_patterns, target_patterns)
            if not operations:
                raise ValueError("WeightConverter requires at least one operation")
            self.operations = operations

        def convert(
            self,
            layer_name: str,
            model: Any = None,
            config: Any = None,
            hf_quantizer: Any = None,
            loading_info: Any = None,
        ) -> dict[str, torch.Tensor]:
            """Materialize the collected tensors and apply each operation."""
            collected: dict[str, Any] = self._materialize_tensors()
            for operation in self.operations:
                collected = operation.convert(
                    collected,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=(
                        loading_info.missing_keys if loading_info is not None else None
                    ),
                )
            del hf_quantizer
            full_name = layer_name.replace(".*.", ".0.")
            try:
                prefix, _, suffix = next(
                    full_name.partition(pattern)
                    for pattern in collected
                    if pattern in full_name
                )
            except StopIteration:
                return collected
            return {f"{prefix}{name}{suffix}": value for name, value in collected.items()}


    def rename_source_key(
        source_key: str,
        weight_renamings: list[WeightRenaming],
        weight_converters: list[WeightConverter],
        base_model_prefix: str | None = None,
        meta_state_dict: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Apply renamings, one converter, and base-prefix normalization."""
        renamed_key = source_key
        for renaming in weight_renamings:
            renamed_key, _ = renaming.rename_source_key(renamed_key)
        source_pattern = None
        for converter in weight_converters:
            renamed_key, source_pattern = converter.rename_source_key(renamed_key)
            if source_pattern is not None:
                break
        if base_model_prefix is not None and meta_state_dict is not None:
            prefix = f"{base_model_prefix}."
            without_prefix = renamed_key.removeprefix(prefix)
            if renamed_key.startswith(prefix) and without_prefix in meta_state_dict:
                renamed_key = without_prefix
            elif f"{prefix}{renamed_key}" in meta_state_dict:
                renamed_key = f"{prefix}{renamed_key}"
        return renamed_key, source_pattern


    def dot_natural_key(value: str) -> list[tuple[int, int | str]]:
        """Sort dotted tensor names with numeric path components numerically."""
        return [
            (0, int(part)) if part.isdigit() else (1, part)
            for part in value.split(".")
        ]


    def get_model_conversion_mapping(
        model: Any,
        key_mapping: dict[str, str] | None = None,
        hf_quantizer: Any = None,
        add_legacy: bool = True,
    ) -> list[WeightRenaming | WeightConverter]:
        """Normalize available Transformers mappings to scoped transforms."""
        if _hf_get_model_conversion_mapping is None:
            return [
                WeightRenaming(source_patterns=source, target_patterns=target)
                for source, target in (key_mapping or {}).items()
            ]
        transforms = _hf_get_model_conversion_mapping(
            model,
            key_mapping=key_mapping,
            hf_quantizer=hf_quantizer,
            add_legacy=add_legacy,
        )
        normalized = []
        for transform in transforms:
            if isinstance(transform, _HFWeightConverter):
                normalized.append(
                    WeightConverter(
                        source_patterns=transform.source_patterns,
                        target_patterns=transform.target_patterns,
                        operations=deepcopy(transform.operations),
                    )
                )
            elif isinstance(transform, _HFWeightRenaming):
                normalized.append(
                    WeightRenaming(
                        source_patterns=transform.source_patterns,
                        target_patterns=transform.target_patterns,
                    )
                )
            else:
                normalized.append(transform)
        return normalized


    def revert_weight_conversion(
        model: Any,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Restore checkpoint names and layouts used before weight conversion."""
        weight_conversions = getattr(model, "_weight_conversions", None)
        if weight_conversions is None:
            weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
        if not weight_conversions:
            return state_dict

        reverse_conversions = [
            conversion.reverse_transform() for conversion in weight_conversions
        ]
        renamings = [
            conversion
            for conversion in reverse_conversions
            if isinstance(conversion, WeightRenaming)
        ]
        converters = [
            conversion
            for conversion in reverse_conversions
            if isinstance(conversion, WeightConverter)
        ]
        conversion_mapping = {}
        for original_key, tensor in sorted(
            state_dict.items(), key=lambda item: dot_natural_key(item[0])
        ):
            renamed_key = original_key
            for renaming in renamings:
                renamed_key, _ = renaming.rename_source_key(renamed_key)

            source_pattern = None
            matched_converter = None
            for converter in converters:
                renamed_key, source_pattern = converter.rename_source_key(renamed_key)
                if source_pattern is not None:
                    matched_converter = converter
                    break

            if matched_converter is not None:
                mapping = conversion_mapping.setdefault(
                    renamed_key, deepcopy(matched_converter)
                )
            else:
                mapping = conversion_mapping.setdefault(
                    renamed_key, WeightRenaming(original_key, renamed_key)
                )
                source_pattern = original_key
            mapping.add_tensor(
                renamed_key, original_key, source_pattern, tensor
            )

        converted_state_dict = {}
        for first_param_name, reverse_conversion in conversion_mapping.items():
            realized = reverse_conversion.convert(
                first_param_name, model=model, config=model.config
            )
            for target_name, parameter in realized.items():
                converted_state_dict[target_name] = (
                    parameter[0] if isinstance(parameter, list) else parameter
                )
        return converted_state_dict


__all__ = [
    "ConversionOps",
    "Transpose",
    "WeightConverter",
    "WeightRenaming",
    "dot_natural_key",
    "get_model_conversion_mapping",
    "rename_source_key",
    "revert_weight_conversion",
]
