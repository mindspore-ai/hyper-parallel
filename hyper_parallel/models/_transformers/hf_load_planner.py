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
"""Hugging Face checkpoints loaded into a finalized model through distributed checkpoint reads.

:class:`HFLoadPlanner` loads what :meth:`CheckpointManager.load_checkpoint` loads, with the same
renaming rules, converters and replacement conversions. The difference is in how the values get
there. Every conversion is run once on :class:`RegionTensor` inputs while the load is planned, which
turns it into the regions of checkpoint tensors each model tensor is copied from. Each rank then reads
only the regions its own shards need, and replicated shards are read once and broadcast. A conversion
that cannot be traced that way reads its checkpoint tensors whole and runs on them for real, on every
rank, as the legacy loader runs every conversion.

Rules come from the model's Transformers conversion mapping unless they are passed in, and a subclass
can override how the mapping is built, how checkpoint keys are named before the rules see them, and
which converter a matched key goes to. See :class:`HFLoadPlanner`.
"""
import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import DTensor
from hyper_parallel.components.checkpoint.region_tensor import RegionTensor, materialize
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    get_model_conversion_mapping,
)
from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.core.distributed_checkpoint.hf_storage import HuggingFaceStorageReader
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata, TensorStorageMetadata
from hyper_parallel.core.distributed_checkpoint.remap_planner import DeferredRead, RemapBlock, RemapLoadPlanner
from hyper_parallel.core.distributed_checkpoint.utils import all_gather_object, str_to_dtype
from hyper_parallel.models._transformers.checkpoint_conversion import (
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
    first_tensor,
    local_target_tensor,
    reject_partial_layout,
    resolve_checkpoint_location,
    select_unique_converter,
    sorted_checkpoint_keys,
    target_layout,
    validate_load_result,
)

logger = logging.getLogger(__name__)

Transform = WeightRenaming | WeightConverter


def _identity(value: Any) -> Any:
    """``value``, handed to a transform as a loader it calls when it converts."""
    return value


@dataclass(eq=False)
class _Conversion:
    """
    One conversion of the load: a transform, and the values it collects in the order it collects them.

    Attributes:
        template (Transform): The transform, with nothing collected. Every run converts a copy of it.
        first_target_name (str): Name the conversion is known by, as its group is.
        model (Any): The model, or the view of it, handed to the conversion operations.
        inputs (list[tuple[str, Any]]): ``(source pattern, value)`` of every collected tensor, where a
            value is a :class:`RegionTensor` or a :class:`_Deferred` output of another conversion.
    """

    template: Transform
    first_target_name: str
    model: Any
    inputs: list[tuple[str, Any]]

    def sources(self) -> frozenset[str]:
        """Names of the checkpoint tensors the conversion reads, through its inputs."""
        return frozenset().union(*(value.sources() for _, value in self.inputs))

    def run(self, values: Sequence[Any]) -> dict[str, Any]:
        """
        Convert ``values`` in place of the inputs, on a fresh copy of the transform.

        Args:
            values (Sequence[Any]): One tensor per input: regions, meta tensors or real tensors.

        Returns:
            dict[str, Any]: What the transform returns, by model tensor name.
        """
        transform = deepcopy(self.template)
        for (pattern, _), value in zip(self.inputs, values):
            transform.add_tensor(self.first_target_name, self.first_target_name, pattern, partial(_identity, value))
        return convert_group(LoadGroup(self.first_target_name, transform), self.model)


@dataclass(frozen=True, eq=False)
class _Deferred:
    """
    An output of a conversion that could not be traced, computed for real once its sources are read.

    Attributes:
        shape (tuple[int, ...]): Shape the conversion gives the output, as run on meta tensors.
        dtype (torch.dtype): Dtype it gives the output.
        conversion (_Conversion): The conversion.
        name (str): Name of the output among those of the conversion.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    conversion: _Conversion
    name: str

    def sources(self) -> frozenset[str]:
        """Names of the checkpoint tensors the output is computed from."""
        return self.conversion.sources()


def _meta_value(value: Any) -> torch.Tensor:
    """A meta tensor of the shape and dtype of a planned value."""
    return torch.empty(tuple(value.shape), dtype=value.dtype, device="meta")


def _real_value(value: Any, fetched: Mapping[str, torch.Tensor], cache: dict[int, dict[str, Any]]) -> Any:
    """The values of a planned value, out of the whole checkpoint tensors it reads."""
    if isinstance(value, RegionTensor):
        return materialize(value, fetched)
    return first_tensor(_run_real(value.conversion, fetched, cache)[value.name])


def _run_real(
        conversion: _Conversion, fetched: Mapping[str, torch.Tensor], cache: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Run ``conversion`` on real tensors, once for however many of its outputs are asked for."""
    key = id(conversion)
    if key not in cache:
        cache[key] = conversion.run([_real_value(value, fetched, cache) for _, value in conversion.inputs])
    return cache[key]


def _complete_conversion(
        conversion: _Conversion,
        claims: Sequence[tuple[str, str]],
        targets: Mapping[str, torch.Tensor],
        fetched: Mapping[str, torch.Tensor],
) -> None:
    """
    Run a conversion that could not be traced, and copy the outputs it was given into their targets.

    Args:
        conversion (_Conversion): The conversion.
        claims (Sequence[tuple[str, str]]): ``(output name, model tensor name)`` of every output to copy.
        targets (Mapping[str, torch.Tensor]): Model tensors by name.
        fetched (Mapping[str, torch.Tensor]): The whole checkpoint tensors the conversion reads.
    """
    outputs = _run_real(conversion, fetched, {})
    for output_name, target_name in claims:
        copy_into_target(target_name, first_tensor(outputs[output_name]), targets[target_name])


def _dcp_view(target: torch.Tensor) -> Any:
    """
    What the distributed checkpoint load fills for a model tensor.

    A DTensor is filled as it is. A parameter that holds its local shard as a plain tensor beside its
    layout, which fully sharded parameters do between materialization and their first use, is filled
    through a DTensor over the same storage, so that its shard is found in the whole tensor.
    """
    if isinstance(target, DTensor):
        return target
    layout = getattr(target, "_sharding_spec", None)
    if layout is not None:
        return DTensor.from_local_with_layout(target.detach(), layout)
    return target


@dataclass
class _Plan:
    """
    What planning the load has decided so far.

    Attributes:
        table (dict[str, list[RemapBlock]]): Blocks of every model tensor loaded by region reads, by the
            name the load's state dict gives it.
        deferred (dict[int, tuple[_Conversion, list[tuple[str, str]]]]): Conversions run for real, by
            identity, with the outputs each one copies into model tensors.
        loaded_ids (set[int]): Identity of every model tensor something loads.
        loaded_keys (set[str]): Every name those tensors are registered under.
        unexpected (list[str]): Checkpoint keys and converted names that match no model tensor.
    """

    table: dict[str, list[RemapBlock]] = field(default_factory=dict)
    deferred: dict[int, tuple[_Conversion, list[tuple[str, str]]]] = field(default_factory=dict)
    loaded_ids: set[int] = field(default_factory=set)
    loaded_keys: set[str] = field(default_factory=set)
    unexpected: list[str] = field(default_factory=list)


class HFLoadPlanner(RemapLoadPlanner):
    """
    Load planner for a Hugging Face checkpoint read into a finalized model.

    Tensors are renamed and converted by ``weights_mapping`` exactly as the legacy loader does it, and a
    model whose replacement modules convert their weights again is loaded in the same two stages. Every
    rank has to plan the load, and a load that fails to plan on one rank fails on all of them.

    The rules, in the order they are tried:

    * ``extra_weights_mapping``, ahead of everything else, for rules a model needs on top of the default.
    * ``weights_mapping`` when given, which replaces the default. Otherwise :meth:`default_weights_mapping`
      builds it: the model's Transformers conversion mapping, with ``key_mapping`` renames first.

    A rule listed twice is used once. A subclass can also override:

    * :meth:`map_checkpoint_key`, to rename a checkpoint key before any rule sees it, or skip it.
    * :meth:`select_converter`, to pick the converter of a key several converters match.
    * :meth:`supports_symbolic`, to have a transform always run on real tensors.

    Example::

        planner = HFLoadPlanner(model)
        load(planner.state_dict_for_load(), storage_reader=HuggingFaceStorageReader(path), planner=planner)
        report = planner.finish(path)
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            weights_mapping: Optional[Sequence[Transform]] = None,
            extra_weights_mapping: Sequence[Transform] = (),
            key_mapping: Optional[Mapping[str, str]] = None,
            strict: bool = True,
            broadcast_replicated_tensors: bool = False,
    ) -> None:
        """
        Args:
            model (nn.Module): The finalized model, its tensors materialized.
            weights_mapping (Optional[Sequence[Transform]]): Rules replacing the default ones. Default None.
            extra_weights_mapping (Sequence[Transform]): Rules tried before the others. Default ().
            key_mapping (Optional[Mapping[str, str]]): Regex renames of checkpoint keys, applied before the
                other rules. Default None.
            strict (bool): Fail the load when a model tensor would be left unloaded. Default True.
            broadcast_replicated_tensors (bool): See :class:`StandardLoadPlanner`. Default False.
        """
        super().__init__(broadcast_replicated_tensors=broadcast_replicated_tensors)
        self.model = model
        self.weights_mapping = None if weights_mapping is None else list(weights_mapping)
        self.extra_weights_mapping = list(extra_weights_mapping)
        self.key_mapping = dict(key_mapping or {})
        self.strict = strict
        self.report: Optional[LoadReport] = None
        self._targets = build_load_targets(model)
        self._aliases = alias_names_by_target(self._targets)
        self._state_keys: dict[int, str] = {}
        for name, target in self._targets.items():
            self._state_keys.setdefault(id(target), name)
        self._load_state = {key: _dcp_view(self._targets[key]) for key in self._state_keys.values()}
        self._loaded_ids: set[int] = set()
        self._used: tuple[list[Transform], Optional[list[Transform]]] = ([], None)

    def default_weights_mapping(self, model: nn.Module) -> list[Transform]:
        """
        The rules used when none are passed in.

        Args:
            model (nn.Module): The model being loaded.

        Returns:
            list[Transform]: The model's Transformers conversion mapping, with ``key_mapping`` first.
        """
        return get_model_conversion_mapping(model, key_mapping=self.key_mapping or None, hf_quantizer=None)

    def map_checkpoint_key(self, key: str) -> Optional[str]:
        """
        The name the rules see a checkpoint key under.

        Args:
            key (str): The key as the checkpoint stores it.

        Returns:
            Optional[str]: The key itself by default. None skips the key, which is then neither loaded
            nor reported as unexpected.
        """
        return key

    def select_converter(
            self, source_pattern: str, target_name: str, candidates: Sequence[WeightConverter]
    ) -> Optional[WeightConverter]:
        """
        The converter a checkpoint key matched by ``source_pattern`` belongs to.

        Args:
            source_pattern (str): The source pattern the key matched.
            target_name (str): The model tensor the key was renamed to.
            candidates (Sequence[WeightConverter]): Every converter with that source pattern.

        Returns:
            Optional[WeightConverter]: The only converter scoped to a module holding the target, else the
            only unscoped one, as the legacy loader picks it. None fails the load.
        """
        return select_unique_converter(source_pattern, target_name, candidates)

    def supports_symbolic(self, transform: Transform) -> bool:
        """
        Whether a transform may be traced into region reads, rather than always run on real tensors.

        Args:
            transform (Transform): The transform.

        Returns:
            bool: True by default. A traced transform is still run on real tensors when tracing fails or
            disagrees with torch on the shapes and dtypes of what it returns.
        """
        del transform
        return True

    def state_dict_for_load(self) -> dict[str, Any]:
        """
        The state dict to pass to :func:`load` with this planner.

        Returns:
            dict[str, Any]: Every model tensor once, under the first name it is registered under.
        """
        return dict(self._load_state)

    def configure_planner(self, state_dict: dict[str, Any], metadata: Metadata, **kwargs: Any) -> None:
        """
        Configure the planner, then plan how every model tensor is loaded out of the checkpoint.

        Args:
            state_dict (dict[str, Any]): What :meth:`state_dict_for_load` returned.
            metadata (Metadata): Metadata of the checkpoint, as :class:`HuggingFaceStorageReader` builds it.
            **kwargs (Any): As :func:`load` passes them.

        Raises:
            RuntimeError: If planning fails on another rank.
            Exception: Whatever planning raised on this rank.
        """
        super().configure_planner(state_dict, metadata, **kwargs)
        error = None
        try:
            self._compile(metadata)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = exc
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        message = None if error is None else f"{type(error).__name__}: {error}"
        failures = all_gather_object(message, world_size, kwargs.get("use_collectives", False))
        if error is not None:
            raise error
        failed = [(rank, failure) for rank, failure in enumerate(failures) if failure is not None]
        if failed:
            raise RuntimeError(f"Planning the pretrained load failed on rank {failed[0][0]}: {failed[0][1]}")

    def finish(self, pretrained_path: str = "") -> LoadReport:
        """
        Complete a load this planner planned and :func:`load` executed.

        Marks the loaded tensors initialized and records the conversions used on the model, where
        saving converts back through them, as the legacy loader does.

        Args:
            pretrained_path (str): Where the checkpoint was loaded from, for the log. Default "".

        Returns:
            LoadReport: What was loaded, what is missing and what the checkpoint holds beyond the model.
        """
        if self.report is None:
            raise RuntimeError("HFLoadPlanner.finish() needs a load planned with this planner first")
        for target in self._targets.values():
            if id(target) in self._loaded_ids:
                target._is_hf_initialized = True  # pylint: disable=protected-access
        self.apply_used_conversions(self.model)
        validate_load_result(self.report.missing_keys, self.report.unexpected_keys, strict=False)
        logger.info("Loaded %d model tensors from %s", len(self.report.loaded_keys), pretrained_path)
        return self.report

    def apply_used_conversions(self, model: nn.Module) -> None:
        """
        Record on ``model`` the conversions the load used, where saving in the original format finds them.

        Args:
            model (nn.Module): The model loaded.
        """
        used, used_replacements = self._used
        if used_replacements is None:
            model._weight_conversions = list(used)  # pylint: disable=protected-access
            return
        model._hp_used_base_weight_conversions = list(used)  # pylint: disable=protected-access
        model._hp_used_replacement_weight_conversions = list(used_replacements)  # pylint: disable=protected-access
        model._weight_conversions = list(used) + list(used_replacements)  # pylint: disable=protected-access

    def _resolved_mapping(self) -> list[Transform]:
        """The rules in the order they are tried, each once."""
        if self.weights_mapping is None:
            base = list(self.default_weights_mapping(self.model))
        else:
            renames = [
                WeightRenaming(source_patterns=source, target_patterns=target)
                for source, target in self.key_mapping.items()
            ]
            base = renames + self.weights_mapping
        mapping, seen = [], set()
        for transform in (*self.extra_weights_mapping, *base):
            if id(transform) not in seen:
                seen.add(id(transform))
                mapping.append(transform)
        return mapping

    def _checkpoint_keys(self, metadata: Metadata) -> dict[str, str]:
        """Every checkpoint tensor the rules see, by the name they see it under."""
        keys: dict[str, str] = {}
        for key, md in metadata.state_dict_metadata.items():
            mapped = self.map_checkpoint_key(key) if isinstance(md, TensorStorageMetadata) else None
            if mapped is None:
                continue
            if mapped in keys:
                raise ValueError(f"Checkpoint keys {keys[mapped]!r} and {key!r} both map to {mapped!r}")
            keys[mapped] = key
        return keys

    def _compile(self, metadata: Metadata) -> None:
        """Plan the load of every model tensor, and fill the table and deferred reads it is executed from."""
        keys = self._checkpoint_keys(metadata)
        make_loader = partial(self._leaf_loader, metadata, keys)
        mapping = self._resolved_mapping()
        replacement_mapping = getattr(self.model, "_hp_replacement_weight_conversions", None)
        source_shapes = getattr(self.model, "_hp_checkpoint_source_shapes", None)
        plan = _Plan()
        if replacement_mapping and source_shapes:
            self._compile_with_replacements(plan, keys, make_loader, mapping, (replacement_mapping, source_shapes))
        else:
            groups, unexpected, mapping = build_load_groups(
                self.model, sorted_checkpoint_keys(keys), self._targets, weights_mapping=mapping,
                make_loader=make_loader, select_converter=self.select_converter,
            )
            plan.unexpected.extend(keys[key] for key in unexpected)
            for group in groups:
                for name, value in self._plan_conversion(_conversion_of(group, self.model)).items():
                    self._claim(plan, name, value)
            self._used = ([transform for transform in mapping if transform.was_used()], None)
        self._adopt(plan)

    def _compile_with_replacements(
            self,
            plan: _Plan,
            keys: dict[str, str],
            make_loader: Callable[[str], Any],
            mapping: list[Transform],
            replacements: tuple[list[Transform], dict[str, tuple[int, ...]]],
    ) -> None:
        """Plan a model whose replacement modules convert the normalized weights a second time."""
        replacement_mapping, source_shapes = replacements
        base_mapping = base_weights_mapping(mapping, replacement_mapping)
        source_model = SourceModelView(self.model, source_shapes)
        base_groups, unexpected, _ = build_load_groups(
            source_model, sorted_checkpoint_keys(keys), source_model.targets, weights_mapping=base_mapping,
            make_loader=make_loader, select_converter=self.select_converter,
        )
        plan.unexpected.extend(keys[key] for key in unexpected)
        routes = build_replacement_routes(self.model, tuple(source_shapes), self._targets, replacement_mapping)
        pending: dict[int, list[tuple[str, Any]]] = {}
        used_replacements = []
        for group in base_groups:
            for source_name, value in self._plan_conversion(_conversion_of(group, source_model)).items():
                route = routes.get(source_name)
                if route is None:
                    self._claim(plan, source_name, value)
                    continue
                state, _, source_pattern = route
                pending.setdefault(id(state), []).append((source_pattern, value))
                state.received[source_pattern] += 1
                if state.completed or state.received != state.expected:
                    continue
                conversion = _Conversion(
                    state.group.transform, state.group.first_target_name, self.model, pending.pop(id(state))
                )
                for name, converted in self._plan_conversion(conversion).items():
                    self._claim(plan, name, converted)
                state.completed = True
                used_replacements.append(state.group.transform)
        self._used = ([transform for transform in base_mapping if transform.was_used()], used_replacements)

    def _leaf_loader(self, metadata: Metadata, keys: dict[str, str], mapped: str) -> Callable[[], RegionTensor]:
        """A loader handing a transform the whole of the checkpoint tensor the rules see as ``mapped``."""
        key = keys[mapped]
        md = metadata.state_dict_metadata[key]
        return partial(_identity, RegionTensor.leaf(key, md.size, str_to_dtype(md.properties.dtype)))

    def _plan_conversion(self, conversion: _Conversion) -> dict[str, Any]:
        """
        Trace a conversion into regions, or plan it to run on real tensors when it cannot be traced.

        The conversion is also run on meta tensors, which both checks what tracing returns against what
        torch computes and gives the shapes of what an untraced conversion will return.

        Args:
            conversion (_Conversion): The conversion.

        Returns:
            dict[str, Any]: Every output by name: a :class:`RegionTensor` when traced, else a
            :class:`_Deferred`.

        Raises:
            RuntimeError: If the conversion fails on meta tensors as well as when traced.
        """
        values = [value for _, value in conversion.inputs]
        traced = None
        if self.supports_symbolic(conversion.template) and all(isinstance(v, RegionTensor) for v in values):
            try:
                traced = {name: first_tensor(value) for name, value in conversion.run(values).items()}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Tracing %s failed, running it on real tensors: %s", conversion.first_target_name, exc)
        try:
            expected = {
                name: first_tensor(value) for name, value in conversion.run([_meta_value(v) for v in values]).items()
            }
        except RuntimeError:
            if traced is not None and all(isinstance(value, RegionTensor) for value in traced.values()):
                return traced
            raise
        if traced is not None and _same_outputs(traced, expected):
            return traced
        return {
            name: _Deferred(tuple(value.shape), value.dtype, conversion, name) for name, value in expected.items()
        }

    def _claim(self, plan: _Plan, name: str, value: Any) -> None:
        """Have ``value`` load the model tensor ``name``, unless something planned earlier already does."""
        target = self._targets.get(name)
        if target is None:
            plan.unexpected.append(name)
            return
        target_id = id(target)
        if target_id not in plan.loaded_ids:
            key = self._state_keys[target_id]
            self._check_target(name, target, self._load_state[key], value)
            if isinstance(value, RegionTensor):
                plan.table[key] = value.remap_blocks()
            else:
                plan.deferred.setdefault(id(value.conversion), (value.conversion, []))[1].append((value.name, name))
            plan.loaded_ids.add(target_id)
        plan.loaded_keys.update(self._aliases[target_id])

    @staticmethod
    def _check_target(name: str, target: torch.Tensor, view: Any, value: Any) -> None:
        """Refuse a model tensor that cannot take ``value``, as the legacy loader refuses it."""
        if local_target_tensor(target).is_meta:
            raise ValueError(f"Target must be materialized before loading: {name}")
        layout = target_layout(target)
        if layout is not None:
            reject_partial_layout(name, layout)
        if tuple(value.shape) != tuple(view.shape):
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {tuple(value.shape)} vs target {tuple(view.shape)}"
            )

    def _adopt(self, plan: _Plan) -> None:
        """Take a finished plan on as the table and deferred reads of the load, and report on it."""
        missing = tuple(sorted(set(self._targets) - plan.loaded_keys, key=dot_natural_key))
        unexpected = tuple(sorted(set(plan.unexpected), key=dot_natural_key))
        validate_load_result(missing, (), self.strict)
        deferred = []
        for conversion, claims in plan.deferred.values():
            complete = partial(_complete_conversion, conversion, tuple(claims), self._targets)
            sources = tuple(sorted(conversion.sources()))
            if sources:
                deferred.append(DeferredRead(sources, complete))
            else:
                # Reads nothing, as a conversion of empty tensors does, so there is nothing to wait for.
                complete({})
        self.table = {key: tuple(blocks) for key, blocks in plan.table.items()}
        self.deferred = tuple(deferred)
        self._loaded_ids = set(plan.loaded_ids)
        self.report = LoadReport(
            loaded_keys=tuple(sorted(plan.loaded_keys, key=dot_natural_key)),
            missing_keys=missing,
            unexpected_keys=unexpected,
        )
        logger.info(
            "Planned %d model tensors as region reads and %d through %d conversions on whole tensors",
            len(self.table), sum(len(claims) for _, claims in plan.deferred.values()), len(plan.deferred),
        )


def _conversion_of(group: LoadGroup, model: Any) -> _Conversion:
    """
    The conversion of a load group, taking over what the group's transform collected.

    Args:
        group (LoadGroup): A group :func:`build_load_groups` built with loaders handing out regions.
        model (Any): The model, or the view of it, handed to the conversion operations.

    Returns:
        _Conversion: The conversion, whose template is the group's transform, emptied.
    """
    inputs = [
        (pattern, loader()) for pattern, loaders in group.transform.collected_tensors.items() for loader in loaders
    ]
    group.transform.collected_tensors.clear()
    return _Conversion(group.transform, group.first_target_name, model, inputs)


def _same_outputs(traced: Mapping[str, Any], expected: Mapping[str, torch.Tensor]) -> bool:
    """Whether a traced conversion returns what torch computes, by name, shape and dtype."""
    return traced.keys() == expected.keys() and all(
        isinstance(value, RegionTensor)
        and tuple(value.shape) == tuple(expected[name].shape)
        and value.dtype == expected[name].dtype
        for name, value in traced.items()
    )


class _SingleFileReader(HuggingFaceStorageReader):
    """A Hugging Face checkpoint that is one safetensors file, under whatever name it has."""

    def __init__(self, file_path: Path) -> None:
        """
        Args:
            file_path (Path): The safetensors file.
        """
        super().__init__(file_path.parent)
        self._file_name = file_path.name

    def _weight_files(self) -> list[str]:
        """The one file."""
        return [self._file_name]


def _load_in_process(state_dict: dict[str, Any], reader: HuggingFaceStorageReader, planner: HFLoadPlanner) -> None:
    """Plan and execute a load in a process with no process group, which has nobody to share reads with."""
    metadata = reader.load_metadata()
    planner.configure_planner(state_dict, metadata, is_coordinator=True, rank=0, use_collectives=False)
    reader.configure_reader(metadata, is_coordinator=True, rank=0)
    reader.execute_read(planner.build_local_plan(), planner)


def load_hf_checkpoint(
        model: nn.Module,
        pretrained_path: str,
        *,
        weights_mapping: Optional[Sequence[Transform]] = None,
        strict: bool = True,
        planner: Optional[HFLoadPlanner] = None,
        **load_kwargs: Any,
) -> LoadReport:
    """
    Load a Hugging Face checkpoint into a finalized model through distributed checkpoint reads.

    Every rank of the default process group has to call this together. Without a process group the
    load runs in this process alone.

    Args:
        model (nn.Module): The finalized model, its tensors materialized.
        pretrained_path (str): A safetensors file, a checkpoint directory or a Hub repository id.
        weights_mapping (Optional[Sequence[Transform]]): Rules replacing the model's default ones.
            Default None. Ignored when ``planner`` is given.
        strict (bool): Fail when a model tensor would be left unloaded. Default True. Ignored when
            ``planner`` is given.
        planner (Optional[HFLoadPlanner]): The planner to load with, for rules customized beyond
            ``weights_mapping``. Default None, for ``HFLoadPlanner(model, ...)``.
        **load_kwargs (Any): Passed on to :func:`load`, such as ``broadcast_batch_bytes``.

    Returns:
        LoadReport: What was loaded, what is missing and what the checkpoint holds beyond the model.
    """
    location = resolve_checkpoint_location(pretrained_path)
    reader = _SingleFileReader(location) if location.is_file() else HuggingFaceStorageReader(location)
    if planner is None:
        planner = HFLoadPlanner(model, weights_mapping=weights_mapping, strict=strict)
    state_dict = planner.state_dict_for_load()
    if dist.is_available() and dist.is_initialized():
        load(state_dict, storage_reader=reader, planner=planner, **load_kwargs)
    else:
        _load_in_process(state_dict, reader, planner)
    return planner.finish(pretrained_path)
