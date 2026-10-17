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
"""Choosing every layer's recompute option for one candidate of the search.

The search keeps the candidates that fit with every layer fully recomputed,
the lightest policy. :func:`choose_recompute` gives a candidate's layers the
fastest options that fit instead. Each pipeline stage's layers share what the
stage has left under the config's own recompute, plus what those layers keep
under it: the memory model's peak for the stage is the sum of what the choice
does not change and of what each layer keeps at the micro-batches the stage
holds of it in flight. :func:`~.knapsack.pp_lite` then picks the options,
stage by stage.

At the end of warm-up, the memory model also charges each stage one layer's
working set at one micro-batch: the plain layer's if the stage's last layer is
fully recomputed, the layer's own otherwise, what it keeps then and, under FSDP
that reshards, the gathered parameters its backward holds besides. So that
layer's options each keep the working set they end warm-up on too, and the
knapsack weighs it with them.

A runtime that runs every layer one way, such as HyperParallel's trainer with
its ``activation_checkpoint.mode``, gets the fastest of its modes that fits
instead, from the same budgets: see :data:`MODES`. One that runs a mode per
layer, as the same trainer does with ``activation_checkpoint.layers``, gets
each layer the fastest of its modes, chosen as a layer's options are, and
:func:`trainer_plan` states the choice as the trainer reads it, where the
installed trainer declares that field (:func:`trainer_runs_layer_plans`). A model
priced on several submodules, a vision tower beside a language model, gets no
choice per layer, whose budgets would be built from one submodule's layers;
its runtime's mode is stated on every submodule, and the whole model priced
under each (:func:`mode_ranges`, :func:`whole_modes`).

With a host link, a choice per layer may also offload: a stage's first layers
run plain and move what they keep per micro-batch to the host, tensor by
tensor, and back before their backward, so that they keep only the rest. A
micro-batch's copies to the host keep pace with its own forward: one stream
copies what each layer keeps once the layer's forward ends, in the order the
layers run, and is done when the forward is, in the link's share of the
forward left. So a stage's first layers move all or part of what they keep,
and its last layers nothing when their window is too short. The copies back
run in the longer backward, one layer ahead, while this micro-batch's
backward has released what the layers after the next one keep; where that
does not cover them, the stage keeps the largest layer's offloaded bytes in
transit. Each byte moved costs the step the link's copy cost, out in the
forward and back in the backward, so where the copies cost time a stage
offloads no more than it needs to fit. Offload is priced at one chunk per
stage.

Where a stage's FSDP holds each layer's gradient output until the backward
ends, and accumulates gradients over several micro-batches, its peak can also
come as a later micro-batch's backward ends: the memory model then counts
what the micro-batches still in flight keep, beside every layer's gradient
output and the working set of the first layer's backward, the last the stage
runs. The stage's layers then fit at both points. Each pair of options of the
first layer and of the layer that ends warm-up leaves the others one budget,
since they keep a share of what they keep as warm-up ends: exact at one chunk
per stage, and with two, the largest share of the chunks'.
"""
import importlib
import itertools
import math
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Hashable, Iterator, List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._exec_spec import RECOMPUTE_OPS, RecomputeRange
from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import HostLink
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import (
    LayerOption,
    build_front,
    configured_switches,
    layer_profiles,
    price_option,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import (
    MEGABYTE,
    Assignment,
    Layers,
    Stage,
    StageChoice,
    choose,
    least_times,
    suffix_times,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES

# The pipeline schedules whose micro-batches in flight and end-of-warm-up
# working set the budget follows.
SCHEDULES = ("1f1b",)
# The ways a runtime can run every layer: nothing recomputed, the switches the
# config sets recomputed, or everything recomputed.
MODES = ("off", "selective", "full")
_ENDS = (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER)
# Marks the front of a layer that ends warm-up, whose options keep its working set.
_ENDS_WARM_UP = "ends warm-up"
# A time shorter than another by this share of it or less is the same time: the
# same options' times summed in another order can differ in their last bits.
_ROUNDING = 1e-9


@dataclass(frozen=True)
class LayerRange:
    """Consecutive layers of one kind that run one option.

    Attributes:
        first: The first layer's index, counting the model's layers in order
            without the embedding and output layers.
        count: How many layers.
        kind: Their kind; ``None`` for a model priced on its config as it
            stands.
        option: The option they run.
        mode: For a choice of a mode per layer, the mode they run, one of
            :data:`MODES`; ``None`` otherwise.
    """

    first: int
    count: int
    kind: Optional[LayerKind]
    option: LayerOption
    mode: Optional[str] = None


@dataclass(frozen=True)
class RecomputeChoice:
    """Every layer's option for one candidate, and what the options change.

    Attributes:
        ranges: The layers' options, in model order.
        stage_memory: Each stage's peak memory under the options, in MB.
        stage_savings: The time each stage's layers save against the
            recompute the config gives them, in the performance estimate's
            units.
        mode: The mode every layer runs, one of :data:`MODES`, when one mode
            was chosen for all of them; ``None`` for a choice per layer.
        score: The candidate's score under the choice, where the whole model
            was priced with it, a model priced on several submodules, or
            where a caller priced one alternative on its own; ``None`` where
            the search takes *stage_savings* off the score of the config's
            own recompute.
        dimensions: The parallel degrees the choice was priced at, for an
            alternative weighed against others at one fixed strategy; empty
            for a choice the search did not weigh that way.
        score_parts: The parts *score* splits into, in the order the
            debugger fills them, where a caller kept them for a plot.
    """

    ranges: Tuple[LayerRange, ...]
    stage_memory: Tuple[float, ...]
    stage_savings: Tuple[float, ...]
    mode: Optional[str] = None
    score: Optional[float] = None
    dimensions: Tuple[Tuple[str, int], ...] = ()
    score_parts: Tuple[float, ...] = ()

    @property
    def memory(self) -> float:
        """The peak memory in MB, the heaviest stage's."""
        return max(self.stage_memory)


@dataclass(frozen=True)
class _Layer:
    """A body layer: where it sits, its front's key and how the config runs it."""

    index: int
    key: Hashable
    in_flight: int
    own: LayerOption


@dataclass(frozen=True)
class _Peaks:
    """What a stage keeps whatever its layers run, at each point its peak can come, in bytes.

    Attributes:
        warm_up: As warm-up ends.
        backward: As a later micro-batch's backward ends, for a stage whose
            FSDP holds its gradient outputs until then; ``None`` otherwise.
    """

    warm_up: float
    backward: Optional[float] = None


def _kept(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer running *option* keeps with *in_flight* micro-batches in flight."""
    return option.memory(in_flight)


def _share(in_flight: int) -> float:
    """The share of what a layer keeps that stays as one of its *in_flight* micro-batches' backward ends."""
    return (in_flight - 1) / in_flight


def _time(option: LayerOption) -> float:
    """A layer's forward and backward time under *option*."""
    return option.forward_time + option.backward_time


def micro_batches_in_flight(evaluator: EvaluatorV2) -> List[List[int]]:
    """How many micro-batches each stage keeps in flight of each chunk's layers.

    Counted as the memory model counts them, with the schedule's own formula
    and at least one.

    Args:
        evaluator: The memory evaluator, set to the candidate's strategy.

    Returns:
        ``[stage][chunk]`` counts.
    """
    ccfg = evaluator.ccfg
    count = evaluator.ctx.pp_micro_eval[ccfg.pp_sched]
    less_memory = evaluator.ctx.vpp_less_mem

    def _at(stage: int, chunk: int) -> int:
        """The count of one stage's chunk, on a context of its own."""
        where = SimpleNamespace(current_stage_id=stage, current_chunk_id=chunk, vpp_less_mem=less_memory)
        return max(1, count(ccfg, where))

    return [[_at(stage, chunk) for chunk in range(ccfg.vp)] for stage in range(ccfg.p)]


def mode_recompute(mode: str, configured: Mapping[str, Any]) -> Optional[FrozenSet[str]]:
    """The switches a mode recomputes; ``None`` for full recompute.

    Args:
        mode: One of :data:`MODES`.
        configured: The switches the config sets, 1 to keep an op and 0 to
            recompute it, which say what ``selective`` recomputes.

    Returns:
        The switches.

    Raises:
        ValueError: For a mode not in :data:`MODES`.
    """
    if mode == "off":
        return frozenset()
    if mode == "full":
        return None
    if mode == "selective":
        return frozenset(name for name in SWITCHES if not int(bool(configured.get(name, 1))))
    raise ValueError(f"unknown recompute mode {mode!r}; expected one of {', '.join(MODES)}")


def mode_ranges(mode: str, configured: Mapping[str, Any]) -> Tuple[RecomputeRange, ...]:
    """Every layer of a model running *mode*, as the recompute ranges an ExecSpec states.

    Args:
        mode: One of :data:`MODES`.
        configured: The switches the selective mode sets, 1 to keep an op and
            0 to recompute it.

    Returns:
        No range for a model that recomputes nothing, else one range over
        every layer: full, or selective with each op's state.
    """
    recompute = mode_recompute(mode, configured)
    if recompute is None:
        return (RecomputeRange(option="full"),)
    if not recompute:
        return ()
    return (RecomputeRange(option="selective",
                           ops={op: "recompute" if op in recompute else "keep" for op in RECOMPUTE_OPS}),)


def census_prices(ccfg: Any) -> bool:
    """Whether a census prices every layer kind of *ccfg*, HyperParallel's selective policy among what it measures."""
    kinds = layer_kinds(ccfg)
    if not kinds:
        return False
    if any(kind is None for kind in kinds):
        return isinstance(getattr(ccfg, "kind_activations", None), KindActivations)
    census = getattr(ccfg, "census", None)
    census = census if isinstance(census, Mapping) else {}
    return all(isinstance(census.get(kind.name), KindActivations) for kind in kinds)


def whole_modes(
    configs: Sequence[Any], modes: Sequence[str], selective: Optional[Mapping[str, int]]
) -> Tuple[str, ...]:
    """*modes*, less a runtime's own selective mode unless a census prices every kind of every one of *configs*.

    The modes a model priced whole is offered, as :func:`choose_recompute`
    offers them a model priced layer by layer: the formulas do not price the
    matmuls HyperParallel's policy recomputes.
    """
    if selective is None or "selective" not in modes or all(census_prices(ccfg) for ccfg in configs):
        return tuple(modes)
    return tuple(mode for mode in modes if mode != "selective")


def _own(node: LayerType, options: Sequence[LayerOption]) -> Optional[LayerOption]:
    """The option a layer the partition gives *node* runs, if its kind's front has it."""
    for option in options:
        if node == LayerType.FULL_REC_LAYER and option.recompute is None:
            return option
        if node == LayerType.NOT_REC_LAYER and option.recompute == frozenset():
            return option
        if node == LayerType.SEL_REC_LAYER and "SLCT" in option.names:
            return option
    return None


def _overhead_position(chunk: Sequence[LayerType], has_mtp: bool) -> Optional[int]:
    """The position in a stage's last chunk of the layer whose working set warm-up ends on, if a body layer."""
    if not chunk:
        return None
    if chunk[-1] not in _ENDS:
        return len(chunk) - 1
    if chunk[-1] == LayerType.OUTPUT_LAYER and has_mtp and len(chunk) > 1 and chunk[-2] not in _ENDS:
        return len(chunk) - 2
    return None


def _body_layers(
    evaluator: EvaluatorV2, fronts: Dict[Hashable, Tuple[LayerOption, ...]], counts: List[List[int]]
) -> Optional[Tuple[List[List[_Layer]], List[List[_Layer]]]]:
    """Each stage's body layers, and the layers each stage's warm-up ends on.

    The memory model visits chunk after chunk, and within a chunk stage after
    stage, which is the model's order of layers; it gives each body layer the
    next kind in that order.

    Returns:
        ``(layers, ends)`` per stage, or ``None`` when some layer runs an
        option its kind's front does not have.
    """
    ccfg = evaluator.ccfg
    partition = ccfg.generate_partitions_vpp()
    kinds = layer_kinds(ccfg)
    layers = [[] for _ in range(ccfg.p)]
    at = {}
    index = 0
    for chunk in range(ccfg.vp):
        for stage in range(ccfg.p):
            for position, node in enumerate(partition[stage][chunk]):
                if node in _ENDS:
                    continue
                key = (ccfg.model_name, kinds[index])
                own = _own(node, fronts[key])
                if own is None:
                    return None
                layer = _Layer(index, key, counts[stage][chunk], own)
                layers[stage].append(layer)
                at[stage, chunk, position] = layer
                index += 1
    ends = [[] for _ in range(ccfg.p)]
    if not ccfg.freeze:
        last = ccfg.vp - 1
        for stage in range(ccfg.p):
            position = _overhead_position(partition[stage][last], ccfg.n_mtp > 0)
            if position is not None:
                ends[stage].append(at[stage, last, position])
    return layers, ends


def _plain(options: Sequence[LayerOption]) -> LayerOption:
    """A kind's plain option."""
    return next(option for option in options if option.recompute == frozenset())


def _working(option: LayerOption, options: Sequence[LayerOption], first: bool = False) -> float:
    """The working set a layer running *option* ends warm-up on, or with *first*, the backward on a stage's first layer.

    What it keeps at one micro-batch, the plain layer for full recompute,
    *options* being its kind's, and what its backward holds beyond.
    """
    run = option if option.recompute is not None else _plain(options)
    return _kept(run, 1) + (option.first_working_extra if first else option.working_extra)


def _charge_working_sets(
    layers: List[List[_Layer]],
    ends: Sequence[Sequence[_Layer]],
    fronts: Dict[Hashable, Tuple[LayerOption, ...]],
    by_mode: Optional[Dict[Hashable, Dict[str, LayerOption]]],
) -> Tuple[Any, ...]:
    """Give each layer that ends warm-up a front whose options keep its working set too.

    Its kind's options, each with the working set it ends warm-up on added to
    what it keeps once, so that the knapsack weighs the working set with the
    option instead of keeping room for the heaviest.

    Returns:
        ``(layers, fronts, by_mode, own)``: the layers, those that end warm-up
        on their new front; the fronts and the options by mode with the new
        fronts added; and each added option's option of its kind.
    """
    fronts = dict(fronts)
    by_mode = None if by_mode is None else dict(by_mode)
    own = {}
    ending = {layer.index for stage_ends in ends for layer in stage_ends}
    for stage_layers in layers:
        for position, layer in enumerate(stage_layers):
            if layer.index not in ending:
                continue
            key = layer.key + (_ENDS_WARM_UP,)
            options = fronts[layer.key]
            charged = {
                option: replace(option, memory_once=option.memory_once + _working(option, options))
                for option in options
            }
            fronts[key] = tuple(charged.values())
            if by_mode is not None:
                by_mode[key] = {mode: charged[option] for mode, option in by_mode[layer.key].items()}
            own.update({new: old for old, new in charged.items()})
            stage_layers[position] = replace(layer, key=key, own=charged[layer.own])
    return layers, fronts, by_mode, own


def _kind_key(key: Hashable) -> Hashable:
    """The key of a layer's kind's own front, for a layer on a front charged its working set."""
    return key[:-1] if isinstance(key, tuple) and key and key[-1] == _ENDS_WARM_UP else key


def _first_working(
    layer: _Layer,
    option: LayerOption,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """The working set of a stage's first layer's backward, the last the stage runs, under *option*.

    As for the layer that ends warm-up (:func:`_working`); for an offloaded
    layer, the plain layer's, whose activations come back for it.
    """
    options = fronts[_kind_key(layer.key)]
    option = own.get(option, option)
    return _working(_plain(options) if option.link_bandwidth else option, options, first=True)


def _backward_kept(
    stage_layers: Sequence[_Layer],
    chosen: Mapping[int, LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """What a stage's layers keep as a later micro-batch's backward ends, running *chosen*.

    Each keeps what it keeps of the micro-batches still in flight, a share of
    what it keeps as warm-up ends, and the first layer the working set of its
    backward, the last the stage runs.
    """
    if not stage_layers:
        return 0.0
    kept = sum(
        _kept(own.get(chosen[layer.index], chosen[layer.index]), layer.in_flight) * _share(layer.in_flight)
        for layer in stage_layers
    )
    first = stage_layers[0]
    return kept + _first_working(first, chosen[first.index], fronts, own)


def _stage_memory(
    stage_layers: Sequence[_Layer],
    peaks: _Peaks,
    chosen: Mapping[int, LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """A stage's peak in bytes, its layers running *chosen*: the higher of the points it can come at."""
    memory = peaks.warm_up + sum(_kept(chosen[layer.index], layer.in_flight) for layer in stage_layers)
    if peaks.backward is None:
        return memory
    return max(memory, peaks.backward + _backward_kept(stage_layers, chosen, fronts, own))


def _groups(layers: Sequence[_Layer]) -> Tuple[Layers, ...]:
    """Layers grouped by kind and micro-batches in flight, for the knapsack."""
    groups: Dict[Tuple[Hashable, int], int] = {}
    for layer in layers:
        groups[layer.key, layer.in_flight] = groups.get((layer.key, layer.in_flight), 0) + 1
    return tuple(Layers(key, count, flight) for (key, flight), count in groups.items())


def _stage(layers: Sequence[_Layer], peak: float, capacity: float) -> Tuple[Stage, float]:
    """A stage for the knapsack, and the bytes it keeps outside the choice as warm-up ends.

    Args:
        layers: The stage's body layers, the one that ends warm-up charged
            its working set (:func:`_charge_working_sets`).
        peak: The memory model's stage memory as warm-up ends, in MB.
        capacity: The device's memory, in MB.

    Returns:
        ``(stage, fixed)``: *fixed* is what the stage keeps whatever its
        layers run.
    """
    fixed = peak * MEGABYTE - sum(_kept(layer.own, layer.in_flight) for layer in layers)
    return Stage(groups=_groups(layers), budget=capacity * MEGABYTE - fixed), fixed


def _assign(layers: Sequence[_Layer], choices: Sequence[StageChoice]) -> Dict[int, LayerOption]:
    """Each layer's option: a group's layers take its options in model order, the slower first."""
    by_group: Dict[Tuple[Hashable, int], List[int]] = {}
    for layer in layers:
        by_group.setdefault((layer.key, layer.in_flight), []).append(layer.index)
    chosen = {}
    for choice in choices:
        for assignment in choice.assignments:
            waiting = by_group[assignment.layers.kind, assignment.layers.in_flight]
            for _ in range(assignment.count):
                chosen[waiting.pop(0)] = assignment.option
    return chosen


def _assign_counted(layers: Sequence[_Layer], chosen: Mapping[int, LayerOption]) -> Dict[int, LayerOption]:
    """Assign a group's option counts to its layers in the order the normal optimizer uses.

    The knapsack hands a group's options to its layers in model order, the
    slower first (:func:`_assign`), so a plan built by changing one layer is
    re-assigned the same way: two plans that run the same counts then name
    the same layers whichever layer was changed.

    Args:
        layers: One group's layers, of one kind, stage and in-flight count.
        chosen: Each layer's option, the group's among them.

    Returns:
        The group's layers and the option each runs.
    """
    group = Layers(layers[0].key, len(layers), layers[0].in_flight)
    counts = Counter(chosen[layer.index] for layer in layers)
    assignments = tuple(
        Assignment(group, option, count)
        for option, count in sorted(counts.items(), key=lambda item: -_time(item[0]))
    )
    choice = StageChoice(
        assignments,
        sum(count * _kept(option, group.in_flight) for option, count in counts.items()),
        sum(count * _time(option) for option, count in counts.items()),
    )
    return _assign(layers, [choice])


def _equivalent_cost(left: RecomputeChoice, right: RecomputeChoice) -> bool:
    """Whether two plans of the same mode counts also have equal stage peaks and time saved."""
    return all(
        math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9)
        for first, second in zip(left.stage_memory, right.stage_memory)
    ) and all(
        math.isclose(first, second, rel_tol=_ROUNDING, abs_tol=1e-9)
        for first, second in zip(left.stage_savings, right.stage_savings)
    )


def _ranges(
    layers: Sequence[_Layer], chosen: Dict[int, LayerOption], modes: Optional[Mapping[LayerOption, str]] = None
) -> Tuple[LayerRange, ...]:
    """The options as ranges of consecutive layers of one kind, in model order, with *modes* each option's mode."""
    ranges = []
    for layer in sorted(layers, key=lambda layer: layer.index):
        option = chosen[layer.index]
        kind = layer.key[1]
        last = ranges[-1] if ranges else None
        if last is not None and last.option == option and last.kind == kind and last.first + last.count == layer.index:
            ranges[-1] = replace(last, count=last.count + 1)
        else:
            ranges.append(LayerRange(layer.index, 1, kind, option, None if modes is None else modes[option]))
    return tuple(ranges)


def _result(
    layers: Sequence[Sequence[_Layer]],
    peaks: Sequence[_Peaks],
    chosen: Dict[int, LayerOption],
    mode: Optional[str],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    modes: Optional[Mapping[LayerOption, str]] = None,
) -> RecomputeChoice:
    """The choice of *chosen* options, with each stage's memory and time saved.

    *own* maps an option charged a working set to its option of its kind,
    which the ranges report; for a choice of a mode per layer, *modes* maps
    each kind's option to its mode, which they report too.
    """
    every = [layer for stage_layers in layers for layer in stage_layers]
    return RecomputeChoice(
        ranges=_ranges(every, {index: own.get(option, option) for index, option in chosen.items()}, modes),
        stage_memory=tuple(
            _stage_memory(stage_layers, stage_peaks, chosen, fronts, own) / MEGABYTE
            for stage_layers, stage_peaks in zip(layers, peaks)
        ),
        stage_savings=tuple(
            sum(_time(layer.own) - _time(chosen[layer.index]) for layer in stage_layers) for stage_layers in layers
        ),
        mode=mode,
    )


def _one_mode(
    layers: Sequence[Sequence[_Layer]],
    peaks: Sequence[_Peaks],
    by_mode: Dict[Hashable, Dict[str, LayerOption]],
    modes: Sequence[str],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    capacity: float,
) -> Optional[Tuple[str, Dict[int, LayerOption]]]:
    """The fastest of *modes* that every stage fits when every layer runs it, and each layer's option."""
    fastest, best = math.inf, None
    for mode in modes:
        chosen = {layer.index: by_mode[layer.key][mode] for stage_layers in layers for layer in stage_layers}
        fits = all(
            _stage_memory(stage_layers, stage_peaks, chosen, fronts, own) <= capacity
            for stage_layers, stage_peaks in zip(layers, peaks)
        )
        time = sum(_time(option) for option in chosen.values())
        if fits and time < fastest:
            fastest, best = time, (mode, chosen)
    return best


def _priced_modes(
    modes: Optional[Sequence[str]], profiles: Mapping[Hashable, Any], selective: Optional[Mapping[str, int]]
) -> Optional[Sequence[str]]:
    """*modes*, less a runtime's own selective mode unless every kind's profile measures its setting whole."""
    if modes is None or selective is None or "selective" not in modes:
        return modes
    setting = mode_recompute("selective", selective)
    if all(setting in profile.whole for profile in profiles.values()):
        return modes
    return tuple(mode for mode in modes if mode != "selective")


def _offered(
    profiles: Dict[Hashable, Any],
    configured: Mapping[str, Any],
    modes: Optional[Sequence[str]],
    selective: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[Hashable, Tuple[LayerOption, ...]], Optional[Dict[Hashable, Dict[str, LayerOption]]]]:
    """Each kind's options: its front, or with *modes*, the option of each mode the profiles can price.

    The selective mode recomputes what *selective* sets, else the config's
    switches.

    Returns:
        ``(options, by_mode)``: each kind's options, and with *modes* each
        kind's option by mode.
    """
    if modes is None:
        return {key: build_front(profile, configured) for key, profile in profiles.items()}, None
    offered = MODES if "selective" in modes else tuple(mode for mode in MODES if mode != "selective")
    switches = {mode: selective if mode == "selective" and selective is not None else configured for mode in offered}
    by_mode = {
        key: {mode: price_option(profile, mode_recompute(mode, switches[mode]), configured) for mode in offered}
        for key, profile in profiles.items()
    }
    return {key: tuple(options.values()) for key, options in by_mode.items()}, by_mode


@dataclass(frozen=True)
class _Link:
    """A host link in the performance estimate's time units.

    Attributes:
        per_byte: The time one byte takes over the link, one way.
        overlap: The share of the compute time the copies may take.
        cost_per_byte: The step time one byte moved costs, either way.
    """

    per_byte: float
    overlap: float
    cost_per_byte: float = 0.0


def _link(link: HostLink, evaluator: EvaluatorV2) -> _Link:
    """*link* in the estimate's units: FLOPs of forward and backward, times the precision's bytes, per device.

    A link calibrated with a COMPUTE ratio converts with it instead
    (:meth:`HostLink.units_per_second`), and the copies' cost with the
    score's own ratio where one is stated
    (:meth:`HostLink.score_units_per_second`).
    """
    bytes_p = evaluator.ccfg.bytes_p
    return _Link(link.seconds_per_byte() * link.units_per_second(bytes_p), link.overlap,
                 link.cost_seconds_per_byte() * link.score_units_per_second(bytes_p))


def _offloaded(option: LayerOption, share: float, cost_per_byte: float = 0.0) -> LayerOption:
    """*option* with *share* of what it keeps per micro-batch moved to the host, so that it keeps the rest.

    The copies cost the step *cost_per_byte* a byte, out in the forward and
    back in the backward.
    """
    return replace(option, memory_per_micro_batch=option.memory_per_micro_batch - share, link_bandwidth=share,
                   forward_time=option.forward_time + share * cost_per_byte,
                   backward_time=option.backward_time + share * cost_per_byte, names=())


def _offloadable(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer running *option* keeps per micro-batch, less any the two memories charge beyond them."""
    return max(0.0, _kept(option, in_flight) - option.memory_once) / in_flight


def _offload_shares(stage_layers: Sequence[_Layer], plain: Sequence[LayerOption], link: _Link) -> List[float]:
    """The bytes per micro-batch each of a stage's first layers can move to the host running plain.

    One stream copies what each layer keeps once the layer's forward ends, in
    the order the layers run, and is done when the forward is, in the link's
    share of the forward left: each layer moves what the window after it
    leaves, all it keeps or part of it, once the earlier layers' copies take
    their part. The layer that ends warm-up, and every layer after it, keep
    their activations.

    Returns:
        A share for each layer before the one that ends warm-up, 0 for one
        whose window is spent.
    """
    ending = next((position for position, layer in enumerate(stage_layers) if _ENDS_WARM_UP in layer.key),
                  len(stage_layers))
    after = list(itertools.accumulate((option.forward_time for option in reversed(plain[1:])), initial=0.0))[::-1]
    shares, room = [], math.inf
    for position in range(ending):
        room = min(room, link.overlap * after[position])
        whole = _offloadable(plain[position], stage_layers[position].in_flight)
        share = whole if link.per_byte == 0 else min(whole, room / link.per_byte)
        room = max(0.0, room - share * link.per_byte)
        shares.append(share)
    return shares


def _backward_can_bind(
    stage_layers: Sequence[_Layer],
    room: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> bool:
    """Whether some choice of a stage's layers keeps more than *room* bytes as a later micro-batch's backward ends."""
    first = stage_layers[0]
    most = max(
        _kept(own.get(option, option), first.in_flight) * _share(first.in_flight)
        + _first_working(first, option, fronts, own)
        for option in fronts[first.key]
    )
    most += sum(
        max(_kept(own.get(option, option), layer.in_flight) for option in fronts[layer.key]) * _share(layer.in_flight)
        for layer in stage_layers[1:]
    )
    return most > room


def _ends(stage_layers: Sequence[_Layer]) -> List[_Layer]:
    """A stage's first layer and the layer that ends its warm-up, once each."""
    first = stage_layers[0]
    ending = next((layer for layer in stage_layers if _ENDS_WARM_UP in layer.key), first)
    return [first] if ending.index == first.index else [first, ending]


def _middle_budget(
    ends: Sequence[_Layer],
    chosen: Mapping[int, LayerOption],
    budgets: Tuple[float, float],
    share: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """What a stage's other layers may keep as warm-up ends, its ends running *chosen*; negative if none fits.

    *budgets* are the bytes all its layers may keep as warm-up ends and as a
    later micro-batch's backward ends, and *share* the share the other layers
    keep of the one at the other.
    """
    left = budgets[0] - sum(_kept(chosen[end.index], end.in_flight) for end in ends)
    left_back = budgets[1] - _backward_kept(ends, chosen, fronts, own)
    if left < 0 or left_back < 0:
        return -1.0
    return min(left, left_back / share) if share else left


def _two_peaks(
    stage_layers: Sequence[_Layer],
    budget: float,
    room: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    bucket: float,
) -> Optional[Dict[int, LayerOption]]:
    """The fastest options of a stage's layers within *budget* bytes as warm-up ends and *room* as a backward ends.

    What the layers keep as a later micro-batch's backward ends is a share of
    what they keep as warm-up ends, but for the first layer, which adds its
    working set, and the layer that ends warm-up, which leaves its own out.
    So each pair of options of those two leaves the other layers one budget:
    exact when they keep as many micro-batches in flight, at one chunk per
    stage, and with the largest share of theirs otherwise, which fits.
    """
    ends = _ends(stage_layers)
    middle = [layer for layer in stage_layers if layer not in ends]
    share = max((_share(layer.in_flight) for layer in middle), default=0.0)
    groups = _groups(middle)
    within = least_times(groups, fronts, budget, bucket)
    fastest, best, best_budget = math.inf, None, 0.0
    for picks in itertools.product(*(fronts[end.key] for end in ends)):
        chosen = {end.index: option for end, option in zip(ends, picks)}
        limit = _middle_budget(ends, chosen, (budget, room), share, fronts, own)
        time = within(limit) + sum(_time(option) for option in picks)
        if time < fastest:
            fastest, best, best_budget = time, chosen, limit
    if best is None:
        return None
    choice = choose(groups, fronts, best_budget, bucket)
    if choice is None:
        return None
    best.update(_assign(middle, [choice]))
    return best


def _choose_stage(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    bucket: float,
    capacity: float,
) -> Optional[Dict[int, LayerOption]]:
    """The fastest options of a stage's layers that keep it within *capacity* bytes wherever its peak comes."""
    if peaks.backward is not None and stage_layers:
        room = capacity - peaks.backward
        if _backward_can_bind(stage_layers, room, fronts, own):
            return _two_peaks(stage_layers, stage.budget, room, fronts, own, bucket)
    choice = choose(stage.groups, fronts, stage.budget, bucket)
    return None if choice is None else _assign(stage_layers, [choice])


def _lightest_working(stage_layers: Sequence[_Layer], fronts: Mapping[Hashable, Tuple[LayerOption, ...]]) -> float:
    """The least working set the layer that ends a stage's warm-up can end it on; 0 without one."""
    ending = next((layer for layer in stage_layers if _ENDS_WARM_UP in layer.key), None)
    if ending is None:
        return 0.0
    options = fronts[_kind_key(ending.key)]
    return min(_working(option, options) for option in options)


def _offload_budgets(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    offloaded: Sequence[LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    sizes: Tuple[float, bool],
) -> List[float]:
    """For every count of a stage's first layers offloaded, what the other layers may keep as warm-up ends.

    Less what the offloaded layers keep on the device and, held in transit,
    the largest of their offloaded bytes. Where the stage's peak can also
    come as a later micro-batch's backward ends, the other layers keep room
    for it as though the layer that ends warm-up ran its lightest option
    there, which fits.

    Args:
        offloaded: The options the first layers offload with, as many as
            may offload.
        sizes: ``(capacity, in_transit)``: the device's memory in bytes, and
            whether the offloaded bytes are held in transit.

    Returns:
        A budget for every count, from none to every layer, negative for a
        count beyond those that may offload.
    """
    capacity, in_transit = sizes
    room = math.inf if peaks.backward is None else capacity - peaks.backward
    share = _share(stage_layers[0].in_flight)
    lightest = _lightest_working(stage_layers, fronts)
    first_working = _first_working(stage_layers[0], offloaded[0], fronts, own)
    budgets, held, largest = [stage.budget], 0.0, 0.0
    for layer, option in zip(stage_layers, offloaded):
        held += _kept(option, layer.in_flight)
        largest = max(largest, option.link_bandwidth if in_transit else 0.0)
        left_back = room - held * share - first_working - largest
        rest_back = left_back / share + lightest if share else (math.inf if left_back >= 0 else -1.0)
        budgets.append(min(stage.budget - held - largest, rest_back))
    return budgets + [-1.0] * (len(stage_layers) + 1 - len(budgets))


def _returns_fit(stage_layers: Sequence[_Layer], chosen: Mapping[int, LayerOption], count: int) -> bool:
    """Whether each of a stage's *count* offloaded layers' bytes come back in room the backward has released.

    A layer's bytes come back while the next layer's backward runs, beside
    the next layer's own when it offloads too, once this micro-batch's
    backward has released its share of what the layers after the next one
    keep, *chosen* their options.
    """
    released = list(itertools.accumulate(
        (_kept(chosen[layer.index], layer.in_flight) / layer.in_flight for layer in reversed(stage_layers)),
        initial=0.0))[::-1]
    for position in range(count):
        back = sum(chosen[layer.index].link_bandwidth for layer in stage_layers[position:position + 2])
        if back > released[min(position + 2, len(stage_layers))]:
            return False
    return True


def _fastest_count(times: Sequence[float], rest: Sequence[float], most: int, fastest: float) -> int:
    """How many of a stage's first layers to offload: the count, up to *most*, that runs fastest below *fastest*.

    Args:
        times: The time the first layers take offloaded, by their count.
        rest: The time the other layers take in what each count leaves.
        most: The most first layers the link reaches.
        fastest: The time to beat, keeping every layer.

    Returns:
        The count, 0 where none beats *fastest*.
    """
    count, best = 0, fastest
    for first in range(1, most + 1):
        if times[first] + rest[first] < best:
            count, best = first, times[first] + rest[first]
    return count


def _offload_choice(stage_layers: Sequence[_Layer], offloaded: Sequence[LayerOption], choice: Any,
                    in_transit: bool) -> Tuple[Dict[int, LayerOption], float]:
    """Each layer's option, and what the offloaded layers hold in transit.

    The stage's first layers offload with *offloaded*, one option each, and
    the others run *choice*'s options. In transit, they hold the largest of
    their offloaded bytes, where *in_transit*.
    """
    chosen = {layer.index: option for layer, option in zip(stage_layers, offloaded)}
    chosen.update(_assign(stage_layers[len(offloaded):], [choice]))
    transit = max(option.link_bandwidth for option in offloaded) if in_transit else 0.0
    return chosen, transit


def _offload_stage(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    link: _Link,
    bucket: float,
    capacity: float,
) -> Optional[Tuple[Dict[int, LayerOption], float]]:
    """The fastest choice of a stage's layers when its first layers may offload.

    Every count of first layers the link reaches is weighed against keeping
    them, the rest of the layers choosing their options in what the
    offloaded ones leave (:func:`_offload_budgets`): first with nothing in
    transit, which holds where the backward releases room for every copy
    back (:func:`_returns_fit`), else with the largest offloaded bytes held.

    Returns:
        ``(chosen, transit)``: each layer's option by index, and the memory
        the offloaded layers keep in transit; ``None`` when the stage cannot
        fit.
    """
    stay = _choose_stage(stage_layers, stage, peaks, fronts, own, bucket, capacity)
    kept = None if stay is None else (stay, 0.0)
    plain = [_plain(fronts[layer.key]) for layer in stage_layers]
    shares = _offload_shares(stage_layers, plain, link)
    most = next((position for position, share in enumerate(shares) if share <= 0), len(shares))
    if not most:
        return kept
    offloaded = [_offloaded(option, share, link.cost_per_byte) for option, share in zip(plain, shares[:most])]
    times = list(itertools.accumulate((_time(option) for option in offloaded), initial=0.0))
    # Offload only where it saves time, more than rounding.
    fastest = math.inf if stay is None else sum(_time(option) for option in stay.values()) * (1 - _ROUNDING)
    for in_transit in (False, True):
        budgets = _offload_budgets(stage_layers, stage, peaks, offloaded, fronts, own, (capacity, in_transit))
        rest = suffix_times([Layers(layer.key, 1, layer.in_flight) for layer in stage_layers], fronts, budgets,
                            bucket)
        count = _fastest_count(times, rest, most, fastest)
        if not count:
            return kept
        choice = choose(_groups(stage_layers[count:]), fronts, budgets[count], bucket)
        if choice is None:
            return None
        chosen, transit = _offload_choice(stage_layers, offloaded[:count], choice, in_transit)
        if in_transit or _returns_fit(stage_layers, chosen, count):
            held = _Peaks(peaks.warm_up + transit, None if peaks.backward is None else peaks.backward + transit)
            return _trim_offload(stage_layers[:count], chosen, plain, held, (stage_layers, fronts, own),
                                 (capacity, link.cost_per_byte)), transit
    return kept


def _trim_offload(
    offloading: Sequence[_Layer],
    chosen: Dict[int, LayerOption],
    plain: Sequence[LayerOption],
    held: _Peaks,
    stage: Tuple[Sequence[_Layer], Mapping[Hashable, Tuple[LayerOption, ...]], Mapping[LayerOption, LayerOption]],
    sizes: Tuple[float, float],
) -> Dict[int, LayerOption]:
    """*chosen* with no more offloaded than the stage needs, where the copies cost time.

    What the stage still has under the device's memory once its layers run
    their choice would be offloaded for nothing but the copies' cost, so the
    last offloaded layers keep it instead, a layer that keeps all it holds
    running plain.

    Args:
        offloading: The stage's layers that offload, in order.
        chosen: Each layer's option by index, those layers' offloading.
        plain: Each of the stage's layers' plain option, in order.
        held: What the stage keeps whatever its layers run, transit included.
        stage: ``(stage_layers, fronts, own)``, as for :func:`_stage_memory`.
        sizes: ``(capacity, cost_per_byte)``: the device's memory in bytes,
            and the step time one byte moved costs.

    Returns:
        The trimmed choice; *chosen* itself where the copies cost nothing.
    """
    capacity, cost_per_byte = sizes
    if cost_per_byte <= 0:
        return chosen
    stage_layers, fronts, own = stage
    # Trimmed to the byte, a stage would fill a budget of a fraction of a megabyte to that fraction, and the
    # search's result, reported in whole megabytes, would read over the budget its own fit check accepted.
    capacity = math.floor(capacity / MEGABYTE) * MEGABYTE
    slack = capacity - _stage_memory(stage_layers, held, chosen, fronts, own)
    chosen = dict(chosen)
    for position in range(len(offloading) - 1, -1, -1):
        if slack <= 0:
            break
        layer = offloading[position]
        option = chosen[layer.index]
        give = min(option.link_bandwidth, slack / layer.in_flight)
        share = option.link_bandwidth - give
        chosen[layer.index] = _offloaded(plain[position], share, cost_per_byte) if share > 0 else plain[position]
        slack -= give * layer.in_flight
    return chosen


def _stages(
    evaluator: EvaluatorV2,
    layers: Sequence[Sequence[_Layer]],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> Tuple[List[Stage], List[_Peaks]]:
    """Every stage for the knapsack, and what each keeps whatever its layers run, at each point its peak can come."""
    capacity = evaluator.ccfg.device_capacity.to_mb().size
    insights = evaluator.estimate_peak_insight()
    stages, peaks = [], []
    for stage_layers, insight, (warm_up, backward) in zip(layers, insights, evaluator.peak_points):
        stage, fixed = _stage(stage_layers, insight["Static"] + warm_up, capacity)
        stages.append(stage)
        second = None
        if backward:
            configured = {layer.index: layer.own for layer in stage_layers}
            second = (insight["Static"] + backward) * MEGABYTE - _backward_kept(stage_layers, configured, fronts, own)
        peaks.append(_Peaks(fixed, second))
    return stages, peaks


def choose_recompute(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    bucket: float = MEGABYTE,
    modes: Optional[Sequence[str]] = None,
    link: Optional[HostLink] = None,
    selective: Optional[Mapping[str, int]] = None,
    per_layer: bool = False,
) -> Optional[RecomputeChoice]:
    """The fastest recompute option of every layer that fits, at the evaluator's current strategy.

    Args:
        evaluator: The memory evaluator, set to the candidate's strategy.
        device_type: The device the times are priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        bucket: The knapsack's memory granularity, in bytes.
        modes: For a runtime that runs every layer one way, the modes it
            runs, of :data:`MODES`: the fastest that fits is chosen for every
            layer. Without them, each layer gets its own option from its
            kind's front.
        link: For a choice per layer at one chunk per stage, the host link
            each stage's first layers may offload all or part of what they
            keep over; ``None`` keeps every layer's activations on the
            device.
        selective: With *modes*, the switches the runtime's own selective
            mode sets, 1 to keep an op and 0 to recompute it, where they are
            not the config's, as HyperParallel's trainer runs its policy
            whatever its train yaml's mode. The mode is offered where every
            kind's profile measures that setting whole, as a census prices
            it; the formulas do not price the matmuls the policy recomputes.
        per_layer: With *modes*, give each layer the fastest of them that
            fits rather than one for every layer, for a runtime that runs a
            mode per layer, as HyperParallel's trainer does with
            ``activation_checkpoint.layers``. Each range then names its mode
            (:func:`trainer_plan`). With *link*, and off among *modes*, each
            stage's first layers may also run off and offload all or part of
            what they keep, reporting the mode off: priced as offload should
            run, for a runtime that offloads them, which the trainer does
            not yet.

    Returns:
        The choice, or ``None`` when there is none to make: a multimodal
        model, a schedule other than those in :data:`SCHEDULES`, a layer
        that runs an option its kind cannot offer, or a stage that does not
        fit even with the lightest option or mode.
    """
    config = evaluator.ccfg
    if config.multimodal or config.pp_sched not in SCHEDULES:
        return None
    counts = micro_batches_in_flight(evaluator)
    profiles = layer_profiles(evaluator, device_type, ccfg, most_in_flight=max(max(row) for row in counts),
                              each_switch=modes is None or ("selective" in modes and selective is None),
                              in_flight=[count for row in counts for count in row])
    modes = _priced_modes(modes, profiles, selective)
    fronts, by_mode = _offered(profiles, configured_switches(evaluator), modes, selective)
    found = _body_layers(evaluator, fronts, counts)
    if found is None:
        return None
    layers, fronts, by_mode, own = _charge_working_sets(*found, fronts, by_mode)
    stages, peaks = _stages(evaluator, layers, fronts, own)
    capacity = config.device_capacity.to_mb().size * MEGABYTE
    if by_mode is not None:
        if per_layer:
            option_modes = _option_modes(by_mode)
            if link is not None and config.vp == 1 and "off" in modes:
                return _offload_result(layers, stages, peaks, fronts, own, _link(link, evaluator), bucket, capacity,
                                       option_modes)
            return _each_layer(layers, stages, peaks, fronts, own, (bucket, capacity), option_modes)
        one = _one_mode(layers, peaks, by_mode, modes, fronts, own, capacity)
        return None if one is None else _result(layers, peaks, one[1], one[0], fronts, own)
    if link is not None and config.vp == 1:
        return _offload_result(layers, stages, peaks, fronts, own, _link(link, evaluator), bucket, capacity)
    return _each_layer(layers, stages, peaks, fronts, own, (bucket, capacity))


class RecomputeAnalysis:
    """The plans one more recomputed layer reaches at a fixed strategy, with the layer prices kept.

    The layer options, the fixed stage memory and the peak constraints are
    priced once for the strategy, as :func:`choose_recompute` prices them,
    and every alternative reuses them. The alternatives are returned
    unscored: what each would cost the step is the caller's estimate to
    make, and it is the cost model's estimate, not a measured figure.
    """

    def __init__(
        self,
        evaluator: EvaluatorV2,
        device_type: Any,
        modes: Sequence[str],
        selective: Optional[Mapping[str, int]] = None,
        dimensions: Tuple[Tuple[str, int], ...] = (),
    ) -> None:
        """Price the layer options and the fixed stage memory once for one strategy.

        Args:
            evaluator: The memory evaluator, set to the fixed strategy.
            device_type: The device the times are priced on.
            modes: The named recompute modes a layer may run, which must
                include full for a layer to be turned up to it.
            selective: The switches the runtime's own selective mode sets,
                as :func:`choose_recompute` takes them.
            dimensions: The strategy's parallel degrees, which every
                alternative reports.

        Raises:
            ValueError: No modes, or a mode that is not one of :data:`MODES`.
        """
        if not modes or set(modes) - set(MODES):
            raise ValueError(f"recompute analysis requires modes from {', '.join(MODES)}")
        self._config = evaluator.ccfg
        self._device_type = device_type
        self._dimensions = dimensions
        self._layers: List[List[_Layer]] = []
        self._fronts: Dict[Hashable, Tuple[LayerOption, ...]] = {}
        self._by_mode: Dict[Hashable, Dict[str, LayerOption]] = {}
        self._own: Dict[LayerOption, LayerOption] = {}
        self._groups: List[List[_Layer]] = []
        self._peaks: List[_Peaks] = []
        self._capacity = self._config.device_capacity.to_mb().size
        if self._config.multimodal or self._config.pp_sched not in SCHEDULES or "full" not in modes:
            return
        self._price(evaluator, modes, selective)

    @property
    def config(self) -> Any:
        """The config the layer options were priced on, which a caller scores an alternative with."""
        return self._config

    @property
    def dimensions(self) -> Tuple[Tuple[str, int], ...]:
        """The parallel degrees the analysis is fixed at."""
        return self._dimensions

    def _price(
        self, evaluator: EvaluatorV2, modes: Sequence[str], selective: Optional[Mapping[str, int]]
    ) -> None:
        """Price every kind's options and each stage's fixed memory, then group the equivalent layers."""
        counts = micro_batches_in_flight(evaluator)
        profiles = layer_profiles(
            evaluator, self._device_type, most_in_flight=max(max(row) for row in counts),
            each_switch="selective" in modes and selective is None,
            in_flight=[count for row in counts for count in row],
        )
        modes = _priced_modes(modes, profiles, selective)
        fronts, by_mode = _offered(profiles, configured_switches(evaluator), modes, selective)
        found = _body_layers(evaluator, fronts, counts)
        if found is None:
            return
        self._layers, self._fronts, self._by_mode, self._own = _charge_working_sets(*found, fronts, by_mode)
        _, self._peaks = _stages(evaluator, self._layers, self._fronts, self._own)
        self._groups = _equivalent_groups(self._layers, self._peaks)

    def alternatives(self, current: RecomputeChoice) -> List[RecomputeChoice]:
        """The distinct plans that fully recompute one more layer of *current* and still fit.

        For each group of equivalent layers and each mode its layers start
        in, one layer is turned up to full and the group's new counts are
        assigned to its layers as the normal optimizer assigns them. Index
        swaps within a group give one plan; different kinds, stages,
        in-flight counts and peak boundaries stay distinct. Every plan is
        weighed against the strategy's own device capacity.

        Args:
            current: A plan of a named mode per layer, every body layer in it.

        Returns:
            The fitting plans, without scores, each reporting the analysis's
            degrees; empty where no group can be turned up, where this
            strategy has no plan to extend, or where *current* runs one mode
            for every layer.

        Raises:
            ValueError: *current* was priced at other degrees, leaves a body
                layer out or names one twice, runs a mode this analysis does
                not price, or offloads activations.
        """
        if current.dimensions and current.dimensions != self._dimensions:
            raise ValueError("recompute alternatives must keep the same parallel dimensions")
        if not self._layers or current.mode is not None:
            return []
        defined = self._defined(current)
        chosen = self._chosen(defined)
        option_modes = _option_modes(self._by_mode)
        results: List[RecomputeChoice] = []
        seen: Dict[Tuple[FrozenSet[Any], ...], List[RecomputeChoice]] = {}
        for group in self._groups:
            for result, counts in self._turned_up(group, defined, chosen, option_modes):
                if any(_equivalent_cost(result, previous) for previous in seen.get(counts, ())):
                    continue
                seen.setdefault(counts, []).append(result)
                results.append(replace(result, dimensions=self._dimensions))
        return results

    def _defined(self, current: RecomputeChoice) -> Dict[int, LayerRange]:
        """Each body layer's range in *current*, refusing a plan that does not cover them once each."""
        defined: Dict[int, LayerRange] = {}
        for item in current.ranges:
            if item.count <= 0 or item.first < 0 or item.option.link_bandwidth:
                raise ValueError("recompute analysis requires positive layer ranges without activation offload")
            for index in range(item.first, item.first + item.count):
                if index in defined:
                    raise ValueError("recompute layer ranges must not overlap")
                defined[index] = item
        every = {layer.index for stage in self._layers for layer in stage}
        if set(defined) != every:
            raise ValueError("the recompute plan must define every body layer exactly once")
        return defined

    def _chosen(self, defined: Mapping[int, LayerRange]) -> Dict[int, LayerOption]:
        """Each body layer's option under *defined*, refusing a kind or a mode this analysis does not price."""
        chosen = {}
        for stage in self._layers:
            for layer in stage:
                item = defined[layer.index]
                if item.kind != layer.key[1] or item.mode not in self._by_mode[layer.key]:
                    raise ValueError("the recompute plan must use the fixed model's kinds and allowed modes")
                chosen[layer.index] = self._by_mode[layer.key][item.mode]
        return chosen

    def _turned_up(
        self,
        group: Sequence[_Layer],
        defined: Mapping[int, LayerRange],
        chosen: Mapping[int, LayerOption],
        option_modes: Mapping[LayerOption, str],
    ) -> Iterator[Tuple[RecomputeChoice, Tuple[FrozenSet[Any], ...]]]:
        """Yield each fitting plan that turns one of *group*'s layers up to full, with its mode counts."""
        started: set = set()
        for layer in group:
            mode = defined[layer.index].mode
            if mode == "full" or mode in started:
                continue
            started.add(mode)
            changed = dict(chosen)
            changed[layer.index] = self._by_mode[layer.key]["full"]
            changed.update(_assign_counted(group, changed))
            result = _result(self._layers, self._peaks, changed, None, self._fronts, self._own, option_modes)
            if result.memory > self._capacity:
                continue
            yield result, self._mode_counts(changed, option_modes)

    def _mode_counts(
        self, chosen: Mapping[int, LayerOption], option_modes: Mapping[LayerOption, str]
    ) -> Tuple[FrozenSet[Any], ...]:
        """Each stage's count of kinds and modes, the key plans of the same shape share."""
        return tuple(
            frozenset(Counter((layer.key[1], option_modes[chosen[layer.index]]) for layer in stage).items())
            for stage in self._layers
        )


def _equivalent_groups(layers: Sequence[Sequence[_Layer]], peaks: Sequence[_Peaks]) -> List[List[_Layer]]:
    """The layers that are interchangeable: one kind, stage and in-flight count, either side of a peak boundary.

    A stage whose peak can also come as a later backward ends keeps its
    first layer apart, since that layer's option is weighed at that peak too.
    """
    groups: List[List[_Layer]] = []
    for stage_layers, stage_peaks in zip(layers, peaks):
        grouped: Dict[Tuple[Hashable, int, bool], List[_Layer]] = {}
        for layer in stage_layers:
            at_boundary = stage_peaks.backward is not None and layer.index == stage_layers[0].index
            grouped.setdefault((layer.key, layer.in_flight, at_boundary), []).append(layer)
        groups.extend(grouped.values())
    return groups


def _option_modes(by_mode: Mapping[Hashable, Mapping[str, LayerOption]]) -> Dict[LayerOption, str]:
    """Each option of every kind's modes, by the mode it runs."""
    return {option: mode for options in by_mode.values() for mode, option in options.items()}


def _each_layer(
    layers: Sequence[Sequence[_Layer]],
    stages: Sequence[Stage],
    peaks: Sequence[_Peaks],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    sizes: Tuple[float, float],
    modes: Optional[Mapping[LayerOption, str]] = None,
) -> Optional[RecomputeChoice]:
    """Each layer's fastest option of its kind's that fits, stage by stage.

    Args:
        sizes: ``(bucket, capacity)``, the knapsack's granularity and the
            device's memory, in bytes.
        modes: For a choice of a mode per layer, each option's mode.

    Returns:
        The choice, or ``None`` when some stage does not fit.
    """
    bucket, capacity = sizes
    chosen: Dict[int, LayerOption] = {}
    for stage_layers, stage, stage_peaks in zip(layers, stages, peaks):
        stage_chosen = _choose_stage(stage_layers, stage, stage_peaks, fronts, own, bucket, capacity)
        if stage_chosen is None:
            return None
        chosen.update(stage_chosen)
    return _result(layers, peaks, chosen, None, fronts, own, modes)


def _offload_result(
    layers: Sequence[Sequence[_Layer]],
    stages: Sequence[Stage],
    peaks: Sequence[_Peaks],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    link: _Link,
    bucket: float,
    capacity: float,
    modes: Optional[Mapping[LayerOption, str]] = None,
) -> Optional[RecomputeChoice]:
    """The choice per layer, each stage's first layers offloading where that is faster.

    Args:
        modes: For a choice of a mode per layer, each option's mode. An
            offloaded layer runs its kind's plain option, and reports that
            option's mode.
    """
    chosen: Dict[int, LayerOption] = {}
    held = []
    for stage_layers, stage, stage_peaks in zip(layers, stages, peaks):
        found = _offload_stage(stage_layers, stage, stage_peaks, fronts, own, link, bucket, capacity)
        if found is None:
            return None
        chosen.update(found[0])
        transit = found[1]
        held.append(_Peaks(stage_peaks.warm_up + transit,
                           None if stage_peaks.backward is None else stage_peaks.backward + transit))
    if modes is not None:
        modes = dict(modes)
        for layer in (layer for stage_layers in layers for layer in stage_layers):
            if chosen[layer.index].link_bandwidth:
                plain = _plain(fronts[layer.key])
                modes[chosen[layer.index]] = modes[own.get(plain, plain)]
    return _result(layers, held, chosen, None, fronts, own, modes)


def offloaded_share(option: LayerOption) -> float:
    """The share of what a layer running *option* keeps per micro-batch that it moves to the host; 0 without offload."""
    if not option.link_bandwidth:
        return 0.0
    return option.link_bandwidth / (option.link_bandwidth + max(0.0, option.memory_per_micro_batch))


def option_label(option: LayerOption) -> str:
    """How an option reads in the search's output."""
    if option.recompute is None:
        label = "full recompute"
    elif not option.recompute:
        label = "no recompute"
    else:
        label = "recompute " + "+".join(name for name in SWITCHES if name in option.recompute)
    if not option.link_bandwidth:
        return label
    return f"{label}, {offloaded_share(option):.0%} offloaded to the host"


def option_record(option: LayerOption) -> Any:
    """An option as plain data: ``"none"``, ``"full"``, or the switches it recomputes, in switch order."""
    if option.recompute is None:
        return "full"
    if not option.recompute:
        return "none"
    return [name for name in SWITCHES if name in option.recompute]


def to_records(choice: RecomputeChoice) -> List[Dict[str, Any]]:
    """The choice's ranges as plain data, for a result file.

    Returns:
        One ``{"first", "count", "kind", "recompute"}`` per range, in model
        order, with ``"offload": True`` and ``"offloaded_bytes"``, what each
        of its layers moves per micro-batch, for a range that offloads, and
        ``"mode"`` for one a choice of a mode per layer gave its mode;
        ``kind`` is the kind's name, or ``None``.
    """
    records = []
    for item in choice.ranges:
        record = {
            "first": item.first,
            "count": item.count,
            "kind": item.kind.name if item.kind is not None else None,
            "recompute": option_record(item.option),
        }
        if item.option.link_bandwidth:
            record["offload"] = True
            record["offloaded_bytes"] = item.option.link_bandwidth
        if item.mode is not None:
            record["mode"] = item.mode
        records.append(record)
    return records


def trainer_plan(choice: RecomputeChoice) -> Tuple[str, Dict[str, str]]:
    """The choice as HyperParallel's trainer states it: ``activation_checkpoint.mode`` and ``.layers``.

    The trainer runs every layer its mode but those its plan names, by index
    or by an inclusive range ``"first-last"``. The mode is the one most
    layers run of those that recompute, full on a tie, or off where none
    recomputes; the plan names the other layers, consecutive layers of one
    mode in one entry whatever their kinds.

    The mode is a mode that recomputes, rather than off with the recomputed
    layers named, because the trainer accepts an entry equal to the mode: a
    runner that states the mode itself, as a sweep forcing its own full
    default does, then still runs the layers this plan states off. Named the
    other way round, the same run would recompute every layer and invert the
    choice without a word.

    The indices are the choice's, which count the body layers in model order,
    MTP's after the decoder's: the indices of the trainer's blocks, for a
    model without MTP layers.

    Args:
        choice: A choice of one mode for every layer, or of a mode per layer
            (:func:`choose_recompute` with *per_layer*).

    Returns:
        ``(mode, layers)``, *layers* empty when every layer runs *mode*.

    Raises:
        ValueError: For a choice of options no mode stands for, each layer's
            own from its kind's front.
    """
    if choice.mode is not None:
        return choice.mode, {}
    if any(item.mode is None for item in choice.ranges):
        raise ValueError("the trainer runs a mode per layer, and this choice gives some layers options that are not "
                         "one; choose among its modes with per_layer")
    counts = Counter()
    for item in choice.ranges:
        counts[item.mode] += item.count
    recomputing = [mode for mode in ("full", "selective") if counts[mode]]
    if not recomputing:
        return "off", {}
    mode = max(recomputing, key=lambda name: counts[name])
    runs: List[List[Any]] = []
    for item in sorted(choice.ranges, key=lambda item: item.first):
        if runs and runs[-1][2] == item.mode and runs[-1][1] + 1 == item.first:
            runs[-1][1] = item.first + item.count - 1
        else:
            runs.append([item.first, item.first + item.count - 1, item.mode])
    return mode, {
        (str(first) if first == last else f"{first}-{last}"): run_mode
        for first, last, run_mode in runs
        if run_mode != mode
    }


# Where the installed trainer configures activation checkpointing: the module,
# and the dataclass whose fields are the keys ``activation_checkpoint`` takes.
_TRAINER_CHECKPOINT_CONFIG = ("hyper_parallel.trainer.config.parallelism", "ActivationCheckpointConfig")


def trainer_runs_layer_plans() -> bool:
    """Whether the installed trainer runs a recompute mode per layer: its ``activation_checkpoint`` takes ``layers``.

    A trainer whose ``ActivationCheckpointConfig`` declares no ``layers``
    field refuses a yaml that states one, so a plan per layer
    (:func:`trainer_plan`) is then advice the trainer cannot run: it runs
    every layer the plan's mode.  The trainer's configuration loads only
    when asked, as importing it costs about as much as the cost model.

    Returns:
        True where the dataclass declares ``layers``; False where it does
        not, or where this tree has no such trainer.
    """
    module, name = _TRAINER_CHECKPOINT_CONFIG
    try:
        config = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return False
    return is_dataclass(config) and any(field.name == "layers" for field in fields(config))


def describe(choice: RecomputeChoice) -> str:
    """The choice as one line per range of layers, after its mode when one was chosen for every layer.

    A range a choice of a mode per layer gave its mode reads as the mode.
    """
    lines = [] if choice.mode is None else [f"every layer: {choice.mode}"]
    for item in choice.ranges:
        last = item.first + item.count - 1
        layers = f"layer {item.first}" if item.count == 1 else f"layers {item.first}-{last}"
        kind = f" ({item.kind.name})" if item.kind is not None else ""
        lines.append(f"{layers}{kind}: {option_label(item.option) if item.mode is None else item.mode}")
    return "\n".join(lines)
