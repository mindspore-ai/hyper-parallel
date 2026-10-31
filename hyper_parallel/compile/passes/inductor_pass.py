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
"""
Inductor Compilation Pass - terminal backend compilation of the joint graph.

The pass lowers the (already partitioned and overlapped) joint FX graph to
TorchInductor kernels via ``torch.fx.passes.regional_inductor``, which scoops
nodes annotated with ``meta["custom"]["compile_with_inductor"]`` into regions
and compiles each region with ``torch._inductor.standalone_compile``. This
brings kernel fusion, c10d functionalization, PG unboxing and Inductor caching
to the graph-mode stack without touching eager semantics.

Modes (``PassConfig.inductor_mode``):

- ``"off"`` (default): the pass is not added to the pipeline.
- ``"regional"``: tag functional compute nodes as compile regions while
  keeping communication (``_c10d_functional`` collectives + ``wait_tensor``)
  and side-effectful nodes (e.g. the FSDP reshard storage free) eager. This
  preserves the comm scheduling surface for the overlap passes.
- ``"full"``: tag every node so the whole graph compiles as one region
  (c10d functional collectives included), mirroring torchtitan's
  ``full_inductor_compilation_pass``.

Pipeline-parallel support: when ``PpPass`` has already split the joint graph,
the outer graph is a ``call_module`` stub around the pipeline schedule. The
compute lives in the schedule's per-virtual-stage forward/backward subgraphs,
so the pass detects the stub and compiles every stage subgraph independently
(both modes), instead of compiling — or rejecting — the stub. Cross-stage
fusion is inherently impossible; the schedule's P2P stays outside the
compiled regions. Unknown ``call_module`` graphs are still rejected. PP
forward subgraphs export saved values verbatim, which can include multi-result
ops (``nll_loss_forward``); their nested output is split into ``getitem``
leaves for compilation and rebuilt by a thin wrapper, because AOTAutograd's
regional path requires flat outputs.

Numerics policy: ``inductor.eager_numerics.division_rounding`` is forced on so
compiled regions stay bitwise-equal to eager execution (same guarantee
torchtitan relies on).

Compatibility: ``torch.fx.passes.regional_inductor`` only exists in newer
torch builds (it is absent from 2.6 / 2.7 / 2.9). On such builds the pass logs
a warning, leaves the graph untouched and records ``inductor_compiled =
"unavailable"`` instead of raising ``ModuleNotFoundError``.

Contract notes:

- The pass must run LAST: after compilation the FX graph is a thin wrapper
  around the compiled callable and is no longer authoritative for
  graph-level passes. PP stage subgraphs are compiled through the schedule's
  ``map_stage_graphs`` hook, so the stub remains the outer entry point.
- ``regional_inductor`` switches a compiled graph module to the boxed calling
  convention (``forward(self, args_list)``); the pass resets it to the
  default positional convention so ``graph_module(*flat_inputs)`` (and the
  schedule's direct ``fwd_gm(*args)`` / ``bwd_gm(*args)`` calls) keep
  working unchanged.
- The graph must be traced with ``make_fx`` (placeholders carry fake
  ``meta["val"]``). Nodes inserted by earlier passes lack fake values, so the
  pass first re-propagates them with ``FakeTensorProp`` using the graph's own
  ``FakeTensorMode``.
"""

__all__ = ["InductorPass"]

import importlib
import logging
import operator
from typing import Any, Dict, Optional, Sequence

import torch
from torch import fx, nn
from torch._subclasses import FakeTensor
from torch.fx.graph import CodeGen
from torch.fx.node import _side_effectful_functions
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from ..pass_config import PassConfig
from .base import GraphPass

_LOG = logging.getLogger(__name__)

# Communication nodes stay outside compiled regions in "regional" mode. The
# launch/wait pair is the scheduling surface the overlap passes reorder; a
# compiled region would fuse a collective into surrounding compute and remove
# the explicit reorder point. "full" mode does compile them (Inductor's c10d
# functionalization supports that), matching torchtitan's full pass.
# ``str(OpOverload)`` has no ``torch.ops.`` prefix, so match the bare namespace.
_C10D_FUNCTIONAL_NS = "_c10d_functional."

# Meta key recording that (and how) a graph module was compiled.
_INDUCTOR_COMPILED_META = "inductor_compiled"

# Meta value when the running torch predates regional_inductor (<= 2.9.x).
_INDUCTOR_UNAVAILABLE = "unavailable"


def _regional_inductor_available() -> bool:
    """Whether this torch build ships ``torch.fx.passes.regional_inductor``.

    The module first appeared after torch 2.9, while this project still
    supports 2.6 / 2.7 / 2.9. Importing it lazily here lets the pass degrade to
    a documented no-op instead of raising ``ModuleNotFoundError`` on those
    builds.
    """
    try:
        importlib.import_module("torch.fx.passes.regional_inductor")
    except ImportError:
        return False
    return True


def _get_fake_mode_from_gm(gm: fx.GraphModule) -> Optional[Any]:
    """Return the ``FakeTensorMode`` shared by the graph's placeholder vals."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if isinstance(val, FakeTensor):
                return val.fake_mode
    return None


def _placeholder_fake_args(gm: fx.GraphModule) -> list:
    """Return the placeholder values in order, requiring fake ``meta["val"]``."""
    args = []
    for node in gm.graph.find_nodes(op="placeholder"):
        val = node.meta.get("val")
        if val is None:
            raise ValueError(
                f"InductorPass requires a fake meta['val'] on placeholder "
                f"'{node.name}' to size compiled regions; the graph must be "
                f"make_fx-traced (PP stage subgraphs copy their placeholders' "
                f"meta from the joint graph)"
            )
        args.append(val)
    return args


def _propagate_fake_vals(gm: fx.GraphModule) -> fx.GraphModule:
    """Fill ``meta["val"]`` on nodes inserted after tracing (FSDP collectives,
    pad views, reshard frees) so ``regional_inductor`` can size each region.

    Propagation reuses the graph's own ``FakeTensorMode``: placeholder vals
    already belong to it and mixing modes is unsupported by ``FakeTensorProp``.
    Side-effectful free nodes return ``None`` and legitimately carry no val.
    """
    fake_mode = _get_fake_mode_from_gm(gm)
    if fake_mode is None:
        raise ValueError(
            "InductorPass requires a make_fx-traced graph whose placeholders "
            "carry fake meta['val'] tensors"
        )
    fake_args = _placeholder_fake_args(gm)
    FakeTensorProp(gm, fake_mode).propagate_dont_convert_inputs(*fake_args)
    return gm


def _node_is_comm(node: fx.Node) -> bool:
    """Whether the node is a functional collective or its wait."""
    return str(node.target).startswith(_C10D_FUNCTIONAL_NS)


def _node_is_side_effectful(node: fx.Node) -> bool:
    """Whether the node targets a function registered as side-effectful.

    ``FSDPPass`` registers the reshard storage-free op there so DCE keeps it;
    compiling such a node would let Inductor drop or defer the free.
    """
    return node.op == "call_function" and node.target in _side_effectful_functions


def _flatten_nested_outputs(
    graph_module: fx.GraphModule,
) -> tuple:
    """Rewrite tuple/list graph outputs into a flat tuple of leaf values.

    AOTAutograd's regional compilation expects flat tensor outputs. The joint
    graph is flat, but PP stage forward subgraphs export every value the
    backward references verbatim — including multi-result ops such as
    ``nll_loss_forward`` — so their output tuple can contain a nested
    ``(loss, total_weight)`` segment, which trips AOT's output-spec check.
    Splitting nested values into ``getitem`` leaves keeps every region output
    flat; the returned spec lets the caller rebuild the original structure
    for the schedule / backward subgraph, whose contract is unchanged.

    Returns:
        ``(spec, graph_module)`` where ``spec`` is ``None`` when the output is
        already flat, otherwise a nested marker tree consumed by
        ``_rebuild_nested_output``.
    """
    graph = graph_module.graph
    output = next((n for n in graph.nodes if n.op == "output"), None)
    returned = output.args[0] if output is not None else None
    if not isinstance(returned, (tuple, list)):
        return None, graph_module
    if not any(
        isinstance(node.meta.get("val"), (tuple, list))
        for node in returned
        if isinstance(node, fx.Node)
    ):
        return None, graph_module

    flat_nodes: list = []

    def build(node: fx.Node) -> tuple:
        """Emit ``getitem`` leaves for a nested value, else record the leaf."""
        val = node.meta.get("val")
        if isinstance(val, (tuple, list)) and val:
            children = []
            for index, element in enumerate(val):
                sub = graph.call_function(operator.getitem, (node, index))
                sub.meta["val"] = element
                children.append(build(sub))
            kind = "tuple" if isinstance(val, tuple) else "list"
            return kind, tuple(children)
        flat_nodes.append(node)
        return "leaf", len(flat_nodes) - 1

    with graph.inserting_before(output):
        spec = tuple(build(node) for node in returned)
    output.args = (tuple(flat_nodes),)
    graph.lint()
    graph_module.recompile()
    return spec, graph_module


def _rebuild_nested_output(spec: tuple, flat: Sequence[Any]) -> list:
    """Rebuild the original output structure from ``spec`` and flat leaves."""
    leaves = iter(flat)
    rebuilt = []
    for kind, payload in spec:
        if kind == "leaf":
            rebuilt.append(next(leaves))
        else:
            children = _rebuild_nested_output(payload, leaves)
            rebuilt.append(tuple(children) if kind == "tuple" else children)
    return rebuilt


class _NestedOutputWrapper(nn.Module):
    """Thin wrapper restoring a compiled graph's nested output structure."""

    def __init__(
        self,
        inner: nn.Module,
        spec: tuple,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store the compiled inner module and its output marker tree."""
        super().__init__()
        self.inner = inner
        self._output_spec = spec
        # Mirrors GraphModule.meta so callers can introspect compilation mode.
        self.meta = dict(meta or {})

    def forward(self, *args: Any, **kwargs: Any) -> tuple:
        """Run the flat compiled graph and rebuild the original structure."""
        flat = self.inner(*args, **kwargs)
        return tuple(_rebuild_nested_output(self._output_spec, flat))


def _tag_node(node: fx.Node, inductor_configs: Optional[Dict[str, Any]]) -> None:
    """Annotate ``node`` for regional_inductor with the user's Inductor config."""
    node.meta.setdefault("custom", {})["compile_with_inductor"] = {
        "inductor_configs": dict(inductor_configs or {}),
    }


class InductorPass(GraphPass):
    """Terminal backend compilation pass (TorchInductor).

    See the module docstring for the mode semantics, the numerics policy and
    the calling-convention contract.
    """

    @property
    def name(self) -> str:
        """Pass name."""
        return "inductor_compilation"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        example_inputs: Optional[tuple] = None,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Compile the joint graph (or its PP stages) with Inductor.

        Args:
            graph_module: The joint FX graph produced by the tracer and
                transformed by the earlier partitioning / overlap passes. When
                ``PpPass`` already ran, this is the schedule stub and its
                stage subgraphs are compiled instead.
            pass_config: Graph-mode configuration; ``inductor_mode`` selects
                the behavior, ``inductor_configs`` overrides Inductor options.
            example_inputs: Fake inputs matching the joint graph's
                placeholders in order (the tracer's
                ``JointGraph.example_inputs``). Ignored for PP stage
                subgraphs, whose inputs are derived from each subgraph's
                placeholder metas.

        Returns:
            The same graph module with compiled regions spliced in and the
            positional calling convention restored. On torch builds without
            regional Inductor support it is returned untouched, logging a
            warning and recording ``"unavailable"`` in its meta.

        Raises:
            ValueError: When ``example_inputs`` is missing for a non-PP graph,
                or a ``call_module`` stub is not a pipeline schedule (unknown
                split).
        """
        mode = pass_config.inductor_mode
        if mode == "off":
            return graph_module

        compiled = False
        if graph_module.graph.find_nodes(op="call_module"):
            compiled = self._compile_pipeline_stages(graph_module, pass_config)
        else:
            if example_inputs is None:
                raise ValueError(
                    "InductorPass requires example_inputs (the tracer's "
                    "JointGraph.example_inputs) to size compiled regions"
                )
            if _regional_inductor_available():
                self._compile_graph(graph_module, pass_config, example_inputs)
                compiled = True
            else:
                self._warn_unavailable(mode)

        graph_module.meta[_INDUCTOR_COMPILED_META] = (
            mode if compiled else _INDUCTOR_UNAVAILABLE
        )
        return graph_module

    @staticmethod
    def _warn_unavailable(mode: str) -> None:
        """Warn that the running torch predates regional Inductor compilation."""
        _LOG.warning(
            "InductorPass: torch %s does not provide "
            "torch.fx.passes.regional_inductor; skipping %r compilation "
            "(requires a newer torch build)",
            torch.__version__,
            mode,
        )

    def _compile_pipeline_stages(
        self, graph_module: fx.GraphModule, pass_config: PassConfig
    ) -> bool:
        """Compile every stage subgraph installed by ``PpPass``.

        ``PpPass`` replaces the joint graph with a ``call_module`` stub around
        the pipeline schedule; the FX compute lives in the schedule's
        per-virtual-stage forward/backward subgraphs (see
        ``PipelineScheduleBase.map_stage_graphs``). Compiling the stub itself
        would be a no-op and the stub must remain the outer entry point.

        Returns:
            ``True`` when the stage graphs were compiled, ``False`` when the
            running torch does not provide regional Inductor compilation (the
            pass then no-ops after ``_warn_unavailable``).
        """
        schedules = []
        for node in graph_module.graph.find_nodes(op="call_module"):
            submodule = graph_module.get_submodule(node.target)
            if not callable(getattr(submodule, "map_stage_graphs", None)):
                raise ValueError(
                    f"InductorPass cannot compile call_module '{node.target}': "
                    f"only PpPass's pipeline schedule exposes stage subgraphs "
                    f"(map_stage_graphs) for backend compilation"
                )
            schedules.append(submodule)

        if not _regional_inductor_available():
            self._warn_unavailable(pass_config.inductor_mode)
            return False

        for schedule in schedules:
            schedule.map_stage_graphs(
                lambda stage_gm: self._compile_graph(
                    stage_gm, pass_config, example_inputs=None
                )
            )
        _LOG.info(
            "InductorPass: compiled %d PP schedule(s) in %r mode",
            len(schedules),
            pass_config.inductor_mode,
        )
        return True

    def _compile_graph(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        example_inputs: Optional[Sequence[Any]],
    ) -> fx.GraphModule:
        """Compile one FX graph (joint graph or one PP stage subgraph).

        ``example_inputs`` is ``None`` for PP stage subgraphs, whose inputs
        are derived from their own placeholder metas (the joint graph's
        inputs do not match a stage's placeholder list).
        """
        mode = pass_config.inductor_mode
        output_spec, graph_module = _flatten_nested_outputs(graph_module)
        _propagate_fake_vals(graph_module)
        if example_inputs is None:
            example_inputs = _placeholder_fake_args(graph_module)
        _LOG.info(
            "InductorPass: compiling graph in %r mode (%d placeholders, %d nodes)",
            mode,
            len(graph_module.graph.find_nodes(op="placeholder")),
            len(graph_module.graph.nodes),
        )

        if mode == "full":
            self._tag_full(graph_module, pass_config.inductor_configs)
        else:
            self._tag_regional(graph_module, pass_config.inductor_configs)

        # Imported lazily so this module does not pull in the (heavy)
        # TorchInductor stack unless compilation actually runs.
        import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel
        from torch._guards import (  # pylint: disable=import-outside-toplevel
            TracingContext,
            tracing,
        )
        from torch.fx.passes.regional_inductor import (  # pylint: disable=import-outside-toplevel
            regional_inductor,
        )

        # Bitwise-equal numerics vs eager (torchtitan's regional_inductor
        # policy). emulate_precision_casts / disable_ftz stay off: enabling
        # them is a numerics-changing opt-in, not this pass's default. Scope
        # the override so it cannot leak into other Inductor compiles.
        numerics = {"eager_numerics.division_rounding": True}

        fake_mode = _get_fake_mode_from_gm(graph_module)
        tracing_ctx = TracingContext(fake_mode)
        with (
            inductor_config.patch(  # pylint: disable=no-member
                numerics
            ),
            tracing(tracing_ctx),
        ):
            # ``regional_inductor`` consumes these on torch releases whose
            # signature expects example inputs; newer releases size regions
            # from the nodes' fake ``meta["val"]`` instead.
            regional_inductor(graph_module, example_inputs)

        # regional_inductor compiles regions with the boxed calling
        # convention (one mutable args list); reset to the default positional
        # convention so run_traced_graph's graph_module(*flat_inputs) (and the
        # PP schedule's fwd_gm(*args) / bwd_gm(*args)) keeps working. Setting
        # the default unconditionally is a no-op when the region compiler
        # already used positional codegen and avoids peeking at private state.
        graph_module.graph.set_codegen(CodeGen())
        graph_module.recompile()
        graph_module.meta[_INDUCTOR_COMPILED_META] = mode
        if output_spec is not None:
            # Rebuild the stage's original (possibly nested) output structure
            # for the schedule / backward subgraph; only the compiled graph is
            # flat.
            return _NestedOutputWrapper(graph_module, output_spec, graph_module.meta)
        return graph_module

    @staticmethod
    def _tag_full(
        graph_module: fx.GraphModule, inductor_configs: Optional[Dict[str, Any]]
    ) -> None:
        """Tag every node so the whole graph compiles as one region.

        Mirrors torchtitan's ``full_inductor_compilation_pass``: Inductor's
        ``reorder_for_peak_memory`` restores the fwd/bwd interleaving that
        AOTAutograd flattening would otherwise drop. Side-effectful nodes
        (the FSDP reshard storage free) stay untagged: AOTAutograd's
        functionalization would re-execute them on functional tensors and
        lose the storage release.
        """
        configs = dict(inductor_configs or {})
        configs.setdefault("reorder_for_peak_memory", True)
        # Flat graph (joint graph or one PP stage subgraph): tag its nodes
        # directly (no nested ``call_module`` submodules to recurse into).
        for node in graph_module.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if _node_is_side_effectful(node):
                continue
            _tag_node(node, configs)

    @staticmethod
    def _tag_regional(
        graph_module: fx.GraphModule, inductor_configs: Optional[Dict[str, Any]]
    ) -> None:
        """Tag functional compute nodes, keeping comms and side effects eager.

        Untagged nodes become region boundaries: contiguous runs of tagged
        compute between collectives / frees are each fused and compiled, and
        the collective launch/wait nodes remain plain FX nodes so the overlap
        passes' scheduling decisions stay observable at runtime.
        """
        for node in graph_module.graph.nodes:
            if node.op not in ("call_function", "call_method"):
                continue
            if _node_is_comm(node) or _node_is_side_effectful(node):
                continue
            _tag_node(node, inductor_configs)
