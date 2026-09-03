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
PP Pass - Pipeline-Parallel Graph Split Pass

Splits the (already FSDP-transformed) joint fwd+bwd FX graph into this
rank's pipeline stage, entirely at graph level — no model deepcopy, no
``torch.distributed.pipelining`` dependency. Where torchtitan splits at the
*module* level (``_split_module`` deepcopy + layer deletion, then wraps each
chunk in a ``PipelineStage``), this pass:

1. Attributes every FX node to a stage via ``nn_module_stack`` metadata
   (parameters are static placeholders, attributed by FQN ancestor walk).
   Root-level nodes the stack cannot resolve (residual adds written in the
   parent's ``forward``, loss glue) are attributed by their CONSUMERS, so a
   residual add sandwiched between two stages rides with the stage that
   consumes it.
2. Detects the stage boundary as the set of values crossing each cut —
   activations forward, gradients backward. Real ATen graphs cross with
   more than the trunk tensor: dynamic-shape ``sym_size`` scalars and
   parent-level views all cross, so every crossing value is shipped (int
   scalars are packed as 0-d int64 tensors by the schedule and unwrapped
   with ``item()`` on arrival).
3. Slices stage ``k`` into a forward subgraph and a backward subgraph. The
   forward outputs (all crossing activations plus every intermediate the
   backward references) become backward placeholders, so the GPipe schedule
   can run all forwards before all backwards.
4. Prunes foreign-stage submodules from the live model (torch's
   ``named_parameters`` then yields only this stage's entries, keeping the
   trainer / optimizer FSDP- and PP-agnostic).
5. Installs a self-contained ``ScheduleGPipe`` (eager ``dist.isend`` /
   ``irecv`` P2P) as a ``call_module`` node in a stub graph — the trainer's
   ``graph_module(*flat_inputs)`` dispatches to it unchanged, and the stub
   survives any later ``recompile()`` because the schedule lives in the
   graph itself, not in an overwritten ``forward``.

Run order in ``PassPipeline``: AFTER ``FSDPPass``. The FSDP collectives are
value-level FX nodes that follow the data flow: this stage's all_gather /
reduce_scatter nodes land inside its slices untouched, while foreign
stages' collectives are simply never copied. Splitting first (PP before
FSDP) would instead hand ``FSDPPass`` two subgraphs and a loss-less output
list, breaking its grad-index contract.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
from torch import fx, nn

from ...parallel_config import PassConfig
from ..base import GraphPass
from ...sharding_config import PassPlan
from .pp_schedule import ScheduleGPipe

_LOG = logging.getLogger(__name__)

_FWD = "fwd"
_BWD = "bwd"


def _sanitize(name: str) -> str:
    """Turn a dotted FQN into a codegen-safe placeholder name."""
    return name.replace(".", "_")


def _iter_node_args(node: fx.Node) -> Iterator[fx.Node]:
    """Yield every ``fx.Node`` reachable in ``node.args`` / ``node.kwargs``."""
    stack: List[Any] = list(node.args) + list(node.kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, fx.Node):
            yield item
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())


def _tensor_arg_stages(node: fx.Node, node_stage: Dict[fx.Node, int]) -> List[int]:
    """Stages of ``node``'s TENSOR-valued resolved args.

    Shape-metadata args (dynamic-shape ``sym_size`` scalars) are excluded:
    they are born wherever shapes are known, carry no data dependency, and
    would otherwise drag a node's placement against its dataflow (e.g. a
    boundary-gradient view whose ``sym_size`` args are stage-0 born while
    the gradient itself is born downstream).
    """
    stages = []
    for arg in _iter_node_args(node):
        stage = node_stage.get(arg)
        if stage is None:
            continue
        val = arg.meta.get("val")
        if val is not None and not isinstance(val, torch.Tensor):
            continue
        stages.append(stage)
    return stages


def _innermost_fqns(node: fx.Node) -> List[str]:
    """Module FQNs on ``node``'s ``nn_module_stack``, innermost first."""
    stack = node.meta.get("nn_module_stack") or {}
    return [fqn for fqn, _ in reversed(list(stack.values()))]


def _even_split(names: List[str], pp_degree: int, prefix: str = "") -> List[List[str]]:
    """Distribute ``names`` across ``pp_degree`` stages as evenly as possible.

    The first ``len(names) % pp_degree`` stages get one extra entry; each
    entry is prefixed (``prefix``) so bare child names become FQNs.
    """
    base, rem = divmod(len(names), pp_degree)
    stages: List[List[str]] = [[] for _ in range(pp_degree)]
    cursor = 0
    for s in range(pp_degree):
        size = base + (1 if s < rem else 0)
        stages[s] = [f"{prefix}{name}" for name in names[cursor : cursor + size]]
        cursor += size
    return stages


def _auto_stage_split(model: nn.Module, pp_degree: int) -> List[List[str]]:
    """Default PP split: even distribution of the model's layer container.

    Mirrors torchtitan's ``_generate_llm_fqn_per_model_part`` weighting
    without hardcoding LLM module names:

    - The first top-level child that is an ``nn.ModuleList`` with at least
      ``pp_degree`` entries is the *layer container* (typically ``layers``).
    - Top-level children registered before it go to stage 0 (input side,
      e.g. embeddings); children after it go to the last stage (output
      side, e.g. norm / lm_head).
    - Container elements are distributed as evenly as possible (see
      ``_even_split``).

    Without a qualifying container, all top-level children are split evenly
    in registration order.

    Returns:
        ``pp_degree`` lists of exact module FQNs, one per stage.
    """
    if pp_degree < 2:
        raise ValueError(f"pp_degree must be >= 2 for a PP split, got {pp_degree}")

    children = [name for name, _ in model.named_children()]

    container: Optional[str] = None
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) >= pp_degree:
            container = name
            break

    if container is None:
        if len(children) < pp_degree:
            raise ValueError(
                f"Cannot split {len(children)} top-level children across "
                f"{pp_degree} PP stages; provide a manual stage plan via "
                f"PassPlan.pp_stage()"
            )
        return _even_split(children, pp_degree)

    container_mod = getattr(model, container)
    if isinstance(container_mod, nn.ModuleDict):
        elem_keys = list(container_mod.keys())
    else:
        elem_keys = [str(i) for i in range(len(container_mod))]

    idx = children.index(container)
    before, after = children[:idx], children[idx + 1 :]

    stages = _even_split(elem_keys, pp_degree, prefix=f"{container}.")
    stages[0] = before + stages[0]
    stages[-1] = stages[-1] + after
    return stages


@dataclass
class _RunContext:
    """This rank's PP execution context, resolved and guarded for ``run``."""

    model: nn.Module
    pp_degree: int
    group: Any
    stage_idx: int


@dataclass
class _StageSplit:
    """Stage plan and per-node classification produced by ``_classify_stage``.

    Bundles what the boundary-detection and subgraph-construction phases
    need, so ``run`` orchestrates phases instead of threading a dozen
    parallel locals through every call.
    """

    state_fqns: List[str]
    state_is_param: List[bool]
    stage_plan: List[List[str]]
    node_stage: Dict[fx.Node, int]
    node_phase: Dict[fx.Node, str]
    input_ph: fx.Node
    label_ph: fx.Node
    state_phs: List[fx.Node]
    stage_state_indices: List[int]
    stage_state_fqns: List[str]


class PpPass(GraphPass):
    """Pipeline-parallel partitioning pass (graph-level stage split)."""

    name = "pp_parallel"

    def __init__(self, pass_plan: Optional[PassPlan] = None) -> None:
        """Initialize PP pass state.

        Args:
            pass_plan: Declarative plan; ``pp_module_fqns_per_stage`` (from
                ``pp_stage()``) overrides the automatic even-by-layers
                split.
        """
        super().__init__()
        self._pass_plan = pass_plan

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Split the joint graph to this rank's stage and install the schedule.

        Args:
            graph_module: Joint fwd+bwd FX graph (post-``FSDPPass``).
            pass_config: Parallel configuration; ``pp_degree`` (``None``
                resolves to ``world_size``) and ``pp_microbatch_size`` are
                read directly.
            **kwargs: Must include ``model`` (the live ``nn.Module``); may
                include ``pass_plan``.

        Returns:
            The same graph module, rewritten into a stage stub whose
            ``call_module`` node invokes the installed ``ScheduleGPipe``.
        """
        run_ctx = self._resolve_run_context(pass_config, kwargs)
        if run_ctx is None:
            return graph_module

        split = self._classify_stage(graph_module, run_ctx)
        boundaries = self._find_boundaries(
            graph_module,
            split.node_stage,
            split.node_phase,
            run_ctx.pp_degree,
            run_ctx.stage_idx,
        )
        self._build_and_install_stage(
            graph_module, pass_config, run_ctx, split, boundaries
        )
        return graph_module

    def _resolve_run_context(
        self,
        pass_config: PassConfig,
        kwargs: Dict[str, Any],
    ) -> Optional[_RunContext]:
        """Guard and resolve the model, PP group, and this rank's stage.

        Returns:
            The run context, or ``None`` when the pass is skipped
            (distributed not initialized, ``world_size == 1``, or
            ``pp_degree < 2``).
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            _LOG.info("Skipped: distributed not initialized or world_size=1")
            return None

        model = kwargs.get("model")
        if model is None:
            raise ValueError(
                "PpPass requires the live model via kwargs (model=...) so it "
                "can resolve the stage plan and prune foreign-stage modules"
            )
        self._pass_plan = kwargs.get("pass_plan", self._pass_plan)

        # Skip BEFORE _resolve_group_and_stage so a disabled pass does not
        # execute the collective dist.new_group.
        pp_degree = pass_config.pp_degree or dist.get_world_size()
        if pp_degree < 2:
            _LOG.info("Skipped: pp_degree=%s < 2", pp_degree)
            return None
        pp_degree, group, stage_idx = self._resolve_group_and_stage(pass_config)
        _LOG.info(
            "Running: stage %s/%s, world_size=%s",
            stage_idx,
            pp_degree,
            dist.get_world_size(),
        )
        return _RunContext(
            model=model, pp_degree=pp_degree, group=group, stage_idx=stage_idx
        )

    def _classify_stage(
        self, graph_module: fx.GraphModule, run_ctx: _RunContext
    ) -> _StageSplit:
        """Resolve the stage plan and classify every node's stage + phase.

        Chains the state-list contract checks, the stage-plan resolution,
        node classification, the gradient-output anchoring with its stage
        propagation, and the adjacent-dataflow validation.

        Args:
            graph_module: Joint fwd+bwd FX graph (post-``FSDPPass``).
            run_ctx: This rank's PP execution context.

        Returns:
            The ``_StageSplit`` bundling the plan and per-node
            classification for the construction phase.
        """
        state_fqns = self._require_state_list(graph_module)
        state_is_param = self._require_is_param(graph_module, state_fqns)
        num_state_inputs = getattr(graph_module, "num_state_inputs", len(state_fqns))

        stage_plan = self._resolve_stage_plan(run_ctx.model, run_ctx.pp_degree)
        stage_of_fqn = {fqn: s for s, fqns in enumerate(stage_plan) for fqn in fqns}

        (
            stage_state_indices,
            node_stage,
            node_phase,
            input_ph,
            label_ph,
            state_phs,
        ) = self._classify_graph(
            graph_module,
            stage_of_fqn,
            run_ctx.pp_degree,
            run_ctx.stage_idx,
            state_fqns,
            num_state_inputs,
        )

        self._anchor_grad_outputs(
            graph_module, node_stage, stage_of_fqn, state_fqns, run_ctx.model
        )
        self._propagate_anchor_stages(graph_module, node_stage, node_phase)
        self._validate_adjacent_dataflow(graph_module, node_stage)

        return _StageSplit(
            state_fqns=state_fqns,
            state_is_param=state_is_param,
            stage_plan=stage_plan,
            node_stage=node_stage,
            node_phase=node_phase,
            input_ph=input_ph,
            label_ph=label_ph,
            state_phs=state_phs,
            stage_state_indices=stage_state_indices,
            stage_state_fqns=[state_fqns[i] for i in stage_state_indices],
        )

    def _build_and_install_stage(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        run_ctx: _RunContext,
        split: _StageSplit,
        boundaries: Tuple[
            List[fx.Node], List[fx.Node], List[fx.Node], List[fx.Node], List[fx.Node]
        ],
    ) -> None:
        """Build the stage subgraphs, prune the live model, install the stub.

        Trainable-state resolution must precede ``_prune_live_model``: the
        gradient-output nodes map positionally onto the FULL trainable list,
        so pruning foreign-stage parameters first would shift the positions.

        Args:
            graph_module: Joint fwd+bwd FX graph, rewritten in place.
            pass_config: Parallel configuration (microbatch size read).
            run_ctx: This rank's PP execution context.
            split: Stage plan and per-node classification.
            boundaries: ``(act_in_list, act_out_list, grad_in_list,
                grad_out_list, saved)`` from ``_find_boundaries``.
        """
        act_in_list, act_out_list, grad_in_list, grad_out_list, saved = boundaries

        trainable_indices = self._trainable_state_indices(
            split.state_fqns, split.state_is_param, run_ctx.model
        )
        stage_state_set = set(split.stage_state_indices)
        stage_trainable = [i for i in trainable_indices if i in stage_state_set]

        fwd_gm, bwd_gm = self._build_stage_graphs(
            graph_module,
            split.node_stage,
            split.node_phase,
            run_ctx.stage_idx,
            run_ctx.pp_degree,
            split.state_phs,
            split.stage_state_indices,
            split.stage_state_fqns,
            split.input_ph,
            split.label_ph,
            act_in_list,
            act_out_list,
            grad_in_list,
            grad_out_list,
            saved,
            trainable_indices,
        )

        self._prune_live_model(run_ctx.model, split.stage_plan[run_ctx.stage_idx])

        sched = self._build_schedule(
            fwd_gm,
            bwd_gm,
            pass_config,
            run_ctx.stage_idx,
            run_ctx.pp_degree,
            run_ctx.group,
            len(split.stage_state_fqns),
            len(stage_trainable),
            act_in_list,
            act_out_list,
            grad_in_list,
            grad_out_list,
        )
        self._install_stub(
            graph_module,
            sched,
            split.stage_state_fqns,
            split.state_is_param,
            split.stage_state_indices,
        )

        _LOG.info(
            "Completed: stage %s owns %d state entries, %d trainable params, "
            "%d fwd / %d bwd boundary values",
            run_ctx.stage_idx,
            len(split.stage_state_fqns),
            len(stage_trainable),
            len(act_out_list),
            len(grad_out_list),
        )

    def _build_stage_graphs(
        self,
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
        stage_idx: int,
        pp_degree: int,
        state_phs: List[fx.Node],
        stage_state_indices: List[int],
        stage_state_fqns: List[str],
        input_ph: fx.Node,
        label_ph: fx.Node,
        act_in_list: List[fx.Node],
        act_out_list: List[fx.Node],
        grad_in_list: List[fx.Node],
        grad_out_list: List[fx.Node],
        saved: List[fx.Node],
        trainable_indices: List[int],
    ) -> Tuple[fx.GraphModule, fx.GraphModule]:
        """Build this stage's forward and backward subgraphs."""
        fwd_gm, fwd_out_orig = self._build_fwd_graph(
            graph_module,
            node_stage,
            node_phase,
            stage_idx,
            pp_degree,
            state_phs,
            stage_state_indices,
            stage_state_fqns,
            input_ph,
            label_ph,
            act_in_list,
            act_out_list,
            self._loss_node(graph_module),
            saved,
        )
        bwd_gm = self._build_bwd_graph(
            graph_module,
            node_stage,
            node_phase,
            stage_idx,
            pp_degree,
            state_phs,
            stage_state_indices,
            stage_state_fqns,
            grad_in_list,
            grad_out_list,
            fwd_out_orig,
            trainable_indices,
        )
        return fwd_gm, bwd_gm

    # ------------------------------------------------------------------
    # Tracer contract helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_state_list(graph_module: fx.GraphModule) -> List[str]:
        """Return the graph's live ``state_fqns`` list (shared with JointGraph).

        The tracer attaches the SAME list object to the GraphModule and the
        ``JointGraph``; mutating it in place (done in ``_install_stub``)
        keeps ``run_traced_graph``'s validation consistent with the pruned
        model. Copying here would desynchronize the two.
        """
        state_fqns = getattr(graph_module, "state_fqns", None)
        if state_fqns is None:
            raise ValueError(
                "PpPass requires the tracer's state_fqns attribute on the "
                "joint graph (attach it via trace_model_graph)"
            )
        return state_fqns

    @staticmethod
    def _require_is_param(
        graph_module: fx.GraphModule, state_fqns: List[str]
    ) -> List[bool]:
        """Return the graph's live ``state_is_param`` list, defaulting to
        all-params (matching the tracer's fallback for older traces)."""
        state_is_param = getattr(graph_module, "state_is_param", None)
        if state_is_param is None:
            state_is_param = [True] * len(state_fqns)
            graph_module.state_is_param = state_is_param
        return state_is_param

    @staticmethod
    def _loss_node(graph_module: fx.GraphModule) -> Optional[fx.Node]:
        """The original joint graph's loss node (output index 0)."""
        output = next((n for n in graph_module.graph.nodes if n.op == "output"), None)
        if output is None:
            return None
        returned = output.args[0]
        if isinstance(returned, (list, tuple)) and returned:
            first = returned[0]
            if isinstance(first, fx.Node):
                return first
        return None

    # ------------------------------------------------------------------
    # Group / plan resolution
    # ------------------------------------------------------------------

    def _resolve_group_and_stage(self, pass_config: PassConfig) -> Tuple[int, Any, int]:
        """Resolve PP degree, process group, and this rank's stage index.

        v1 supports the pure-PP layout (``pp_degree == world_size``, one
        stage per rank); ``pp_degree == 1`` disables the pass. Any other
        layout is rejected loudly: DP replication over one stage would
        funnel every replica's P2P into a single peer rank (the group is
        the whole world), and the PP+FSDP hybrid requires a mesh
        expressing both dimensions (``mesh_context`` path) — the trainer's
        1-D fallback mesh shards FSDP across the whole world and cannot
        host PP, so that combination would silently mis-shard.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        pp_degree = pass_config.pp_degree or world_size

        if pp_degree not in (1, world_size):
            raise ValueError(
                f"v1 supports pure-PP only: pp_degree={pp_degree} must "
                f"equal world_size={world_size} (or be 1 to disable PP). "
                f"DP-replication and PP+FSDP hybrid layouts require a "
                f"mesh_context with an explicit pp dim."
            )
        if pass_config.fsdp_enabled:
            raise ValueError(
                "PP+FSDP hybrid requires a mesh expressing both the PP and "
                "FSDP dimensions (mesh_context path). The trainer's 1-D "
                "fallback FSDP mesh spans the whole world and cannot host "
                "PP — FSDP would shard parameters across stage boundaries. "
                "Run pure-PP (fsdp_enabled=False, pp_degree == world_size) "
                "or provide a mesh_context with a pp dim."
            )

        group = dist.new_group(list(range(world_size)))
        # v1 is pure-PP (pp_degree == world_size, checked above), so each
        # rank owns exactly one stage. Keep the general formula so a future
        # DP-replication layout (pp_degree < world_size) maps rank->stage
        # contiguously rather than silently assuming 1:1.
        stage_idx = rank * pp_degree // world_size
        return pp_degree, group, stage_idx

    def _resolve_stage_plan(self, model: nn.Module, pp_degree: int) -> List[List[str]]:
        """Resolve the per-stage module FQN lists (manual plan or auto split)."""
        manual = (
            self._pass_plan.pp_module_fqns_per_stage
            if self._pass_plan is not None
            else None
        )
        if manual is not None:
            if len(manual) != pp_degree:
                raise ValueError(
                    f"Manual PP plan declares {len(manual)} stages but "
                    f"pp_degree={pp_degree}"
                )
            seen: Set[str] = set()
            for s, fqns in enumerate(manual):
                for fqn in fqns:
                    if fqn in seen:
                        raise ValueError(
                            f"Module '{fqn}' is assigned to multiple stages "
                            f"(stage {s} repeats it)"
                        )
                    seen.add(fqn)
            stage_plan = [list(fqns) for fqns in manual]
        else:
            stage_plan = _auto_stage_split(model, pp_degree)

        known = dict(model.named_modules())
        for s, fqns in enumerate(stage_plan):
            for fqn in fqns:
                if fqn not in known:
                    raise ValueError(f"Stage {s} declares unknown module '{fqn}'")
        self._warn_unassigned_modules(model, stage_plan)
        return stage_plan

    @staticmethod
    def _warn_unassigned_modules(model: nn.Module, stage_plan: List[List[str]]) -> None:
        """Warn about live modules no stage declares.

        A module that is neither declared itself, nor lives inside a
        declared one, nor is an ancestor of a declared one (a container
        whose elements are declared) has its graph nodes attributed by
        the dataflow fallbacks (consumer stage, then last stage) — rarely
        what a manual plan intends. Auto splits declare every top-level
        child, so they never warn.
        """
        declared = [fqn for fqns in stage_plan for fqn in fqns]
        unassigned: List[str] = []
        for fqn, _ in model.named_modules():
            if not fqn:
                # Root: parent-level glue is attributed by dataflow.
                continue
            covered = any(
                fqn == d or fqn.startswith(d + ".") or d.startswith(fqn + ".")
                for d in declared
            )
            if not covered:
                unassigned.append(fqn)
        if not unassigned:
            return
        shown = unassigned[:10]
        if len(unassigned) > len(shown):
            shown.append(f"... {len(unassigned) - len(shown)} more")
        _LOG.warning(
            "No PP stage declares modules %s — their nodes ride with the "
            "dataflow attribution (consumer stage, then last stage). "
            "Declare them via PassPlan.pp_stage() if that is not intended.",
            shown,
        )

    # ------------------------------------------------------------------
    # Node classification (stage + phase, topo order)
    # ------------------------------------------------------------------

    def _classify_graph(
        self,
        graph_module: fx.GraphModule,
        stage_of_fqn: Dict[str, int],
        num_stages: int,
        stage_idx: int,
        state_fqns: Sequence[str],
        num_state_inputs: int,
    ) -> Tuple[
        List[int],
        Dict[fx.Node, int],
        Dict[fx.Node, str],
        fx.Node,
        fx.Node,
        List[fx.Node],
    ]:
        """Resolve every node's stage and fwd/bwd phase in topological order.

        Phase: explicit ``autograd_backward`` tag, else inherited from args
        (post-trace inserts such as FSDP's reduce_scatter/wait inherit the
        bwd phase through their grad arguments).

        Stage resolution, in priority order:

        1. ``nn_module_stack`` walk (traced nodes, innermost module first)
        2. argument inheritance (post-trace inserts — state placeholders
           win, since backward compute nodes reference their params)
        3. consumer attribution for stack-less root-level glue (residual
           adds written in the parent's forward): the stage of their
           consumers, minimum when several — resolved in reverse topo
        4. last stage as the final fallback (loss chain, dead nodes)

        ``stage_state_indices`` lists the state entries owned by THIS
        stage (``stage_idx``); every state placeholder still gets a stage
        so foreign consumers can be attributed.

        Root-level state (no stage-owning module ancestor) is assigned to
        the stage of its (single) consumer; multi-stage consumption is the
        RoPE-cache-shaped error case and fails with an actionable message.
        """
        graph = graph_module.graph
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        state_ph_set = set(placeholders[:num_state_inputs])

        node_stage: Dict[fx.Node, int] = {}
        node_phase: Dict[fx.Node, str] = {}

        (
            state_phs,
            deferred_state,
            stage_state_indices,
            input_ph,
            label_ph,
        ) = self._classify_placeholders(
            placeholders,
            node_stage,
            node_phase,
            state_fqns,
            stage_of_fqn,
            num_stages,
            num_state_inputs,
            stage_idx,
        )
        self._classify_forward_nodes(
            graph, state_ph_set, stage_of_fqn, num_stages, node_stage, node_phase
        )
        self._classify_backward_nodes(
            graph, state_ph_set, num_stages, node_stage, node_phase
        )
        self._assign_deferred_state(
            deferred_state,
            state_phs,
            state_fqns,
            node_stage,
            stage_idx,
            stage_state_indices,
        )

        if input_ph is None or label_ph is None:
            raise ValueError(
                "Joint graph must expose (state..., input, label) placeholders"
            )
        return (
            sorted(stage_state_indices),
            node_stage,
            node_phase,
            input_ph,
            label_ph,
            state_phs,
        )

    def _classify_placeholders(
        self,
        placeholders: List[fx.Node],
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
        state_fqns: Sequence[str],
        stage_of_fqn: Dict[str, int],
        num_stages: int,
        num_state_inputs: int,
        stage_idx: int,
    ) -> Tuple[
        List[fx.Node], List[int], List[int], Optional[fx.Node], Optional[fx.Node]
    ]:
        """Classify placeholder nodes and collect this stage's state entries.

        Returns:
            ``(state_phs, deferred_state, stage_state_indices, input_ph,
            label_ph)``. ``deferred_state`` lists root-level state indices
            whose stage is resolved later from their consumers.
        """
        state_phs: List[fx.Node] = []
        deferred_state: List[int] = []
        stage_state_indices: List[int] = []
        input_ph: Optional[fx.Node] = None
        label_ph: Optional[fx.Node] = None
        for idx, ph in enumerate(placeholders):
            if idx < num_state_inputs:
                stage = self._state_stage(state_fqns[idx], stage_of_fqn)
                state_phs.append(ph)
                if stage is None:
                    deferred_state.append(idx)
                else:
                    node_stage[ph] = stage
                    if stage == stage_idx:
                        stage_state_indices.append(idx)
            elif idx == num_state_inputs:
                input_ph = ph
                node_stage[ph] = 0
            else:
                label_ph = ph
                node_stage[ph] = num_stages - 1
            node_phase[ph] = _FWD
        return state_phs, deferred_state, stage_state_indices, input_ph, label_ph

    def _classify_forward_nodes(
        self,
        graph: fx.Graph,
        state_ph_set: Set[fx.Node],
        stage_of_fqn: Dict[str, int],
        num_stages: int,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
    ) -> None:
        """Assign stages to forward nodes in topological order.

        Anchors: module stack, then state-ph args, then max-args over
        TENSOR args (data flows forward; shape-metadata scalars carry no
        data dependency). Leftover root-level glue rides with its
        consumers (reverse topo), else the last stage.
        """
        floating: List[fx.Node] = []
        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            phase = _BWD if node.meta.get("autograd_backward", False) else _FWD
            if phase == _FWD:
                for arg in _iter_node_args(node):
                    if node_phase.get(arg) == _BWD:
                        phase = _BWD
                        break
            node_phase[node] = phase
            if phase == _FWD:
                stage = self._stage_from_stack(node, stage_of_fqn)
                if stage is None:
                    stage = self._stage_from_state_arg(node, node_stage, state_ph_set)
                if stage is None:
                    arg_stages = _tensor_arg_stages(node, node_stage)
                    if arg_stages:
                        stage = max(arg_stages)
                if stage is None:
                    floating.append(node)
                else:
                    node_stage[node] = stage

        # Leftover fwd glue: consumers first (reverse topo), else last.
        for node in reversed(list(graph.nodes)):
            if node not in floating or node_phase[node] != _FWD:
                continue
            consumer_stages = {
                node_stage[u]
                for u in node.users
                if u.op != "output" and node_stage.get(u) is not None
            }
            node_stage[node] = (
                min(consumer_stages) if consumer_stages else num_stages - 1
            )

    def _classify_backward_nodes(
        self,
        graph: fx.Graph,
        state_ph_set: Set[fx.Node],
        num_stages: int,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
    ) -> None:
        """Assign stages to backward nodes in topological order.

        A bwd node runs where its earliest TENSOR input lives: gradients
        born downstream are shipped down during the backward sweep, and
        upstream activations arrive as replayed forward outputs. Weight-
        gradient nodes that consume early activations are corrected by
        the positional anchor (the param's stage owns its grad).
        """
        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node_phase[node] != _BWD:
                continue
            stage = self._stage_from_state_arg(node, node_stage, state_ph_set)
            if stage is None:
                arg_stages = _tensor_arg_stages(node, node_stage)
                if arg_stages:
                    stage = min(arg_stages)
            if stage is None:
                # Dead-end bwd glue: ride with consumers if any, else last.
                consumer_stages = {
                    node_stage[u]
                    for u in node.users
                    if u.op != "output" and node_stage.get(u) is not None
                }
                stage = min(consumer_stages) if consumer_stages else num_stages - 1
            node_stage[node] = stage

    def _assign_deferred_state(
        self,
        deferred_state: List[int],
        state_phs: List[fx.Node],
        state_fqns: Sequence[str],
        node_stage: Dict[fx.Node, int],
        stage_idx: int,
        stage_state_indices: List[int],
    ) -> None:
        """Assign root-level state to the stage of its (single) consumer.

        Multi-stage consumption is the RoPE-cache-shaped error case and
        fails with an actionable message.
        """
        for idx in deferred_state:
            ph = state_phs[idx]
            consumer_stages = {node_stage[u] for u in ph.users if u.op != "output"}
            if len(consumer_stages) > 1:
                raise ValueError(
                    f"Root-level state '{state_fqns[idx]}' is consumed by "
                    f"stages {sorted(consumer_stages)} — v1 requires state "
                    f"to live under a single stage's module (e.g. move a "
                    f"shared RoPE cache under a per-stage module or split "
                    f"it per stage)"
                )
            stage = next(iter(consumer_stages), 0)
            node_stage[ph] = stage
            if stage == stage_idx:
                stage_state_indices.append(idx)
            _LOG.info(
                "Root-level state '%s' assigned to stage %s by its consumer",
                state_fqns[idx],
                stage,
            )

    def _state_stage(self, fqn: str, stage_of_fqn: Dict[str, int]) -> Optional[int]:
        """Stage of a state entry via longest-ancestor walk over its FQN."""
        parts = fqn.split(".")
        for i in range(len(parts) - 1, 0, -1):
            stage = stage_of_fqn.get(".".join(parts[:i]))
            if stage is not None:
                return stage
        return stage_of_fqn.get(fqn)

    def _stage_from_stack(
        self, node: fx.Node, stage_of_fqn: Dict[str, int]
    ) -> Optional[int]:
        """Stage from ``nn_module_stack``, innermost module first."""
        for fqn in _innermost_fqns(node):
            stage = stage_of_fqn.get(fqn)
            if stage is not None:
                return stage
        return None

    def _stage_from_state_arg(
        self,
        node: fx.Node,
        node_stage: Dict[fx.Node, int],
        state_ph_set: Set[fx.Node],
    ) -> Optional[int]:
        """Stage from a state-placeholder argument, if any.

        Post-trace inserts (FSDP collectives, weight-grad compute) reference
        their owning stage's parameters — a state placeholder arg pins the
        node to that stage. Generic argument inheritance is deliberately
        NOT done here: it would let an early-stage argument pull a node
        against the dataflow (see the max/min-args rules in
        ``_classify_graph``).
        """
        for arg in _iter_node_args(node):
            if arg in state_ph_set:
                return node_stage.get(arg)
        return None

    @staticmethod
    def _propagate_anchor_stages(
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
    ) -> None:
        """Pull bwd args up to anchored consumers (one reverse-topo pass).

        A bwd value consumed by a DOWNSTREAM-stage bwd node must be
        recomputed there: gradients flow downstream -> upstream, so a
        downstream consumer cannot receive an upstream-born bwd value in
        time (its own bwd sweep runs first). Fwd-phase args are untouched —
        they arrive as replayed forward outputs; downstream-born args are
        untouched — they arrive via the grad crossings.

        A single REVERSE topological sweep reaches the fixpoint: a node is
        visited only after every user of it has been visited and pulled it
        to its final stage, so the node propagates its final value to its
        own args. The previous iterate-to-fixpoint loop was quadratic in
        the worst case on large joint graphs.
        """
        for node in reversed(list(graph_module.graph.nodes)):
            if node.op in ("placeholder", "output"):
                continue
            if node_phase.get(node) != _BWD:
                continue
            stage = node_stage.get(node)
            if stage is None:
                continue
            for arg in _iter_node_args(node):
                if (
                    isinstance(arg, fx.Node)
                    and node_phase.get(arg) == _BWD
                    and node_stage.get(arg) is not None
                    and node_stage[arg] < stage
                ):
                    node_stage[arg] = stage

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_adjacent_dataflow(
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
    ) -> None:
        """Reject skip-stage dataflow before it reaches the slice copy.

        v1 ships boundary values between NEIGHBOURING stages only: a value
        produced on stage ``j`` and consumed on stage ``k`` with
        ``abs(j - k) > 1`` has no route (it would have to transit
        intermediate stages untouched). Without this check such graphs die
        later inside ``_copy_node`` / the fwd-output assembly with a
        generic "outside this stage's slice" error that hides the root
        cause.

        Raises:
            ValueError: When any producer-consumer pair spans more than
                one stage cut (e.g. a dense skip connection across two
                stages, or a stage plan that leaves a module's nodes to
                the dataflow fallbacks).
        """
        for node in graph_module.graph.nodes:
            if node.op == "output":
                continue
            producer = node_stage.get(node)
            if producer is None:
                continue
            for user in node.users:
                if user.op == "output":
                    continue
                consumer = node_stage.get(user)
                if consumer is None or abs(consumer - producer) <= 1:
                    continue
                raise ValueError(
                    f"Value '{node.name}' is produced on stage {producer} "
                    f"but consumed on stage {consumer}: v1 ships boundary "
                    f"values between neighbouring stages only. A skip "
                    f"connection spanning 2+ stages or a stage plan that "
                    f"leaves modules unattributed produces this — adjust "
                    f"PassPlan.pp_stage() so the dataflow crosses one cut "
                    f"at a time."
                )

    def _anchor_grad_outputs(
        self,
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        stage_of_fqn: Dict[str, int],
        state_fqns: Sequence[str],
        model: Optional[nn.Module],
    ) -> None:
        """Pin each gradient-output node to its parameter's stage.

        Gradient-output node i differentiates the i-th trainable parameter
        (positional mapping, mirroring the tracer); it runs on the stage
        owning that parameter even when min-args would place it upstream
        (a later stage's weight-grad consumes an earlier stage's
        activation, replayed from the forward outputs).
        """
        output = next((n for n in graph_module.graph.nodes if n.op == "output"), None)
        if output is None or model is None:
            return
        returned = output.args[0]
        if not isinstance(returned, (list, tuple)):
            return
        state_is_param = getattr(
            graph_module, "state_is_param", [True] * len(state_fqns)
        )
        trainable = self._trainable_state_indices(state_fqns, state_is_param, model)
        for i, grad_node in enumerate(returned[1:]):
            if i >= len(trainable) or not isinstance(grad_node, fx.Node):
                continue
            pstage = self._state_stage(state_fqns[trainable[i]], stage_of_fqn)
            if pstage is not None:
                node_stage[grad_node] = pstage

    def _find_boundaries(
        self,
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
        pp_degree: int,
        stage_idx: int,
    ) -> Tuple[
        List[fx.Node], List[fx.Node], List[fx.Node], List[fx.Node], List[fx.Node]
    ]:
        """Locate the value lists crossing this stage's cuts.

        Returns:
            ``(act_in_list, act_out_list, grad_in_list, grad_out_list,
            saved)``, all in joint-graph topological order so the producer
            and consumer of a cut agree on P2P ordering:

            - ``act_in_list``: stage-(k-1) values consumed by stage k
              (tensors, dynamic-shape scalars, parent-level views —
              everything a real ATen graph pushes across a cut).
            - ``act_out_list``: stage-k values consumed by stage k+1.
            - ``grad_in_list``: stage-(k+1) backward values consumed by
              stage k.
            - ``grad_out_list``: stage-k backward values consumed by
              stage k-1.
            - ``saved``: forward-phase values referenced by this stage's
              backward nodes — exported as forward outputs and replayed as
              backward inputs.
        """
        k = stage_idx
        is_last = k == pp_degree - 1
        nodes = [
            n for n in graph_module.graph.nodes if n.op not in ("placeholder", "output")
        ]

        def crossing(
            producer: int, consumer: int, phase: Optional[str]
        ) -> List[fx.Node]:
            """Values produced by one stage and consumed by another.

            ``phase=None`` matches any phase (backward crossings may carry
            gradients as well as functionalized fwd views).
            """
            found = []
            for v in nodes:
                if node_stage[v] != producer:
                    continue
                if phase is not None and node_phase[v] != phase:
                    continue
                if any(node_stage.get(u) == consumer for u in v.users):
                    found.append(v)
            return found

        # FWD-phase only: values born in this stage's forward sweep and
        # consumed downstream. Bwd-phase values cannot ship forward (they
        # do not exist yet during the forward sweep); downstream-born
        # gradients flow back via grad crossings below.
        act_in_list = crossing(k - 1, k, _FWD) if k > 0 else []
        act_out_list = [] if is_last else crossing(k, k + 1, _FWD)
        # Backward crossings are phase-AGNOSTIC: besides plain gradients,
        # functionalized views (fwd nodes) can be consumed by an earlier
        # stage's backward (weight-grad compute) — those values flow back
        # during the backward sweep just the same.
        grad_in_list = [] if is_last else crossing(k + 1, k, None)
        grad_out_list = crossing(k, k - 1, None) if k > 0 else []

        # Forward-phase values the backward references. Stage-owned
        # placeholders (state / input / label of this stage) may appear and
        # round-trip through the forward outputs; foreign placeholders
        # cannot be shipped and fail loudly during the slice copy.
        stage_owned_phs = {
            p
            for p in graph_module.graph.nodes
            if p.op == "placeholder" and node_stage.get(p) == k
        }
        saved: List[fx.Node] = []
        seen: Set[fx.Node] = set()
        act_in_set = set(act_in_list)
        for node in nodes:
            if node_phase[node] != _BWD or node_stage[node] != k:
                continue
            for arg in _iter_node_args(node):
                if not isinstance(arg, fx.Node) or arg in seen:
                    continue
                # Fwd-phase values replay from the fwd outputs; bwd-phase
                # values are only shippable when they arrived as boundary
                # inputs (e.g. gradient-shape sym_size scalars).
                if node_phase.get(arg) != _FWD and arg not in act_in_set:
                    continue
                if arg.op == "placeholder" and arg not in stage_owned_phs:
                    continue
                seen.add(arg)
                saved.append(arg)
        # Fwd-phase values this stage must ship BACKWARD (consumed by the
        # previous stage's backward) ride the fwd outputs as pass-throughs:
        # computed once in the fwd subgraph, replayed into the bwd
        # subgraph, and appended to its outputs.
        for v in grad_out_list:
            if node_phase[v] == _FWD and v not in seen:
                seen.add(v)
                saved.append(v)
        return act_in_list, act_out_list, grad_in_list, grad_out_list, saved

    # ------------------------------------------------------------------
    # Subgraph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _unique(name: str, used: Set[str]) -> str:
        """Return ``name`` (or a suffixed variant) not yet in ``used``."""
        candidate = name
        suffix = 1
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    def _copy_node(
        self,
        g: fx.Graph,
        node: fx.Node,
        env: Dict[fx.Node, fx.Node],
        foreign: Dict[fx.Node, fx.Node],
    ) -> fx.Node:
        """Copy ``node`` into ``g``, remapping args through env/foreign maps.

        ``env`` maps already-copied same-slice nodes; ``foreign`` maps
        boundary values to their placeholder stand-ins.
        """
        if node.op == "get_attr":
            raise ValueError(
                f"Unexpected get_attr node '{node.name}' — the tracer "
                f"contracts parameters/buffers to be static placeholders"
            )

        def map_fn(a: Any) -> Any:
            """Remap a single arg: env first, then foreign placeholders."""
            if isinstance(a, fx.Node):
                if a in env:
                    return env[a]
                if a in foreign:
                    return foreign[a]
                raise ValueError(
                    f"Node '{node.name}' references '{a.name}' which lies "
                    f"outside this stage's slice — boundary detection or "
                    f"the stage plan is inconsistent"
                )
            return a

        new_args = fx.map_arg(node.args, map_fn)
        new_kwargs = fx.map_arg(node.kwargs, map_fn)
        new_node = g.create_node(node.op, node.target, new_args, new_kwargs)
        new_node.meta = dict(node.meta)
        env[node] = new_node
        return new_node

    def _build_fwd_graph(
        self,
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
        stage_idx: int,
        pp_degree: int,
        state_phs: List[fx.Node],
        stage_state_indices: List[int],
        stage_state_fqns: List[str],
        input_ph: fx.Node,
        label_ph: fx.Node,
        act_in_list: List[fx.Node],
        act_out_list: List[fx.Node],
        loss_node: Optional[fx.Node],
        saved: List[fx.Node],
    ) -> Tuple[fx.GraphModule, List[fx.Node]]:
        """Build this stage's forward subgraph.

        Outputs: ``(*act_out_list, *saved)`` for non-last stages,
        ``(loss, *saved)`` for the last stage — where ``saved`` contains
        every forward-phase value the backward references that is not
        already crossing. The schedule ships the crossing prefix and
        replays the full output tuple into the backward subgraph.

        Returns:
            ``(fwd_gm, fwd_out_orig)`` — the subgraph module together with
            the ORIGINAL nodes at each output position, so the backward
            builder can key its placeholder mapping on them.
        """
        graph = graph_module.graph
        k = stage_idx
        is_last = k == pp_degree - 1
        g = fx.Graph()
        env: Dict[fx.Node, fx.Node] = {}
        foreign: Dict[fx.Node, fx.Node] = {}
        used: Set[str] = set()

        self._create_fwd_placeholders(
            g,
            used,
            env,
            foreign,
            stage_state_fqns,
            state_phs,
            stage_state_indices,
            act_in_list,
            input_ph,
            label_ph,
            k,
            is_last,
        )

        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node_phase[node] != _FWD or node_stage[node] != k:
                continue
            self._copy_node(g, node, env, foreign)

        out_orig = self._resolve_fwd_outputs(act_out_list, saved, loss_node, is_last)
        out_nodes = [env[n] if n in env else foreign[n] for n in out_orig]
        g.output(tuple(out_nodes))

        fwd_gm = fx.GraphModule(nn.Module(), g)
        return fwd_gm, out_orig

    def _create_fwd_placeholders(
        self,
        g: fx.Graph,
        used: Set[str],
        env: Dict[fx.Node, fx.Node],
        foreign: Dict[fx.Node, fx.Node],
        stage_state_fqns: List[str],
        state_phs: List[fx.Node],
        stage_state_indices: List[int],
        act_in_list: List[fx.Node],
        input_ph: fx.Node,
        label_ph: fx.Node,
        k: int,
        is_last: bool,
    ) -> None:
        """Create the fwd subgraph's placeholders (state / boundary / input).

        Incoming boundary values (stage k > 0) become placeholders; stage 0
        takes the full input and the last stage the label.
        """
        for i, orig_idx in enumerate(stage_state_indices):
            ph = g.placeholder(self._unique(_sanitize(stage_state_fqns[i]), used))
            ph.meta = dict(state_phs[orig_idx].meta)
            env[state_phs[orig_idx]] = ph

        for i, act in enumerate(act_in_list):
            ph = g.placeholder(self._unique(f"pp_act_in_{i}", used))
            ph.meta = dict(act.meta)
            foreign[act] = ph

        if k == 0:
            in_ph = g.placeholder(self._unique("pp_input", used))
            in_ph.meta = dict(input_ph.meta)
            env[input_ph] = in_ph

        if is_last:
            lbl_ph = g.placeholder(self._unique("pp_label", used))
            lbl_ph.meta = dict(label_ph.meta)
            env[label_ph] = lbl_ph

    @staticmethod
    def _resolve_fwd_outputs(
        act_out_list: List[fx.Node],
        saved: List[fx.Node],
        loss_node: Optional[fx.Node],
        is_last: bool,
    ) -> List[fx.Node]:
        """Order the fwd subgraph's outputs: boundary values first, then saved."""
        if is_last:
            if loss_node is None:
                raise ValueError("Last PP stage requires a loss node in the output")
            out_orig: List[fx.Node] = [loss_node]
        else:
            if not act_out_list:
                raise ValueError(
                    "Stage ships no boundary values but is not the last "
                    "stage — the stage plan does not cut the graph"
                )
            out_orig = list(act_out_list)

        crossing_ids = {id(v) for v in act_out_list}
        for node in saved:
            if id(node) in crossing_ids or (is_last and node is loss_node):
                continue
            out_orig.append(node)
        return out_orig

    def _build_bwd_graph(
        self,
        graph_module: fx.GraphModule,
        node_stage: Dict[fx.Node, int],
        node_phase: Dict[fx.Node, str],
        stage_idx: int,
        pp_degree: int,
        state_phs: List[fx.Node],
        stage_state_indices: List[int],
        stage_state_fqns: List[str],
        grad_in_list: List[fx.Node],
        grad_out_list: List[fx.Node],
        fwd_out_orig: List[fx.Node],
        trainable_indices: List[int],
    ) -> fx.GraphModule:
        """Build this stage's backward subgraph.

        Placeholders: ``(*state, *grad_in, *fwd_outputs)`` — the schedule
        passes the received boundary gradients and then the forward
        outputs verbatim (on the last stage a single ``ones_like(loss)``
        takes the gradient slots' place). Outputs: ``(*param_grads,)``
        plus the boundary gradients for every stage except stage 0.
        """
        graph = graph_module.graph
        k = stage_idx
        g = fx.Graph()
        env: Dict[fx.Node, fx.Node] = {}
        foreign: Dict[fx.Node, fx.Node] = {}
        used: Set[str] = set()

        self._create_bwd_placeholders(
            g,
            used,
            env,
            foreign,
            graph_module,
            stage_state_fqns,
            state_phs,
            stage_state_indices,
            grad_in_list,
            fwd_out_orig,
            k,
            pp_degree,
        )

        # Sets, not lists: membership is probed once per graph node below
        # and real joint graphs carry tens of thousands of nodes.
        stage_bwd_set = {
            n
            for n in graph.nodes
            if n.op not in ("placeholder", "output")
            and node_phase[n] == _BWD
            and node_stage[n] == k
        }

        # Gradient-output nodes of this stage's trainable params (by state
        # index identity, not module attribution — post-trace inserts like
        # zeros_like carry no module stack). A LIST — output order is the
        # positional gradient contract; only membership uses the set.
        stage_state_set = set(stage_state_indices)
        stage_grad_nodes = self._stage_grad_nodes(
            graph, trainable_indices, stage_state_set
        )
        stage_grad_set = set(stage_grad_nodes)

        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node not in stage_bwd_set and node not in stage_grad_set:
                continue
            self._copy_node(g, node, env, foreign)

        out_nodes = [env[gn] for gn in stage_grad_nodes]
        out_nodes.extend(env[gn] for gn in grad_out_list)
        g.output(tuple(out_nodes))

        bwd_gm = fx.GraphModule(nn.Module(), g)
        return bwd_gm

    def _create_bwd_placeholders(
        self,
        g: fx.Graph,
        used: Set[str],
        env: Dict[fx.Node, fx.Node],
        foreign: Dict[fx.Node, fx.Node],
        graph_module: fx.GraphModule,
        stage_state_fqns: List[str],
        state_phs: List[fx.Node],
        stage_state_indices: List[int],
        grad_in_list: List[fx.Node],
        fwd_out_orig: List[fx.Node],
        k: int,
        pp_degree: int,
    ) -> None:
        """Create the bwd subgraph's placeholders (state / grads / fwd outs).

        The schedule passes the received boundary gradients and then the
        forward outputs verbatim (on the last stage a single
        ``ones_like(loss)`` takes the gradient slots' place). The
        ``foreign`` entries double as the remap for backward nodes
        referencing forward values.
        """
        for i, orig_idx in enumerate(stage_state_indices):
            ph = g.placeholder(self._unique(_sanitize(stage_state_fqns[i]), used))
            ph.meta = dict(state_phs[orig_idx].meta)
            env[state_phs[orig_idx]] = ph

        for i, grad in enumerate(grad_in_list):
            ph = g.placeholder(self._unique(f"pp_grad_in_{i}", used))
            ph.meta = dict(grad.meta)
            foreign[grad] = ph
        if not grad_in_list and k == pp_degree - 1:
            # Last stage: the schedule feeds ones_like(loss) into the
            # single gradient slot.
            ph = g.placeholder(self._unique("pp_grad_ones", used))
            loss = self._loss_node(graph_module)
            if loss is not None:
                ph.meta = dict(loss.meta)

        # One placeholder per forward output, same order — the schedule
        # passes the forward outputs verbatim after the gradients.
        for orig in fwd_out_orig:
            ph = g.placeholder(self._unique(f"pp_fwd_{orig.name}", used))
            ph.meta = dict(orig.meta)
            foreign[orig] = ph

    @staticmethod
    def _stage_grad_nodes(
        graph: fx.Graph, trainable_indices: List[int], stage_state_set: Set[int]
    ) -> List[fx.Node]:
        """Gradient-output nodes of the stage's trainable params."""
        output = next((n for n in graph.nodes if n.op == "output"), None)
        returned = output.args[0] if output is not None else []
        stage_grad_nodes: List[fx.Node] = []
        for i, grad_node in enumerate(returned[1:]):
            if i >= len(trainable_indices):
                break
            if trainable_indices[i] in stage_state_set:
                stage_grad_nodes.append(grad_node)
        return stage_grad_nodes

    # ------------------------------------------------------------------
    # Live model pruning
    # ------------------------------------------------------------------

    def _prune_live_model(self, model: nn.Module, keep_fqns: List[str]) -> None:
        """Remove foreign-stage submodules from the live model in place.

        Mirrors torchtitan's ``_split_module`` (``None``-out unkept
        children, filter ModuleList/ModuleDict entries) while PRESERVING
        original index keys — reindexing a ModuleList would rename the
        surviving parameters' FQNs and break ``run_traced_graph``'s state
        validation against the sliced ``state_fqns``. Manual plans may cut
        inside a container element; such elements are recursed into
        (mixed subtree), exactly like plain modules.
        """
        keep = set(keep_fqns)

        def is_kept(fqn: str) -> bool:
            """True when ``fqn`` equals or descends from a stage module."""
            return any(fqn == kf or fqn.startswith(kf + ".") for kf in keep)

        def prune_container_child(child: nn.Module, cfqn: str) -> None:
            """Prune a ModuleList/ModuleDict child, preserving index keys.

            Reindexing would rename the surviving parameters' FQNs and break
            ``run_traced_graph``'s state validation. An element is kept
            whole, deleted, or recursed into when a manual plan cuts inside
            it (same rule as plain modules).
            """
            children = child._modules  # pylint: disable=protected-access
            keys = list(children.keys())
            if all(is_kept(f"{cfqn}.{key}") for key in keys):
                return
            for key in keys:
                elem_fqn = f"{cfqn}.{key}"
                if is_kept(elem_fqn):
                    continue
                elem = children[key]
                descendants = [f for f, _ in elem.named_modules() if f]
                if any(is_kept(f"{elem_fqn}.{f}") for f in descendants):
                    prune(elem, elem_fqn + ".")
                else:
                    del children[key]

        def prune_plain_child(
            mod: nn.Module, name: str, child: nn.Module, cfqn: str
        ) -> None:
            """Prune a non-container child: delete it or recurse when mixed."""
            descendants = [f for f, _ in child.named_modules() if f]
            if not descendants:
                # Leaf module (e.g. a bare Linear): keep only if declared.
                if not is_kept(cfqn):
                    setattr(mod, name, None)
                return
            if not any(is_kept(f"{cfqn}.{f}") for f in descendants):
                setattr(mod, name, None)
            else:
                # Mixed subtree (manual plans may cut inside a child).
                prune(child, cfqn + ".")

        def prune(mod: nn.Module, prefix: str) -> None:
            """Recursively prune non-stage children under ``prefix``."""
            for name, child in list(mod.named_children()):
                cfqn = f"{prefix}{name}"
                if is_kept(cfqn):
                    # Declared (or ancestor of a declared) stage module —
                    # keep the whole subtree.
                    continue
                if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    prune_container_child(child, cfqn)
                    continue
                prune_plain_child(mod, name, child, cfqn)

        prune(model, "")

    # ------------------------------------------------------------------
    # Schedule + stub installation
    # ------------------------------------------------------------------

    def _trainable_state_indices(
        self,
        state_fqns: Sequence[str],
        state_is_param: Sequence[bool],
        model: Optional[nn.Module],
    ) -> List[int]:
        """Trainable-parameter state indices, mirroring the tracer's order."""
        param_lookup = (
            dict(model.named_parameters(remove_duplicate=False))
            if model is not None
            else {}
        )
        trainable: List[int] = []
        for idx, fqn in enumerate(state_fqns):
            if idx < len(state_is_param) and not state_is_param[idx]:
                continue
            param = param_lookup.get(fqn)
            if param is not None and param.requires_grad:
                trainable.append(idx)
        return trainable

    @staticmethod
    def _value_spec(
        nodes: Sequence[fx.Node], stage_idx: int, kind: str
    ) -> List[Tuple[Any, ...]]:
        """Build the schedule's recv descriptor for a boundary value list.

        Tensor values yield ``("tensor", shape, dtype, device)``; int/SymInt
        scalars (dynamic-shape ``sym_size`` nodes) yield ``("scalar",)``
        and are shipped as 0-d int64 tensors, unwrapped via ``item()``.
        Scalars carry no device of their own — the schedule anchors them to
        the first tensor entry's device, so a non-empty list must contain
        at least one tensor.
        """
        spec: List[Tuple[Any, ...]] = []
        for v in nodes:
            val = v.meta.get("val")
            if isinstance(val, torch.Tensor):
                # int(s): trace-time dims are SymInts (symbolic over the
                # sample input); the schedule allocates real buffers from
                # this spec, and ATen rejects symbolic sizes at runtime.
                # Training uses static shapes per compile, so the hinted
                # concrete values are stable.
                shape = tuple(int(s) for s in val.shape)
                spec.append(("tensor", shape, val.dtype, val.device))
            elif val is None:
                raise ValueError(
                    f"Boundary {kind} value '{v.name}' on stage {stage_idx} "
                    f"has no 'val' meta (traced without FakeTensor "
                    f"metadata) — cannot size the P2P receive buffer"
                )
            else:
                spec.append(("scalar",))
        if nodes and not any(entry[0] == "tensor" for entry in spec):
            raise ValueError(
                f"Boundary {kind} list on stage {stage_idx} carries only "
                f"scalar values — no tensor entry to anchor the P2P "
                f"receive device on"
            )
        return spec

    def _build_schedule(
        self,
        fwd_gm: fx.GraphModule,
        bwd_gm: fx.GraphModule,
        pass_config: PassConfig,
        stage_idx: int,
        pp_degree: int,
        group: Any,
        num_state: int,
        num_trainable: int,
        act_in_list: List[fx.Node],
        act_out_list: List[fx.Node],
        grad_in_list: List[fx.Node],
        grad_out_list: List[fx.Node],
    ) -> ScheduleGPipe:
        """Assemble the ``ScheduleGPipe`` for this stage.

        Receive descriptors come from the boundary values' fake-tensor
        ``val`` metas; the send side packs by value type at runtime.
        """
        recv_spec = (
            self._value_spec(act_in_list, stage_idx, "activation")
            if stage_idx > 0
            else []
        )
        grad_recv_spec = (
            self._value_spec(grad_in_list, stage_idx, "gradient")
            if stage_idx < pp_degree - 1
            else []
        )
        return ScheduleGPipe(
            fwd_gm,
            bwd_gm,
            stage_idx=stage_idx,
            pp_degree=pp_degree,
            num_state=num_state,
            num_trainable=num_trainable,
            num_send=len(act_out_list),
            grad_send_count=len(grad_out_list),
            microbatch_size=pass_config.pp_microbatch_size,
            pp_group=group,
            recv_spec=recv_spec,
            grad_recv_spec=grad_recv_spec,
        )

    def _install_stub(
        self,
        graph_module: fx.GraphModule,
        sched: ScheduleGPipe,
        stage_state_fqns: List[str],
        state_is_param: Sequence[bool],
        stage_state_indices: List[int],
    ) -> None:
        """Rewrite the graph module into a schedule-dispatching stub.

        The schedule is added as a submodule and invoked through a
        ``call_module`` node, so it survives any later ``recompile()`` and
        the trainer's ``graph_module(*flat_inputs)`` dispatches to it
        unchanged. ``state_fqns`` / ``state_is_param`` are updated IN PLACE
        (the JointGraph shares the list objects).
        """
        g = fx.Graph()
        used: Set[str] = set()
        state_phs = [
            g.placeholder(self._unique(_sanitize(fqn), used))
            for fqn in stage_state_fqns
        ]
        in_ph = g.placeholder(self._unique("pp_input", used))
        lbl_ph = g.placeholder(self._unique("pp_label", used))

        graph_module.add_module("pp_schedule", sched)
        call = g.call_module("pp_schedule", args=tuple(state_phs) + (in_ph, lbl_ph))
        g.output(call)

        graph_module.graph = g
        graph_module.state_fqns[:] = stage_state_fqns
        graph_module.state_is_param[:] = [
            state_is_param[i] for i in stage_state_indices
        ]
        graph_module.num_state_inputs = len(stage_state_fqns)
        graph_module.recompile()


__all__ = ["PpPass", "_auto_stage_split"]
