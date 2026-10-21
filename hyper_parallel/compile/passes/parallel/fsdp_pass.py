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
FSDP / DDP / HSDP Pass - Data Parallel Partitioning Pass

Operates on the joint fwd+bwd graph produced by the tracer, where
parameters/buffers are **static inputs** (leading placeholders), not get_attr
nodes. The data-parallel mode comes from ``PassConfig.dp_mode`` and mirrors
simplefsdp's ``data_parallel`` modes:

- ``"fsdp"`` (fully_shard): parameters sharded on dim 0; forward all_gather,
  backward reduce_scatter.
- ``"ddp"`` (replicate): parameters stay replicated; no all_gather, gradients
  are all-reduced on the ``dp_replicate`` axis.
- ``"hsdp"`` (hybrid_shard): parameters replicated on ``dp_replicate`` and
  sharded on ``fsdp``; forward all_gather on ``fsdp``, backward reduce_scatter
  on ``fsdp`` then all_reduce on ``dp_replicate``.

Responsibilities:
1. Identify parameter placeholders belonging to DP-wrapped modules
   (via GraphParallelPlan, exact FQN or pattern)
2. Insert AllGather after each such placeholder (Shard -> Replicate), so the
   computation body operates on full parameters while the graph input stays
   sharded (skipped for ``"ddp"``)
3. Insert gradient reduction on the gradient outputs of DP parameters —
   reduce_scatter (FSDP/HSDP) and/or all_reduce (DDP/HSDP)
4. Physically shard the *live model's* parameters in place (dim 0, by FSDP
   rank) for FSDP/HSDP, so ``model.parameters()`` already holds the local
   shard and the trainer / optimizer need no DP awareness at all
5. Reshard (``fsdp_reshard_after_forward``): sink each AllGather to its first
   forward use, free the replicated parameter after its last forward read,
   and, for parameters the backward still needs, re-gather plus rematerialize
   the saved view chain just before the first backward consumer. Peak memory
   tracks the forward working set instead of every replicated parameter.

All DP logic (which parameters, the collectives, and the sharding itself)
lives in this pass; the trainer simply feeds ``model.parameters()``.
"""

__all__ = ["FSDPPass"]

import logging
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import fx, nn
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.fx.node import _side_effectful_functions
from torch.ops import _c10d_functional

from ...pass_config import PassConfig, normalize_dp_mode
from ..base import GraphPass
from ...graph_parallel_plan import GraphParallelPlan

_LOG = logging.getLogger(__name__)

_GETITEM = operator.getitem

# Gradient reduction op for every DP axis. Mirrors eager ``fully_shard``'s
# default (``core/fully_shard/hsdp_state.py``: ``set_reduce_op_type("avg")``):
# the backward produces the full (replicated) gradient on every rank, so each
# axis must be AVERAGED (not summed) to keep the step gradient at the
# DP-world-mean scale. HSDP applies it on both axes, yielding the mean over the
# shard times the mean over the replicate axis.
_GRAD_REDUCE_OP = "avg"


def _free_tensor_storage(tensor: torch.Tensor) -> None:
    """Release ``tensor``'s storage in place (FSDP reshard free step).

    Inserted as a side-effecting FX node between an unsharded parameter's last
    forward reader and its backward re-gather, so the replicated full
    parameter (and every view aliasing it) does not stay resident for the
    whole joint graph. Returns ``None``; the node exists only for its effect.
    Mirrors ``fully_shard``'s ``free_unsharded_param`` storage release.
    """
    if isinstance(tensor, torch.Tensor) and tensor.untyped_storage().size() > 0:
        tensor.untyped_storage().resize_(0)


# FX dead-code elimination drops call_function nodes it judges pure; our free
# op is pure to FX but has a real side effect. Register it so a later
# ``eliminate_dead_code`` (a future pass, or a user) cannot delete the free.
_side_effectful_functions.add(_free_tensor_storage)


@dataclass
class _UnshardedParam:
    """Forward all_gather that materializes one FSDP parameter.

    ``param_node`` is the sharded placeholder (graph input); ``ag_node`` /
    ``wait_node`` produce and expose the replicated (unsharded) full parameter
    the forward — and, after a reshard re-gather, the backward — read.
    """

    param_node: fx.Node
    ag_node: fx.Node
    wait_node: fx.Node


@dataclass
class _DpParamClassification:
    """How DP handles each trainable parameter's gradient.

    Produced by ``_classify_dp_params``:

    - ``sharded_nodes``: marked, trainable parameter placeholders that get an
      AllGather (Shard -> Replicate) and a reduce_scatter gradient, and are
      physically sharded in the live model. Dim-0-divisible params are chunked
      directly; non-divisible ones are pad-sharded (see ``_PaddedParam``).
    - ``replicated_indices``: state indices of parameters kept replicated (never
      chunked) whose gradient is averaged over the DP group with ``all_reduce``
      instead of reduce-scattered — a marked 0-dim scalar (cannot be sharded), or
      a partial-plan leftover that falls back to DDP.
    - ``unmarked_trainable``: ``(state_idx, fqn)`` for the partial-plan leftovers
      (a subset of ``replicated_indices``), kept only so the caller can log that
      the plan did not cover them.
    """

    sharded_nodes: List[fx.Node]
    replicated_indices: Set[int]
    unmarked_trainable: List[Tuple[int, str]]


@dataclass
class _PaddedParam:
    """A marked parameter that cannot be sharded evenly, pad-sharded instead.

    Mirrors eager ``fully_shard`` (``core/fully_shard/hsdp_param.py``): dim 0 of
    ``n`` elements is zero-padded to ``n_pad = ceil(n / degree) * degree`` so
    every rank owns a uniform ``n_pad / degree``-element shard. The graph
    narrows the gathered ``[n_pad, ...]`` back to ``[n, ...]`` for the compute,
    and the backward pads the ``[n, ...]`` gradient back to ``[n_pad, ...]``
    before reduce_scatter, so the returned local gradient matches the padded
    shard (the padding region gets an all-zero gradient, hence never moves).
    """

    n: int
    n_pad: int
    dim0_pad: int
    ndim: int


def _iter_arg_nodes(node: fx.Node) -> Iterator[fx.Node]:
    """Yield every ``fx.Node`` in ``node.args`` / ``node.kwargs`` (nested)."""
    stack: List[Any] = list(node.args) + list(node.kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, fx.Node):
            yield item
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())


@contextmanager
def _insertion_point(
    graph: fx.Graph, anchor: Optional[fx.Node], fallback: fx.Node
) -> Iterator[None]:
    """Yield a graph insertion context.

    Owns the ``anchor``-vs-``fallback`` choice so the gather-insertion body
    stays a single code path: when ``anchor`` is given, new nodes land right
    before it; otherwise they land right after ``fallback``.
    """
    if anchor is None:
        with graph.inserting_after(fallback):
            yield
    else:
        with graph.inserting_before(anchor):
            yield


class FSDPPass(GraphPass):
    """
        FSDP Partitioning Pass (Module-Level, static-input graph)

        Responsibilities:
        1. Only shard parameters in modules marked for FSDP wrapping
        2. Insert AllGather after each FSDP parameter placeholder
           (Shard -> Replicate)
        3. Insert ReduceScatter after Backward (Replicate -> Shard)
        4. Physically shard the live model's parameters in place so the
           trainer / optimizer stay FSDP-agnostic

    Key Difference from Old Implementation:
    - Old: shard all parameters via get_attr nodes and physically rewrite the
      GraphModule's stored tensors
    - New: parameters are static graph inputs; the pass inserts collectives
      into the graph and shards the live model's parameters in place
    """

    name = "fsdp_parallel"

    def __init__(
        self,
        fsdp_group_name: Optional[str] = None,
        parallel_plan: Optional[GraphParallelPlan] = None,
    ) -> None:
        """Initialize FSDP pass state.

        Args:
            fsdp_group_name: Process-group name registered for FSDP
                collectives. Defaults to ``"fsdp"``. May be overridden
                per-run via the ``fsdp_group_name`` kwarg in ``run``.
            parallel_plan: Declarative plan identifying which modules to
                shard. When ``None``, all parameters are sharded.
        """
        super().__init__()
        self._fsdp_group_name = fsdp_group_name or "fsdp"
        self._parallel_plan = parallel_plan
        # Resolved from ``pass_config.fsdp_degree`` (or world_size) at
        # ``run`` entry; left as ``None`` here so a stray access before
        # ``run`` fails loudly instead of silently using a wrong default.
        self._fsdp_degree: Optional[int] = None
        self._expert_param_names: Set[str] = set()
        self._expert_fsdp_group_name = "fsdp_expert"
        self._expert_fsdp_degree: Optional[int] = None
        # Data-parallel mode ("fsdp" | "ddp" | "hsdp") and the replicate-axis
        # group/degree used by DDP / HSDP gradient reduction.
        self._dp_mode = "fsdp"
        self._dp_replicate_group_name = "dp_replicate"
        self._dp_replicate_degree: Optional[int] = None
        self._processed_params: Set[str] = set()
        self._fsdp_modules: Set[str] = set()
        # Per-param forward all_gather record (the unsharded parameter),
        # populated by ``_insert_all_gather_for_params`` and consumed by the
        # reshard step.
        self._unsharded_params: Dict[str, _UnshardedParam] = {}
        # Non-divisible marked params keyed by FQN; populated by
        # ``_classify_dp_params`` and consumed by the all_gather / grad-reduce /
        # live-shard steps so they are pad-sharded like eager ``fully_shard``.
        self._padded_params: Dict[str, _PaddedParam] = {}

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Insert AllGather/ReduceScatter and shard the live model.

        Args:
            graph_module: Joint fwd+bwd FX graph from the tracer.
            pass_config: Parallel configuration. ``fsdp_degree`` is
                read directly; ``None`` falls back to ``world_size`` (the
                FSDP-only path). A ``TypeError`` here means a non-Protocol
                config was passed — fix at the caller, do not paper over.
            **kwargs: Must include ``model`` (the live ``nn.Module``) and
                ``fsdp_group_name`` / ``parallel_plan`` as needed.

        Returns:
            The transformed graph module.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            _LOG.info("Skipped: distributed not initialized or world_size=1")
            return graph_module

        model = self._setup_dp_config(pass_config, kwargs)

        # Identify DP parameter placeholders. The joint graph's
        # parameters/buffers are static inputs (leading placeholders, not
        # get_attr nodes): locate them by position via the state layout the
        # tracer attached, then keep only those in DP-marked modules. For
        # FSDP/HSDP a marked param whose leading dim is not divisible by the
        # shard degree is pad-sharded (like eager fully_shard) so it is still
        # sharded and gradient-reduced; scalars stay replicated. DDP has no
        # shard axis so every marked parameter is reduced.
        state_fqns = getattr(graph_module, "state_fqns", [])
        num_state_inputs = getattr(graph_module, "num_state_inputs", 0)
        # Param-vs-buffer flag per leading state input. Absent on graphs
        # traced before this flag existed; default to all-params so old
        # traces keep the previous (parameter-only) behaviour.
        state_is_param = getattr(graph_module, "state_is_param", None)

        self._reset_run_state()
        classification = self._classify_dp_params(
            graph_module, state_fqns, num_state_inputs, state_is_param, model
        )
        param_nodes = classification.sharded_nodes

        _LOG.info(
            "Identified %d sharded (incl. %d pad-sharded) + %d replicated DP "
            "parameter nodes out of %d total state inputs",
            len(param_nodes),
            len(self._padded_params),
            len(classification.replicated_indices),
            num_state_inputs,
        )

        # A partial plan that leaves trainable params unmarked keeps their local
        # gradient (never reduced) — a silent divergence source across DP ranks,
        # so warn loudly before continuing.
        if classification.unmarked_trainable:
            self._warn_unmarked_trainable(classification.unmarked_trainable)

        if not param_nodes and not classification.replicated_indices:
            _LOG.warning(
                "No DP parameters found, check GraphParallelPlan or model structure"
            )
            return graph_module

        graph_module = self._apply_dp_transforms(
            graph_module,
            model,
            classification,
            pass_config,
            state_fqns,
            num_state_inputs,
            state_is_param,
        )

        _LOG.info(
            "Completed dp_mode=%s, processed %d parameters",
            self._dp_mode,
            len(param_nodes),
        )

        graph_module.recompile()
        return graph_module

    def _setup_dp_config(
        self, pass_config: PassConfig, kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Resolve run-time DP config, stash it on ``self``, return the model.

        Reads ``dp_mode`` / ``fsdp_degree`` / ``dp_replicate_degree`` off
        ``pass_config`` (the last two fall back to ``world_size`` when unset)
        and the group names / ``parallel_plan`` / ``model`` off ``kwargs``.

        Returns:
            The live ``nn.Module`` to physically shard.

        Raises:
            ValueError: If ``model`` is missing from ``kwargs``.
        """
        self._dp_mode = normalize_dp_mode(getattr(pass_config, "dp_mode", "fsdp"))
        # FSDP shard-axis group size: use the explicitly configured degree when
        # present (required for TP+FSDP / HSDP, where the FSDP group is a
        # proper sub-group of the world — using world_size would over-shard
        # along the TP axis).
        configured = pass_config.fsdp_degree
        self._fsdp_degree = configured if configured else dist.get_world_size()
        self._fsdp_group_name = kwargs.get("fsdp_group_name", self._fsdp_group_name)
        self._expert_param_names = set(kwargs.get("expert_param_names", ()))
        self._expert_fsdp_group_name = kwargs.get("expert_fsdp_group_name", self._expert_fsdp_group_name)
        self._expert_fsdp_degree = (
            dist.get_world_size(_resolve_process_group(self._expert_fsdp_group_name))
            if self._expert_param_names else None
        )
        self._dp_replicate_group_name = kwargs.get(
            "dp_replicate_group_name", self._dp_replicate_group_name
        )
        # Replicate-axis degree used by DDP / HSDP gradient reduction.
        replicate_degree = getattr(pass_config, "dp_replicate_degree", None)
        if self._dp_mode == "ddp":
            self._dp_replicate_degree = replicate_degree or dist.get_world_size()
        elif self._dp_mode == "hsdp":
            self._dp_replicate_degree = replicate_degree or (
                dist.get_world_size() // self._fsdp_degree
            )
        else:
            self._dp_replicate_degree = replicate_degree
        self._parallel_plan = kwargs.get("parallel_plan", self._parallel_plan)
        model = kwargs.get("model")
        if model is None:
            raise ValueError(
                "FSDPPass requires the live model via kwargs (model=...) so it "
                "can physically shard parameters; the trainer passes it through "
                "in compile()"
            )

        _LOG.info(
            "Running dp_mode=%s fsdp_degree=%s dp_replicate_degree=%s world_size=%s",
            self._dp_mode,
            self._fsdp_degree,
            self._dp_replicate_degree,
            dist.get_world_size(),
        )
        return model

    def _reset_run_state(self) -> None:
        """Clear per-run state so a reused pass does not carry names across graphs."""
        self._processed_params = set()
        self._fsdp_modules = set()
        self._unsharded_params = {}
        self._padded_params = {}

    def _apply_dp_transforms(
        self,
        graph_module: fx.GraphModule,
        model: nn.Module,
        classification: _DpParamClassification,
        pass_config: PassConfig,
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]],
    ) -> fx.GraphModule:
        """Insert gather / grad-reduce / reshard and shard the live model.

        DDP replicates parameters: no all_gather, no live-model sharding and no
        reshard — only the gradient all-reduce path runs. FSDP/HSDP also
        all_gather each parameter for the forward and pad/shard the live model
        in place.
        """
        param_nodes = classification.sharded_nodes
        if self._dp_mode != "ddp":
            graph_module = self._insert_all_gather_for_params(
                graph_module, param_nodes, pass_config
            )

        sharded_param_indices = frozenset(
            node.meta["state_idx"] for node in param_nodes
        )
        graph_module = self._insert_grad_reduction(
            graph_module,
            sharded_param_indices,
            classification.replicated_indices,
            state_fqns,
            num_state_inputs,
            state_is_param,
            model,
        )

        if self._dp_mode != "ddp":
            # Reshard: free each replicated parameter once forward is done and
            # re-gather + rematerialize it for the backward. Runs after the
            # grad reduction so the grad path is untouched; no-op when disabled.
            graph_module = self._insert_reshard_logic(
                graph_module, param_nodes, pass_config
            )

            # Shard the live model's parameters in place (dim 0, by this rank's
            # index in the FSDP group). ``model.parameters()`` then yields the
            # shards, so the trainer's optimizer / grad accumulation stay
            # DP-agnostic; the graph re-gathers each step.
            self._shard_live_model_params(model)

        return graph_module

    def _param_group(self, name: str) -> tuple[str, int]:
        """Choose the expert or dense shard axis for one parameter."""
        if name in self._expert_param_names:
            return self._expert_fsdp_group_name, self._expert_fsdp_degree
        return self._fsdp_group_name, self._fsdp_degree

    def _classify_dp_params(
        self,
        graph_module: fx.GraphModule,
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]] = None,
        model: Optional[nn.Module] = None,
    ) -> _DpParamClassification:
        """Classify parameter placeholders by how DP handles their gradient.

        Parameters are the leading ``num_state_inputs`` placeholders of the
        joint graph, in ``state_fqns`` order. A parameter is DP-marked when any
        ancestor module FQN (e.g. ``layers.0.attention.wq`` for
        ``layers.0.attention.wq.weight``) is marked in the GraphParallelPlan —
        either exactly or via pattern (e.g. ``layers.*``).

        Buffers (e.g. RoPE's non-persistent ``cache``) and frozen parameters
        are skipped: the tracer omits them from the gradient list and the
        live-model sharder leaves them full-size. The pad-shard and unmarked-
        fallback decisions are delegated to ``_marked_param_stays_replicated``
        and ``_unmarked_disposition``.

        Returns:
            A ``_DpParamClassification`` (see its docstring).
        """
        sharded_nodes: List[fx.Node] = []
        replicated_indices: Set[int] = set()
        unmarked_trainable: List[Tuple[int, str]] = []

        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        if len(placeholders) < num_state_inputs:
            return _DpParamClassification(
                sharded_nodes, replicated_indices, unmarked_trainable
            )

        # Build a FQN -> Parameter lookup so the divisibility gate uses the
        # same source of truth as ``_shard_live_model_params`` (which iterates
        # ``model.named_parameters``). ``state_fqns`` covers both parameters
        # and buffers; only parameters are reachable here because buffers are
        # filtered out by ``state_is_param`` above.
        param_lookup: Dict[str, nn.Parameter] = (
            dict(model.named_parameters(remove_duplicate=False))
            if model is not None
            else {}
        )

        for idx in range(num_state_inputs):
            if state_is_param is not None and not state_is_param[idx]:
                # Buffer, not a parameter -> never FSDP-sharded.
                continue

            fqn = state_fqns[idx]
            param = param_lookup.get(fqn)
            if model is not None and param is None:
                continue

            # Frozen params are not trainable: the tracer omits them from the
            # gradient list and ``_shard_live_model_params`` leaves them
            # full-size, so there is no gradient to classify.
            if param is not None and not param.requires_grad:
                _LOG.info("Skip %s: requires_grad=False (frozen)", fqn)
                continue

            _, degree = self._param_group(fqn)
            if fqn in self._expert_param_names and degree == 1:
                continue

            fallback = self._unmarked_disposition(fqn, param, model)
            if fallback is not None:
                if fallback:
                    replicated_indices.add(idx)
                    unmarked_trainable.append((idx, fqn))
                continue

            if self._marked_param_stays_replicated(fqn, param):
                replicated_indices.add(idx)
                continue

            node = placeholders[idx]
            node.meta["state_idx"] = idx
            node.meta["param_name"] = fqn
            node.meta["fsdp_degree"] = degree
            node.meta["is_param"] = True
            sharded_nodes.append(node)
            self._fsdp_modules.add(self._get_parent_module_fqn(fqn))

        return _DpParamClassification(
            sharded_nodes, replicated_indices, unmarked_trainable
        )

    def _unmarked_disposition(
        self,
        fqn: str,
        param: Optional[nn.Parameter],
        model: Optional[nn.Module],
    ) -> Optional[bool]:
        """Decide how a trainable param a partial plan left unmarked is handled.

        Rather than leave it with a local, unreduced gradient (silent
        divergence across DP ranks), fall it back to DDP: keep it replicated
        and average its gradient over the DP group.

        Returns:
            ``True`` to replicate it and warn the caller, ``False`` to skip it
            silently (``param is None`` with a live model means a buffer, only
            reachable when an old trace lacks ``state_is_param``), or ``None``
            when the param is marked and the caller should shard it normally.
        """
        if self._parallel_plan is None or self._param_belongs_to_fsdp_module(fqn):
            return None
        return model is None or param is not None

    def _marked_param_stays_replicated(
        self, fqn: str, param: Optional[nn.Parameter]
    ) -> bool:
        """Decide whether a marked param is kept replicated instead of sharded.

        A marked param whose dim 0 is not divisible by ``fsdp_degree`` cannot be
        chunked evenly. A 0-dim scalar cannot be padded along dim 0, so it stays
        replicated and its gradient is averaged over the shard axis. Every
        other non-divisible param is recorded for pad-sharding (mirroring eager
        ``fully_shard``): dim 0 is zero-padded to a multiple of the degree, so
        it is still sharded and gradient-reduced. DDP has no shard axis, so
        every marked param is handled the same way regardless of divisibility.
        """
        if self._dp_mode == "ddp" or param is None:
            return False
        _, degree = self._param_group(fqn)
        shape = param.shape
        if fqn in self._expert_param_names and (not shape or shape[0] % degree):
            raise ValueError(
                f"Graph FSDP+EP expert parameter {fqn} with shape {tuple(shape)} "
                f"cannot shard across edp_shard degree {degree}"
            )
        if shape and shape[0] % degree == 0:
            return False
        if not shape:
            _LOG.info(
                "Scalar param %s stays replicated: 0-dim cannot be "
                "pad-sharded; its gradient will be all-reduced",
                fqn,
            )
            return True
        n = int(shape[0])
        n_pad = -(-n // degree) * degree
        self._padded_params[fqn] = _PaddedParam(
            n=n, n_pad=n_pad, dim0_pad=n_pad - n, ndim=param.ndim
        )
        _LOG.info(
            "Param %s pad-sharded: dim 0 %d -> %d (fsdp_degree %d)",
            fqn,
            n,
            n_pad,
            degree,
        )
        return False

    @staticmethod
    def _warn_unmarked_trainable(entries: List[Tuple[int, str]]) -> None:
        """Report trainable params a partial plan left unmarked.

        They are not FSDP-sharded, so they fall back to DDP: kept replicated and
        their gradient averaged over the DP group. The warning is informational —
        the plan simply did not cover them, and they cost no sharded memory but a
        replicated gradient all-reduce.
        """
        shown = [fqn for _, fqn in entries[:10]]
        remaining = len(entries) - len(shown)
        _LOG.warning(
            "%d trainable parameter(s) are not marked for DP; falling back to "
            "DDP (kept replicated, gradient all-reduced over the DP group): "
            "%s%s. Mark them via GraphParallelPlan.fsdp_mark()/"
            "fsdp_mark_pattern() to FSDP-shard them instead.",
            len(entries),
            ", ".join(shown),
            f" (+{remaining} more)" if remaining > 0 else "",
        )

    def _shard_live_model_params(self, model: nn.Module) -> None:
        """
        Physically shard the live model's parameters in place (dim 0).

        After this, model.parameters() yields the local shards.
        """
        # Use the FSDP group's local rank (NOT the global rank) as the
        # chunk index — when fsdp_degree < world_size (TP+FSDP), the global
        # rank exceeds the chunk count and causes IndexError.
        fsdp_pg = _resolve_process_group(self._fsdp_group_name)
        group_ranks = {self._fsdp_group_name: dist.get_rank(group=fsdp_pg)}
        sharded_count = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            group_name, degree = self._param_group(name)
            if name in self._expert_param_names and degree == 1:
                continue

            # Same plan gate as ``_classify_dp_params``: only parameters inside
            # FSDP-marked modules are sharded. Skipping the gate here would
            # shard params the graph still expects full-rank
            # (no AllGather inserted), causing a shape mismatch at
            # ``run_traced_graph`` time on any non-``*`` plan.
            if (
                self._parallel_plan is not None
                and not self._param_belongs_to_fsdp_module(name)
            ):
                _LOG.info(
                    "Skip %s: not in an FSDP-wrapped module",
                    name,
                )
                continue

            if group_name not in group_ranks:
                group_ranks[group_name] = dist.get_rank(group=_resolve_process_group(group_name))
            rank = group_ranks[group_name]

            # Pad-shard a non-divisible param (recorded by
            # ``_classify_dp_params``): zero-pad dim 0 to a multiple of the
            # degree, then take this rank's equal-size chunk — exactly what
            # eager ``fully_shard`` does. A 0-dim scalar cannot be padded and
            # stays replicated.
            pad_info = self._padded_params.get(name)
            if pad_info is not None:
                original_shape = param.shape
                full = param.detach()
                if pad_info.dim0_pad:
                    full = torch.cat(
                        [full, full.new_zeros((pad_info.dim0_pad, *full.shape[1:]))],
                        dim=0,
                    )
                param.data = full.chunk(degree, dim=0)[rank].clone()
                sharded_count += 1
                _LOG.info(
                    "Pad-sharded %s: %s -> %s",
                    name,
                    list(original_shape),
                    list(param.shape),
                )
                continue

            # Mirror ``_classify_dp_params``: a 0-dim scalar stays replicated.
            if not param.shape or param.shape[0] % degree != 0:
                _LOG.info(
                    "Skip %s: dim 0 (%s) not divisible by fsdp_degree (%s)",
                    name,
                    param.shape[0] if param.shape else "scalar",
                    degree,
                )
                continue

            original_shape = param.shape
            param.data = param.detach().chunk(degree, dim=0)[rank].clone()
            sharded_count += 1
            _LOG.info(
                "Sharded %s: %s -> %s",
                name,
                list(original_shape),
                list(param.shape),
            )

        _LOG.info("Total sharded parameters: %d", sharded_count)

    def _param_belongs_to_fsdp_module(self, param_fqn: str) -> bool:
        """Check if a parameter FQN belongs to an FSDP-wrapped module.

        The parameter's own module FQN (``layers.0.attention.wq.weight`` ->
        ``layers.0.attention.wq``) plus every ancestor (``layers.0``, ...) is
        tested, so both ``fsdp_mark("layers.0.attention.wq")`` and
        ``fsdp_mark_pattern("layers.*")`` match.

        A parameter directly on the root (``weight``) has no module ancestor;
        its own FQN is still tested first so it can be marked by its name
        (``fsdp_mark("weight")``) or by a ``*`` pattern.
        """
        if self._parallel_plan.is_marked_for_fsdp(param_fqn):
            return True

        parts = param_fqn.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_fqn = ".".join(parts[:i])
            if self._parallel_plan.is_marked_for_fsdp(module_fqn):
                return True
        return False

    def _get_parent_module_fqn(self, param_fqn: str) -> str:
        """
        Get parent module FQN from parameter FQN.

        e.g., "layers.0.attention.wq.weight" -> "layers.0.attention.wq"
        """
        parts = param_fqn.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else param_fqn

    def _insert_all_gather_for_params(
        self,
        graph_module: fx.GraphModule,
        param_nodes: List[fx.Node],
        pass_config: PassConfig,
    ) -> fx.GraphModule:
        """
        Insert AllGather for each FSDP parameter placeholder.

        Parameter state: Shard -> Replicate. All subsequent uses of the
        placeholder are rewired to the unsharded (replicated) tensor, so the
        computation body keeps operating on full parameters while the graph
        input stays sharded. The gather is placed after the placeholder, or
        sunk to just before the first consumer when reshard is enabled.

        When reshard is enabled the gather is **sunk** to just before the
        parameter's first consumer instead of sitting next to the placeholder.
        All state placeholders lead the graph, so an unsunk gather would
        materialize every full parameter before any compute runs, leaving the
        reshard free step nothing to reclaim; sinking is what makes peak
        memory track the forward working set.
        """
        graph = graph_module.graph

        for param_node in param_nodes:
            if param_node.name in self._processed_params:
                continue

            sink = pass_config.fsdp_reshard_after_forward
            if sink and not param_node.users:
                # Dead parameter: a gather would materialize a full replica
                # nobody reads, and reshard would never free it. Leave the
                # placeholder sharded.
                self._processed_params.add(param_node.name)
                continue
            anchor = self._first_user(param_node) if sink else None

            group_name, degree = self._param_group(param_node.meta.get("param_name", ""))

            # AllGather + immediate wait for correctness; AutoOverlapPass may
            # sink the wait past independent compute later.
            with _insertion_point(graph, anchor, param_node):
                ag_node = graph.call_function(
                    _c10d_functional.all_gather_into_tensor,
                    args=(param_node, degree, group_name),
                )
                ag_node.meta["comm_type"] = "fsdp_all_gather"
                ag_node.meta["comm_group"] = group_name
                ag_node.meta["param_node"] = param_node.name
                ag_node.meta["param_name"] = param_node.meta.get("param_name")
                ag_node.meta["state_idx"] = param_node.meta.get("state_idx")
                ag_node.meta["fsdp_degree"] = degree

            # Insert the wait in its own block: a wait placed in the same block
            # as the gather would land *before* the gather it depends on (the
            # first insert inside ``inserting_after(X)`` lands right after
            # ``X``), and codegen would emit
            # ``wait_tensor(all_gather_into_tensor)`` before that variable is
            # assigned.
            with graph.inserting_after(ag_node):
                wait_node = graph.call_function(
                    _c10d_functional.wait_tensor,
                    args=(ag_node,),
                )
                wait_node.meta["wait_for"] = ag_node.name

            param_node.meta["fsdp_sharded"] = True
            param_node.meta["fsdp_ag_node"] = ag_node.name
            self._unsharded_params[param_node.name] = _UnshardedParam(
                param_node=param_node, ag_node=ag_node, wait_node=wait_node
            )

            # Pad-sharded parameter: the gather produced the padded full tensor
            # ``[n_pad, ...]``; narrow it back to the real ``[n, ...]`` and route
            # every consumer to that view. Use the ``aten`` overload (not the
            # ``torch.narrow`` builtin) so its schema's ``alias_info`` marks it a
            # view: ``_propagates_alias`` keys off that schema, so the reshard
            # alias walk (free / rematerialize) tracks the narrow and does not
            # free the padded gather out from under its forward consumers.
            replacement = wait_node
            pad_info = self._padded_params.get(param_node.meta.get("param_name"))
            if pad_info is not None:
                with graph.inserting_after(wait_node):
                    replacement = graph.call_function(
                        torch.ops.aten.narrow.default,
                        args=(wait_node, 0, 0, pad_info.n),
                    )
                replacement.meta["param_name"] = param_node.meta.get("param_name")
                replacement.meta["state_idx"] = param_node.meta.get("state_idx")

            for user in list(param_node.users.keys()):
                if user not in (ag_node, wait_node):
                    user.replace_input_with(param_node, replacement)

            self._processed_params.add(param_node.name)

        return graph_module

    @staticmethod
    def _first_user(param_node: fx.Node) -> Optional[fx.Node]:
        """Earliest (in graph order) consumer of ``param_node``.

        Placeholders are all at the top; their consumers are spread through
        the forward. Returning the earliest consumer lets the gather sink
        next to the first real use. ``None`` when the placeholder is unused
        (then the gather stays next to the placeholder).
        """
        users = [u for u in param_node.users if u.op != "output"]
        if not users:
            return None
        forward_users = [u for u in users if not u.meta.get("autograd_backward", False)]
        candidates = forward_users or users
        order = {n: i for i, n in enumerate(param_node.graph.nodes)}
        return min(candidates, key=lambda u: order.get(u, len(order)))

    def _insert_grad_reduction(  # pylint: disable=too-many-locals
        self,
        graph_module: fx.GraphModule,
        sharded_param_indices: Set[int],
        replicated_param_indices: Set[int],
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]] = None,
        model: Optional[nn.Module] = None,
    ) -> fx.GraphModule:
        """Insert gradient reduction on the gradient outputs of DP parameters.

        The joint graph returns ``[loss, grad0, grad1, ...]`` from the fwd+bwd
        function; gradient ``i`` (output index ``i+1``) corresponds to the
        ``i``-th trainable parameter (``state_fqns`` order, buffers and frozen
        params skipped, mirrored by ``_build_trainable_state_indices``). For a
        parameter in ``sharded_param_indices`` (a real shard axis) the mode
        decides the collective chain: ``fsdp`` reduce_scatters on the ``fsdp``
        group (Replicate -> Shard); ``ddp`` all_reduces on ``dp_replicate``;
        ``hsdp`` reduce_scatters on ``fsdp`` then all_reduces on
        ``dp_replicate`` (shard axis first, so the cross-axis message is
        already sharded). Pad-sharded params (``_padded_params``) zero-pad their
        ``[n, ...]`` gradient back to ``[n_pad, ...]`` first so the local
        gradient matches the padded shard.

        For a parameter in ``replicated_param_indices`` (a marked 0-dim scalar,
        or a partial-plan leftover that fell back to DDP) no shard axis exists,
        so the gradient is AVERAGED (all_reduce) over the DP group instead;
        FSDP/HSDP also average the shard axis, HSDP the replicate axis.
        Gradients of parameters outside the DP modules stay full.
        """
        graph = graph_module.graph

        output_node = next((n for n in graph.nodes if n.op == "output"), None)
        if output_node is None:
            return graph_module

        returned = output_node.args[0]
        if not isinstance(returned, (list, tuple)):
            # Model returned a single value (e.g. just loss) -> no grads.
            return graph_module

        trainable_state_indices = self._build_trainable_state_indices(
            state_fqns, num_state_inputs, state_is_param, model
        )

        new_returned = list(returned)
        # ``train_fn`` may append the traced ``loss_dict`` outputs AFTER the
        # gradients (see ``trace_model_graph``); those are not gradients and
        # must stay outside the reduce_scatter range.
        num_loss_outputs = len(getattr(graph_module, "loss_dict_keys", []))
        num_grads = len(new_returned) - 1 - num_loss_outputs  # index 0 is the loss
        if num_grads != len(trainable_state_indices):
            raise ValueError(
                f"Gradient count ({num_grads}) does not match trainable "
                f"parameter count ({len(trainable_state_indices)}). The "
                f"traced graph and the live model disagree on which "
                f"parameters are trainable; refusing to insert gradient "
                f"reduction to avoid silent gradient/state misalignment."
            )

        # Index 0 is the loss; gradients start at index 1. Gradient i+1
        # corresponds to trainable parameter i, whose state_idx is
        # trainable_state_indices[i]. The trailing loss_dict outputs are
        # named loss values, not gradients.
        for i in range(1, len(new_returned) - num_loss_outputs):
            grad_node = new_returned[i]
            if not isinstance(grad_node, fx.Node):
                continue

            state_idx = trainable_state_indices[i - 1]
            if state_idx in sharded_param_indices:
                replicated = False
                pad_info = self._padded_params.get(state_fqns[state_idx])
            elif state_idx in replicated_param_indices:
                replicated = True
                pad_info = None
            else:
                continue

            group_name, degree = self._param_group(state_fqns[state_idx])
            with graph.inserting_before(output_node):
                new_returned[i] = self._build_grad_reduction_nodes(
                    graph, grad_node, replicated=replicated, pad_info=pad_info,
                    group_name=group_name, degree=degree,
                )

        output_node.args = (type(returned)(new_returned),) + tuple(output_node.args[1:])

        return graph_module

    def _build_grad_reduction_nodes(
        self,
        graph: fx.Graph,
        grad_node: fx.Node,
        replicated: bool = False,
        pad_info: Optional[_PaddedParam] = None,
        group_name: Optional[str] = None,
        degree: Optional[int] = None,
    ) -> fx.Node:
        """Build and return the waited gradient-reduction node for ``grad_node``.

        The returned (wait) node replaces ``grad_node`` in the graph output.
        Emits the mode-specific chain (see ``_insert_grad_reduction``). When
        ``pad_info`` is given the parameter is pad-sharded: the real ``[n, ...]``
        gradient is zero-padded on dim 0 to ``[n_pad, ...]`` before the
        reduce_scatter, so the scattered local gradient matches the padded live
        shard (mirrors eager ``fully_shard``'s padded grad buffer). When
        ``replicated`` is True the parameter is a 0-dim scalar that could not be
        sharded, so its gradient is AVERAGED over the shard group with
        ``all_reduce`` instead of reduce-scattered — the graph and the live
        model both keep the parameter full, so a reduce_scatter would change the
        shape and desync them.

        Every collective is waited on before its result is consumed. A chained
        collective runs on a different process-group stream (HSDP's
        reduce_scatter is on ``fsdp`` while the all_reduce is on
        ``dp_replicate``), so without the intermediate wait the all_reduce could
        read a partially written shard. This mirrors eager ``fully_shard``,
        which orders the all_reduce stream after the reduce_scatter stream.
        """
        group_name = group_name or self._fsdp_group_name
        degree = degree or self._fsdp_degree
        reduced = grad_node
        if pad_info is not None:
            # constant_pad_nd's pad tuple runs last-dim-first: pad dim 0 on the
            # right by ``dim0_pad`` and leave every other dim untouched.
            pad_tuple = tuple([0, 0] * (pad_info.ndim - 1) + [0, pad_info.dim0_pad])
            reduced = graph.call_function(
                torch.constant_pad_nd, args=(reduced, pad_tuple)
            )
            reduced.meta["comm_type"] = "fsdp_grad_pad"
        if replicated:
            # Replicated parameter (a 0-dim scalar, or a partial-plan leftover
            # that fell back to DDP): average the full gradient over the shard
            # group so every replica takes the same optimizer step. DDP mode has
            # no shard group — the replicate all_reduce below handles it.
            if self._dp_mode in ("fsdp", "hsdp"):
                ar_node = graph.call_function(
                    _c10d_functional.all_reduce,
                    args=(reduced, _GRAD_REDUCE_OP, group_name),
                )
                ar_node.meta["comm_type"] = "fsdp_all_reduce"
                ar_node.meta["comm_group"] = group_name
                ar_node.meta["fsdp_degree"] = degree
                reduced = ar_node
                # HSDP chains the replicate all_reduce (a different process-group
                # stream) after the shard all_reduce, so wait first.
                if self._dp_mode == "hsdp":
                    reduced = self._insert_wait(graph, reduced)
        elif self._dp_mode in ("fsdp", "hsdp"):
            # Matmul backward can return a transposed view on NPU, including dense router weights.
            reduced = graph.call_method("contiguous", args=(reduced,))
            rs_node = graph.call_function(
                _c10d_functional.reduce_scatter_tensor,
                args=(
                    reduced,
                    _GRAD_REDUCE_OP,
                    degree,
                    group_name,
                ),
            )
            rs_node.meta["comm_type"] = "fsdp_reduce_scatter"
            rs_node.meta["comm_group"] = group_name
            rs_node.meta["fsdp_degree"] = degree
            reduced = rs_node
            # HSDP chains all_reduce (a different process-group stream) after
            # the reduce_scatter, so wait first; plain FSDP waits on the final
            # node only.
            if self._dp_mode == "hsdp":
                reduced = self._insert_wait(graph, reduced)

        if self._dp_mode in ("ddp", "hsdp"):
            ar_node = graph.call_function(
                _c10d_functional.all_reduce,
                args=(reduced, _GRAD_REDUCE_OP, self._dp_replicate_group_name),
            )
            ar_node.meta["comm_type"] = "dp_all_reduce"
            ar_node.meta["comm_group"] = self._dp_replicate_group_name
            ar_node.meta["dp_replicate_degree"] = self._dp_replicate_degree
            reduced = ar_node

        return self._insert_wait(graph, reduced)

    @staticmethod
    def _insert_wait(graph: fx.Graph, tensor_node: fx.Node) -> fx.Node:
        """Append ``wait_tensor(tensor_node)`` and return the waited node."""
        wait_node = graph.call_function(
            _c10d_functional.wait_tensor,
            args=(tensor_node,),
        )
        wait_node.meta["wait_for"] = tensor_node.name
        return wait_node

    def _build_trainable_state_indices(
        self,
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]],
        model: Optional[nn.Module],
    ) -> List[int]:
        """
        Return state indices of trainable parameters, in ``state_fqns`` order.

        Mirrors the tracer's ``params`` list construction (skip buffers via
        ``state_is_param``, skip frozen params via ``requires_grad``) so
        gradient ``i`` maps to ``trainable_state_indices[i]``. Reproducing this
        filter here is what keeps reduce_scatter aligned for models with
        buffers or frozen params.
        """
        if model is None:
            # Without the model we cannot inspect requires_grad; fall back to
            # the all-trainable assumption. The grad-count check in the caller
            # raises if that assumption is wrong, so misalignment is not silent.
            return list(range(num_state_inputs))

        param_lookup = dict(model.named_parameters(remove_duplicate=False))
        trainable: List[int] = []
        for idx in range(num_state_inputs):
            if state_is_param is not None and not state_is_param[idx]:
                continue
            param = param_lookup.get(state_fqns[idx])
            if param is not None and param.requires_grad:
                trainable.append(idx)
        return trainable

    def _insert_reshard_logic(
        self,
        graph_module: fx.GraphModule,
        param_nodes: List[fx.Node],
        pass_config: PassConfig,
    ) -> fx.GraphModule:
        """Reshard each FSDP parameter after forward (FSDP release).

        For every FSDP parameter whose all_gather was inserted by
        ``_insert_all_gather_for_params``:

        1. Free the replicated full parameter once its last forward reader —
           including forward views aliasing it — has run.
        2. If the backward still needs the parameter (directly or through a
           saved forward view), insert a fresh all_gather just before the
           first backward consumer and rematerialize the saved view chain, so
           the backward computes the exact same values from the re-gathered
           (unsharded) parameter.
        3. Free that backward re-gather once its last backward reader runs.

        Steps 1-3 keep only the forward working set resident during forward and
        only the active layer's parameter resident during backward, instead of
        every replicated parameter for the whole joint graph.

        No-op when ``fsdp_reshard_after_forward`` is disabled.
        """
        if not pass_config.fsdp_reshard_after_forward:
            return graph_module

        for param_node in param_nodes:
            record = self._unsharded_params.get(param_node.name)
            if record is None:
                continue
            self._reshard_param(graph_module.graph, record)
        return graph_module

    def _reshard_param(self, graph: fx.Graph, record: _UnshardedParam) -> None:
        """Reshard one FSDP parameter (free + backward re-gather/remat)."""
        wait = record.wait_node
        alias_nodes = self._alias_descendants(wait)
        order = {n: i for i, n in enumerate(graph.nodes)}

        # Alias (view) descendants the backward reads must be rematerialized
        # from the re-gathered parameter; the unsharded tensor itself may be
        # consumed directly by the backward in some models.
        needed = [
            n for n in [wait, *alias_nodes] if self._feeds_backward(n, alias_nodes)
        ]

        # Remat plan first: the forward free is only safe once every backward
        # consumer has been rewired onto fresh (re-gathered) storage. If the
        # view chain cannot be cloned, keep the previous resident-for-the-graph
        # behavior — a missed free costs memory, a misordered one is unsound.
        if self._rematerialize_for_backward(graph, record, needed, order):
            # 1. Free after the last forward data read of the unsharded storage.
            self._insert_reshard_free(graph, wait, record, order, backward=False)

    def _rematerialize_for_backward(
        self,
        graph: fx.Graph,
        record: _UnshardedParam,
        needed: List[fx.Node],
        order: Dict[fx.Node, int],
    ) -> bool:
        """Re-gather ahead of the first backward consumer and remat views.

        The saved forward view chain is cloned against the re-gathered
        parameter so the backward reads identical values from a fresh,
        correctly-sized tensor.

        Returns:
            ``True`` when the free may proceed — no backward consumer exists,
            or every one of them was rewired onto rematerialized storage.
            ``False`` when a needed view cannot be cloned; the caller must
            keep the unsharded parameter resident.
        """
        backward_users = [
            u for n in needed for u in n.users if u.meta.get("autograd_backward", False)
        ]
        if not backward_users:
            return True

        wait = record.wait_node
        uncloneable = [n for n in needed if n is not wait and not self._cloneable(n)]
        if uncloneable:
            _LOG.warning(
                "FSDP reshard: cannot rematerialize %s for '%s'; keeping the "
                "unsharded parameter resident",
                [n.format_node() for n in uncloneable],
                record.param_node.meta.get("param_name"),
            )
            return False

        anchor = min(backward_users, key=lambda u: order.get(u, len(order)))
        wait2 = self._insert_backward_gather(graph, record, anchor)

        recreate: Dict[fx.Node, fx.Node] = {wait: wait2}
        last_inserted = wait2
        for node in sorted(
            (n for n in needed if n is not wait), key=lambda n: order[n]
        ):
            last_inserted = self._clone_with_remap(
                graph, node, recreate, wait, wait2, last_inserted
            )

        self._rewire_backward_users(recreate)
        self._insert_reshard_free(graph, wait2, record, order, backward=True)
        return True

    @staticmethod
    def _cloneable(node: fx.Node) -> bool:
        """Whether ``node`` can be cloned as a rematerialized view op."""
        return node.op == "call_function" and callable(node.target)

    def _insert_backward_gather(
        self, graph: fx.Graph, record: _UnshardedParam, anchor: fx.Node
    ) -> fx.Node:
        """Insert the backward all_gather + wait pair before ``anchor``."""
        group_name, degree = self._param_group(record.param_node.meta.get("param_name", ""))
        with graph.inserting_before(anchor):
            ag_node = graph.call_function(
                _c10d_functional.all_gather_into_tensor,
                args=(record.param_node, degree, group_name),
            )
            ag_node.meta["comm_type"] = "fsdp_all_gather_backward"
            ag_node.meta["comm_group"] = group_name
            ag_node.meta["param_node"] = record.param_node.name
            ag_node.meta["param_name"] = record.param_node.meta.get("param_name")
            ag_node.meta["fsdp_degree"] = degree
            # This gather lives in the backward half; tag it so downstream
            # passes (e.g. PpPass phase classification) do not mistake it and
            # its remat clones for forward nodes — their only arg is a forward
            # state placeholder, which would otherwise imply the forward phase.
            ag_node.meta["autograd_backward"] = True
        with graph.inserting_before(anchor):
            wait_node = graph.call_function(
                _c10d_functional.wait_tensor, args=(ag_node,)
            )
            wait_node.meta["wait_for"] = ag_node.name
            wait_node.meta["autograd_backward"] = True
        return wait_node

    def _clone_with_remap(
        self,
        graph: fx.Graph,
        node: fx.Node,
        recreate: Dict[fx.Node, fx.Node],
        wait: fx.Node,
        wait2: fx.Node,
        insert_after: fx.Node,
    ) -> fx.Node:
        """Clone ``node`` after ``insert_after``, remapping args to re-gathers."""
        args = tuple(self._remap_arg(a, recreate, wait, wait2) for a in node.args)
        kwargs = {
            k: self._remap_arg(v, recreate, wait, wait2) for k, v in node.kwargs.items()
        }
        with graph.inserting_after(insert_after):
            clone = graph.call_function(node.target, args=args, kwargs=kwargs)
        self._copy_recreated_meta(node, clone)
        # Rematerialized views serve the backward half, so they carry the
        # backward phase (see ``_insert_backward_gather``).
        clone.meta["autograd_backward"] = True
        recreate[node] = clone
        return clone

    @staticmethod
    def _rewire_backward_users(recreate: Dict[fx.Node, fx.Node]) -> None:
        """Point every backward consumer of an original node at its clone."""
        for original, clone in recreate.items():
            for user in list(original.users):
                if user.meta.get("autograd_backward", False):
                    user.replace_input_with(original, clone)

    def _insert_reshard_free(
        self,
        graph: fx.Graph,
        tensor: fx.Node,
        record: _UnshardedParam,
        order: Dict[fx.Node, int],
        backward: bool,
    ) -> None:
        """Free ``tensor``'s storage after its last reader in the phase."""
        last_reader = self._last_data_reader(tensor, order, backward=backward)
        if last_reader is None:
            return
        comm_type = "fsdp_reshard_free_backward" if backward else "fsdp_reshard_free"
        with graph.inserting_after(last_reader):
            free_node = graph.call_function(_free_tensor_storage, args=(tensor,))
            free_node.meta["comm_type"] = comm_type
            free_node.meta["param_name"] = record.param_node.meta.get("param_name")
            if backward:
                free_node.meta["autograd_backward"] = True

    def _alias_descendants(self, root: fx.Node) -> Set[fx.Node]:
        """Forward view descendants of ``root`` (nodes aliasing its storage).

        Only aliasing ops propagate: an op that allocates a new tensor
        (``mm``, ``addmm``, ``relu``, ...) copies the data out, so freeing
        ``root``'s storage does not invalidate it and it need not be
        rematerialized. Backward nodes terminate the walk.
        """
        alias: Set[fx.Node] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            for user in current.users:
                if user is root or user in alias:
                    continue
                if user.meta.get("autograd_backward", False):
                    continue
                if current not in set(_iter_arg_nodes(user)):
                    continue
                if not self._propagates_alias(user):
                    continue
                alias.add(user)
                stack.append(user)
        return alias

    @staticmethod
    def _propagates_alias(node: fx.Node) -> bool:
        """Whether ``node``'s output aliases its already-aliasing input.

        Two forms propagate. Schema-declared view returns (``aten.t``,
        ``aten.slice``, ...), and ``operator.getitem`` picking one output of a
        multi-output view op such as ``aten.chunk`` / ``aten.split`` /
        ``aten.unbind`` — getitem itself has no schema, so the alias must be
        inherited from the indexed producer (which the caller has already
        established is in the alias closure).
        """
        if node.op != "call_function":
            return False
        if node.target is _GETITEM:
            return True
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            return False
        return any(ret.alias_info is not None for ret in schema.returns)

    def _feeds_backward(self, node: fx.Node, alias_nodes: Set[fx.Node]) -> bool:
        """Whether ``node`` (or an aliasing descendant) is read by backward."""
        memo: Dict[fx.Node, bool] = {}

        def visit(current: fx.Node) -> bool:
            """Whether ``current`` or an aliasing descendant reaches backward.

            Args:
                current: Node to test.

            Returns:
                True when a backward node consumes the node or its aliases.
            """
            if current in memo:
                return memo[current]
            memo[current] = False  # cycle guard (aliasing is a DAG)
            result = False
            for user in current.users:
                if user.meta.get("autograd_backward", False):
                    result = True
                    break
                if user in alias_nodes and visit(user):
                    result = True
                    break
            memo[current] = result
            return result

        return visit(node)

    def _last_data_reader(
        self, root: fx.Node, order: Dict[fx.Node, int], backward: bool
    ) -> Optional[fx.Node]:
        """Latest, in graph order, op that reads ``root``'s storage data.

        Walks the alias (view) chain so a free is not placed after a view
        *creation* but before the non-view op that actually reads it: freeing
        between ``t_2 = t(w)`` and ``mm(grad, t_2)`` would invalidate ``t_2``.
        Multi-output views reach their consumers through ``getitem``, so the
        walk continues through those picks instead of mistaking them for the
        final reader. ``backward=False`` picks the last forward read (free
        point for the unsharded parameter); ``backward=True`` the last
        backward read (free point for the backward re-gather).
        """
        last: Optional[fx.Node] = None
        last_pos = -1
        seen: Set[fx.Node] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for user in current.users:
                if user.op == "output":
                    continue
                if self._propagates_alias(user) and current in set(
                    _iter_arg_nodes(user)
                ):
                    stack.append(user)
                    continue
                if bool(user.meta.get("autograd_backward", False)) != backward:
                    continue
                pos = order.get(user, -1)
                if pos > last_pos:
                    last_pos = pos
                    last = user
        return last

    @classmethod
    def _remap_arg(
        cls,
        arg: Any,
        recreate: Dict[fx.Node, fx.Node],
        wait: fx.Node,
        wait2: fx.Node,
    ) -> Any:
        """Rewrite an op arg for a rematerialized clone (nested containers)."""
        if arg is wait:
            return wait2
        if isinstance(arg, fx.Node):
            return recreate.get(arg, arg)
        if isinstance(arg, tuple):
            return tuple(cls._remap_arg(a, recreate, wait, wait2) for a in arg)
        if isinstance(arg, list):
            return [cls._remap_arg(a, recreate, wait, wait2) for a in arg]
        if isinstance(arg, dict):
            return {k: cls._remap_arg(v, recreate, wait, wait2) for k, v in arg.items()}
        return arg

    @staticmethod
    def _copy_recreated_meta(original: fx.Node, clone: fx.Node) -> None:
        """Copy the module/stack/shape metadata a remat clone needs."""
        for key in (
            "nn_module_stack",
            "custom",
            "stack_trace",
            "tensor_meta",
            "val",
        ):
            value = original.meta.get(key)
            if value is not None:
                clone.meta[key] = value
