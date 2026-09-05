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
FSDP Pass - Fully Sharded Data Parallel Partitioning Pass

Operates on the joint fwd+bwd graph produced by the tracer, where
parameters/buffers are **static inputs** (leading placeholders), not get_attr
nodes.

Responsibilities:
1. Identify parameter placeholders belonging to FSDP-wrapped modules
   (via PassPlan, exact FQN or pattern)
2. Insert AllGather after each such placeholder (Shard -> Replicate), so the
   computation body operates on full parameters while the graph input stays
   sharded
3. Insert ReduceScatter on the gradient outputs of FSDP parameters, then
   synchronize HSDP replica gradients with AllReduce
   (Replicate -> Shard); gradients of non-FSDP parameters stay full
4. Physically shard the *live model's* parameters in place (dim 0, by FSDP
   rank), so ``model.parameters()`` already holds the local shard and the
   trainer / optimizer need no FSDP awareness at all

All FSDP logic (which parameters, the collectives, and the sharding itself)
lives in this pass; the trainer simply feeds ``model.parameters()``.

Partitioning:
- Parameters sharded on dim 0
- Forward: all_gather parameters, compute, optional release
- Backward: reduce_scatter gradients
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import torch.distributed as dist
from torch import fx, nn
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.ops import _c10d_functional

from ...parallel_config import PassConfig
from ..base import GraphPass
from ...sharding_config import PassPlan

_LOG = logging.getLogger(__name__)


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
        pass_plan: Optional[PassPlan] = None,
    ) -> None:
        """Initialize FSDP pass state.

        Args:
            fsdp_group_name: Process-group name registered for FSDP
                collectives. Defaults to ``"fsdp"``. May be overridden
                per-run via the ``fsdp_group_name`` kwarg in ``run``.
            pass_plan: Declarative plan identifying which modules to
                shard. When ``None``, all parameters are sharded.
        """
        super().__init__()
        self._fsdp_group_name = fsdp_group_name or "fsdp"
        self._pass_plan = pass_plan
        # Resolved from ``pass_config.fsdp_degree`` (or world_size) at
        # ``run`` entry; left as ``None`` here so a stray access before
        # ``run`` fails loudly instead of silently using a wrong default.
        self._fsdp_degree: Optional[int] = None
        self._processed_params: Set[str] = set()
        self._fsdp_modules: Set[str] = set()
        self._group_infos: Dict[str, Tuple[int, int]] = {}
        self._param_groups: Dict[str, str] = {}
        self._replicate_groups: Dict[str, str] = {}

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
                ``fsdp_group_name`` / ``pass_plan`` as needed.

        Returns:
            The transformed graph module.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            _LOG.info("Skipped: distributed not initialized or world_size=1")
            return graph_module

        # FSDP group size: use the explicitly configured degree when present
        # (required for TP+FSDP, where the FSDP group is a proper sub-group
        # of the world — using world_size would over-shard along the TP
        # axis).
        configured = pass_config.fsdp_degree
        self._fsdp_degree = configured if configured else dist.get_world_size()
        self._fsdp_group_name = kwargs.get("fsdp_group_name", self._fsdp_group_name)
        self._pass_plan = kwargs.get("pass_plan", self._pass_plan)
        self._configure_groups(kwargs)
        model = kwargs.get("model")
        if model is None:
            raise ValueError(
                "FSDPPass requires the live model via kwargs (model=...) so it "
                "can physically shard parameters; the trainer passes it through "
                "in compile()"
            )

        _LOG.info(
            "Running with fsdp_degree=%s, world_size=%s",
            self._fsdp_degree,
            dist.get_world_size(),
        )

        # Identify FSDP parameter placeholders. The joint graph's
        # parameters/buffers are static inputs (leading placeholders, not
        # get_attr nodes): locate them by position via the state layout the
        # tracer attached, then keep only those in FSDP-marked modules whose
        # leading dim is divisible by the FSDP degree (non-divisible params
        # stay replicated, in both graph and live model).
        state_fqns = getattr(graph_module, "state_fqns", [])
        num_state_inputs = getattr(graph_module, "num_state_inputs", 0)
        # Param-vs-buffer flag per leading state input. Absent on graphs
        # traced before this flag existed; default to all-params so old
        # traces keep the previous (parameter-only) behaviour.
        state_is_param = getattr(graph_module, "state_is_param", None)
        param_nodes = self._identify_params_in_fsdp_modules(
            graph_module, state_fqns, num_state_inputs, state_is_param, model
        )

        _LOG.info(
            "Identified %d FSDP parameter nodes out of %d total state inputs",
            len(param_nodes),
            num_state_inputs,
        )

        if not param_nodes and not self._replicate_groups:
            _LOG.warning(
                "No FSDP parameters found, check PassPlan or model structure"
            )
            return graph_module

        if param_nodes:
            graph_module = self._insert_all_gather_for_params(
                graph_module, param_nodes
            )

        sharded_param_groups = {
            node.meta["state_idx"]: (
                node.meta["fsdp_group_name"],
                node.meta["fsdp_degree"],
            )
            for node in param_nodes
        }
        graph_module = self._insert_reduce_scatter_for_grads(
            graph_module,
            sharded_param_groups,
            state_fqns,
            num_state_inputs,
            state_is_param,
            model,
        )

        # Shard the live model's parameters in place (dim 0, by this rank's
        # index in the FSDP group). ``model.parameters()`` then yields the
        # shards, so the trainer's optimizer / grad accumulation stay
        # FSDP-agnostic; the graph re-gathers each step.
        self._shard_live_model_params(model)
        _LOG.info("Completed, sharded %d parameters", len(param_nodes))

        graph_module.recompile()
        return graph_module

    def _configure_groups(self, kwargs: Mapping[str, Any]) -> None:
        """Resolve communication degree and local rank for each FSDP group."""
        raw_group_infos = kwargs.get("fsdp_group_infos")
        if raw_group_infos is None:
            self._group_infos = {
                self._fsdp_group_name: (
                    self._fsdp_degree,
                    kwargs.get("fsdp_group_rank", dist.get_rank()),
                )
            }
        else:
            self._group_infos = {}
            for group_name, group_info in raw_group_infos.items():
                degree = group_info["degree"]
                rank = group_info["rank"]
                if (
                    isinstance(degree, bool)
                    or not isinstance(degree, int)
                    or degree < 1
                ):
                    raise ValueError(
                        f"FSDP group {group_name!r} degree must be a positive integer"
                    )
                if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < degree:
                    raise ValueError(
                        f"FSDP group {group_name!r} rank must be in [0, {degree}), got {rank}"
                    )
                self._group_infos[group_name] = (degree, rank)
            if self._fsdp_group_name not in self._group_infos:
                raise ValueError(
                    f"Default FSDP group {self._fsdp_group_name!r} is missing from fsdp_group_infos"
                )
        self._param_groups = dict(kwargs.get("fsdp_param_groups", {}))
        unknown_groups = set(self._param_groups.values()) - set(self._group_infos)
        if unknown_groups:
            raise ValueError(
                f"Parameter FSDP groups are not registered: {sorted(unknown_groups)}"
            )
        self._configure_replicate_groups(kwargs)

    def _configure_replicate_groups(self, kwargs: Mapping[str, Any]) -> None:
        """Validate the shard-to-replica group mapping used by HSDP."""
        self._replicate_groups = dict(kwargs.get("fsdp_replicate_groups", {}))
        unknown_shard_groups = set(self._replicate_groups) - set(self._group_infos)
        unknown_replicate_groups = (
            set(self._replicate_groups.values()) - set(self._group_infos)
        )
        if unknown_shard_groups or unknown_replicate_groups:
            raise ValueError(
                "FSDP replica groups reference unregistered groups: "
                f"shard={sorted(unknown_shard_groups)}, "
                f"replicate={sorted(unknown_replicate_groups)}"
            )

    def _group_for_param(self, param_fqn: str) -> Tuple[str, int, int]:
        """Return group name, degree, and group-local rank for a parameter."""
        group_name = self._param_groups.get(param_fqn, self._fsdp_group_name)
        degree, rank = self._group_infos[group_name]
        return group_name, degree, rank

    def _replicate_group_for_param(
        self, param_fqn: str
    ) -> Optional[Tuple[str, int]]:
        """Return the replica group name and degree for a parameter."""
        shard_group_name = self._param_groups.get(
            param_fqn, self._fsdp_group_name
        )
        replicate_group_name = self._replicate_groups.get(shard_group_name)
        if replicate_group_name is None:
            return None
        replicate_degree, _ = self._group_infos[replicate_group_name]
        return replicate_group_name, replicate_degree

    def _identify_params_in_fsdp_modules(
        self,
        graph_module: fx.GraphModule,
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]] = None,
        model: Optional[nn.Module] = None,
    ) -> List[fx.Node]:
        """
        Identify parameter placeholder nodes belonging to FSDP-wrapped modules.

        Parameters are the leading ``num_state_inputs`` placeholders of the
        joint graph, in ``state_fqns`` order. A parameter is FSDP-sharded when
        any ancestor module FQN (e.g. ``layers.0.attention.wq`` for
        ``layers.0.attention.wq.weight``) is marked in the PassPlan —
        either exactly or via pattern (e.g. ``layers.*``).

        Only parameters are sharded. Buffers (e.g. RoPE's non-persistent
        ``cache``) are full-rank by construction and must not be all-gathered;
        they are skipped using the ``state_is_param`` flag the tracer attaches
        to the graph. When the flag is unavailable, every state input is
        treated as a parameter (previous behaviour).

        The dim-0 divisibility gate mirrors ``_shard_live_model_params``: a
        parameter whose leading dim is not divisible by ``fsdp_degree`` stays
        replicated on both sides. Skipping it here would make the graph expect
        a sharded input (AllGather reshapes ``[N/world, ...] -> [N, ...]``)
        while the live model still holds the full ``[N, ...]`` tensor, causing
        a shape mismatch at ``run_traced_graph`` time.
        """
        param_nodes: List[fx.Node] = []

        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        if len(placeholders) < num_state_inputs:
            return param_nodes

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

            if (
                self._pass_plan is not None
                and not self._param_belongs_to_fsdp_module(fqn)
            ):
                continue

            group_name, group_degree, _ = self._group_for_param(fqn)
            if group_degree == 1:
                continue

            # Divisibility gate: must match ``_shard_live_model_params`` so the
            # graph and the live model agree on which parameters are sharded.
            param = param_lookup.get(fqn)
            if (
                param is not None
                and param.shape
                and param.shape[0] % group_degree != 0
            ):
                _LOG.info(
                    "Skip %s: dim 0 (%s) not divisible by fsdp_degree (%s)",
                    fqn,
                    param.shape[0],
                    group_degree,
                )
                continue

            node = placeholders[idx]
            node.meta["state_idx"] = idx
            node.meta["param_name"] = fqn
            node.meta["fsdp_degree"] = group_degree
            node.meta["fsdp_group_name"] = group_name
            node.meta["is_param"] = True
            param_nodes.append(node)
            self._fsdp_modules.add(self._get_parent_module_fqn(fqn))

        return param_nodes

    def _shard_live_model_params(self, model: nn.Module) -> None:
        """
        Physically shard the live model's parameters in place (dim 0).

        After this, model.parameters() yields the local shards.
        """
        sharded_count = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Same plan gate as ``_identify_params_in_fsdp_modules``: only
            # parameters inside FSDP-marked modules are sharded. Skipping the
            # gate here would shard params the graph still expects full-rank
            # (no AllGather inserted), causing a shape mismatch at
            # ``run_traced_graph`` time on any non-``*`` plan.
            if (
                self._pass_plan is not None
                and not self._param_belongs_to_fsdp_module(name)
            ):
                _LOG.info(
                    "Skip %s: not in an FSDP-wrapped module",
                    name,
                )
                continue

            group_name, group_degree, group_rank = self._group_for_param(name)
            if group_degree == 1:
                continue

            # Mirror ``_identify_params_in_fsdp_modules``: scalar params
            # (empty shape) and non-divisible dim-0 params stay replicated.
            if not param.shape or param.shape[0] % group_degree != 0:
                _LOG.info(
                    "Skip %s: dim 0 (%s) not divisible by fsdp_degree (%s)",
                    name,
                    param.shape[0] if param.shape else "scalar",
                    group_degree,
                )
                continue

            original_shape = param.shape
            param.data = param.detach().chunk(group_degree, dim=0)[group_rank].clone()
            sharded_count += 1
            _LOG.info(
                "Sharded %s on %s: %s -> %s",
                name,
                group_name,
                list(original_shape),
                list(param.shape),
            )

        _LOG.info("Total sharded parameters: %d", sharded_count)

    def _param_belongs_to_fsdp_module(self, param_fqn: str) -> bool:
        """Check if a parameter FQN belongs to an FSDP-wrapped module.

        The parameter's own module FQN (``layers.0.attention.wq.weight`` ->
        ``layers.0.attention.wq``) plus every ancestor (``layers.0``, ...) is
        tested, so both ``fsdp_wrap("layers.0.attention.wq")`` and
        ``fsdp_wrap_pattern("layers.*")`` match.

        A parameter directly on the root (``weight``) has no module ancestor;
        its own FQN is still tested first so it can be wrapped by its name
        (``fsdp_wrap("weight")``) or by a ``*`` pattern.
        """
        if self._pass_plan.is_fsdp_module(param_fqn):
            return True

        parts = param_fqn.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_fqn = ".".join(parts[:i])
            if self._pass_plan.is_fsdp_module(module_fqn):
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
    ) -> fx.GraphModule:
        """
        Insert AllGather after each FSDP parameter placeholder.

        Parameter state: Shard -> Replicate. All subsequent uses of the
        placeholder are rewired to the gathered (replicated) tensor, so the
        computation body keeps operating on full parameters while the graph
        input stays sharded.
        """
        graph = graph_module.graph

        for param_node in param_nodes:
            if param_node.name in self._processed_params:
                continue

            group_name = param_node.meta["fsdp_group_name"]
            group_degree = param_node.meta["fsdp_degree"]
            # AllGather + immediate wait for correctness; AutoOverlapPass may
            # sink the wait past independent compute later.
            with graph.inserting_after(param_node):
                ag_node = graph.call_function(
                    _c10d_functional.all_gather_into_tensor,
                    args=(param_node, group_degree, group_name),
                )
                ag_node.meta["comm_type"] = "fsdp_all_gather"
                ag_node.meta["comm_group"] = group_name
                ag_node.meta["param_node"] = param_node.name
                ag_node.meta["param_name"] = param_node.meta.get("param_name")
                ag_node.meta["state_idx"] = param_node.meta.get("state_idx")
                ag_node.meta["fsdp_degree"] = group_degree

            # Insert the wait in its own ``inserting_after(ag_node)`` block: a
            # wait placed in the same block as the gather would land *before*
            # the gather it depends on (the first insert inside
            # ``with inserting_after(X)`` lands right after ``X``), and
            # codegen would emit ``wait_tensor(all_gather_into_tensor)``
            # before that variable is assigned.
            with graph.inserting_after(ag_node):
                wait_node = graph.call_function(
                    _c10d_functional.wait_tensor,
                    args=(ag_node,),
                )
                wait_node.meta["wait_for"] = ag_node.name

            param_node.meta["fsdp_sharded"] = True
            param_node.meta["fsdp_ag_node"] = ag_node.name

            for user in list(param_node.users.keys()):
                if user not in (ag_node, wait_node):
                    user.replace_input_with(param_node, wait_node)

            self._processed_params.add(param_node.name)

        return graph_module

    def _insert_gradient_collectives(
        self,
        graph: fx.Graph,
        grad_node: fx.Node,
        param_fqn: str,
        shard_group_info: Optional[Tuple[str, int]],
        replicate_group_info: Optional[Tuple[str, int]],
    ) -> fx.Node:
        """Insert FSDP reduce-scatter followed by HSDP replica reduction."""
        synchronized_grad = grad_node
        if shard_group_info is not None:
            group_name, group_degree = shard_group_info
            reduce_scatter_node = graph.call_function(
                _c10d_functional.reduce_scatter_tensor,
                args=(synchronized_grad, "sum", group_degree, group_name),
            )
            reduce_scatter_node.meta.update(
                comm_type="fsdp_reduce_scatter",
                comm_group=group_name,
                fsdp_degree=group_degree,
                param_name=param_fqn,
            )
            synchronized_grad = graph.call_function(
                _c10d_functional.wait_tensor,
                args=(reduce_scatter_node,),
            )
            synchronized_grad.meta["wait_for"] = reduce_scatter_node.name

        if replicate_group_info is not None:
            replicate_group_name, replicate_degree = replicate_group_info
            all_reduce_node = graph.call_function(
                _c10d_functional.all_reduce,
                args=(synchronized_grad, "sum", replicate_group_name),
            )
            all_reduce_node.meta.update(
                comm_type="fsdp_all_reduce",
                comm_group=replicate_group_name,
                fsdp_degree=replicate_degree,
                param_name=param_fqn,
            )
            synchronized_grad = graph.call_function(
                _c10d_functional.wait_tensor,
                args=(all_reduce_node,),
            )
            synchronized_grad.meta["wait_for"] = all_reduce_node.name
        return synchronized_grad

    def _insert_reduce_scatter_for_grads(
        self,
        graph_module: fx.GraphModule,
        sharded_param_groups: Mapping[int, Tuple[str, int]],
        state_fqns: List[str],
        num_state_inputs: int,
        state_is_param: Optional[List[bool]] = None,
        model: Optional[nn.Module] = None,
    ) -> fx.GraphModule:
        """
        Insert replica AllReduce and shard ReduceScatter on parameter gradients.

        The joint graph returns ``[loss, grad0, grad1, ...]`` from the fwd+bwd
        function; gradient ``i`` (output index ``i+1``) corresponds to the
        ``i``-th trainable parameter. Gradients of FSDP-sharded parameters are
        reduce-scattered (Replicate -> Shard); gradients of parameters outside
        FSDP modules stay full. Only a subset of parameters is typically
        wrapped, so this is a per-parameter decision rather than
        scatter-everything. HSDP dense parameters reduce-scatter across the
        shard group first, then sum the local shards across the replica group.

        The tracer emits gradients in ``state_fqns`` order, skipping buffers
        and frozen (``requires_grad=False``) parameters.
        ``_build_trainable_state_indices`` mirrors that filter so gradient
        ``i`` maps to the correct ``state_idx`` even when the model has buffers
        or frozen params; the previous ``param_idx = i - 1`` only held for the
        all-trainable, no-buffer case and silently misaligned reduce_scatter
        otherwise.
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
        num_grads = len(new_returned) - 1  # index 0 is the loss
        if num_grads != len(trainable_state_indices):
            raise ValueError(
                f"Gradient count ({num_grads}) does not match trainable "
                f"parameter count ({len(trainable_state_indices)}). The "
                f"traced graph and the live model disagree on which "
                f"parameters are trainable; refusing to insert "
                f"reduce_scatter to avoid silent gradient/state misalignment."
            )

        # Index 0 is the loss; gradients start at index 1. Gradient i+1
        # corresponds to trainable parameter i, whose state_idx is
        # trainable_state_indices[i].
        for i in range(1, len(new_returned)):
            grad_node = new_returned[i]
            if not isinstance(grad_node, fx.Node):
                continue

            state_idx = trainable_state_indices[i - 1]
            group_info = sharded_param_groups.get(state_idx)
            param_fqn = state_fqns[state_idx]
            replicate_group_info = self._replicate_group_for_param(param_fqn)
            if group_info is None:
                _, shard_degree, _ = self._group_for_param(param_fqn)
                if shard_degree > 1:
                    replicate_group_info = None
            if group_info is None and replicate_group_info is None:
                continue
            if (
                self._pass_plan is not None
                and not self._param_belongs_to_fsdp_module(param_fqn)
            ):
                continue

            with graph.inserting_before(output_node):
                synchronized_grad = self._insert_gradient_collectives(
                    graph,
                    grad_node,
                    param_fqn,
                    group_info,
                    replicate_group_info,
                )

            new_returned[i] = synchronized_grad

        output_node.args = (type(returned)(new_returned),) + tuple(output_node.args[1:])

        return graph_module

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
    ) -> fx.GraphModule:
        """Placeholder for a future reshard pass (release gathered params
        after forward to cut peak memory). Currently a no-op."""
        return graph_module


__all__ = ["FSDPPass"]
