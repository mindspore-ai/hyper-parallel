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
Graph Compiler - compile a model into a parallel graph and run fwd+bwd.

Compiler surface of the graph-mode stack:

- ``compile`` traces the model into a joint forward+backward FX graph and
  runs the partitioning passes (FSDP / PP / overlap) on it.
- ``forward_backward`` executes the compiled graph against the live model
  state and deposits gradients into ``param.grad`` (accumulating, so several
  micro-batch calls may run before the caller's optimizer step).

Training-policy concerns (optimizer, dataloader loop, logging, batch device
placement) are deliberately out of scope: ``GraphTrainer`` composes this
class and owns them.
"""

__all__ = ["GraphCompiler"]

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _register_process_group

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

from .pass_config import PassConfig, normalize_dp_mode
from .graph_parallel_plan import GraphParallelPlan
from .passes.pipeline import PassPipeline
from .size_specialization import SizeSpecializationDispatcher, validate_compile_sizes
from .tracer.dynamic_shapes import DynamicArgDims, normalize_dynamic_arg_dims
from .tracer.graph_tracer import run_traced_graph, trace_model_graph

_LOG = logging.getLogger(__name__)


class GraphCompiler:
    """
    Graph-mode Compiler

    Users provide model code and parallel configuration.
    The compiler handles graph capture and all parallel logic
    (FSDP / PP / overlap passes), then runs forward+backward per step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_fn: Callable,
        pass_config: PassConfig,
        parallel_plan: Optional[GraphParallelPlan] = None,
        device: Optional[torch.device] = None,
        mesh_context: Optional[Any] = None,
        dynamic: bool = False,
        dynamic_arg_dims: Optional[DynamicArgDims] = None,
        compile_sizes: Optional[List[int]] = None,
        compile_size_input: Optional[str] = None,
        compile_size_dim: int = -1,
        max_specializations: int = 8,
    ) -> None:
        """
        Args:
            model: Model to compile
            train_fn: Training function signature: (model, input, label) -> loss
            pass_config: Parallel configuration
            parallel_plan: GraphParallelPlan declaring which modules to shard
                (optional; enables declarative sharding)
            device: Device to place the model and run the graph on. Defaults
                to the NPU device when available, otherwise CPU.
            mesh_context: Optional automodel ``MeshContext`` carrying a
                pre-built TP/FSDP mesh. When provided, the TP group is reused
                as-is (boundary forwards already hold the group object) and
                only the FSDP shard sub-mesh is registered under ``"fsdp"``.
                Use this to feed an automodel TP-sharded model into the
                graph-mode FSDP pass.
            dynamic: Symbolize user input dimensions (default False).
            dynamic_arg_dims: Dotted input paths to dynamic dimensions; overrides automatic selection.
            compile_sizes: Sizes for lazy FX specialization (disabled by default).
            compile_size_input: Dotted tensor path for dispatch; defaults to the first symbolic input axis.
            compile_size_dim: Dispatch axis for an explicit path (default -1).
            max_specializations: Maximum cached full input signatures (default 8).
        """
        self.dynamic = dynamic
        if not isinstance(self.dynamic, bool):
            raise ValueError("dynamic must be a bool")
        self.dynamic_arg_dims = normalize_dynamic_arg_dims(dynamic_arg_dims)
        self.compile_size_input = compile_size_input
        self.compile_size_dim = compile_size_dim
        self.max_specializations = max_specializations
        self.compile_sizes = validate_compile_sizes(compile_sizes, self.max_specializations)
        if self.compile_sizes and not (self.dynamic or self.dynamic_arg_dims is not None):
            raise ValueError("compile_sizes requires dynamic=True or dynamic_arg_dims")
        self.model = model
        self.train_fn = train_fn
        self.pass_config = pass_config
        self.parallel_plan = parallel_plan
        self._mesh_context = mesh_context
        self.device = device or (
            torch.device("npu")
            if (hasattr(torch, "npu") and torch.npu.is_available())
            else torch.device("cpu")
        )

        pass_config.validate()

        self._joint_graph = None
        self._size_dispatcher = None

    @property
    def is_compiled(self) -> bool:
        """Whether a joint graph has already been compiled."""
        return self._joint_graph is not None

    @property
    def specialization_stats(self) -> dict[str, Any]:
        """Return lazy size-specialization counters, or an empty mapping when disabled."""
        return self._size_dispatcher.stats if self._size_dispatcher is not None else {}

    def compile(self, **inputs: Any) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled
        at first forward_backward

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments and used to trace the joint graph
        """
        if (self.dynamic or self.dynamic_arg_dims is not None) and self.pass_config.pp_enabled:
            raise ValueError(
                "Dynamic shapes with pipeline parallel are not supported yet; disable PP or dynamic shapes"
            )
        if self.pass_config.fsdp_enabled and dist.is_initialized():
            # Only build the FSDP mesh when distributed is actually up.
            # ``FSDPPass`` early-returns when ``world_size == 1``, so a
            # single-process run (no dist, or a single rank) compiles and
            # runs as plain graph mode without sharding.
            self._init_device_mesh(self._mesh_context)

        trace_kwargs = {}
        if self.dynamic or self.dynamic_arg_dims is not None:
            trace_kwargs = {"dynamic": self.dynamic, "dynamic_arg_dims": self.dynamic_arg_dims}
        joint_graph = trace_model_graph(self.model, self.train_fn, inputs, **trace_kwargs)
        # Validate selection before a parallel pass mutates the live model.
        size_dispatcher = None
        if self.compile_sizes:
            size_dispatcher = SizeSpecializationDispatcher(
                joint_graph, inputs, self.compile_sizes, self.compile_size_input,
                self.compile_size_dim, self.max_specializations,
            )
        pipeline = PassPipeline.from_config(self.pass_config, self.parallel_plan)

        pass_kwargs = self._build_pass_kwargs()

        # Passes mutate ``graph_module`` in place and return it, so the
        # transformed graph lives on ``joint_graph`` for ``forward_backward``.
        pipeline.run(joint_graph.graph_module, **pass_kwargs)

        if joint_graph.input_guards is not None:
            joint_graph.input_guards.refresh()
        self._joint_graph = joint_graph
        self._size_dispatcher = size_dispatcher

    def forward_backward(self, **inputs: Any) -> Any:
        """
        Execute one compiled forward+backward step.

        Compiles lazily on the first call. Gradients are ACCUMULATED into
        ``param.grad`` (not overwritten), so several micro-batch calls may
        run before the caller's optimizer step / zero_grad.

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments (must live on the compiler's device)

        Returns:
            loss: Loss value
        """
        if self._joint_graph is None:
            self.compile(**inputs)

        loss, grads = self._run_graph(**inputs)

        self._accumulate_grads(grads)

        return loss

    def to(self, device: torch.device) -> "GraphCompiler":
        """Move the model and subsequent graph execution to the requested device.

        Args:
            device: Target device for model state and graph execution.

        Returns:
            This compiler, for chained configuration.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def _init_device_mesh(self, mesh_context: Optional[Any] = None):
        """Initialize the DP process group(s) for the configured ``dp_mode``.

        Three modes (see ``PassConfig.dp_mode``), matching simplefsdp's
        ``data_parallel`` and the eager ``fully_shard`` conventions:

        * ``"fsdp"`` (default): a 1-D ``("fsdp_shard",)`` mesh; the group is
          registered under the graph-mode name ``"fsdp"``.
        * ``"ddp"``: a 1-D ``("fsdp_replicate",)`` mesh; registered under
          ``"dp_replicate"`` (no parameter sharding, grads all-reduced).
        * ``"hsdp"``: a 2-D ``("fsdp_replicate", "fsdp_shard")`` mesh
          (replicate, shard); both groups registered.

        An external automodel ``MeshContext`` is reused when provided — its
        ``fsdp_replicate`` / ``fsdp_shard`` sub-meshes are registered under the
        graph-mode names, mirroring the eager ``FSDP2Manager``. The fallback
        builds the mesh with the repo ``init_device_mesh`` (the same builder the
        eager stack uses) and only supports **pure DP** (``tp_size == 1``, PP
        off); TP/PP topologies must supply a ``MeshContext``.
        """
        dp_mode = normalize_dp_mode(self.pass_config.dp_mode)
        if mesh_context is not None:
            self._init_mesh_from_context(mesh_context, dp_mode)
            return

        self._require_pure_dp_fallback()
        device_type = (
            "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
        )
        world_size = dist.get_world_size()
        rank_list = tuple(range(world_size))

        if dp_mode == "ddp":
            mesh = init_device_mesh(
                device_type,
                (world_size,),
                mesh_dim_names=("fsdp_replicate",),
                rank_list=rank_list,
                init_backend=True,
            )
            self._register_mesh_group("dp_replicate", mesh["fsdp_replicate"])
            self.pass_config.dp_replicate_degree = mesh["fsdp_replicate"].size()
            return

        if dp_mode == "hsdp":
            shard, replicate = self._resolve_hsdp_degrees(world_size)
            mesh = init_device_mesh(
                device_type,
                (replicate, shard),
                mesh_dim_names=("fsdp_replicate", "fsdp_shard"),
                rank_list=rank_list,
                init_backend=True,
            )
            self._register_mesh_group("dp_replicate", mesh["fsdp_replicate"])
            self._register_mesh_group("fsdp", mesh["fsdp_shard"])
            self.pass_config.dp_replicate_degree = mesh["fsdp_replicate"].size()
            self.pass_config.fsdp_degree = mesh["fsdp_shard"].size()
            return

        # dp_mode == "fsdp" (default): 1-D shard mesh over the whole world.
        mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=("fsdp_shard",),
            rank_list=rank_list,
            init_backend=True,
        )
        self._register_mesh_group("fsdp", mesh["fsdp_shard"])
        # Back-fill, mirroring the external-mesh branch: FSDPPass resolves the
        # group size from ``fsdp_degree`` (falling back to world_size when
        # ``None``), so setting it here keeps the two paths consistent.
        self.pass_config.fsdp_degree = world_size

    def _require_pure_dp_fallback(self) -> None:
        """Reject TP/PP topologies for the self-built DP mesh.

        The fallback mesh spans the whole world; with a TP or PP axis it would
        shard/reduce across the wrong ranks. Those topologies must supply an
        automodel ``MeshContext`` (which carries the proper DP sub-mesh).

        Raises:
            ValueError: When ``tp_size != 1`` or ``pp_enabled``.
        """
        cfg = self.pass_config
        if cfg.tp_size != 1 or cfg.pp_enabled:
            raise ValueError(
                "Graph-mode DP without a MeshContext only supports pure data "
                f"parallelism (tp_size=1, PP disabled); got tp_size={cfg.tp_size}, "
                f"pp_enabled={cfg.pp_enabled}. Pass a MeshContext carrying the DP "
                "sub-mesh for TP/PP hybrids."
            )

    def _init_mesh_from_context(self, mesh_context: Any, dp_mode: str) -> None:
        """Resolve and register the DP group(s) from an automodel ``MeshContext``.

        Reuses automodel's already-built mesh (boundary forwards hold the TP
        group object directly) and registers the ``fsdp_replicate`` /
        ``fsdp_shard`` sub-meshes under the graph-mode names ``"dp_replicate"``
        / ``"fsdp"``. The effective mode is first reconciled with the mesh
        topology (see ``_reconcile_mode_with_context``), mirroring the eager
        ``FSDP2Manager`` which picks HSDP from ``dp_replicate_size > 1``.
        """
        fsdp_mesh = (
            getattr(mesh_context, "fsdp_non_moe_mesh", None) or mesh_context.device_mesh
        )
        names = tuple(getattr(fsdp_mesh, "mesh_dim_names", ()) or ())
        dp_mode = self._reconcile_mode_with_context(mesh_context, dp_mode)
        self.pass_config.dp_mode = dp_mode

        if dp_mode == "ddp":
            dim = self._first_dim(names, ("fsdp_replicate", "dp_replicate", "dp"))
            if dim is None:
                raise ValueError(
                    "dp_mode='ddp' needs a replicate axis among "
                    "('fsdp_replicate', 'dp_replicate', 'dp') in the mesh "
                    f"dims {names}"
                )
            sub = fsdp_mesh[dim]
            self._register_mesh_group("dp_replicate", sub)
            self.pass_config.dp_replicate_degree = sub.size()
            return

        if dp_mode == "hsdp":
            rep_dim = self._first_dim(names, ("fsdp_replicate", "dp_replicate"))
            shard_dim = self._first_dim(names, ("fsdp_shard", "fsdp", "dp"))
            if rep_dim is None or shard_dim is None:
                raise ValueError(
                    "dp_mode='hsdp' needs both a replicate axis "
                    "('fsdp_replicate'/'dp_replicate') and a shard axis "
                    f"('fsdp_shard'/'fsdp'/'dp') in the mesh dims {names}"
                )
            rep_sub = fsdp_mesh[rep_dim]
            shard_sub = fsdp_mesh[shard_dim]
            self._register_mesh_group("dp_replicate", rep_sub)
            self._register_mesh_group("fsdp", shard_sub)
            self.pass_config.dp_replicate_degree = rep_sub.size()
            self.pass_config.fsdp_degree = shard_sub.size()
            return

        # dp_mode == "fsdp": automodel's fsdp_non_moe_mesh is
        # ("fsdp_replicate","fsdp_shard","tp"); device_mesh (cp=1) is
        # ("dp","cp","tp") and "dp" is the FSDP axis.
        dim = "fsdp_shard" if "fsdp_shard" in names else "dp"
        sub = fsdp_mesh[dim]
        self._register_mesh_group("fsdp", sub)
        self.pass_config.fsdp_degree = sub.size()

    def _reconcile_mode_with_context(self, mesh_context: Any, dp_mode: str) -> str:
        """Align ``dp_mode`` with the ``MeshContext`` topology (eager parity).

        The eager ``fully_shard`` path chooses FSDP vs HSDP from the mesh
        (``dp_replicate_size > 1`` means HSDP), not from a config flag. Mirror
        that so a mesh carrying a replicate axis is never silently treated as
        plain FSDP, and a 1-wide replicate axis is treated as FSDP.

        Raises:
            ValueError: When ``dp_mode='ddp'`` but the mesh shards
                (``dp_shard_size > 1``).
        """
        rep = getattr(mesh_context, "dp_replicate_size", None)
        shard = getattr(mesh_context, "dp_shard_size", None)
        if not isinstance(rep, int):
            # Not a real MeshContext (e.g. a test stub): leave the mode as-is.
            return dp_mode
        if dp_mode == "fsdp" and rep > 1:
            _LOG.warning(
                "dp_mode='fsdp' but MeshContext.dp_replicate_size=%d > 1; "
                "using 'hsdp' to match the mesh",
                rep,
            )
            return "hsdp"
        if dp_mode == "hsdp" and rep <= 1:
            _LOG.warning(
                "dp_mode='hsdp' but MeshContext.dp_replicate_size=%d <= 1; "
                "using 'fsdp'",
                rep,
            )
            return "fsdp"
        if dp_mode == "ddp" and isinstance(shard, int) and shard > 1:
            raise ValueError(
                "dp_mode='ddp' requires a non-sharding mesh "
                f"(dp_shard_size=1), got dp_shard_size={shard}; use 'hsdp'"
            )
        return dp_mode

    @staticmethod
    def _first_dim(names: tuple, candidates: tuple) -> Optional[str]:
        """Return the first candidate axis present in ``names`` (or None)."""
        for cand in candidates:
            if cand in names:
                return cand
        return None

    @staticmethod
    def _register_mesh_group(name: str, sub_mesh: Any) -> None:
        """Register ``sub_mesh``'s process group under ``name``."""
        _register_process_group(name, sub_mesh.get_group())

    def _resolve_hsdp_degrees(self, world_size: int) -> Tuple[int, int]:
        """Resolve ``(shard, replicate)`` for a fallback 2-D HSDP mesh.

        Either degree may be left ``None`` and is derived from ``world_size``
        and the other; both must be consistent with ``world_size``.

        Raises:
            ValueError: When both are ``None``, a degree does not divide
                ``world_size``, or the product does not equal ``world_size``.
        """
        shard = self.pass_config.fsdp_degree
        replicate = self.pass_config.dp_replicate_degree
        if shard is None and replicate is None:
            raise ValueError(
                "dp_mode='hsdp' requires fsdp_degree and/or "
                "dp_replicate_degree to build the 2-D (dp_replicate, fsdp) mesh"
            )
        if shard is None:
            if world_size % replicate != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by "
                    f"dp_replicate_degree ({replicate})"
                )
            shard = world_size // replicate
        if replicate is None:
            if world_size % shard != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by fsdp_degree ({shard})"
                )
            replicate = world_size // shard
        if shard * replicate != world_size:
            raise ValueError(
                f"hsdp degrees fsdp_degree={shard} x "
                f"dp_replicate_degree={replicate} != world_size={world_size}"
            )
        return shard, replicate

    def _build_pass_kwargs(self) -> dict:
        """
        Build kwargs to pass to passes.
        """
        kwargs = {}

        # Live model: partitioning passes (FSDPPass) physically shard
        # parameters in place, keeping the compiler FSDP-agnostic.
        kwargs["model"] = self.model

        if self.pass_config.fsdp_enabled:
            kwargs["fsdp_group_name"] = "fsdp"
            # DDP / HSDP also reduce gradients on the replicate axis.
            if normalize_dp_mode(self.pass_config.dp_mode) in ("ddp", "hsdp"):
                kwargs["dp_replicate_group_name"] = "dp_replicate"

        return kwargs

    def _run_graph(self, **inputs):
        """Execute compiled graph"""
        if self._joint_graph is None:
            raise RuntimeError(
                "Graph not compiled. Call compiler.compile() or "
                "compiler.forward_backward() first."
            )

        # The joint graph's parameters/buffers are static inputs: feed the
        # live (FSDP-sharded) model state in FQN order each step.
        return run_traced_graph(
            self._joint_graph,
            self.model,
            inputs,
            **({"graph_dispatcher": self._size_dispatcher} if self._size_dispatcher is not None else {}),
        )

    def _accumulate_grads(self, grads: List[torch.Tensor]) -> None:
        """Accumulate graph-computed gradients into the live model's parameters.

        The graph emits gradients in ``state_fqns`` order (trainable
        parameters only, shared parameters included once per FQN), which
        diverges from ``model.parameters()`` (deduplicated) when the model
        ties weights. Mapping by FQN keeps every gradient on the right
        parameter; the count check refuses to assign on mismatch instead of
        letting ``zip`` silently truncate.

        Accumulation (not overwrite) keeps ``forward_backward`` composable:
        several micro-batch steps may run before the caller's optimizer step
        (which ends with ``zero_grad``), so per-step grads must sum into
        ``param.grad``.
        """
        # ``state_is_param`` is attached to the traced GraphModule by
        # ``trace_model_graph``, not to the JointGraph dataclass itself.
        state_is_param = getattr(self._joint_graph.graph_module, "state_is_param", None)
        fqn_to_param = dict(self.model.named_parameters(remove_duplicate=False))
        trainable = self._trainable_params_in_state_order(
            self._joint_graph.state_fqns, state_is_param, fqn_to_param
        )
        if len(trainable) != len(grads):
            raise ValueError(
                f"Gradient count ({len(grads)}) does not match trainable "
                f"parameter count ({len(trainable)}). The traced graph and "
                f"the live model disagree on which parameters are trainable; "
                f"refusing to assign gradients to avoid silent misalignment."
            )

        for param, grad in zip(trainable, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    @staticmethod
    def _trainable_params_in_state_order(
        state_fqns: List[str],
        state_is_param: Optional[List[bool]],
        fqn_to_param: Dict[str, torch.nn.Parameter],
    ) -> List[torch.nn.Parameter]:
        """Return the live trainable parameters in the graph's state order.

        Mirrors the tracer's gradient emission order (``state_fqns`` order,
        parameters only, trainable only, shared parameters kept per FQN), so
        gradient ``i`` belongs to the returned parameter ``i``. Buffers share
        ``state_fqns`` but are absent from the parameter lookup; they are
        skipped explicitly so a missing ``state_is_param`` flag (old traces)
        degrades to parameter-only instead of raising KeyError.
        """
        trainable: List[torch.nn.Parameter] = []
        for idx, fqn in enumerate(state_fqns):
            if state_is_param is not None and not state_is_param[idx]:
                continue
            param = fqn_to_param.get(fqn)
            if param is not None and param.requires_grad:
                trainable.append(param)
        return trainable
