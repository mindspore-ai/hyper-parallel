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
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _register_process_group

from .pass_config import PassConfig, build_pass_config_from_trainer_config
from .graph_parallel_plan import GraphParallelPlan
from .passes.pipeline import PassPipeline
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
        pass_config: Optional[PassConfig] = None,
        parallel_plan: Optional[GraphParallelPlan] = None,
        trainer_config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        mesh_context: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model: Model to compile
            train_fn: Training function signature: (model, input, label) -> loss
            pass_config: Parallel configuration. When omitted, the compiler
                projects ``trainer_config`` onto a graph-mode ``PassConfig``;
                if both are omitted, it uses ``PassConfig()``.
            parallel_plan: GraphParallelPlan declaring which modules to shard
                (optional; enables declarative sharding)
            trainer_config: Optional trainer topology used only to infer a
                default ``PassConfig`` when ``pass_config`` is omitted.
            device: Device to place the model and run the graph on. Defaults
                to the NPU device when available, otherwise CPU.
            mesh_context: Optional automodel ``MeshContext`` carrying a
                pre-built TP/FSDP mesh. When provided, the TP group is reused
                as-is (boundary forwards already hold the group object) and
                only the FSDP shard sub-mesh is registered under ``"fsdp"``.
                Use this to feed an automodel TP-sharded model into the
                graph-mode FSDP pass.
        """
        self.model = model
        self.train_fn = train_fn
        self.parallel_plan = parallel_plan
        self.pass_config = self._resolve_pass_config(
            pass_config=pass_config,
            trainer_config=trainer_config,
        )
        self._mesh_context = mesh_context
        self.device = device or (
            torch.device("npu")
            if (hasattr(torch, "npu") and torch.npu.is_available())
            else torch.device("cpu")
        )

        self.pass_config.validate()

        self._joint_graph = None
        self._pytree_pre_hook: Optional[Callable[[], None]] = None

    @staticmethod
    def _resolve_pass_config(
        *,
        pass_config: Optional[PassConfig],
        trainer_config: Optional[Any],
    ) -> PassConfig:
        """Resolve pass configuration from explicit or trainer-level input."""
        if pass_config is not None:
            return pass_config
        if trainer_config is not None:
            return build_pass_config_from_trainer_config(trainer_config)
        return PassConfig()

    @staticmethod
    def _resolve_pass_config(
        *,
        pass_config: Optional[PassConfig],
        trainer_config: Optional[Any],
    ) -> PassConfig:
        """Resolve pass configuration from explicit or trainer-level input."""
        if pass_config is not None:
            return pass_config
        if trainer_config is not None:
            return build_pass_config_from_trainer_config(trainer_config)
        return PassConfig()

    @property
    def is_compiled(self) -> bool:
        """Whether a joint graph has already been compiled."""
        return self._joint_graph is not None

    def compile(self, **inputs: Any) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled
        at first forward_backward

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments and used to trace the joint graph
        """
        if self._pytree_pre_hook is not None:
            self._pytree_pre_hook()

        if self.pass_config.fsdp_enabled and dist.is_initialized():
            # Only build the FSDP mesh when distributed is actually up.
            # ``FSDPPass`` early-returns when ``world_size == 1``, so a
            # single-process run (no dist, or a single rank) compiles and
            # runs as plain graph mode without sharding.
            self._init_device_mesh(self._mesh_context)

        joint_graph = trace_model_graph(self.model, self.train_fn, inputs)

        pipeline = PassPipeline.from_config(self.pass_config, self.parallel_plan)

        pass_kwargs = self._build_pass_kwargs()

        # Passes mutate ``graph_module`` in place and return it, so the
        # transformed graph lives on ``joint_graph`` for ``forward_backward``.
        pipeline.run(joint_graph.graph_module, **pass_kwargs)

        self._joint_graph = joint_graph

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
            tuple: ``(loss, loss_dict)`` — the backward loss and the named
            loss values emitted by ``train_fn`` as extra graph outputs after
            the gradients (empty when ``train_fn`` returned a bare loss
            tensor).
        """
        if self._joint_graph is None:
            self.compile(**inputs)

        loss, grads, loss_dict = self._run_graph(**inputs)

        self._accumulate_grads(grads)

        return loss, loss_dict

    def set_pytree_pre_hook(self, hook: Callable[[], None]) -> "GraphCompiler":
        """Register a no-arg hook run immediately before first compilation."""
        self._pytree_pre_hook = hook
        return self

    def to(self, device: torch.device) -> "GraphCompiler":
        """Move the model to ``device`` and remember it for graph execution."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def _init_device_mesh(self, mesh_context: Optional[Any] = None):
        """Initialize the FSDP process group.

        Two modes:

        * **External mesh** (``mesh_context`` from automodel): the TP group is
          already created by automodel (the boundary forward holds the group
          object directly), so we only resolve the FSDP shard sub-mesh and
          register it under the name ``"fsdp"`` so ``FSDPPass``'s functional
          collectives resolve it by name. ``fsdp_degree`` is back-filled on
          ``pass_config`` from the sub-mesh size — essential for a TP+FSDP
          hybrid, where the FSDP group is a proper sub-group of the world and
          must NOT be confused with ``world_size``.
        * **Fallback** (no mesh): build a 1-D ``("fsdp",)`` mesh over the
          whole world (the original FSDP-only path).
        """
        if mesh_context is not None:
            fsdp_mesh = (
                getattr(mesh_context, "fsdp_non_moe_mesh", None)
                or mesh_context.device_mesh
            )
            names = tuple(getattr(fsdp_mesh, "mesh_dim_names", ()) or ())
            # automodel's fsdp_non_moe_mesh is ("fsdp_replicate","fsdp_shard","tp");
            # device_mesh (cp=1) is ("dp","cp","tp") and "dp" is the FSDP axis.
            dim = "fsdp_shard" if "fsdp_shard" in names else "dp"
            sub = fsdp_mesh[dim]
            pg = sub.get_group()
            _register_process_group("fsdp", pg)
            self.pass_config.fsdp_degree = sub.size()
            return

        device_type = (
            "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
        )
        world_size = dist.get_world_size()

        mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=("fsdp",),
        )

        pg = mesh["fsdp"].get_group()
        _register_process_group("fsdp", pg)
        # Back-fill, mirroring the external-mesh branch: FSDPPass resolves the
        # group size from ``fsdp_degree`` (falling back to world_size when
        # ``None``), so setting it here keeps the two paths consistent.
        self.pass_config.fsdp_degree = world_size

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
