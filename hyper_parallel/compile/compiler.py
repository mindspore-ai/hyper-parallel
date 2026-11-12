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

from __future__ import annotations

__all__ = ["GraphCompiler"]

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.distributed_c10d import _register_process_group

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.external_state import get_model_external_state

from .graph_parallel_plan import GraphParallelPlan
from .mixed_precision import GraphMixedPrecision
from .pass_config import (
    GradientReduction,
    PassConfig,
    build_pass_config_from_trainer_config,
    normalize_dp_mode,
)
from .passes.pipeline import PassPipeline
from .passes.activation_checkpoint import prepare_activation_checkpoint
from .passes.parallel.fsdp_pass import FSDPPass
from .tracer.dynamic_shapes import normalize_dynamic_arg_dims
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
        dynamic: Optional[bool] = None,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
        source_shard_info: Optional[dict] = None,
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
            mesh_context: Optional automodel ``MeshContext`` carrying the
                pre-built parallel meshes. Model forwards retain their TP/EP
                groups; the compiler registers the dense shard sub-mesh under
                ``"fsdp"`` and, for basic EP, the expert EDP shard sub-mesh
                under ``"fsdp_expert"`` for the graph-mode FSDP pass.
            dynamic: Symbolize user input dimensions; inherits trainer_config.compile.dynamic when omitted.
            dynamic_arg_dims: Optional dotted input paths to dynamic dimensions, e.g. {"x": [0, 1]}.
                Overrides automatic dimension selection and inherits trainer configuration when omitted.
            source_shard_info: Source placements and meshes returned by apply_sharding_plan,
                before FSDP cuts the localized parameters.
        """
        compile_config = getattr(trainer_config, "compile", None)
        self.dynamic = (
            getattr(compile_config, "dynamic", False) if dynamic is None else dynamic
        )
        if not isinstance(self.dynamic, bool):
            raise ValueError("dynamic must be a bool")
        if dynamic_arg_dims is None:
            dynamic_arg_dims = getattr(compile_config, "dynamic_arg_dims", None)
        self.dynamic_arg_dims = normalize_dynamic_arg_dims(dynamic_arg_dims)
        self.model = model
        self.train_fn = train_fn
        self.parallel_plan = parallel_plan
        self.pass_config = self._resolve_pass_config(
            pass_config=pass_config,
            trainer_config=trainer_config,
            mesh_context=mesh_context,
        )
        self._mesh_context = mesh_context
        self.source_shard_info = source_shard_info
        self._fsdp_mesh = None
        self._expert_fsdp_mesh = None
        self._fsdp_config = getattr(trainer_config, 'fsdp_config', None)
        self._parameter_meshes = {}
        self._parameter_groups = {}
        self._replicate_param_names = frozenset()
        self._parameter_shapes = {}
        self._parameter_layouts = None
        self._layout_signature = None
        self._storage_ready = False
        self._storage_failed = False
        self._storage_signature = None
        self._sharded_parameter_names = frozenset()
        self._prepared_pass_kwargs = None
        self._mixed_precision = None
        self._fp32_main_params = getattr(getattr(trainer_config, "optimizer", None), "fp32_main_params", False)
        self.device = device or (
            torch.device("npu")
            if (hasattr(torch, "npu") and torch.npu.is_available())
            else torch.device("cpu")
        )

        self.pass_config.validate()

        self._joint_graph = None
        self._expert_param_names: set[str] = set()
        self._pytree_pre_hook: Optional[Callable[[], None]] = None
        # Set when the PP hybrid mesh registered a "pp" process group; makes
        # ``_build_pass_kwargs`` hand the group name to ``PpPass``.
        self._pp_group_name: Optional[str] = None

    @staticmethod
    def _resolve_pass_config(
        *,
        pass_config: Optional[PassConfig],
        trainer_config: Optional[Any],
        mesh_context: Optional[Any] = None,
    ) -> PassConfig:
        """Resolve pass configuration from explicit or trainer-level input."""
        if pass_config is not None:
            return pass_config
        if trainer_config is not None:
            return build_pass_config_from_trainer_config(trainer_config, mesh_context=mesh_context)
        return PassConfig()

    @property
    def is_compiled(self) -> bool:
        """Whether a joint graph has already been compiled."""
        return self._joint_graph is not None

    @property
    def execution_backend(self) -> Optional[str]:
        """Actual terminal backend, or None before the graph is ready."""
        if self._joint_graph is None:
            return None
        return getattr(self._joint_graph.graph_module, "meta", {}).get("inductor_compiled", "fx")

    @property
    def parameter_layouts(self) -> Optional[dict]:
        """Final parameter-identity layouts, available after storage preparation.

        Cache metadata only: gradients may be replaced on every optimizer step.
        Replacing a parameter invalidates the mapping, as does compiling again.
        """
        if not self._storage_ready:
            return None
        parameters = dict(self.model.named_parameters())
        signature = tuple((id(param), tuple(param.shape), param.dtype, param.device) for param in parameters.values())
        if signature != self._layout_signature:
            self._parameter_layouts = self._build_parameter_layouts(parameters)
            self._layout_signature = signature
            if self._joint_graph is not None:
                self._joint_graph.graph_module.parameter_layouts = self._parameter_layouts
        return self._parameter_layouts

    @property
    def parameter_shapes(self) -> dict:
        """Source-local logical shapes before FSDP padding and slicing."""
        return dict(self._parameter_shapes)

    @property
    def sharded_parameter_names(self) -> frozenset:
        """Canonical names of parameters physically partitioned by FSDP."""
        return self._sharded_parameter_names

    def prepare_model(self) -> None:
        """Prepare local parameter storage once, before optimizer construction.

        Values may subsequently be loaded or updated in place. Changing a
        parameter identity, shape, dtype or device requires a new model and
        compiler. A failed preparation is terminal because storage may have
        already been partitioned.
        """
        external = get_model_external_state(self.model)
        if external is not None and external.parameters:
            raise ValueError("GraphCompiler does not support externally managed parameters yet")
        if self._storage_failed:
            raise RuntimeError("Graph storage preparation failed; rebuild the model and compiler")
        signature = self._model_storage_signature()
        if self._storage_ready:
            if signature != self._storage_signature:
                raise ValueError("Prepared graph parameter storage changed; rebuild the model and compiler")
            return
        try:
            self._prepare_model_storage()
        except Exception:
            self._storage_failed = True
            raise
        self._storage_signature = self._model_storage_signature()
        self._storage_ready = True

    def _model_storage_signature(self) -> tuple:
        """Describe storage identity without retaining any parameter values."""
        return tuple((name, id(param), tuple(param.shape), param.dtype, param.device, param.requires_grad)
                     for name, param in self.model.named_parameters(remove_duplicate=False))

    def _prepare_model_storage(self) -> None:
        """Reuse FSDP ownership and slicing before creating optimizer state."""
        self._validate_ep_topology()
        tp_size = getattr(self._mesh_context, "tp_size", 1)
        if isinstance(tp_size, int) and tp_size > 1 and not self.source_shard_info:
            raise ValueError("Graph TP requires source_shard_info from the sharding plan")
        if self.pass_config.fsdp_enabled and dist.is_initialized():
            self._init_device_mesh(self._mesh_context)
        self._bind_parameter_meshes()
        precision = getattr(self._fsdp_config, "mix_precision", None)
        if precision is not None and any((precision.param_dtype, precision.reduce_dtype, precision.output_dtype,
                                          precision.custom_params, self._fp32_main_params)):
            manager = FSDP2Manager(self._fsdp_config, self._mesh_context or MeshContext(), self._fp32_main_params)
            policy, modules = manager.mixed_precision_bindings(self.model, self.source_shard_info)
            self._mixed_precision = GraphMixedPrecision(self.model, policy, modules)
        self._parameter_shapes = {
            name: tuple(param.shape) for name, param in self.model.named_parameters(remove_duplicate=False)
        }
        self._prepared_pass_kwargs = self._build_pass_kwargs()
        if self.pass_config.fsdp_enabled:
            fsdp = FSDPPass(parallel_plan=self.parallel_plan)
            self._sharded_parameter_names = fsdp.prepare_model(self.pass_config, **self._prepared_pass_kwargs)
        self._layout_signature = None

    def compile(self, **inputs: Any) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled
        at first forward_backward

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments and used to trace the joint graph
        """
        external = get_model_external_state(self.model)
        if external is not None and external.parameters:
            raise ValueError("GraphCompiler does not support externally managed parameters yet")

        if (
            self.dynamic or self.dynamic_arg_dims is not None
        ) and self.pass_config.pp_enabled:
            raise ValueError(
                "Dynamic shapes with pipeline parallel are not supported yet; disable PP or dynamic shapes"
            )
        if self._pytree_pre_hook is not None:
            self._pytree_pre_hook()

        self.prepare_model()

        # Activation checkpointing must resolve its scope and run the adapter's
        # runtime hook BEFORE tracing: the hook configures state the model's
        # forward reads. The pass receives the same scope so the hook runs once.
        ac_scope = None
        if self.pass_config.ac_mode != "off":
            ac_scope = prepare_activation_checkpoint(self.model, self.pass_config)

        trace_kwargs = {"state_shapes": self._parameter_shapes, "mixed_precision": self._mixed_precision}
        if self.dynamic or self.dynamic_arg_dims is not None:
            trace_kwargs.update({
                "dynamic": self.dynamic,
                "dynamic_arg_dims": self.dynamic_arg_dims,
            })
        joint_graph = trace_model_graph(
            self.model, self.train_fn, inputs, **trace_kwargs
        )

        pipeline = PassPipeline.from_config(self.pass_config, self.parallel_plan)

        pass_kwargs = dict(self._prepared_pass_kwargs)
        pass_kwargs["storage_prepared"] = True
        # Terminal backend compilation (InductorPass) sizes its regions from
        # the tracer's fake example inputs; every other pass ignores them.
        pass_kwargs["example_inputs"] = joint_graph.example_inputs
        if ac_scope is not None:
            pass_kwargs["ac_scope"] = ac_scope

        # The terminal backend may return a wrapper around the transformed graph.
        runtime_graph = pipeline.run(joint_graph.graph_module, **pass_kwargs)
        if getattr(runtime_graph, "meta", {}).get("inductor_compiled") == "unavailable":
            raise RuntimeError("The requested joint-graph Inductor backend is unavailable in this torch build")
        for name in (
            "state_fqns", "state_is_param", "num_state_inputs", "user_inputs_spec",
            "loss_dict_keys", "gradient_usage", "fsdp_sharded_param_names", "meta",
        ):
            if not hasattr(runtime_graph, name) and hasattr(joint_graph.graph_module, name):
                setattr(runtime_graph, name, getattr(joint_graph.graph_module, name))
        joint_graph.graph_module = runtime_graph

        if joint_graph.input_guards is not None:
            joint_graph.input_guards.refresh()
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

    def source_layout_for_parameter(self, name: str) -> Optional[tuple]:
        """Resolve pre-FSDP placement, including tied and replicated parameters.

        Args:
            name: Parameter FQN in the live model.

        Returns:
            The source ``(placements, mesh)`` pair, or None for a plain local model.
        """
        if self.source_shard_info is not None:
            if name in self.source_shard_info:
                return self.source_shard_info[name]
            parameters = dict(self.model.named_parameters(remove_duplicate=False))
            for alias, layout in self.source_shard_info.items():
                if parameters.get(alias) is parameters.get(name):
                    return layout
        context = self._mesh_context
        if context is None:
            return None
        if name in self._expert_param_names:
            return ((Shard(0),), context.fsdp_moe_mesh["ep"])
        mesh = context.fsdp_non_moe_mesh
        if mesh is not None and "tp" in mesh.mesh_dim_names:
            return ((Replicate(),), mesh["tp"])
        return None

    def _build_parameter_layouts(self, parameters: dict) -> Optional[dict]:
        """Describe actual local storage using common FSDP layout construction."""
        context = self._mesh_context
        if context is None and self._fsdp_mesh is None:
            return None
        if self._fsdp_mesh is None and context is not None:
            # Plain graph mode still needs metadata, without creating collectives.
            self._init_mesh_from_context(context, self.pass_config.dp_mode, register_groups=False)
        sharded = self._sharded_parameter_names
        layouts = {}
        for name, parameter in parameters.items():
            if not parameter.requires_grad:
                continue
            mesh = self._parameter_meshes.get(name)
            if mesh is None:
                mesh = self._expert_fsdp_mesh if name in self._expert_param_names else self._fsdp_mesh
            if mesh is None:
                raise ValueError("Parameter layouts require an initialized FSDP mesh or parameter binding")
            layouts[parameter] = self._build_parameter_layout(
                parameter, mesh, self.source_layout_for_parameter(name), sharded=name in sharded,
                unsharded_shape=self._parameter_shapes.get(name, tuple(parameter.shape)),
            )
        return layouts

    @staticmethod
    def _build_parameter_layout(
        parameter: torch.nn.Parameter, fsdp_mesh: DeviceMesh, source: Optional[tuple], *,
        sharded: bool, unsharded_shape: Tuple[int, ...],
    ) -> Layout:
        """Describe actual local storage without changing parameters or source metadata.

        Args:
            parameter: Live parameter after physical FSDP slicing.
            fsdp_mesh: Selected data-parallel child mesh.
            source: Source TP/EP placements and mesh before FSDP.
            sharded: Whether the pass physically sliced this parameter.
            unsharded_shape: Local shape before FSDP padding and slicing.
        """
        mesh = fsdp_mesh
        placements = [Replicate() for _ in range(mesh.ndim)]
        shape = tuple(unsharded_shape)
        axis = None
        if sharded:
            axis = next((index for index, name in enumerate(mesh.mesh_dim_names or ())
                         if name in ("fsdp_shard", "edp_shard", "fsdp", "dp", "dp_shard")), None)
            if axis is None or parameter.ndim == 0:
                raise ValueError("A physically sharded parameter requires a shard mesh axis and a non-scalar shape")
            shape = (parameter.shape[0] * mesh.size(axis), *shape[1:])
        if source is not None and source[1] is not None:
            source_placements, source_mesh = source
            source_layout = Layout.from_device_mesh(source_mesh)
            source_layout.set_placements(tuple(source_placements))
            source_layout.placement_to_tensor_map(len(shape))
            shape = source_layout.get_global_shape(shape)
            mesh = DeviceMesh.concatenate([fsdp_mesh, source_mesh])
            placements.extend(source_placements)
        if axis is not None:
            split_factor = math.prod(mesh.mesh_shape[index] for index, value in enumerate(placements)
                                     if index != axis and value.is_shard(0))
            placements[axis] = StridedShard(0, split_factor) if split_factor > 1 else Shard(0)
        layout = Layout.from_device_mesh(mesh)
        layout.set_placements(tuple(placements))
        layout.placement_to_tensor_map(len(shape))
        layout.set_tensor_meta(tuple(shape), make_contiguous_strides_for(shape), parameter.dtype)
        return layout

    def _bind_parameter_meshes(self) -> None:
        """Reuse eager unit discovery and managed identities without eager hooks."""
        self._parameter_meshes = {}
        self._parameter_groups = {}
        self._replicate_param_names = frozenset()
        if not isinstance(self._fsdp_config, FSDP2Config):
            return
        manager = FSDP2Manager(self._fsdp_config, self._mesh_context or MeshContext())
        self._replicate_param_names = manager.replicate_parameter_names(self.model)
        if self._mesh_context is None:
            return
        bindings = manager.parameter_mesh_bindings(self.model, self.source_shard_info)
        registered = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            mesh = bindings.get(parameter)
            if mesh is None:
                continue
            self._parameter_meshes[name] = mesh
            shard = mesh[mesh.mesh_dim_names[-1]]
            ranks = tuple(shard.rank_list)
            if ranks not in registered:
                group = f"fsdp_unit_{len(registered)}"
                self._register_mesh_group(group, shard)
                registered[ranks] = group
            self._parameter_groups[name] = (registered[ranks], shard.size())

    def _init_device_mesh(self, mesh_context: Optional[Any] = None):
        """Initialize the DP process group(s) for the configured ``dp_mode``.

        Four modes (see ``PassConfig.dp_mode``), matching simplefsdp's
        ``data_parallel`` and the eager ``fully_shard`` conventions:

        * ``"fsdp"`` (default): a 1-D ``("fsdp_shard",)`` mesh; the group is
          registered under the graph-mode name ``"fsdp"``.
        * ``"ddp"``: a 1-D ``("fsdp_replicate",)`` mesh; registered under
          ``"dp_replicate"`` (no parameter sharding, grads all-reduced).
        * ``"hsdp"``: a 2-D ``("fsdp_replicate", "fsdp_shard")`` mesh
          (replicate, shard); both groups registered.
        * PP hybrid (``pp_enabled``): a 2-D ``(pp, DP)`` mesh — see
          ``_init_pp_dp_mesh``; the intra-stage DP group and the cross-stage
          ``"pp"`` group are both registered.

        An external automodel ``MeshContext`` is reused when provided — its
        ``fsdp_replicate`` / ``fsdp_shard`` sub-meshes are registered under the
        graph-mode names, mirroring the eager ``FSDP2Manager``. The fallback
        builds the mesh with the repo ``init_device_mesh`` (the same builder the
        eager stack uses) and only supports pure DP (``tp_size == 1``) plus the
        PP hybrid; TP topologies must supply a ``MeshContext``.
        """
        dp_mode = normalize_dp_mode(self.pass_config.dp_mode)
        if self.pass_config.pp_enabled:
            self._init_pp_dp_mesh(dp_mode, mesh_context)
            return

        if mesh_context is not None:
            self._init_mesh_from_context(mesh_context, dp_mode)
            self._init_expert_dp(mesh_context)
            return

        device_type = self._mesh_device_type()
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
            self._fsdp_mesh = mesh
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
            self._fsdp_mesh = mesh
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
        self._fsdp_mesh = mesh
        self._register_mesh_group("fsdp", mesh["fsdp_shard"])
        # Back-fill, mirroring the external-mesh branch: FSDPPass resolves the
        # group size from ``fsdp_degree`` (falling back to world_size when
        # ``None``), so setting it here keeps the two paths consistent.
        self.pass_config.fsdp_degree = world_size

    def _validate_ep_topology(self) -> None:
        """Limit graph EP to the synchronous, non-pipelined basic topology."""
        context = self._mesh_context
        if context is None or getattr(context, "ep_size", 1) <= 1:
            return
        if any(
            getattr(context, axis, 1) != 1 for axis in ("cp_size", "pp_size")
        ):
            raise ValueError("Graph EP currently requires CP=PP=1")
        if self.pass_config.enable_overlap:
            raise ValueError("Graph EP currently requires enable_overlap=False")

    def _init_expert_dp(self, context: Any) -> None:
        """Use EDP for the exact parameters owned by bound local experts."""
        if getattr(context, "ep_size", 1) <= 1:
            return
        mesh = context.fsdp_moe_mesh
        if mesh is None or "edp_shard" not in mesh.mesh_dim_names:
            raise ValueError("Graph EP requires the expert data-parallel shard mesh")
        if self.pass_config.dp_mode != "fsdp":
            raise ValueError(
                "Graph EP currently supports sharded data parallelism without replication"
            )
        if "edp_replicate" in mesh.mesh_dim_names and mesh["edp_replicate"].size() > 1:
            raise ValueError(
                "Graph EP currently does not support expert data-parallel replication"
            )
        parameters = {
            id(parameter)
            for module in self.model.modules()
            if hasattr(module, "local_expert_count")
            for parameter in module.parameters(recurse=False)
        }
        self._expert_param_names = {
            name
            for name, parameter in self.model.named_parameters(remove_duplicate=False)
            if id(parameter) in parameters
        }
        if not self._expert_param_names:
            raise ValueError("Graph EP requires bound local expert parameters")
        self._expert_fsdp_mesh = mesh["edp_shard"]
        self._register_mesh_group("fsdp_expert", self._expert_fsdp_mesh)

    def _init_pp_dp_mesh(
        self, dp_mode: str, mesh_context: Optional[Any] = None
    ) -> None:
        """Initialize the 2-D ``(pp, DP)`` mesh for the PP+FSDP/DDP hybrid.

        Builds the mesh over the identity rank list (``rank = pp_idx *
        dp_size + dp_idx``, PP-outer) and registers the intra-stage DP group
        plus the cross-stage ``"pp"`` group that ``PpPass`` pipelines over.

        Args:
            dp_mode: Normalized DP mode (``"fsdp"`` or ``"ddp"``).
            mesh_context: Must be ``None``; an external automodel
                ``MeshContext`` composed with PP is not supported yet.

        Raises:
            ValueError: On a ``mesh_context``, ``"hsdp"``, missing degrees, or
                degrees that do not tile the world.
        """
        if mesh_context is not None:
            raise ValueError(
                "PP composed with an external automodel MeshContext is not "
                "supported yet; run the PP hybrid with the self-built mesh "
                "(omit mesh_context and set pp_degree plus fsdp_degree / "
                "dp_replicate_degree)"
            )
        if dp_mode == "hsdp":
            raise ValueError(
                "PP + HSDP requires a 3-D (pp, dp_replicate, fsdp_shard) "
                "mesh, which is not supported yet; run PP with dp_mode='fsdp' "
                "or 'ddp'"
            )
        pp_degree, dp_degree, dp_mesh_name = self._resolve_pp_dp_degrees(dp_mode)
        world_size = dist.get_world_size()
        mesh = init_device_mesh(
            self._mesh_device_type(),
            (pp_degree, dp_degree),
            mesh_dim_names=("pp", dp_mesh_name),
            rank_list=tuple(range(world_size)),
            init_backend=True,
        )
        self._register_pp_dp_groups(mesh, dp_mode, dp_mesh_name, pp_degree, dp_degree)
        _LOG.info(
            "PP+DP mesh: pp_degree=%s dp_mode=%s dp_degree=%s world_size=%s",
            pp_degree,
            dp_mode,
            dp_degree,
            world_size,
        )

    def _resolve_pp_dp_degrees(self, dp_mode: str) -> Tuple[int, int, str]:
        """Resolve and validate the ``(pp, DP)`` degrees and DP dim name."""
        world_size = dist.get_world_size()
        pp_degree = self.pass_config.pp_degree
        if dp_mode == "ddp":
            dp_degree = self.pass_config.dp_replicate_degree
            dp_mesh_name = "dp_replicate"
        else:
            dp_degree = self.pass_config.fsdp_degree
            dp_mesh_name = "fsdp_shard"
        if pp_degree is None and dp_degree is None:
            raise ValueError(
                "PP + DP hybrid requires pp_degree together with (or instead "
                "of) fsdp_degree / dp_replicate_degree so the 2-D mesh can be "
                "sized; got both None"
            )
        if pp_degree is None:
            if world_size % dp_degree != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by the DP degree "
                    f"({dp_degree}); cannot derive pp_degree"
                )
            pp_degree = world_size // dp_degree
        if dp_degree is None:
            if world_size % pp_degree != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by pp_degree "
                    f"({pp_degree}); cannot derive the DP degree"
                )
            dp_degree = world_size // pp_degree
        if pp_degree * dp_degree != world_size:
            raise ValueError(
                f"PP+DP degrees must tile the world: pp_degree ({pp_degree}) "
                f"x dp_degree ({dp_degree}) != world_size ({world_size})"
            )
        return pp_degree, dp_degree, dp_mesh_name

    def _register_pp_dp_groups(
        self, mesh: Any, dp_mode: str, dp_mesh_name: str, pp_degree: int, dp_degree: int
    ) -> None:
        """Register both hybrid groups and back-fill the pass-config degrees."""
        self._fsdp_mesh = mesh[dp_mesh_name]
        self._register_mesh_group("pp", mesh["pp"])
        self._register_mesh_group(
            "fsdp" if dp_mesh_name == "fsdp_shard" else "dp_replicate",
            mesh[dp_mesh_name],
        )
        self._pp_group_name = "pp"
        self.pass_config.pp_degree = pp_degree
        if dp_mode == "ddp":
            self.pass_config.dp_replicate_degree = dp_degree
        else:
            self.pass_config.fsdp_degree = dp_degree

    @staticmethod
    def _mesh_device_type() -> str:
        """Return the backend device type for the self-built mesh."""
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        return "cpu"

    def _init_mesh_from_context(self, mesh_context: Any, dp_mode: str, *, register_groups: bool = True) -> None:
        """Resolve and register the DP group(s) from an automodel ``MeshContext``.

        Reuses automodel's already-built mesh (boundary forwards hold the TP
        group object directly) and registers the ``fsdp_replicate`` /
        ``fsdp_shard`` sub-meshes under the graph-mode names ``"dp_replicate"``
        / ``"fsdp"``. The effective mode is first reconciled with the mesh
        topology (see ``_reconcile_mode_with_context``), mirroring the eager
        ``FSDP2Manager`` which picks HSDP from ``dp_replicate_size > 1``.

        Args:
            mesh_context: Existing data/model parallel topology.
            dp_mode: Requested data-parallel mode.
            register_groups: False for metadata-only access in plain graph mode.
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
            self._fsdp_mesh = sub
            if register_groups:
                self._register_mesh_group("dp_replicate", sub)
            self.pass_config.dp_replicate_degree = sub.size()
            return

        if dp_mode == "hsdp":
            rep_dim = self._first_dim(names, ("fsdp_replicate", "dp_replicate"))
            shard_dim = self._first_dim(names, ("fsdp_shard", "fsdp", "dp_shard", "dp"))
            if rep_dim is None or shard_dim is None:
                raise ValueError(
                    "dp_mode='hsdp' needs both a replicate axis "
                    "('fsdp_replicate'/'dp_replicate') and a shard axis "
                    f"('fsdp_shard'/'fsdp'/'dp') in the mesh dims {names}"
                )
            rep_sub = fsdp_mesh[rep_dim]
            shard_sub = fsdp_mesh[shard_dim]
            if shard_dim == "dp_shard" and "cp" in names:
                shard_sub = fsdp_mesh[(shard_dim, "cp")].flatten("fsdp_shard")
                self._fsdp_mesh = DeviceMesh.concatenate([rep_sub, shard_sub])
            else:
                self._fsdp_mesh = fsdp_mesh[(rep_dim, shard_dim)]
            if register_groups:
                self._register_mesh_group("dp_replicate", rep_sub)
                self._register_mesh_group("fsdp", shard_sub)
            self.pass_config.dp_replicate_degree = rep_sub.size()
            self.pass_config.fsdp_degree = shard_sub.size()
            return

        # dp_mode == "fsdp": automodel's fsdp_non_moe_mesh is
        # ("fsdp_replicate","fsdp_shard","tp"); device_mesh (cp=1) is
        # ("dp","cp","tp") and "dp" is the FSDP axis.
        dim = self._first_dim(names, ("fsdp_shard", "dp_shard", "dp"))
        sub = fsdp_mesh[dim]
        if dim == "dp_shard" and "cp" in names:
            sub = fsdp_mesh[(dim, "cp")].flatten("fsdp_shard")
        self._fsdp_mesh = sub
        if register_groups:
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
        kwargs = {
            "storage_prepared": self._storage_ready,
            "logical_parameter_shapes": self._parameter_shapes,
            "replicate_param_names": self._replicate_param_names,
        }

        # Live model: partitioning passes (FSDPPass) physically shard
        # parameters in place, keeping the compiler FSDP-agnostic.
        kwargs["model"] = self.model
        if self._mixed_precision is not None:
            kwargs["parameter_dtypes"] = self._mixed_precision.parameter_dtypes

        if self.pass_config.fsdp_enabled:
            kwargs["fsdp_group_name"] = "fsdp"
            if self._parameter_groups:
                kwargs["parameter_groups"] = self._parameter_groups
            if self._expert_param_names:
                kwargs["expert_param_names"] = self._expert_param_names
                kwargs["expert_fsdp_group_name"] = "fsdp_expert"
            kwargs.update(self._source_reduction_kwargs())
            # DDP / HSDP also reduce gradients on the replicate axis.
            if normalize_dp_mode(self.pass_config.dp_mode) in ("ddp", "hsdp"):
                kwargs["dp_replicate_group_name"] = "dp_replicate"

        # The PP hybrid mesh registered the cross-stage "pp" group; PpPass
        # resolves its stage index from it and runs boundary P2P over it.
        # Pure PP (fsdp_enabled=False) leaves it unset — PpPass then builds
        # its world-sized group itself.
        if self.pass_config.pp_enabled and self._pp_group_name is not None:
            kwargs["pp_group_name"] = self._pp_group_name

        return kwargs

    def _source_reduction_kwargs(self) -> dict:
        """Lower eager source-layout normalization and replicated-axis synchronization."""
        names = (
            dict(self.model.named_parameters(remove_duplicate=False))
            if self.source_shard_info is not None else self._expert_param_names
        )
        if not names:
            return {}
        if self._mesh_context is None:
            raise ValueError("Graph source-layout reduction requires MeshContext")
        context = self._mesh_context
        loss_replicas = 1 if context.loss_parallel else context.tp_size
        reduction = GradientReduction("sum", context.dp_size * context.cp_size * loss_replicas)
        result = {"gradient_reductions": dict.fromkeys(names, reduction)}
        if self.source_shard_info is None:
            return result
        groups, registered = {}, {}
        for name in names:
            layout = self.source_layout_for_parameter(name)
            if layout is None:
                continue
            placements, mesh = layout
            if mesh is None or not placements:
                continue
            axes = tuple(axis for index, axis in enumerate(mesh.mesh_dim_names)
                         if placements[index].is_replicate() and mesh.size(index) > 1)
            if not axes:
                continue
            replica_mesh = mesh[axes].flatten()
            ranks = tuple(replica_mesh.rank_list)
            if ranks not in registered:
                group_name = f"fsdp_source_{len(registered)}"
                self._register_mesh_group(group_name, replica_mesh)
                registered[ranks] = group_name
            groups[name] = registered[ranks]
        result["source_gradient_groups"] = groups
        return result

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

        Gradients follow the tracer's trainable state order, with tied
        parameters deduplicated. Keep absent edges distinct from zero-valued
        gradients and accumulate across calls until the optimizer clears them.
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

        usage = getattr(self._joint_graph.graph_module, "gradient_usage", (True,) * len(grads))
        if len(usage) != len(grads):
            raise ValueError("Gradient usage does not match the traced gradient count")
        main_grad = self._mixed_precision is not None and self._mixed_precision.policy.apply_grad_on_fp32_main_grad
        for param, grad, used in zip(trainable, grads, usage):
            if not used:
                continue
            attribute = "main_grad" if main_grad else "grad"
            accumulated = getattr(param, attribute, None)
            # Eager keeps reduction precision for main gradients and casts each
            # ordinary gradient to storage precision before microbatch addition.
            if not main_grad:
                grad = grad.to(param.dtype)
            if accumulated is None:
                # Match eager's gradient layout so optimizer kernels see the
                # parameter's strides rather than a backward matmul's view.
                dtype = grad.dtype
                accumulated = torch.empty_like(param, dtype=dtype)
                accumulated.copy_(grad)
                setattr(param, attribute, accumulated)
            else:
                accumulated += grad

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
