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
Graph Trainer - Graph-mode Trainer

Users provide model code and parallel configuration.
Framework automatically handles all parallel logic.
"""

from typing import Any, Callable, Iterable, Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _register_process_group

from .parallel_config import PassConfig
from .passes.pipeline import PassPipeline
from .sharding_config import PassPlan
from .tracer.graph_tracer import run_traced_graph, trace_model_graph


class GraphTrainer:
    """
    Graph-mode Trainer

    Users provide model code and parallel configuration.
    Framework automatically handles all parallel logic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_fn: Callable,
        pass_config: PassConfig,
        pass_plan: Optional[PassPlan] = None,
        optimizer_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        mesh_context: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model: Model to train
            train_fn: Training function signature: (model, input, label) -> loss
            pass_config: Parallel configuration
            pass_plan: Sharding plan (optional, if provided use declarative sharding)
            optimizer_config: Optimizer configuration
            device: Device to place the model and run training on. Defaults to
                the NPU device when available, otherwise CPU.
            mesh_context: Optional automodel ``MeshContext`` carrying a
                pre-built TP/FSDP mesh. When provided, the TP group is reused
                as-is (boundary forwards already hold the group object) and
                only the FSDP shard sub-mesh is registered under ``"fsdp"``.
                Use this to feed an automodel TP-sharded model into the
                graph-mode FSDP pass.
        """
        self.model = model
        self.train_fn = train_fn
        self.pass_config = pass_config
        self.pass_plan = pass_plan
        self.optimizer_config = optimizer_config or {}
        self._mesh_context = mesh_context
        self.device = device or (
            torch.device("npu")
            if (hasattr(torch, "npu") and torch.npu.is_available())
            else torch.device("cpu")
        )

        pass_config.validate()

        self._joint_graph = None
        self.optimizer = None
        # Optional hook run right before the first compile, for model-specific
        # pytree / tracer registration (e.g. flex-attention BlockMask).
        self._pytree_pre_hook: Optional[Callable[[], None]] = None

    def compile(self, sample_input: torch.Tensor, sample_label: torch.Tensor) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled at first train_step
        """
        if self._pytree_pre_hook is not None:
            self._pytree_pre_hook()

        if self.pass_config.fsdp_enabled and dist.is_initialized():
            # Only build the FSDP mesh when distributed is actually up.
            # ``FSDPPass`` early-returns when ``world_size == 1``, so a
            # single-process run (no dist, or a single rank) compiles and
            # trains as plain graph mode without sharding.
            self._init_device_mesh(self._mesh_context)

        joint_graph = trace_model_graph(
            self.model, self.train_fn, sample_input, sample_label
        )

        pipeline = PassPipeline.from_config(self.pass_config, self.pass_plan)

        pass_kwargs = self._build_pass_kwargs()

        # Passes mutate ``graph_module`` in place and return it, so the
        # transformed graph lives on ``joint_graph`` for ``train_step``.
        pipeline.run(joint_graph.graph_module, **pass_kwargs)

        self._joint_graph = joint_graph

        self._init_optimizer()

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
        # parameters in place, keeping the trainer FSDP-agnostic.
        kwargs["model"] = self.model

        if self.pass_config.fsdp_enabled:
            kwargs["fsdp_group_name"] = "fsdp"

        return kwargs

    def train_step(self, input_batch: torch.Tensor, label_batch: torch.Tensor) -> Any:
        """
        Execute one training step

        Args:
            input_batch: Input batch
            label_batch: Label batch

        Returns:
            loss: Loss value
        """
        if self._joint_graph is None:
            self.compile(input_batch, label_batch)

        loss, grads = self._run_graph(input_batch, label_batch)

        self._accumulate_grads(grads)

        return loss

    def optimizer_step(self) -> None:
        """Optimizer update"""
        if self.optimizer is None:
            return

        if self.optimizer_config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.optimizer_config["grad_clip"]
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

    def to(self, device: torch.device) -> "GraphTrainer":
        """Move the model to ``device`` and remember it for batch placement."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def set_pytree_pre_hook(self, hook: Callable[[], None]) -> "GraphTrainer":
        """Register a no-arg hook run just before the graph is compiled.

        Used for model-specific tracer setup that must happen before the first
        ``compile`` -- e.g. registering flex-attention ``BlockMask`` as a
        pytree node inside ``torch``'s pytree registry. ``train`` triggers
        compilation lazily on the first batch, so the hook fires on that batch.
        """
        self._pytree_pre_hook = hook
        return self

    def _place_on_device(self, batch):
        """Move a ``(input, label)`` batch onto ``self.device``."""
        if self.device is None:
            return batch
        moved = tuple(
            b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch
        )
        return moved

    def train(
        self,
        data_iterable: Iterable,
        max_steps: Optional[int] = None,
        log_interval: Optional[int] = None,
        log_fn: Optional[Callable[[int, Any], None]] = None,
    ) -> List[Any]:
        """Run the full training loop over ``data_iterable``.

        The data iterator must yield ``(input, label)`` pairs (the same two
        positional arguments ``train_fn`` and ``train_step`` consume). Each
        batch is moved onto ``self.device`` (if one is set), then
        ``train_step`` + ``optimizer_step`` are driven. The graph is compiled
        lazily on the first batch via ``train_step``.

        Args:
            data_iterable: An iterable / iterator of ``(input, label)`` pairs.
            max_steps: Stop after this many steps. Runs the whole iterator when
                ``None``.
            log_interval: Log a loss every ``log_interval`` steps (requires
                ``log_fn`` or a rank-0 printer).
            log_fn: Callback ``log_fn(step, loss)`` for progress reporting. When
                ``None`` the loss is printed to stdout on ``log_interval``.

        Returns:
            List of per-step losses.
        """
        if not isinstance(data_iterable, Iterator):
            data_iterable = iter(data_iterable)

        losses: List[Any] = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        for step, batch in enumerate(data_iterable):
            if max_steps is not None and step >= max_steps:
                break

            input_batch, label_batch = batch
            if self.device is not None:
                input_batch, label_batch = self._place_on_device(
                    (input_batch, label_batch)
                )

            loss = self.train_step(input_batch, label_batch)
            self.optimizer_step()
            losses.append(loss)

            if rank == 0 and log_interval and (step + 1) % log_interval == 0:
                if log_fn is not None:
                    log_fn(step + 1, loss)
                else:
                    print(f"Step {step + 1} | Loss: {loss.item():.4f}")

        return losses

    def _init_optimizer(self):
        """Initialize optimizer on the model's (FSDP-sharded) parameters.

        FSDPPass shards ``self.model``'s parameters in place during compile,
        so ``model.parameters()`` already yields the local shards and the
        optimizer needs no FSDP awareness.

        When ``torch_npu`` is installed but no NPU is available (e.g. a
        CPU-only CI run), Adam's automatic foreach/fused kernel selection
        probes ``torch_npu.npu.current_device()`` via ``_lazy_init()``,
        which raises even though the parameters live on CPU. Disable the
        probe explicitly in that case so single-process CPU runs (and the
        UT suite) succeed; NPU runs keep the default foreach/fused path.
        """
        optimizer_class = torch.optim.Adam
        kwargs = {"lr": self.optimizer_config.get("lr", 1e-4)}
        if hasattr(torch, "npu") and not torch.npu.is_available():
            kwargs["foreach"] = False
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def _run_graph(self, input_batch, label_batch):
        """Execute compiled graph"""
        if self._joint_graph is None:
            raise RuntimeError(
                "Graph not compiled. Call trainer.compile() or trainer.train() first."
            )

        # The joint graph's parameters/buffers are static inputs: feed the
        # live (FSDP-sharded) model state in FQN order each step.
        return run_traced_graph(
            self._joint_graph,
            self.model,
            input_batch,
            label_batch,
        )

    def _accumulate_grads(self, grads):
        """Accumulate gradients"""
        params = [p for p in self.model.parameters() if p.requires_grad]

        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad


__all__ = ["GraphTrainer"]
