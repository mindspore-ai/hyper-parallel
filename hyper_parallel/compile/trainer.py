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

The trainer composes ``GraphCompiler``: compilation and graph execution
(``compile`` / ``forward_backward``) are delegated to the compiler, while the
trainer owns the training policy -- optimizer lifecycle, grad clip, batch
device placement, and the ``train`` loop. Model / device / compiled-graph
state lives on the compiler (``trainer._compiler``).
"""

__all__ = ["GraphTrainer"]

import logging
from typing import Any, Callable, Iterable, Iterator, List, Optional

import torch
import torch.distributed as dist

from .compiler import GraphCompiler
from .pass_config import PassConfig
from .pass_plan import PassPlan

_LOG = logging.getLogger(__name__)


class GraphTrainer:
    """
    Graph-mode Trainer

    Users provide model code and parallel configuration.
    Framework automatically handles all parallel logic.

    Holds a ``GraphCompiler`` and delegates compilation and graph execution
    to it; the optimizer and the training loop stay here.
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
            pass_plan: PassPlan declaring which modules to shard (optional;
                enables declarative sharding)
            optimizer_config: Optimizer configuration
            device: Device to place the model and run training on. Defaults to
                the NPU device when available, otherwise CPU.
            mesh_context: Optional automodel ``MeshContext`` carrying a
                pre-built TP/FSDP mesh (forwarded to ``GraphCompiler``). When
                provided, the TP group is reused as-is (boundary forwards
                already hold the group object) and only the FSDP shard
                sub-mesh is registered under ``"fsdp"``. Use this to feed an
                automodel TP-sharded model into the graph-mode FSDP pass.
        """
        # The compiler owns the model, device, config, and the compiled
        # graph; the trainer keeps only the optimizer / loop policy.
        self._compiler = GraphCompiler(
            model=model,
            train_fn=train_fn,
            pass_config=pass_config,
            pass_plan=pass_plan,
            device=device,
            mesh_context=mesh_context,
        )
        self.optimizer_config = optimizer_config or {}
        self.optimizer = None

    def compile(self, input_batch: torch.Tensor, label_batch: torch.Tensor) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled
        at first train_step

        Args:
            input_batch: Input batch used to trace the joint graph
            label_batch: Label batch used to trace the joint graph
        """
        self._compiler.compile(input_batch, label_batch)
        self._init_optimizer()

    def train_step(self, input_batch: torch.Tensor, label_batch: torch.Tensor) -> Any:
        """
        Execute one training step

        Args:
            input_batch: Input batch
            label_batch: Label batch

        Returns:
            loss: Loss value
        """
        return self._compiler.forward_backward(input_batch, label_batch)

    def to(self, device: torch.device) -> "GraphTrainer":
        """Move the model to ``device`` and remember it for batch placement.

        Args:
            device: Target device for the model and subsequent batches

        Returns:
            self, so calls can be chained
        """
        self._compiler.to(device)
        return self

    def optimizer_step(self) -> None:
        """Optimizer update"""
        if self.optimizer is None:
            return

        if self.optimizer_config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                self._compiler.model.parameters(), self.optimizer_config["grad_clip"]
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _place_on_device(self, batch):
        """Move a ``(input, label)`` batch onto the compiler's device."""
        device = self._compiler.device
        if device is None:
            return batch
        moved = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
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
        batch is moved onto the compiler's device, then ``train_step`` +
        ``optimizer_step`` are driven. The graph is compiled lazily on the
        first batch via ``train_step``.

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
            input_batch, label_batch = self._place_on_device((input_batch, label_batch))

            loss = self.train_step(input_batch, label_batch)
            self.optimizer_step()
            losses.append(loss)

            if rank == 0 and log_interval and (step + 1) % log_interval == 0:
                if log_fn is not None:
                    log_fn(step + 1, loss)
                else:
                    _LOG.info("Step %s | Loss: %.4f", step + 1, loss.item())

        return losses

    def _init_optimizer(self):
        """Initialize optimizer on the model's (FSDP-sharded) parameters.

        FSDPPass shards the compiler's model parameters in place during
        compile, so ``model.parameters()`` already yields the local shards
        and the optimizer needs no FSDP awareness.

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
        self.optimizer = optimizer_class(self._compiler.model.parameters(), **kwargs)
