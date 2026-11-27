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
"""Bridge Trainer microbatches and loss semantics to the HP pipeline scheduler."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any, Optional

import torch
from torch import distributed as dist, nn

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context
from hyper_parallel.trainer.callbacks.moe_monitor_callback import MoEMonitorCallback
from hyper_parallel.core.pipeline_parallel import (
    MetaStepType, PipelineStage, Schedule1F1B, ScheduleDualPipeV, ScheduleGPipe,
    ScheduleInterleaved1F1B, ScheduleMPipeTranspose,
)
from hyper_parallel.trainer.runtime.memory import empty_cache
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


class _TrainerStage(PipelineStage):
    """Preserve one auxiliary-loss context through each microbatch's backward."""

    def __init__(self, runtime: "TrainerPipeline", model: nn.Module) -> None:
        mesh = runtime.trainer.mesh
        partition = model._hp_pipeline_partition  # pylint: disable=protected-access
        super().__init__(
            model, partition.stage_index, partition.stage_count,
            device=runtime.trainer.device, group=mesh.pp_mesh.get_group(), mesh=mesh.pp_mesh, dyn_shape=True,
        )
        self.contexts = {}
        self.partition = partition
        self.runtime = runtime
        self.monitor = MoEMonitorCallback(model)

    def _forward_submodule(
            self, *pipeline_inputs: torch.Tensor, inputs: dict, loss_inputs: dict, counts: dict, micro_index: int,
    ) -> Any:
        """Run the local forward and capture stage-local metrics before the next microbatch."""
        runtime = self.runtime
        trainer = runtime.trainer
        inputs = {**inputs, "pipeline_inputs": pipeline_inputs, "use_cache": False}
        prepare = getattr(trainer.loss_fn, "prepare_model_inputs", None)
        if callable(prepare) and self.partition.is_last:
            inputs = prepare(inputs, loss_inputs)
        context = trainer.model_fwd_context
        with context() if callable(context) else context:
            outputs = self.submodule(**inputs)
        router_losses = self.monitor.last_aux_losses()
        if router_losses:
            runtime.router_stats[micro_index, 0] += torch.stack(router_losses).detach().sum()
            runtime.router_stats[micro_index, 1] += len(router_losses)
        if not self.partition.is_last:
            return outputs
        loss, metrics = trainer.postforward(outputs, loss_inputs.get("labels"), current_token_counts=counts)
        runtime.metrics[micro_index] = {
            name: value.detach() for name, value in metrics.items() if name != "router_aux_loss"
        }
        return loss

    def forward_one_chunk(self, micro_index: int, args: Optional[tuple] = None,
                          kwargs: Optional[dict] = None) -> Any:
        """Fetch this DP replica's next batch and run the local stage forward.

        Args:
            micro_index: Forward microbatch index within the optimizer step.
            args: Local positional inputs supplied by the schedule.
            kwargs: Unused schedule inputs. The Trainer supplies the batch fields.
        """
        runtime = self.runtime
        trainer = runtime.trainer
        inputs, loss_inputs, counts, scale = runtime.batch_for_chunk(self, micro_index)
        trainer.current_token_counts = counts
        callback = getattr(trainer, "channel_loss_callback", None)
        tracking = callback.micro_step_context(trainer.state, inputs) if callback is not None else nullcontext()
        with tracking:
            context = copy_context()
            scope = aux_loss_scale_context(scale)
            context.run(scope.__enter__)
            self.contexts[micro_index] = (context, scope)
            if callback is not None:
                inputs = inputs.copy()
                callback.strip_model_inputs(inputs)
            if self is runtime.stages[0]:
                trainer.model_integration.record_batch(inputs, loss_inputs)
            stage_kwargs = {"inputs": inputs, "loss_inputs": loss_inputs, "counts": counts, "micro_index": micro_index}
            result = context.run(super().forward_one_chunk, micro_index, args, stage_kwargs)
        prefetch_plan = getattr(trainer.train_dataloader, "prefetch_plan", None)
        if self is runtime.stages[-1] and callable(prefetch_plan):
            prefetch_plan()
        return result

    def backward_one_chunk(self, micro_index: int) -> None:
        """Run backward with the auxiliary-loss scale captured during forward.

        Args:
            micro_index: Index of the corresponding saved forward context.
        """
        context, scope = self.contexts.pop(micro_index)
        trainer = self.runtime.trainer
        try:
            if trainer.config.training.empty_cache_before_backward:
                empty_cache()
            with trainer.model_bwd_context:
                context.run(super().backward_one_chunk, micro_index)
            prefetch = getattr(trainer.train_dataloader, "prefetch", None)
            if self is self.runtime.stages[0] and callable(prefetch):
                prefetch()
        finally:
            context.run(scope.__exit__, None, None, None)

    def clear_cache(self) -> None:
        """Release microbatch contexts and saved activations after the schedule finishes."""
        for context, scope in self.contexts.values():
            context.run(scope.__exit__, None, None, None)
        self.contexts.clear()
        super().clear_cache()
        self.fwd_cache.clear()
        self.clear_states()
        self.last_stage_outputs = None


class TrainerPipeline:
    """Run a complete optimizer step's microbatches on one local stage.

    The data runtime supplies the same DP-coordinate batches to every PP rank.
    Labels and model-owned metadata remain local; the schedule sends only the
    model adapter's explicit activation/state tensors.
    """

    def __init__(self, trainer: Any) -> None:
        """Bind the local partition and its PP group to the Trainer."""
        self.trainer = trainer
        self.partition = trainer.model._hp_pipeline_partition  # pylint: disable=protected-access
        self.stages = [_TrainerStage(self, part) for part in getattr(trainer.model, "parts", (trainer.model,))]
        self.preprocess = None
        self.visual_inputs = {}
        if self.partition.mpipe:
            self.preprocess = trainer.model.build_mpipe_preprocess()
            # Frozen weights can differ after scratch init or checkpoint restore.
            group = trainer.mesh.pp_mesh.get_group()
            for tensor in (*self.preprocess.parameters(), *self.preprocess.buffers()):
                if not tensor.requires_grad:
                    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                    dist.broadcast(local.detach(), group=group, group_src=0)
        self.batch_cache = {}
        self.schedule = None
        self.batches = None
        self.router_stats = None
        self.metrics = {}

    def batch_for_chunk(self, stage: PipelineStage, micro_index: int) -> tuple:
        """Share a microbatch and its token weight across this rank's virtual stages.

        Args:
            stage: Local chunk about to execute forward.
            micro_index: Microbatch index within the current optimizer step.
        """
        trainer = self.trainer
        if stage is self.stages[0]:
            inputs, loss_inputs = next(self.batches)
            inputs, loss_inputs = trainer.preforward(inputs), trainer.preforward(loss_inputs)
            counts = count_loss_token(loss_inputs)
            for name, value in counts.items():
                trainer.step_token_counts[name] = trainer.step_token_counts.get(name, 0) + value.detach()
            count = counts["foundation_tokens"]
            trainer.on_micro_step_begin({**inputs, **loss_inputs, "token_count": count})
            trainer.begin_fsdp_runtime_diagnostics(micro_index)
            mesh = trainer.mesh
            if mesh.sequence_parallel:
                group = mesh.device_mesh["tp"].get_group()
                count = count.clone()
                dist.all_reduce(count, group=group)
                count = count / dist.get_world_size(group)
            self.batch_cache[micro_index] = (inputs, loss_inputs, counts, count * mesh.dp_size * mesh.cp_size)
        if stage is self.stages[-1]:
            return self.batch_cache.pop(micro_index)
        return self.batch_cache[micro_index]

    def _load_visual_input(self, step, context):
        context.arg_mbs[step.micro_index] = self.visual_inputs.pop(step.micro_index)

    def _finish_step(self, losses: list, count: int) -> tuple[float, dict[str, float]]:
        """Publish per-microbatch callbacks and the same step numerators on every stage."""
        trainer = self.trainer
        group = trainer.mesh.pp_mesh.get_group()
        source_index = 0 if trainer.config.pipeline.schedule == "dualpipev" else -1
        source = trainer.mesh.pp_mesh.rank_list[source_index]
        keys = [tuple(self.metrics[i]) for i in range(count)] if self.partition.is_last else None
        metadata = [(keys, trainer.loss_names) if self.partition.is_last else None]
        dist.broadcast_object_list(metadata, src=source, group=group)
        keys, trainer.loss_names = metadata[0]
        columns = {name: i for i, name in enumerate(dict.fromkeys(name for row in keys for name in row), start=1)}
        values = torch.zeros((count, len(columns) + 1), device=trainer.device, dtype=torch.float32)
        if self.partition.is_last:
            values[:, 0] = torch.stack(losses).detach()
            for i, metrics in self.metrics.items():
                for name, value in metrics.items():
                    values[i, columns[name]] = value
        dist.broadcast(values, src=source, group=group)
        dist.all_reduce(self.router_stats, group=group)
        router_counts = self.router_stats[:, 1].tolist()
        totals = defaultdict(float)
        for i, names in enumerate(keys):
            metrics = {name: values[i, columns[name]] for name in names}
            if router_counts[i]:
                metrics["router_aux_loss"] = self.router_stats[i, 0] / router_counts[i]
            trainer.on_micro_step_end(metrics)
            for name, value in metrics.items():
                totals[name] += value.item()
        return sum(values[:, 0].tolist()), totals

    def run(self, batches: Iterator[tuple[dict, dict]], count: int) -> tuple[float, dict[str, float]]:
        """Consume batches in forward order and schedule their forward/backward work.

        Args:
            batches: Stage-local iterator over Trainer model and loss inputs.
            count: Number of microbatches in this optimizer step.
        """
        if not count:
            raise ValueError("a pipeline optimizer step needs at least one microbatch")
        if self.schedule is None or self.schedule.micro_batch_num != count:
            if self.preprocess is not None:
                self.schedule = ScheduleMPipeTranspose(self.stages, count, self.preprocess, num_transpose_layers=1)
                self.schedule.register_custom_function(MetaStepType.DATA_LOAD, self._load_visual_input)
            else:
                schedule_type = {
                    "1f1b": Schedule1F1B, "gpipe": ScheduleGPipe, "interleaved_1f1b": ScheduleInterleaved1F1B,
                    "dualpipev": ScheduleDualPipeV,
                }[self.trainer.config.pipeline.schedule]
                self.schedule = schedule_type(self.stages, count)
        self.batches = iter(batches)
        self.router_stats = torch.zeros((count, 2), device=self.trainer.device, dtype=torch.float32)
        try:
            if self.preprocess is not None:
                prepared = []
                for index in range(count):
                    inputs, loss_inputs = next(self.batches)
                    inputs = inputs.copy()
                    if self.schedule.owner_of(index) == self.trainer.mesh.pp_rank:
                        self.visual_inputs[index] = self.preprocess.take_inputs(inputs)
                    prepared.append((inputs, loss_inputs))
                self.batches = iter(prepared)
            return self._finish_step(self.schedule.run(), count)
        finally:
            for stage in self.stages:
                stage.clear_cache()
            self.visual_inputs.clear()
            self.batch_cache.clear()
            self.batches = None
            self.router_stats = None
            self.metrics.clear()


__all__ = ["TrainerPipeline"]
