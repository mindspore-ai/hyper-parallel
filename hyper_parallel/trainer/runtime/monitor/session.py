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
"""Trainer-owned recorder lifecycle; device work stays on the training thread."""

import json
import logging
import os
import queue
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import partial
from typing import Any, Iterator

import torch
import torch.distributed as dist

from hyper_parallel.components.monitor import statistics_scope
from hyper_parallel.trainer.config.monitor import MonitorConfig
from .datasets import DatasetStatsRecorder
from .experts import ExpertLoadRecorder
from .flops import DevicePeak, model_blocks
from .output import OutputWriter
from .transfer import SummaryTransfer

logger = logging.getLogger(__name__)


def monitor_phase(trainer: Any, name: str) -> Any:
    """Enter an explicitly instrumented boundary, including legacy trainer fixtures."""
    monitor = getattr(trainer, "monitor", None)
    return monitor.phase(name) if monitor is not None else nullcontext()


def monitor_scope(trainer: Any) -> Any:
    """Install the observer only during original forward/data/optimizer execution."""
    monitor = getattr(trainer, "monitor", None)
    return statistics_scope(monitor if monitor is not None and monitor.enabled else None)


class MonitorSession:
    """Own small detached summaries, CPU events and deterministic sampling decisions."""

    def __init__(self, trainer: Any) -> None:
        """Own recorder state for one trainer without retaining forward activations."""
        self.trainer = trainer
        self.config = getattr(trainer.config, "monitor", MonitorConfig(enabled=False))
        self.enabled = self.config.enabled
        self.rank = int(os.environ.get("RANK", "0"))
        self.run_id = os.environ.get("WANDB_RUN_ID") or uuid.uuid4().hex
        self.load_state_dict({"run_id": self.run_id})
        self.writer = None
        self.transfer = SummaryTransfer()
        self.dataset = DatasetStatsRecorder(self.config.datasets)
        self.step = 0
        self.micro_step = 0
        self.in_step = False
        self.timing = defaultdict(float)
        self.events = []
        self.unavailable_reasons = set()
        self.completions = queue.Queue(maxsize=64)
        self.expert_recorder = ExpertLoadRecorder()
        self.names = {}
        self.parameter_names = {}
        self.ns_tensor_names = {}
        self.gradient_parameters = {}
        self.losses = {}
        self.ns_calls = 0
        self.estimator = None
        self.peak = None
        self.window_flops = 0.0
        self.window_seconds = 0.0
        self.compiled_model = False
        self.closed = False

    def initialize(self) -> None:
        """Share a run identity once after distributed setup, before any output."""
        if not self.enabled:
            return
        self.rank = getattr(self.trainer, "global_rank", self.rank)
        if dist.is_available() and dist.is_initialized():
            identity = [self.run_id if self.rank == 0 else None]
            dist.broadcast_object_list(identity, src=0)
            self.run_id = identity[0]

    def bind_model(self) -> None:
        """Resolve stable module names without installing tensor hooks."""
        if self.enabled:
            self.names = {id(module): name or "<root>" for name, module in self.trainer.model.named_modules()}
            self.parameter_names = {id(parameter): name for name, parameter in self.trainer.model.named_parameters()}
            self.expert_recorder = ExpertLoadRecorder()
            for name, module in self.trainer.model.named_modules():
                if getattr(module, "monitor_expert_count", None) is not None:
                    reference = getattr(module, "tokens_per_expert", None)
                    if reference is None:
                        reference = next(module.parameters())
                    self.expert_recorder.register(name or "<root>", module.monitor_expert_count, reference.device)
            self.gradient_parameters = {
                prefix: [
                    (name, parameter)
                    for name, parameter in self.trainer.model.named_parameters()
                    if parameter.requires_grad and (not prefix or name.startswith(prefix + "."))
                ]
                for prefix in self.config.local_grad_norm.modules or [""]
            }
            if self.config.mfu or self.trainer.config.training.log_tflops:
                self.estimator = model_blocks(self.trainer.model, getattr(self.trainer, "model_config", None))
            peak_spec = os.environ.get("HYPER_PARALLEL_MONITOR_PEAK")
            if self.config.mfu and peak_spec:
                self.peak = DevicePeak(**json.loads(peak_spec))
            if getattr(getattr(self.trainer.config, "compile", None), "enabled", False):
                self.compiled_model = True
                self.unavailable(
                    "compiled_tensor_recorders", "compiled regions omit Python observations; no graph breaks inserted"
                )

    def unavailable(self, metric: str, reason: str) -> None:
        """Expose unsupported coverage without publishing a misleading zero."""
        if (metric, reason) not in self.unavailable_reasons:
            self.unavailable_reasons.add((metric, reason))
            self._emit({"event": "unavailable", "step": self.step, "metric": metric, "reason": reason})

    def _emit(self, event: dict) -> None:
        if not self.enabled or self.closed:
            return
        if self.writer is None:
            # Delay remote initialization until checkpoint restoration chooses the run ID.
            if len(self.events) < 256:
                self.events.append(event)
            return
        self.writer.emit(event)

    def completion(self, name: str) -> Any:
        """Capture submit time; future callbacks enqueue only small immutable CPU records."""
        if not self.enabled or not self.config.timing:
            return None
        step, start = self.step, time.perf_counter()

        def complete(future: Any) -> None:
            """Queue completion metadata without touching training tensors or collectives."""
            outcome = "cancelled" if future.cancelled() else ("failed" if future.exception() else "ok")
            event = {
                "event": "timing",
                "step": step,
                "phase": name,
                "clock": "host_wall",
                "duration_seconds": time.perf_counter() - start,
                "outcome": outcome,
            }
            try:
                self.completions.put_nowait(event)
            except queue.Full:
                logger.warning("Monitor completion queue full; timing event dropped")

        return complete

    def _drain_completions(self) -> None:
        while True:
            try:
                self._emit(self.completions.get_nowait())
            except queue.Empty:
                return

    def start_output(self) -> None:
        """Open sinks after restore, then replay bounded startup events."""
        if not self.enabled or self.writer is not None:
            return
        self.writer = OutputWriter(self.config, self.rank, self.run_id, getattr(self.trainer.config, "wandb", None))
        for event in self.events:
            self.writer.emit(event)
        self.events.clear()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure host wall time; asynchronous device work is not synchronized."""
        if not self.enabled or not self.config.timing:
            yield
            return
        start = time.perf_counter()
        outcome = "ok"
        try:
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            duration = time.perf_counter() - start
            if self.in_step:
                self.timing[name] += duration
            else:
                self._emit(
                    {
                        "event": "timing",
                        "step": self.step,
                        "phase": name,
                        "clock": "host_wall",
                        "duration_seconds": duration,
                        "outcome": outcome,
                    }
                )

    def sampled(self, config: Any) -> bool:
        """Decide at optimizer-step boundaries, identically on all ranks."""
        return (
            self.enabled and self.in_step and getattr(config, "enabled", True) and self.step % config.sampler_steps == 0
        )

    def begin_step(self, step: int) -> None:
        """Poll completed copies and reset all step-owned counters."""
        if not self.enabled:
            return
        self.transfer.poll()
        self._drain_completions()
        self.step, self.micro_step, self.ns_calls = step, 0, 0
        self.ns_tensor_names.clear()
        self.in_step = True
        self.timing.clear()
        self.dataset.reset()
        self.expert_recorder.begin_step(self.sampled(self.config.moe) and not self.compiled_model)
        self.losses.clear()

    def record_component(self, kind: str, owner: Any, values: dict) -> None:
        """Consume borrowed tensors synchronously; retain only bounded summaries."""
        if kind == "gradients" and self.sampled(self.config.local_grad_norm):
            parameters = []
            for parameter in values["parameters"]:
                grad = getattr(parameter, "main_grad", None)
                if grad is None:
                    grad = parameter.grad
                if parameter.requires_grad and grad is not None:
                    parameters.append(parameter)
            self.record_gradients(dict(zip((id(p) for p in parameters), values["gradients"])))
        elif kind == "dataset" and self.sampled(self.config.datasets):
            batch = values.get("batch")
            if batch is not None:
                batch = dict(batch)
                if "labels_are_shifted" in values and isinstance(batch.get("labels"), torch.Tensor):
                    labels = batch["labels"]
                    if labels.device.type == "cpu":
                        batch["loss_mask"] = labels >= 0
                        if values.get("eod_mask_loss"):
                            batch["loss_mask"] &= batch["input_ids"] != values["eod_token_id"]
                        if values["labels_are_shifted"]:
                            batch["shift_labels"] = labels
                    batch["packing_policy"] = values["packing_policy"]
                    batch["reset_position_ids"] = values["reset_position_ids"]
                self.dataset.record(batch)
        elif kind == "expert_routes" and self.sampled(self.config.moe):
            name = self.names.get(id(owner))
            if name is not None:
                self.expert_recorder.record_routes(name, values["indices"], values["attempted"])
        elif kind == "expert_load" and self.sampled(self.config.moe):
            name = self.names.get(id(owner))
            if name is None:
                return
            self.expert_recorder.record(name, values["counts"])
        elif kind == "ns_inputs" and self.sampled(self.config.optimizer.ns_norm):
            name = self.parameter_names.get(id(values["parameter"]), "<unmapped>")
            for index, tensor in enumerate(values["tensors"]):
                self.ns_tensor_names[id(tensor)] = (name, index)
        elif kind == "ns_norm" and self.sampled(self.config.optimizer.ns_norm):
            tensor = values["values"].detach()
            # vector_norm with dtype avoids materializing a full fp32 copy of the update.
            norms = torch.linalg.vector_norm(tensor, dim=(-2, -1), dtype=torch.float32).reshape(-1)
            matrices, offset = [], 0
            for source, size in zip(values.get("tensors", ()), values.get("sizes", ())):
                name, index = self.ns_tensor_names.pop(id(source), ("<unmapped>", None))
                matrices.append({"parameter": name, "transform_index": index, "start": offset, "count": size})
                offset += size
            self._submit(
                "ns_norm",
                norms,
                {
                    "call": self.ns_calls,
                    "stage": "post_ns_pre_scale",
                    "matrix_owners": matrices,
                },
            )
            self.ns_calls += 1

    def record_loss(self, losses: Any, token_counts: dict) -> None:
        """Sample raw pre-reduction micro means plus token-weighted step sums."""
        if not self.sampled(self.config.local_loss):
            return
        mapping = losses if isinstance(losses, dict) else {"foundation_loss": losses}
        for name, value in mapping.items():
            count_name = name.removesuffix("_loss") + "_tokens"
            count = token_counts.get(count_name)
            if count is None or value.numel() != 1:
                self._emit(
                    {
                        "event": "unavailable",
                        "step": self.step,
                        "metric": "local_loss",
                        "name": name,
                        "reason": "scalar loss and matching token count required",
                    }
                )
                continue
            count = torch.as_tensor(count, device=value.device).detach().float().reshape(())
            raw = value.detach().float().reshape(())
            summary = torch.stack((raw, count))
            self._submit("local_micro_loss", summary, {"name": name, "micro_step": self.micro_step})
            weighted = torch.stack((torch.where(count > 0, raw * count, torch.zeros_like(raw)), count))
            if name in self.losses:
                self.losses[name].add_(weighted)
            else:
                self.losses[name] = weighted
        self.micro_step += 1

    def record_gradients(self, gradient_views: dict | None = None) -> None:
        """Compute local shard L2 after reduction and before clipping, without collectives."""
        if not self.sampled(self.config.local_grad_norm):
            return
        selected = self.config.local_grad_norm.modules or [""]
        buckets = defaultdict(list)
        counts = defaultdict(int)
        for prefix, parameters in self.gradient_parameters.items():
            for _, parameter in parameters:
                grad = (
                    gradient_views.get(id(parameter))
                    if gradient_views is not None
                    else getattr(parameter, "main_grad", None)
                )
                if grad is None and gradient_views is None:
                    grad = parameter.grad
                if grad is None:
                    continue
                placements = getattr(grad, "placements", ())
                if gradient_views is None and any(placement.is_partial() for placement in placements):
                    self.unavailable("local_grad_norm", "Partial gradient has no reduced view without clipping")
                    return
                local = getattr(grad, "_local_tensor", grad)
                if local.is_sparse:
                    local = local.coalesce().values()
                buckets[prefix].append(torch.linalg.vector_norm(local.detach(), dtype=torch.float32))
                counts[prefix] += 1
        for prefix in selected:
            if not buckets[prefix]:
                self._emit(
                    {
                        "event": "unavailable",
                        "step": self.step,
                        "metric": "local_grad_norm",
                        "module": prefix or "<root>",
                        "reason": "no local gradients",
                    }
                )
                continue
            by_device = defaultdict(list)
            for norm in buckets[prefix]:
                by_device[norm.device].append(norm)
            for device, norms in by_device.items():
                summary = torch.linalg.vector_norm(torch.stack(norms)).reshape(1)
                self._submit(
                    "local_grad_norm",
                    summary,
                    {
                        "module": prefix or "<root>",
                        "device": str(device),
                        "parameter_count": counts[prefix],
                        "norm_type": 2,
                        "stage": "post_reduce_pre_clip",
                    },
                )

    def _submit(self, kind: str, tensor: torch.Tensor, metadata: dict) -> None:
        event = {"event": kind, "step": self.step, **metadata}
        self.transfer.submit(tensor, partial(self._consume, event))

    def _consume(self, event: dict, values: list) -> None:
        event = dict(event)
        if event["event"] in ("expert_layers", "expert_layers_local"):
            offset = 0
            violations = []
            for name, length in zip(event["modules"], event["lengths"]):
                layer_counts = values[offset : offset + length - 2]
                attempted, observations = values[offset + length - 2 : offset + length]
                if not observations:
                    self._emit(
                        {
                            "event": "unavailable",
                            "step": event["step"],
                            "metric": "expert_load",
                            "module": name,
                            "scope": event["scope"],
                            "reason": "no instrumented forward observation",
                        }
                    )
                    offset += length
                    continue
                total = sum(layer_counts)
                if total:
                    violations.append(max(layer_counts) * len(layer_counts) / total - 1)
                self._consume(
                    {key: value for key, value in event.items() if key not in ("modules", "lengths")}
                    | {
                        "event": "expert_load" if event["event"] == "expert_layers" else "expert_load_local",
                        "module": name,
                        "attempted_assignments": attempted,
                    },
                    layer_counts,
                )
                offset += length
            if event["event"] == "expert_layers" and violations:
                self._emit(
                    {
                        "event": "expert_load_summary",
                        "step": event["step"],
                        "scope": event["scope"],
                        "metrics": {"moe/max_vio": max(violations)},
                    }
                )
            return
        if event["event"] == "expert_load":
            mean = sum(values) / len(values) if values else 0
            event["summary"] = {
                "max": max(values, default=0),
                "mean": mean,
                "max_vio": max(values) / mean - 1 if mean else None,
                "empty_expert_fraction": sum(v == 0 for v in values) / len(values) if values else None,
                "accepted_assignments": sum(values),
                "attempted_assignments": event["attempted_assignments"],
                "dropped_assignments": event["attempted_assignments"] - sum(values),
            }
            if self.config.moe.record_distribution:
                event["counts"] = values
            event["metrics"] = {
                f"moe/{event['module']}/{key}": value for key, value in event["summary"].items() if value is not None
            }
        elif event["event"] == "local_grad_norm":
            event["values"] = values
            event["metrics"] = {"training/grad_norm_local_post_reduce_pre_clip": values[0]}
        elif event["event"] == "ns_norm":
            event["values"] = values
            if values:
                event["metrics"] = {
                    "optimizer/ns_norm_mean": sum(values) / len(values),
                    "optimizer/ns_norm_max": max(values),
                    "optimizer/ns_norm_min": min(values),
                }
        elif event["event"] == "local_step_loss":
            event["loss"] = values[0] / values[1] if values[1] else None
            event["tokens"] = values[1]
            if event["loss"] is not None:
                event["metrics"] = {f"training/local_{event['name']}": event["loss"]}
        else:
            event["values"] = values
        self._emit(event)

    def end_step(self) -> dict:
        """Launch small ordered reductions on the training thread, then async host copies."""
        if not self.enabled:
            return {}
        if self.sampled(self.config.moe) and not self.expert_recorder.counts:
            self.unavailable("expert_load", "no instrumented MoE observation (unsupported or compiled path)")
        if self.sampled(self.config.local_loss) and not self.losses:
            self.unavailable("local_loss", "no pre-reduction loss observation on this rank")
        if self.sampled(self.config.optimizer.ns_norm) and not self.ns_calls:
            self.unavailable("ns_norm", "no instrumented Muon NS call on this rank")
        mesh = getattr(self.trainer, "mesh", None)
        groups = []
        dp_cp = getattr(mesh, "dp_cp_mesh", None)
        if dp_cp is not None:
            groups.append(dp_cp.get_group())
        if getattr(mesh, "sequence_parallel", False):
            groups.append(self.trainer.device_mesh["tp"].get_group())
        self.expert_recorder.finalize(groups, self._submit, self.config.moe.record_distribution)
        for name, values in sorted(self.losses.items()):
            self._submit("local_step_loss", values, {"name": name})
        if self.sampled(self.config.datasets):
            self._emit(
                {"event": "datasets", "step": self.step, "scope": "rank_source_before_cp", **self.dataset.snapshot()}
            )
        metrics = {f"timing/host/{key}": value for key, value in self.timing.items()}
        if self.transfer.dropped:
            metrics["monitor/dropped_summaries"] = self.transfer.dropped
        self.losses.clear()
        self.transfer.poll()
        self.in_step = False
        return metrics

    def flops_metrics(self, tokens: float, lengths_squared: float, seconds: float) -> dict:
        """Publish complete standard-block work; unknown hardware peaks never imply zero MFU."""
        training = self.trainer.config.training
        if not self.enabled or not (self.config.mfu or training.log_tflops):
            return {}
        estimator = self.estimator
        if estimator is None:
            return {}
        work = estimator.estimate(tokens, lengths_squared)
        if not work:
            self._emit(
                {
                    "event": "flops_coverage",
                    "step": self.step,
                    "missing_blocks": list(estimator.missing),
                    "formula": estimator.version,
                    "complete": False,
                }
            )
            return {}
        world = max(1, getattr(self.trainer, "world_size", 1))
        result = {}
        if seconds > 0:
            rate = sum(work.values()) / seconds / world / 1e12
            if training.log_tflops:
                result["performance/tflops_per_device"] = rate
            if self.config.mfu:
                # Peaks are deployment facts, not guessed from accelerator family strings.
                self.window_flops += sum(work.values())
                self.window_seconds += seconds
                if self.peak is not None:
                    result["performance/mfu_percent"] = 100 * rate / self.peak.tflops
                    result["performance/mfu_window_percent"] = (
                        100 * self.window_flops / self.window_seconds / world / 1e12 / self.peak.tflops
                    )
                else:
                    result["performance/mfu_status"] = "unavailable_peak"
        self._emit(
            {
                "event": "flops_coverage",
                "step": self.step,
                "formula": estimator.version,
                "complete": True,
                "blocks": work,
                "peak": asdict(self.peak) if self.peak is not None else None,
            }
        )
        return result

    def log_step(self, state: Any, metrics: dict) -> None:
        """Write one canonical logging event, using only the external cadence."""
        self._emit({"event": "train_step", "step": state.global_step, "epoch": state.epoch, "metrics": metrics})

    def state_dict(self) -> dict:
        """Persist run identity alongside cumulative meter state."""
        return {"run_id": self.run_id}

    def load_state_dict(self, state: dict) -> None:
        """Restore identity before remote sinks open."""
        run_id = state.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id)
        ):
            raise ValueError("monitor checkpoint run_id must be a nonempty safe identifier")
        self.run_id = run_id

    def close(self) -> None:
        """Drain outstanding small transfers at shutdown, never during a training step."""
        if self.closed or not self.enabled:
            return
        self.in_step = False
        self.start_output()
        self.transfer.close()
        self._drain_completions()
        self._emit(
            {
                "event": "monitor_closed",
                "step": self.step,
                "dropped_summaries": self.transfer.dropped,
                "dropped_remote_records": self.writer.dropped,
            }
        )
        self.writer.close()
        self.closed = True
