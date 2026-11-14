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
"""Bounded CPU-only output queue with isolated optional remote sinks."""

import importlib
import json
import logging
import math
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def json_value(value: Any) -> Any:
    """Reject device tensors and represent non-finite scalars explicitly."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Output only accepts CPU values, got {type(value).__name__}")


class OutputWriter:
    """Persist JSONL before asynchronous remote delivery; only rank zero logs."""

    def __init__(self, config: Any, rank: int, run_id: str, remote: Any = None) -> None:
        """Open rank-local output and start optional asynchronous sink delivery."""
        self.config, self.rank, self.run_id = config, rank, run_id
        self.remote_config = remote
        self.directory = Path(config.output_dir) / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = None
        if set(config.sinks) & {"jsonl", "wandb", "tensorboard"}:
            self.file = (self.directory / f"rank_{rank}.jsonl").open("a", encoding="utf-8")
        # One step can publish local/global distributions for every decoder layer.
        # Bound the queue while allowing a whole 40-layer sampling burst.
        self.queue: queue.Queue = queue.Queue(maxsize=256)
        self.failed_sinks = set()
        self.dropped = 0
        self.closed = False
        self.stop = threading.Event()
        self.worker = None
        if rank == 0 and set(config.sinks) & {"wandb", "tensorboard"}:
            self.worker = threading.Thread(target=self._run, name="trainer-monitor-output", daemon=True)
            self.worker.start()

    def emit(self, event: dict) -> None:
        """Write a complete immutable record, then enqueue optional remote work."""
        if self.closed:
            return
        record = json_value({"schema_version": 1, "run_id": self.run_id, "rank": self.rank, **event})
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if self.file is not None:
            self.file.write(payload + "\n")
            self.file.flush()
        if self.rank == 0 and "console" in self.config.sinks and record.get("event") in ("train_step", "timing"):
            logger.info("%s", payload)
        if self.worker is not None:
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1
                if self.dropped == 1:
                    logger.warning("Monitor remote queue full; local JSONL remains authoritative")

    def _open_sinks(self) -> tuple[Any, Any, Any]:
        wandb, run, tensorboard = None, None, None
        if "wandb" in self.config.sinks:
            try:
                wandb = importlib.import_module("wandb")
                run = wandb.init(
                    project=self.config.wandb.project,
                    mode=self.config.wandb.mode,
                    entity=getattr(self.remote_config, "entity", None),
                    id=self.run_id,
                    resume="allow" if self.config.wandb.mode == "online" else None,
                )
                run.define_metric("global_step")
                for prefix in ("training", "performance", "data", "memory", "moe", "optimizer"):
                    run.define_metric(f"{prefix}/*", step_metric="global_step")
            except Exception as exc:  # optional sinks cannot terminate training
                logger.warning("WandB unavailable: %s", exc)
        if "tensorboard" in self.config.sinks:
            try:
                module = importlib.import_module("torch.utils.tensorboard")
                tensorboard = module.SummaryWriter(str(self.directory / "tensorboard"))
            except Exception as exc:
                logger.warning("TensorBoard unavailable: %s", exc)
        return wandb, run, tensorboard

    @staticmethod
    def _tables(wandb: Any, record: dict) -> dict:
        event = record["event"]
        tables = {
            f"events/{event}": wandb.Table(
                columns=["event", "step", "rank", "payload"],
                data=[[event, record.get("step"), record["rank"], json.dumps(record, ensure_ascii=False)]],
            )
        }
        if event == "datasets" and "top_tokens" in record:
            tables[f"data/top_tokens_rank_{record['rank']}"] = wandb.Table(
                columns=["step", "token_id", "count", "fraction"],
                data=[[record["step"], row["token_id"], row["count"], row["fraction"]] for row in record["top_tokens"]],
            )
        if event.startswith("expert_load") and ("counts" in record or "values" in record):
            tables[f"moe/distribution_rank_{record['rank']}"] = wandb.Table(
                columns=["step", "module", "expert_id", "assignments"],
                data=[
                    [record["step"], record["module"], index, count]
                    for index, count in enumerate(record.get("counts", record.get("values", [])))
                ],
            )
        return tables

    @staticmethod
    def _scalars(record: dict) -> dict:
        metrics = record.get("metrics", {})
        scalars = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        if record["event"] != "train_step":
            scope = f"rank_{record['rank']}/{record.get('module', record.get('call', 'all'))}"
            scalars = {f"{key}/{scope}": value for key, value in scalars.items()}
        return scalars

    def _write_wandb(self, wandb: Any, run: Any, record: dict, scalars: dict) -> None:
        if scalars:
            run.log({"global_step": record.get("step", 0), **scalars})
        if record["event"] != "train_step":
            run.log(self._tables(wandb, record))

    @staticmethod
    def _write_tensorboard(tensorboard: Any, record: dict, scalars: dict) -> None:
        for key, value in scalars.items():
            tensorboard.add_scalar(key, value, record.get("step", 0))

    def _sink_failure(self, name: str, exc: Exception) -> None:
        # Repeated filesystem/network errors must not flood the training log.
        if name not in self.failed_sinks:
            self.failed_sinks.add(name)
            logger.warning("Monitor %s output failed; retaining local JSONL: %s", name, exc)

    def _deliver(self, wandb: Any, run: Any, tensorboard: Any, record: dict) -> None:
        scalars = self._scalars(record)
        if run is not None:
            try:
                self._write_wandb(wandb, run, record, scalars)
            except Exception as exc:
                self._sink_failure("wandb", exc)
        if tensorboard is not None:
            try:
                self._write_tensorboard(tensorboard, record, scalars)
            except Exception as exc:
                self._sink_failure("tensorboard", exc)

    def _run(self) -> None:
        wandb, run, tensorboard = self._open_sinks()
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._deliver(wandb, run, tensorboard, record)
                finally:
                    self.queue.task_done()
        finally:
            for name, sink in (("wandb", run), ("tensorboard", tensorboard)):
                if sink is not None:
                    try:
                        if name == "wandb":
                            sink.finish()
                        else:
                            sink.close()
                    except Exception as exc:
                        self._sink_failure(name, exc)

    def close(self) -> None:
        """Bound shutdown latency even if a remote SDK stops responding."""
        if self.closed:
            return
        self.closed = True
        if self.file is not None:
            self.file.close()
        if self.worker is not None:
            self.stop.set()
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                logger.warning("Monitor remote shutdown timed out; local records are persisted")
