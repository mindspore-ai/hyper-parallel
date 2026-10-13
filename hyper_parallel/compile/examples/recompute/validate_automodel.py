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
"""Record exact per-rank parity evidence using the existing AutoModel TextTrainer.

Set VALIDATION_OUTPUT to a directory. Optionally set VALIDATION_REFERENCE to a
baseline directory; loss uses absolute tolerance 1e-5, while gradient norms,
gradient bytes and updated parameter bytes must match exactly. CPU copies/hashing make this unsuitable
for throughput benchmarking. Run under torchrun with the usual Trainer YAML.
"""

# pylint: disable=forbidden-backend-import
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def _fingerprint(model: torch.nn.Module, gradient: bool = False) -> dict[str, Any]:
    """Hash every local tensor with its shape and dtype; reject nonfinite data."""
    result = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradient else parameter
        if tensor is None:
            result[name] = None
            continue
        tensor = tensor.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        tensor = tensor.cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"Nonfinite tensor: {name}")
        result[name] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    if not any(value is not None for value in result.values()):
        raise AssertionError("No tensors were observed")
    return result


def main() -> None:
    """Run the normal trainer and record gradients before clipping and step results."""
    config = parse_training_args()
    output = Path(os.environ["VALIDATION_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    trainer = TextTrainer(config)
    base = trainer.base
    rank = int(os.environ["RANK"])
    records = []
    original_prepare = base.prepare_optimizer_step
    original_step = trainer.train_step
    device_api = getattr(torch, base.device.type)

    def prepare() -> Any:
        """Inspect gradients before the standard clipping path."""
        records[-1]["gradients"] = _fingerprint(base.model, gradient=True)
        return original_prepare()

    def step(iterator: Any) -> dict[str, float]:
        """Run a standard trainer step and persist its verification evidence."""
        record = {"step": base.state.global_step + 1}
        records.append(record)
        device_api.reset_peak_memory_stats()
        record["metrics"] = original_step(iterator)
        if not all(math.isfinite(value) for value in record["metrics"].values()):
            raise AssertionError("Nonfinite training metrics")
        device_api.synchronize()
        record["peak_allocated_bytes"] = device_api.max_memory_allocated()
        record["parameters"] = _fingerprint(base.model)
        payload = {"rank": rank, "recompute_report": base.graph_compiler.recompute_report, "steps": records}
        (output / f"rank{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return record["metrics"]

    base.prepare_optimizer_step = prepare
    trainer.train_step = step
    trainer.train()
    if len(records) != config.training.train_iters:
        raise AssertionError("Training ended before the requested number of steps")
    report = base.graph_compiler.recompute_report
    if base.graph_compiler.pass_config.recompute.mode != "none" and not report["recomputed_nodes"]:
        raise AssertionError("The selected recompute policy produced no duplicated nodes")
    reference_dir = os.environ.get("VALIDATION_REFERENCE")
    if reference_dir:
        reference = json.loads((Path(reference_dir) / f"rank{rank}.json").read_text(encoding="utf-8"))
        if len(reference["steps"]) != len(records):
            raise AssertionError("Reference step count differs")
        for expected, actual in zip(reference["steps"], records):
            for key in ("step", "gradients", "parameters"):
                if expected[key] != actual[key]:
                    raise AssertionError(f"Rank {rank}, step {actual['step']}: {key} differs from reference")
            if expected["metrics"]["grad_norm"] != actual["metrics"]["grad_norm"]:
                raise AssertionError(f"Rank {rank}, step {actual['step']}: gradient norm differs")
            if not math.isclose(expected["metrics"]["loss"], actual["metrics"]["loss"], rel_tol=0, abs_tol=1e-5):
                raise AssertionError(f"Rank {rank}, step {actual['step']}: loss differs beyond atol=1e-5")
    print(f"VALIDATION_COMPLETE rank={rank} steps={len(records)}", flush=True)


if __name__ == "__main__":
    main()
