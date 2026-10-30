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
"""Exercise a wheel installation without importing the source checkout."""

import argparse
from email.parser import BytesParser
from importlib import metadata
import json
import logging
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile

import numpy as np
import torch
import yaml

import hyper_parallel
from hyper_parallel.data.indexed import _indexed_helpers_cpp


LOGGER = logging.getLogger(__name__)
RESOURCE_PATTERNS = (
    "core/shard/ops/yaml/*.yaml",
    "auto_parallel/op_profiles/*.yaml",
    "auto_parallel/op_records.yaml",
    "auto_parallel/sapp_nd/memory_estimation/configs_eval/default.yaml",
    "auto_parallel/sapp_nd/nd/common/framework_parsers/mapping.yaml",  # codespell:ignore nd
    "models/qwen3_moe/recipes/*.yaml",
    "models/deepseek_v3/recipes/*.yaml",
)


def _check_origin(module: ModuleType, package_root: Path, source_root: Path) -> None:
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(package_root) or origin.is_relative_to(source_root):
        raise RuntimeError(f"Expected an installed module, got {origin}")
    LOGGER.info("Installed module: %s", origin)


def _check_distribution(wheel: Path, source_root: Path) -> Path:
    distribution = metadata.distribution("hyper_parallel")
    package_root = Path(distribution.locate_file("hyper_parallel")).resolve()
    if package_root.is_relative_to(source_root) or not package_root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Package must be installed into the isolated environment: {package_root}")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable installations cannot validate a release wheel")
    with ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(records) != 1:
            raise RuntimeError("Expected exactly one wheel metadata record")
        wheel_metadata = BytesParser().parsebytes(archive.read(records[0]))
    if distribution.version != wheel_metadata["Version"]:
        raise RuntimeError("Installed version does not match the wheel being validated")
    _check_origin(hyper_parallel, package_root, source_root)
    _check_origin(_indexed_helpers_cpp, package_root, source_root)
    LOGGER.info("Package version: %s; PyTorch: %s", distribution.version, torch.__version__)
    return package_root


def _check_resources(package_root: Path, source_root: Path) -> None:
    source_package = source_root / "hyper_parallel"
    count = 0
    for pattern in RESOURCE_PATTERNS:
        resources = sorted(source_package.glob(pattern))
        if not resources:
            raise RuntimeError(f"No source resources match {pattern}; update the installation contract if intentional")
        for source in resources:
            installed = package_root / source.relative_to(source_package)
            if not installed.is_file() or installed.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Missing or changed package resource: {installed}")
            with installed.open(encoding="utf-8") as resource_file:
                yaml.safe_load(resource_file)
            count += 1
    LOGGER.info("Validated %d packaged YAML resources", count)


def _check_cpu_model() -> None:
    torch.manual_seed(0)
    with hyper_parallel.init_empty_weights():
        empty_model = torch.nn.Linear(4, 2)
    if not all(parameter.is_meta for parameter in empty_model.parameters()):
        raise RuntimeError("Empty model initialization did not create meta parameters")
    with hyper_parallel.init_on_device(torch.device("cpu")):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    loss = model(inputs).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    ):
        raise RuntimeError("CPU forward/backward produced missing or non-finite gradients")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    if all(torch.equal(previous, current) for previous, current in zip(before, model.parameters())):
        raise RuntimeError("CPU optimizer step did not update any parameters")
    LOGGER.info("CPU initialization, forward, backward and optimizer step passed")


def _check_native_dataset_helper() -> None:
    # Import the compiled module directly above so a source JIT build cannot hide a missing payload.
    dataset_index = np.empty(4, dtype=np.int16)
    sample_index = np.empty(4, dtype=np.int64)
    _indexed_helpers_cpp.build_blending_indices(dataset_index, sample_index, np.array([1.0], dtype=np.float64))
    np.testing.assert_array_equal(dataset_index, np.zeros(4, dtype=np.int16))
    np.testing.assert_array_equal(sample_index, np.arange(4, dtype=np.int64))
    LOGGER.info("Installed indexed dataset C++ extension passed")


def main() -> None:
    """Validate wheel metadata, resources and CPU behavior in an isolated interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    source_root = arguments.source_root.resolve()
    if not sys.flags.isolated or Path.cwd().resolve().is_relative_to(source_root):
        raise RuntimeError("Run with python -I from a directory outside the source checkout")
    if any(Path(entry).resolve().is_relative_to(source_root) for entry in sys.path):
        raise RuntimeError("The source checkout must not be on the import path")
    package_root = _check_distribution(arguments.wheel.resolve(), source_root)
    _check_resources(package_root, source_root)
    _check_cpu_model()
    _check_native_dataset_helper()
    LOGGER.info("Package installation validation passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
