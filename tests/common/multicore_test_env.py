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
"""Shared environment helpers for multicore system tests."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANN_SET_ENV = "/usr/local/Ascend/cann/set_env.sh"
HP_ACTIVATION_SCRIPT = "hyper_parallel_multicore_set_env.bash"
HP_ACTIVATION_VARIABLES = ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
INHERITED_RANK_VARIABLES = ("RANK_TABLE_FILE", "RANK_ID", "RANK_SIZE")


def multicore_test_environment_is_active() -> bool:
    """Return whether the current process already has the packaged OPP paths."""
    vendor_roots = [
        Path(value).resolve()
        for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if value and Path(value).name == "hyper_parallel_multicore_nn"
    ]
    library_roots = {
        Path(value).resolve()
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    }
    return any(
        (vendor_root / "op_api" / "lib" / "libcust_opapi.so").is_file()
        and (vendor_root / "op_api" / "lib").resolve() in library_roots
        for vendor_root in vendor_roots
    )


def multicore_activation_scripts() -> list[Path]:
    """Return wheel and source-build activation candidates in priority order."""
    scripts = []
    installed_script = shutil.which(HP_ACTIVATION_SCRIPT)
    if installed_script:
        scripts.append(Path(installed_script))
    source_script = (
        PROJECT_ROOT
        / "build"
        / "native"
        / "payload"
        / "hyper_parallel"
        / "core"
        / "multicore"
        / "lib"
        / "set_env.bash"
    )
    if source_script.is_file() and source_script not in scripts:
        scripts.append(source_script)
    return scripts


def prepare_multicore_test_environment() -> None:
    """Validate CANN and activate the wheel or PYTHONPATH multicore environment."""
    required_cann_variables = ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH")
    missing_cann_variables = [name for name in required_cann_variables if not os.environ.get(name)]
    if missing_cann_variables:
        missing_text = ",".join(missing_cann_variables)
        raise RuntimeError(
            f"[HP-NATIVE-CANN-ENV-NOT-ACTIVATED] missing={missing_text}. "
            f"Run before pytest: source {DEFAULT_CANN_SET_ENV}"
        )

    if multicore_test_environment_is_active():
        return

    activation_scripts = multicore_activation_scripts()
    if not activation_scripts:
        raise RuntimeError(
            f"[HP-NATIVE-ACTIVATION-SCRIPT-NOT-FOUND] {HP_ACTIVATION_SCRIPT} is absent from PATH and the "
            "source-build payload is unavailable. Run ./build.sh, then activate the wheel or PYTHONPATH payload."
        )
    activation_errors = []
    for activation_script in activation_scripts:
        activation_command = f"source {activation_script}"
        result = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null && env -0', "bash", str(activation_script)],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            activation_errors.append(f"command={activation_command} error={error}")
            continue
        activated_environment = {}
        for entry in result.stdout.split(b"\0"):
            name, separator, value = entry.partition(b"=")
            if separator:
                activated_environment[name.decode("utf-8")] = value.decode("utf-8")
        for variable in HP_ACTIVATION_VARIABLES:
            if variable in activated_environment:
                os.environ[variable] = activated_environment[variable]
        if multicore_test_environment_is_active():
            return
        activation_errors.append(f"command={activation_command} error=vendor paths were not activated")

    raise RuntimeError(
        "[HP-NATIVE-OPP-ACTIVATION-FAILED] " + "; ".join(activation_errors)
    )


@contextmanager
def without_inherited_rank_environment() -> Iterator[None]:
    """Prevent explicit local msrun cases from inheriting stale cluster rank metadata."""
    inherited = {
        variable: os.environ[variable]
        for variable in INHERITED_RANK_VARIABLES
        if variable in os.environ
    }
    for variable in INHERITED_RANK_VARIABLES:
        os.environ.pop(variable, None)
    try:
        yield
    finally:
        for variable in INHERITED_RANK_VARIABLES:
            os.environ.pop(variable, None)
        os.environ.update(inherited)
