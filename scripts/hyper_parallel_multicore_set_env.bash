#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Stable wheel-installed locator for the packaged multicore custom OPP environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: this script must be sourced so it can update the calling shell." >&2
    echo "Run: source \"\$(command -v hyper_parallel_multicore_set_env.bash)\"" >&2
    exit 1
fi

_hp_multicore_python=$(command -v python3 2>/dev/null || true)
if [[ -z "${_hp_multicore_python}" ]]; then
    echo "[HP-NATIVE-PYTHON-NOT-FOUND] python3 is not available in the active PATH." >&2
    unset _hp_multicore_python
    return 1
fi

_hp_multicore_set_env=$("${_hp_multicore_python}" -c '
import importlib.util
import os
from pathlib import Path
import sys

# Do not let the shell working directory shadow an installed wheel. Keep
# explicit PYTHONPATH entries so pip --target installations remain supported.
cwd = Path.cwd().resolve()
explicit_pythonpath = {
    Path(entry).resolve()
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if entry
}
sys.path[:] = [
    entry
    for entry in sys.path
    if entry and (Path(entry).resolve() != cwd or cwd in explicit_pythonpath)
]
spec = importlib.util.find_spec("hyper_parallel")
if spec is None or spec.origin is None:
    raise SystemExit("hyper_parallel is not installed in the active Python environment")
print(Path(spec.origin).resolve().parent / "core" / "multicore" / "lib" / "set_env.bash")
' 2>&1)
_hp_multicore_status=$?
if [[ ${_hp_multicore_status} -ne 0 ]]; then
    echo "[HP-NATIVE-PACKAGE-NOT-FOUND] ${_hp_multicore_set_env}" >&2
    unset _hp_multicore_python _hp_multicore_set_env _hp_multicore_status
    return 1
fi
if [[ ! -f "${_hp_multicore_set_env}" ]]; then
    echo "[HP-NATIVE-PAYLOAD-MISSING] the installed wheel does not contain multicore/HyperMegaMoe." >&2
    echo "Expected activation script: ${_hp_multicore_set_env}" >&2
    echo "Check the wheel build warning or install a wheel built with multicore enabled." >&2
    unset _hp_multicore_python _hp_multicore_set_env _hp_multicore_status
    return 1
fi

source "${_hp_multicore_set_env}"
_hp_multicore_status=$?
unset _hp_multicore_python _hp_multicore_set_env
return "${_hp_multicore_status}"
