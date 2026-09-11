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
"""indexed: indexed (.idx/.bin) dataset implementation and C++ helpers.

Submodules are imported by full path; this package keeps no flat re-exports.
Offline indexed I/O tools live in ``hyper_parallel.data.tools.io``.
"""

from pathlib import Path


def _extend_native_path(package_file: str, package_path: list[str]) -> None:
    """Make the source-build helper visible without copying binaries into source."""
    repository = Path(package_file).resolve().parents[3]
    native = repository / "build/native/payload/hyper_parallel/data/indexed"
    if (repository / "setup.py").is_file() and native.is_dir() and str(native) not in package_path:
        package_path.append(str(native))


_extend_native_path(__file__, __path__)
