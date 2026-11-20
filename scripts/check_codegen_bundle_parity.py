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
"""Check fresh CLI, Trainer-sidecar and tracked Codegen bundles byte for byte."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def bundle_hashes(directory: Path) -> dict[str, str]:
    """Hash the regular bundle files and reject transient publication directories."""
    if not directory.is_dir():
        raise ValueError(f"Missing bundle directory {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink() or path.name.endswith(".lock") or path.name.startswith((".codegen_", ".staging")):
            raise ValueError(f"Unexpected bundle artifact {path}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path.read_bytes()).hexdigest()
    modeling = [name for name in result if name.startswith("modeling_") and name.endswith("_gen_npu.py")]
    if len(modeling) != 1 or "__init__.py" not in result or f"{modeling[0]}.diff" not in result:
        raise ValueError(f"Incomplete modeling bundle {directory}")
    return result


def _read_tracked(root: Path) -> set[str]:
    """Use only checkout-owned artifacts and recipes when discovering scope."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "hyper_parallel/models", "examples/training_demo"], cwd=root,
    )
    return set(output.decode("utf-8").rstrip("\0").split("\0"))


def _recipes(tracked: set[str]) -> dict[str, set[str]]:
    patterns = ("hyper_parallel/models/", "examples/training_demo/")
    recipes: dict[str, set[str]] = {}
    for path in tracked:
        if not path.endswith((".yaml", ".yml")):
            continue
        for prefix in patterns:
            if path.startswith(prefix):
                model, _, relative = path[len(prefix):].partition("/")
                if prefix == patterns[1] or relative.startswith("recipes/"):
                    recipes.setdefault(model, set()).add(path)
    return recipes


def _verify_model(
    row: dict, recipes: set[str], tracked_bundle: Path, cli_bundle: Path, sidecar_bundle: Path,
) -> None:
    """Compare the published files for one scoped model."""
    name = row["model"]
    if row["recipe"] not in recipes or row["cli_exit"] != 0 or row["sidecar"] != "published":
        raise ValueError(f"Unverified CLI or Trainer-sidecar result for {name}")
    native = bundle_hashes(tracked_bundle)
    if native != bundle_hashes(cli_bundle) or native != bundle_hashes(sidecar_bundle):
        raise ValueError(f"CLI, Trainer-sidecar or tracked bundle differs for {name}")
    if native != row["cli_files"] or native != row["sidecar_files"]:
        raise ValueError(f"Recorded SHA256 differs from the files for {name}")
    print(f"{name}: {len(native)} files match")


def verify_matrix(root: Path, cli: Path, sidecar: Path, manifest: Path) -> None:
    """Verify the three artifact trees against the discovered tracked model scope."""
    tracked = _read_tracked(root)
    candidates = {
        path.split("/")[2] for path in tracked if path.startswith("hyper_parallel/models/")
        and "/generated/modeling_" in path and path.endswith("_gen_npu.py")
    }
    recipes = _recipes(tracked)
    scope = candidates.intersection(recipes)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    names = {row["model"] for row in rows}
    if not scope or names != scope or len(rows) != len(scope):
        raise ValueError(f"Expected one verification row for {sorted(scope)}, got {sorted(names)}")
    for row in rows:
        name = row["model"]
        tracked_bundle = root / "hyper_parallel" / "models" / name / "generated"
        _verify_model(row, recipes[name], tracked_bundle, cli / name, sidecar / name)


def main() -> None:
    """Run the model parity gate from a clean checkout and a verification report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--cli-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    verify_matrix(args.repository, args.cli_root, args.sidecar_root, args.manifest)


if __name__ == "__main__":
    main()
