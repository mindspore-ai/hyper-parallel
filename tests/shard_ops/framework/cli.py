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
"""CLI for local reproduction — pytest-style and flag-style.

Pytest-style (file path, with optional ::case_name)::

    python -m tests.shard_ops.framework tests/torch/.../case_sort.py
    python -m tests.shard_ops.framework tests/torch/.../case_sort.py::sort_2d_dp_last_dim

Flag-style::

    python -m tests.shard_ops.framework --case sort_2d_dp_last_dim
    python -m tests.shard_ops.framework --case-glob "sort_*" --tag sort
    python -m tests.shard_ops.framework --framework mindspore --case sort_2d_dp_last_dim

framework is auto-derived from the file path, ``HYPER_PARALLEL_PLATFORM``
env var, or defaults to ``torch``.  ``--device-type`` defaults from
``HYPER_PARALLEL_TEST_DEVICE_TYPE`` or the framework default (cpu for
torch, npu for mindspore).
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

from tests.shard_ops.framework.case_spec import OpShardCase
from tests.shard_ops.framework.registry import load_cases_from_package
from tests.shard_ops.framework.runner import RUNNER
from tests.shard_ops.framework.utils import match_filter

_FW_CFG: Dict[str, dict] = {
    "torch": {
        "cases_pkg": "tests.torch.shard.ops.cases",
        "platform_pkg": ["tests.torch.shard.ops.framework"],
        "device_type": "cpu",
    },
    "mindspore": {
        "cases_pkg": "tests.mindspore.st.shard.ops.cases",
        "platform_pkg": ["tests.mindspore.st.shard.ops.framework"],
        "device_type": "npu",
    },
}


def _parse_target(target: Optional[str]) -> dict:
    """Parse a pytest-style target: ``[PATH][::CASE_NAME]``.

    Returns a dict with keys ``framework``, ``cases_pkg``, ``module_filter``
    and ``case_name``.  Unresolved keys are ``None``.
    """
    result: dict = {"framework": None, "cases_pkg": None,
                    "module_filter": None, "case_name": None}
    if not target:
        return result

    path, sep, case = target.partition("::")
    result["case_name"] = case if sep else None
    file_path = path if path and path != "::" else None

    if file_path:
        result["framework"] = _framework_from_path(file_path)
        result["cases_pkg"] = _cases_pkg_from_framework(result["framework"])
        result["module_filter"] = _module_from_path(file_path)
    return result


def _framework_from_path(file_path: str) -> str:
    """Derive framework from a file path like ``.../torch/...``."""
    parts = file_path.replace("\\", "/").split("/")
    if "mindspore" in parts:
        return "mindspore"
    return "torch"


def _cases_pkg_from_framework(framework: str) -> str:
    return _FW_CFG[framework]["cases_pkg"]


def _module_from_path(file_path: str) -> str:
    """Convert ``tests/torch/shard/ops/cases/case_sort.py`` → module name."""
    clean = file_path.replace("\\", "/").rstrip("/")
    if clean.endswith(".py"):
        clean = clean[:-3]
    return clean.replace("/", ".")


def _filter(cases: List[OpShardCase], args,
            module_filter: Optional[str] = None) -> List[OpShardCase]:
    """Filter cases by name, glob, tag, and source module."""
    exact = set(args.case) if args.case else None
    selected = []
    for c in cases:
        if module_filter and c.source_module != module_filter:
            continue
        if match_filter(c.name, c.tags,
                        exact=exact,
                        glob_pat=args.case_glob,
                        tag=args.tag):
            selected.append(c)
    return selected


def _resolve_config(args, target_info: dict) -> dict:
    """Resolve framework, device_type, cases_pkg, platform_pkg."""
    framework = target_info["framework"] or \
                os.environ.get("HYPER_PARALLEL_PLATFORM") or \
                args.framework

    cfg = _FW_CFG[framework]
    if args.device_type is not None:
        device_type = args.device_type
    else:
        device_type = os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE") or \
                      cfg["device_type"]

    cases_pkg = target_info["cases_pkg"] or args.cases_pkg or cfg["cases_pkg"]
    platform_pkg = args.platform_pkg or cfg["platform_pkg"]
    return {"framework": framework, "device_type": device_type,
            "cases_pkg": cases_pkg, "platform_pkg": platform_pkg}


def main(argv: List[str] = None) -> int:
    """CLI entry point: parse args, filter cases, drive ``RUNNER.run_single``."""
    p = argparse.ArgumentParser(
        prog="python -m tests.shard_ops.framework",
        description="Local shard-ops case runner — pytest-style target or flags.",
    )
    p.add_argument("target", nargs="?", default=None)
    p.add_argument("--framework", default="torch",
                   choices=["torch", "mindspore"])
    p.add_argument("--device-type", default=None, choices=["npu", "cpu"])
    p.add_argument("--cases-pkg", default=None)
    p.add_argument("--platform-pkg", action="append", default=[])
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--case-glob", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--repeat", type=int, default=1)

    args = p.parse_args(argv)
    target_info = _parse_target(args.target)
    config = _resolve_config(args, target_info)

    for pkg in config["platform_pkg"]:
        __import__(pkg)

    cases = load_cases_from_package(config["cases_pkg"])

    if target_info["case_name"] and target_info["case_name"] not in args.case:
        args.case.insert(0, target_info["case_name"])

    if not (args.case or args.case_glob or args.tag or target_info["module_filter"]):
        print("WARN: no filter given — running all cases, one launcher per "
              "case. This is slow; use --case or a pytest-style target.",
              file=sys.stderr)

    selected = _filter(cases, args, module_filter=target_info["module_filter"])
    if not selected:
        print("no case matched", file=sys.stderr)
        return 2

    for _ in range(args.repeat):
        for case in selected:
            RUNNER.run_single(
                case,
                framework=config["framework"],
                device_type=config["device_type"],
                num_proc=args.num_proc,
                cases_pkg=config["cases_pkg"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
