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
"""Check a model before its layouts are ranked: what ND prices for each part of a layer, beside the model's own.

Run from anywhere, with the Python that has the checkout's dependencies:

    python .agent/skills/sapp-parallel-strategy/scripts/check.py --family deepseek_v41 \\
        --request "the request, in the asker's words"

The run is described as ``ask.py`` describes it, and the check is kept in its run folder (``check.log``, with the
commands in ``command.txt``). The DeepSeek-V4.1 crop, which Transformers does not know, is checked by its own tool on
the V4.1 branch: its description written from the released config at the crop's sizes, then that description beside
the crop's own tensors. Any other model goes through ND's verify mode, which sets each layer kind's parameters and
FLOPs beside those of the layers Transformers builds from its config.
"""
import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ask

V41_TOOL = Path("examples", "training_demo", "deepseek_v41", "nd_describe_deepseek_v41.py")
# One step of a check: its command, and the file its standard output is written to (None: the log).
Step = Tuple[List[str], Optional[Path]]


def _deepseek_v41(doc: Dict[str, Any], run: Path, repo: Path) -> List[Step]:
    """The V4.1 crop's check: its description at the crop's sizes, then that description beside the crop's tensors."""
    model = doc["model"]
    if not (repo / V41_TOOL).is_file() or "engram_assets_path" not in model:
        raise SystemExit(f"the DeepSeek-V4.1 check needs {V41_TOOL.as_posix()} and the model the V4.1 launcher "
                         "builds, neither of which is in this checkout")
    tool = [sys.executable, str(repo / V41_TOOL)]
    sizes = ["--divisor", str(model["text_parameter_divisor"]), "--experts", str(model["num_routed_experts"])]
    return [(tool + ["spec", model["config_path"], *sizes], run / "spec.yaml"),
            (tool + ["verify", model["config_path"], "--assets", model["engram_assets_path"], *sizes,
                     "--spec-yaml", str(run / "spec.yaml")], None)]


def _nd_verify(doc: Dict[str, Any], run: Path, repo: Path) -> List[Step]:
    """ND's own model reader, on the yaml the run folder keeps."""
    del doc, repo
    return [([sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", "-f", "hyper_v2",
              "-y", str(run / "run.yaml"), "-V", "-o", str(run / "verify")], None)]


# A family whose model ND's verify mode cannot read names its own check here.
CHECKS: Dict[str, Callable[[Dict[str, Any], Path, Path], List[Step]]] = {"deepseek_v41": _deepseek_v41}


def _run(steps: Sequence[Step], repo: Path, run: Path, timeout: int) -> int:
    """Run each step from the checkout, every message kept in ``check.log``: the first non-zero exit code, or 0."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with open(run / "check.log", "w", encoding="utf-8") as log:
        for command, out in steps:
            log.write("$ " + " ".join(command) + (f" > {out}" if out else "") + "\n")
            log.flush()
            with contextlib.ExitStack() as stack:
                stdout = stack.enter_context(open(out, "w", encoding="utf-8")) if out else log
                code = subprocess.run(command, cwd=repo, env=env, stdout=stdout,
                                      stderr=log if out else subprocess.STDOUT, timeout=timeout, check=False).returncode
            if code:
                return code
    return 0


def _report(run: Path) -> List[str]:
    """What the check found: the parts the description leaves out and the table for the V4.1 tool, else ND's output."""
    lines = (run / "check.log").read_text(encoding="utf-8", errors="replace").splitlines()
    spec = run / "spec.yaml"
    if spec.is_file():
        unpriced = [line for line in spec.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
        start = next((index for index, line in enumerate(lines) if line.startswith("ND's spec:")), len(lines))
        return unpriced + [""] + lines[start:]
    return [line.split(" - OUTPUT - ", 1)[1] for line in lines if " - OUTPUT - " in line]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Describe the run, check its model, keep the check, print what it found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ask.add_run_arguments(parser)
    args = parser.parse_args(argv)
    if not (args.family or args.yaml):
        parser.error("give --family, or --yaml with the asker's trainer yaml")
    args.recompute = []  # the check prices no layout
    repo = ask.repo_root()
    notes: List[str] = []
    doc = ask.describe(args, notes)
    run = ask.run_folder(args, repo, doc)
    steps = CHECKS.get(args.family or "", _nd_verify)(doc, run, repo)
    (run / "command.txt").write_text(f"cd {repo}\n" + "".join(" ".join(command) + (f" > {out}" if out else "") + "\n"
                                                             for command, out in steps), encoding="utf-8")
    code = _run(steps, repo, run, args.timeout)
    stopped = []
    if code:  # what the check printed before it stopped still holds; the log's last line says why it stopped
        last = (run / "check.log").read_text(encoding="utf-8", errors="replace").splitlines()[-1:]
        stopped = ["", f"The check stopped (exit {code}) before its end: {' '.join(last)}"]
    sys.stdout.write("\n".join(notes + _report(run) + stopped + ["", f"Check kept in {run}"]) + "\n")
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
