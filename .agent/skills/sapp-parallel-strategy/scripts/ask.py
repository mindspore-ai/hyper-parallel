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
"""Ask ND for a layout: describe the run, price every valid layout once, keep the run, print the answer.

Run from the HyperParallel checkout whose ND should answer, with the Python that has its dependencies:

    python .agent/skills/sapp-parallel-strategy/scripts/ask.py --family qwen3_5_moe --cluster a3 \\
        --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 --recompute full \\
        --request "the request, in the asker's words"

``--yaml <trainer yaml>`` describes the run with the asker's own config instead of a template. Everything
lands in ``output/sapp-parallel-strategy/<time>/``: the request, the yaml ND read, the command, the versions,
ND's log and ranking, and the answer, so that any answer can be checked or run again.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

import answer

SKILL = Path(__file__).resolve().parents[1]
MODES = ("off", "selective", "full")
# Request field -> where the trainer yaml states it.
FIELDS = {
    "layers": ("model", "num_hidden_layers"),
    "seq": ("dataset", "data_config", "seq_length"),
    "global_batch": ("training", "global_batch_size"),
    "micro_batch": ("training", "micro_batch_size"),
    "dies": ("context", "device_num"),
}
# Where a trainer yaml states the tokens a sequence, in the order ND reads them.
SEQ_PLACES = (("dataset", "data_transform", "max_seq_len"), ("dataset", "data_config", "seq_length"))
# <family>_description.yaml: the family's model described by hand, for a checkout that cannot build the template's.
DESCRIBED = "_description"
# ND log lines the answer must pass on: a part it did not price, a field it priced as 0, what it refused.
NOTICES = ("MTP layer", "no parser set", "priced fully recomputed", "refus", "does not declare", "Error",
           "WARNING")


def _is_checkout(folder: Path) -> bool:
    """True when ``folder`` is a checkout of the repository."""
    return (folder / "hyper_parallel" / "__init__.py").is_file()


def repo_root() -> Path:
    """The checkout ND is run from: the one the environment names, the installed copy's, or this skill's own.

    A linked skill resolves back into its checkout, so walking up finds it. A copied one cannot,
    and ``install.py --copy`` leaves the checkout's path in ``checkout.txt`` beside the skill.
    """
    stated = os.environ.get("HYPER_PARALLEL_ROOT")
    if stated:
        folder = Path(stated).expanduser()
        if _is_checkout(folder):
            return folder
        raise SystemExit(f"HYPER_PARALLEL_ROOT={stated} is not a HyperParallel checkout")
    pointer = SKILL / "checkout.txt"
    if pointer.is_file():
        folder = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        if _is_checkout(folder):
            return folder
        raise SystemExit(f"{pointer} names {folder}, which is not a HyperParallel checkout")
    for folder in SKILL.parents:
        if _is_checkout(folder):
            return folder
    raise SystemExit(f"no HyperParallel checkout above {SKILL}: set HYPER_PARALLEL_ROOT to one, "
                     "or install the skill again with install.py")


def _set(doc: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set ``doc[path[0]][path[1]]...`` to value, making the missing levels."""
    for key in path[:-1]:
        doc = doc.setdefault(key, {})
    doc[path[-1]] = value


def _get(doc: Dict[str, Any], path: Sequence[str]) -> Any:
    """``doc[path[0]][path[1]]...``, None when a level is missing."""
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def _states(doc: Dict[str, Any], path: Sequence[str]) -> bool:
    """Whether the yaml states the field at *path*, a null included."""
    parent = _get(doc, path[:-1])
    return isinstance(parent, dict) and path[-1] in parent


def _in_checkout(repo: Path, module: str) -> bool:
    """Whether the dotted *module* is a file or a package of the checkout ND runs from."""
    base = repo.joinpath(*module.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def template_for(family: str, repo: Path) -> Tuple[Path, Optional[str]]:
    """The family's template, and the module this checkout lacks when the model described by hand stands in for it.

    A template whose model a factory builds needs the factory's module in the checkout ND runs from. Where the
    module is missing and the family ships ``<family>_description.yaml``, that model, described by hand, is priced.
    """
    template = SKILL / "templates" / f"{family}.yaml"
    if not template.is_file():
        known = sorted(path.stem for path in (SKILL / "templates").glob("*.yaml") if not path.stem.endswith(DESCRIBED))
        raise SystemExit(f"no template for {family!r} (templates: {known}); "
                         "pass the asker's trainer yaml with --yaml, after the model reader checked it")
    model = yaml.safe_load(template.read_text(encoding="utf-8"))["model"]
    module = str(model.get("_target_") or "").rpartition(".")[0]
    described = template.with_name(f"{family}{DESCRIBED}.yaml")
    if module and not _in_checkout(repo, module) and described.is_file():
        return described, module
    return template, None


def _resolve_paths(model: Dict[str, Any], folder: Path) -> None:
    """Point each relative ``*_path`` of a template's model section at the file shipped beside the template."""
    for key, value in model.items():
        if key.endswith("_path") and isinstance(value, str) and not Path(value).is_absolute():
            shipped = folder / value
            if not shipped.exists():
                raise SystemExit(f"the template's model.{key} names {value}, which is not shipped in {folder}")
            model[key] = shipped.as_posix()


def describe(args: argparse.Namespace, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """The trainer yaml ND reads: the template or the asker's yaml, with every field the request states.

    When this checkout cannot build the template's model and the family's model described by hand is priced
    instead, a note saying so, with the trust record's measure of what that costs, goes to *notes*.
    """
    if args.yaml:
        doc = yaml.safe_load(Path(args.yaml).read_text(encoding="utf-8"))
    else:
        template, missing = template_for(args.family, repo_root())
        doc = yaml.safe_load(template.read_text(encoding="utf-8"))
        _resolve_paths(doc["model"], template.parent)
        if missing and notes is not None:
            cost = answer.trust_entry(args.family, args.cluster).get("described_note", "")
            notes.append(f"This checkout has no {missing}, so ND prices the model described by hand in "
                         f"{template.name}, its memory by formulas alone. {cost}".strip())
    target = _get(doc, ("model", "_target_"))
    for name, path in FIELDS.items():
        if getattr(args, name) is None:
            continue
        where = _where(doc, path)
        if where[0] == "model" and target and not _states(doc, where):
            raise SystemExit(f"--{name.replace('_', '-')}: {target} builds this model and the yaml gives it no "
                             f"{where[-1]}, so the factory decides it and the request cannot change it here")
        _set(doc, where, getattr(args, name))
    if len(args.recompute) == 1 and args.recompute[0] in MODES:
        _set(doc, ("activation_checkpoint", "mode"), args.recompute[0])
    return doc


def _where(doc: Dict[str, Any], path: Sequence[str]) -> Sequence[str]:
    """Where this yaml states a field: a model described inline states its shape under config_overrides, and the
    tokens a sequence go where ND reads them first."""
    if path[0] == "model" and _get(doc, ("model", "config_overrides", *path[1:])) is not None:
        return ("model", "config_overrides", *path[1:])
    if tuple(path) == FIELDS["seq"]:
        return next((place for place in SEQ_PLACES if _states(doc, place)), path)
    return path


def _modes(asked: Sequence[str], trust: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """The recompute modes to search, and why any the trust record names were left out or kept."""
    excluded = trust.get("exclude_modes") or {}
    if list(asked) == ["auto"]:
        modes = [mode for mode in MODES if mode not in excluded]
        return modes, [f"Left out of the search: {mode}, because {why}." for mode, why in excluded.items()]
    return list(asked), [f"{mode} was asked for; mind that {why}." for mode, why in excluded.items() if mode in asked]


def library_versions(repo: Path, ratios: Optional[Path]) -> Dict[str, Any]:
    """What the answer depends on besides the request: the ND tree, the libraries, the ratios."""
    def git(*command: str) -> str:
        """What a git command prints in the checkout, empty when git fails."""
        return subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True,
                              check=False).stdout.strip()

    def version(name: str) -> str:
        """The installed version of a distribution, or "not installed"."""
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    return {
        "nd_commit": git("rev-parse", "HEAD"),
        "nd_tree_changed": bool(git("status", "--porcelain", "--", "hyper_parallel", "examples")),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ratios": ratios.as_posix() if ratios else None,
        "ratios_sha256": hashlib.sha256(ratios.read_bytes()).hexdigest() if ratios else None,
    }


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that describe a run, shared by ask.py and plan.py."""
    parser.add_argument("--family", default=None,
                        help="template name, for example qwen3_5_moe; with --yaml it only picks the status line")
    parser.add_argument("--yaml", default=None, help="the asker's trainer yaml, instead of the family's template")
    parser.add_argument("--cluster", default="a3", help="cluster key in calibration/clusters.json")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None, help="tokens a sequence")
    parser.add_argument("--global-batch", dest="global_batch", type=int, default=None, help="sequences a step")
    parser.add_argument("--micro-batch", dest="micro_batch", type=int, default=None, help="sequences a die")
    parser.add_argument("--dies", type=int, default=None, help="total dies")
    parser.add_argument("--request", default="", help="the request in the asker's words, kept with the run")
    parser.add_argument("--out", default=None, help="run folder (default output/sapp-parallel-strategy/<time>)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds ND may take")


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The request's values, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_arguments(parser)
    parser.add_argument("--recompute", nargs="+", default=["auto"], choices=MODES + ("auto",),
                        help="the activation checkpoint mode the run uses, several to compare, auto for all three")
    parser.add_argument("--budget", default=None, help="memory budget a die, for example 58GB")
    args = parser.parse_args(argv)
    if not (args.family or args.yaml):
        parser.error("give --family, or --yaml with the asker's trainer yaml")
    return args


def cluster_record(name: str, trust: Dict[str, Any], budget: Optional[str]) -> Dict[str, Any]:
    """The cluster's record: its ratios only where the trust record says they hold for this family, its budget."""
    clusters = json.loads((SKILL / "calibration" / "clusters.json").read_text(encoding="utf-8"))
    record = dict(clusters.get(name, {}))
    calibrated = bool(trust.get("calibrated")) and bool(record.get("ratios"))
    record["ratios"] = SKILL / "calibration" / record["ratios"] if calibrated else None
    record["budget"] = budget or record.get("budget")
    record.setdefault("device", name.upper())
    return record


def run_folder(args: argparse.Namespace, repo: Path, doc: Dict[str, Any]) -> Path:
    """The folder that keeps this answer, with the yaml ND reads, the request, and the template and cluster whose
    records launch.py and readback.py read for it."""
    stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"  # two askers in one second stay apart
    run = Path(args.out) if args.out else repo / "output" / "sapp-parallel-strategy" / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / "run.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (run / "request.md").write_text(args.request + "\n", encoding="utf-8")
    asked = {"family": args.family, "cluster": args.cluster, "yaml": args.yaml}
    (run / "ask.json").write_text(json.dumps(asked, indent=2) + "\n", encoding="utf-8")
    return run


def _command(doc: Dict[str, Any], run: Path, modes: Sequence[str], cluster: Dict[str, Any]) -> List[str]:
    """The ND command that prices the run ``run/run.yaml`` describes."""
    command = [
        sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", "-f", "hyper_v2",
        "-y", str(run / "run.yaml"), "-d", str(_get(doc, FIELDS["dies"])), "-A", cluster["device"],
        "-b", str(_get(doc, FIELDS["global_batch"])), "-v", "2", "-o", str(run / "nd_out"),
        "--ranking_csv", str(run / "nd_rank.csv"), "--recompute", *modes,
    ]
    if cluster["budget"]:
        command += ["-M", cluster["budget"]]
    if cluster["ratios"]:
        command += ["-c", str(cluster["ratios"])]
    return command


def price(command: Sequence[str], repo: Path, run: Path, timeout: int) -> Tuple[int, float, List[str]]:
    """Run ND from the checkout, its log kept in the run folder: exit code, seconds, log lines."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    started = time.monotonic()
    with open(run / "nd.log", "w", encoding="utf-8") as log:
        result = subprocess.run(list(command), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    lines = (run / "nd.log").read_text(encoding="utf-8", errors="replace").splitlines()
    return result.returncode, time.monotonic() - started, lines


def launch_yaml(doc: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
    """The trainer yaml for one of ND's layouts: the run's own, with the layout's degrees where the trainer reads them.

    The trainer refuses ND's ``context`` section, so it is left out. Under expert parallelism, every layout ND
    prices with ``context.expert_shard: group``, and every run the sweep launched, shards a routed expert over its
    whole expert data-parallel group, DP x CP x TP / EP ranks. The trainer reads that as
    ``fsdp_config.edp_shard_size``, which ND's own resolved yaml leaves as the template states it.
    """
    out = copy.deepcopy(doc)
    context = out.pop("context", None) or {}
    accelerator = out.setdefault("accelerator", {})
    for degree, field in (("MP", "tp_size"), ("CP", "cp_size"), ("EP", "ep_size"), ("PP", "pp_size")):
        accelerator[field] = int(row[degree])
    if int(row["MP"]) > 1:
        accelerator["sequence_parallel"] = row.get("SP") == "True"
    fsdp = out.setdefault("fsdp_config", {})
    fsdp["dp_shard_size"] = int(row["OP"])
    if context.get("expert_shard") == "group":  # at EP 1 the trainer reads none of it, but the runs state it
        fsdp["edp_shard_size"] = int(row["DP"]) * int(row["CP"]) * int(row["MP"]) // int(row["EP"])
    out.setdefault("training", {})["micro_batch_size"] = int(row["MBS"])
    if row.get("recompute"):
        out.setdefault("activation_checkpoint", {})["mode"] = row["recompute"]
    return out


def _write_launches(run: Path, doc: Dict[str, Any], trust: Dict[str, Any]) -> List[str]:
    """The trainer yaml of the layout to launch, and of ND's own pick when that differs: lines naming them."""
    rows = answer.Ranking(str(run / "nd_rank.csv")).distinct()
    if not rows:
        return []
    if trust.get("region") is None:  # nothing was timed for this family here, so nothing is the layout to launch
        files, chosen = [("launch_nd_pick.yaml", rows[0])], None
    else:
        chosen = answer.recommended(rows, trust)
        files = [("launch.yaml", chosen)] + ([("launch_nd_pick.yaml", rows[0])] if chosen is not rows[0] else [])
    lines = []
    for name, row in files:
        text = yaml.safe_dump(launch_yaml(doc, row), sort_keys=False, allow_unicode=True)
        (run / name).write_text(text, encoding="utf-8")
        what = ("the layout to launch" if row is chosen else
                "ND's own pick, which nothing validates" if chosen is None else
                "ND's own pick, outside the timed region")
        lines.append(f"Trainer yaml for {what}: {run / name}; set its tokenizer and data paths on the cluster.")
        if int(row.get("VPP", "1")) > 1:
            lines.append(f"Its virtual pipeline degree, VPP {row['VPP']}, is not written there: set it by hand.")
    return lines


def version_notes(trust: Dict[str, Any], versions: Dict[str, Any]) -> List[str]:
    """A warning for each library installed at a version the trust record was not checked with."""
    notes = []
    for name, checked in (trust.get("checked_with") or {}).items():
        known = [checked] if isinstance(checked, str) else list(checked)
        installed = versions.get(name)
        if installed and installed not in known:
            notes.append(f"{name} {installed} is installed here, and the region was checked with "
                         f"{' and '.join(known)}: ND's census follows the installed library, so this ranking may "
                         "differ from the checked one.")
    return notes


def _report(args: argparse.Namespace, run: Path, facts: Dict[str, Any], lines: Sequence[str]) -> List[str]:
    """The answer: the status line, what priced it, the card, ND's notices and where the run is kept."""
    cluster, trust, versions = facts["cluster"], facts["trust"], facts["versions"]
    notices = [line.strip() for line in lines if any(word in line for word in NOTICES)]
    card = answer.card(answer.Ranking(str(run / "nd_rank.csv")), calibrated=cluster["ratios"] is not None,
                       spread=cluster.get("remeasure_spread_pct"), budget=cluster["budget"], trust=trust)
    changed = " (with local changes)" if versions["nd_tree_changed"] else ""
    scale = "calibrated" if cluster["ratios"] else "no calibration for this family here: relative scores only"
    return [
        f"Status: {trust['status']}. {trust['may_say']}",
        f"Priced by ND {versions['nd_commit'][:8]}{changed}, Transformers {versions['transformers']}, "
        f"cluster {args.cluster} ({scale}), in {facts['seconds']:.0f} s.",
        *facts["notes"],
        *version_notes(trust, versions),
        "",
        card,
        "ND's notices, verbatim:" if notices else "ND printed no notice.",
        *[f"- {line}" for line in notices],
        "",
        *facts["launches"],
        f"Run kept in {run}",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Describe the run, price it with ND, keep everything, print the answer."""
    args = _arguments(argv)
    repo = repo_root()
    trust = answer.trust_entry(args.family or "*", args.cluster)
    cluster = cluster_record(args.cluster, trust, args.budget)
    args.recompute, notes = _modes(args.recompute, trust)
    doc = describe(args, notes)
    run = run_folder(args, repo, doc)
    command = _command(doc, run, args.recompute, cluster)
    versions = library_versions(repo, cluster["ratios"])
    (run / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    (run / "command.txt").write_text(f"cd {repo}\n" + " ".join(command) + "\n", encoding="utf-8")

    code, seconds, lines = price(command, repo, run, args.timeout)
    if code != 0 or not (run / "nd_rank.csv").is_file():
        sys.stdout.write(f"ND failed (exit {code}) after {seconds:.0f} s; the last lines of {run / 'nd.log'}:\n")
        sys.stdout.write("\n".join(lines[-25:]) + "\n")
        return 1
    facts = {"cluster": cluster, "trust": trust, "versions": versions, "seconds": seconds, "notes": notes,
             "launches": _write_launches(run, doc, trust)}
    report = _report(args, run, facts, lines)
    (run / "answer.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
