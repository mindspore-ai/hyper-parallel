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
"""Consolidate Agent-reviewed, portable RSI lessons at experiment boundaries.

The Agent extracts and interprets lessons. This tool discovers repository owners,
checks review coverage and evidence, applies deduplicated changes transactionally,
and produces a separate RSI report. It never infers correctness from a speedup.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterator

MODES = ("acceptance", "continuous")
REGISTRY = ".agent/learning/registry.json"
ROOTS = (".agent/skills", ".agent/agents", ".agent/commands", ".agent/rules", ".agent/scripts",
         ".agent/examples", ".agent/tests", "hyper_parallel", "tests/ut")
SUFFIXES = {".md", ".py", ".json", ".sh", ".yaml", ".yml"}
_LOCAL = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:pod|superpod)[-_ ]?\d+\b|"
    r"\b[A-Za-z]:[\\/]|/(?:home|Users|workspace|mnt|data)/|"
    r"\b[\w.-]+@(?:[\w.-]+\.[\w.-]+)|\b[\w.-]+\.(?:local|internal)\b|"
    r"\b(?:dashboard|control-panel)\.[a-z0-9.-]+|"
    r"(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*[\"']?\w{8,}",
    re.IGNORECASE,
)


def require(condition: bool, message: str) -> None:
    """Reject an invalid learning request."""
    if not condition:
        raise ValueError(message)


def digest(value: object) -> str:
    """Hash finite JSON without depending on dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str | None:
    """Return a file digest, or None for a new destination."""
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    """Atomically save a local record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def contained(root: Path, name: str) -> Path:
    """Resolve a portable relative path without accepting traversal or symlinks."""
    require(isinstance(name, str) and name and "\\" not in name, "Use a nonempty POSIX-relative path")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
            f"Non-relative artifact path: {name}")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes its root: {name}")
    require(not any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(root)),
            f"Symlink destination is not supported: {name}")
    return path


@contextmanager
def exclusive(repo: Path) -> Iterator[None]:
    """Serialize repository learning writes across independent experiment runs."""
    lock = repo / ".agent/.rsi-learning.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def initialize(run: Path, mode: str = "acceptance") -> dict:
    """Freeze the user-selected consolidation mode; acceptance is the default."""
    require(mode in MODES, "RSI mode must be acceptance or continuous")
    path = run / "rsi/config.json"
    if path.exists():
        config = read(path)
        require(config["mode"] == mode, "RSI mode is frozen for this run")
        return config
    config = {"schema": "hp-rsi-learning/v1", "mode": mode}
    save(path, config)
    return config


def inventory(repo: Path) -> dict:
    """Discover current Skills and source owners instead of maintaining a copied catalog."""
    files = {}
    for name in ROOTS:
        directory = repo / name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                relative = path.relative_to(repo).as_posix()
                contained(repo, relative)
                files[relative] = file_hash(path)
    if (repo / REGISTRY).exists():
        files[REGISTRY] = file_hash(contained(repo, REGISTRY))
    skills = sorted(name for name in files if re.fullmatch(r"\.agent/skills/[^/]+/SKILL\.md", name))
    return {"digest": digest(files), "skills": skills, "files": files,
            "roots": [name for name in ROOTS if (repo / name).is_dir()]}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()))


def related(repo: Path, current: dict, lesson: dict) -> list[dict]:
    """Return lexical retrieval candidates for the Agent's semantic duplicate review."""
    query = _words(" ".join(lesson.get("keywords", [])))
    require(len(query) >= 2, "Each lesson needs at least two useful retrieval keywords")
    hits = []
    for name, checksum in current["files"].items():
        if not name.startswith(".agent/") or not name.endswith(".md"):
            continue
        words = _words((repo / name).read_text(encoding="utf-8"))
        score = len(query & words) / len(query)
        if score >= 0.5:
            hits.append({"path": name, "sha256": checksum, "score": round(score, 4)})
    return sorted(hits, key=lambda row: (-row["score"], row["path"]))[:5]


def events(run: Path) -> list[dict]:
    """Inventory every completed experiment, including rejected and failed results."""
    controller = run / "controller.json"
    if controller.exists():
        state = read(controller)
        require(not state.get("pending"), "Do not consolidate while a trial or cleanup is pending")
        result = []
        for trial in state["attempts"]:
            name = f"trials/{trial}/result.json"
            path = contained(run, name)
            checksum = file_hash(path)
            require(checksum == state["result_hashes"][trial], "Trial evidence changed after validation")
            result.append({"id": trial, "sha256": checksum, "path": name, "decision": read(path)["decision"]})
        return result
    ledger = run / "candidates.jsonl"
    results = run / "results.tsv"
    if ledger.exists():
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        if rows:
            # Legacy/non-training baselines have TSV records but no candidate receipt.
            # Match the suffix before including the otherwise unreviewed baseline prefix.
            lines = results.read_text(encoding="utf-8").splitlines()[1:] if results.exists() else []
            prefix = len(lines) - len(rows)
            require(prefix >= 0 and all(line.split("\t")[3] == row["status"]
                                       for line, row in zip(lines[prefix:], rows)),
                    "Candidate and result ledgers disagree; reconcile the experiment inventory")
            return ([{"id": f"result-{index}-{digest(line)[:12]}", "sha256": digest(line),
                      "path": "results.tsv", "decision": line.split("\t")[3]}
                     for index, line in enumerate(lines[:prefix])]
                    + [{"id": f"candidate-{index}-{digest(row)[:12]}", "sha256": digest(row),
                        "path": "candidates.jsonl", "decision": row["status"]} for index, row in enumerate(rows)])
    if results.exists():
        return [{"id": f"result-{index}-{digest(line)[:12]}", "sha256": digest(line),
                 "path": "results.tsv", "decision": line.split("\t")[3]}
                for index, line in enumerate(results.read_text(encoding="utf-8").splitlines()[1:]) if line]
    # External training adapters may supply a complete, hashed experiment index.
    adapter = run / "rsi/experiments.json"
    if adapter.exists():
        rows = read(adapter)["experiments"]
        require(len({row["id"] for row in rows}) == len(rows), "Duplicate external experiment IDs")
        for row in rows:
            require(file_hash(contained(run, row["path"])) == row["sha256"], "External experiment evidence changed")
        return rows
    return []


def inspect(repo: Path, run: Path) -> dict:
    """Write a fresh architecture, experiment inventory, and duplicate-review task."""
    current = inventory(repo)
    review_path = run / "rsi/review.json"
    review = read(review_path) if review_path.exists() else {"lessons": []}
    task = {"schema": "hp-rsi-review-task/v1", "inventory_digest": current["digest"],
            "skills": current["skills"], "roots": current["roots"], "experiments": events(run),
            "related": {row["id"]: related(repo, current, row) for row in review["lessons"]}}
    save(run / "rsi/architecture.json", current)
    save(run / "rsi/review-task.json", task)
    (run / "rsi/REVIEW_REQUIRED.md").write_text(
        "# Agent action: reconcile reusable experience\n\n"
        "Read review-task.json and the listed current Skill entrypoints. Review every experiment, including "
        "failures. Write review.json using the consolidation contract. Classify each lesson as new, update, "
        "reuse, defer, or reject. Inspect all retrieved matches before claiming novelty. Keep cluster identities "
        "and raw deployment details in this run, not in repository changes. Do not ask the user to remind you.\n",
        encoding="utf-8",
    )
    return task


def _portable(text: str) -> None:
    require(_LOCAL.search(text) is None,
            "Repository lesson contains a deployment identifier or credential; generalize it")


def _evidence(run: Path, lesson: dict) -> list[str]:
    records = lesson.get("evidence", [])
    require(bool(records), "Consolidated lessons require hashed evidence")
    for record in records:
        require(file_hash(contained(run, record["path"])) == record["sha256"], "Lesson evidence is missing or changed")
    return sorted({record["sha256"] for record in records})


def _review_coverage(review: dict, observed: list[dict]) -> None:
    require(review.get("schema") == "hp-rsi-learning-review/v1", "Unsupported learning review schema")
    ids = {lesson["id"] for lesson in review["lessons"]}
    require(len(ids) == len(review["lessons"]), "Duplicate lesson IDs")
    coverage = review.get("coverage", [])
    require(len(coverage) == len(observed) and {row["event_id"] for row in coverage} == {row["id"] for row in observed},
            "Review must account for every experiment, including failures")
    expected = {row["id"]: row["sha256"] for row in observed}
    for row in coverage:
        require(row.get("sha256") == expected[row["event_id"]], "Reviewed experiment changed")
        require(bool(row.get("reason")) and set(row.get("lesson_ids", [])) <= ids,
                "Every experiment needs lesson references or a reason for no new experience")
    require(bool(observed), "A final RSI review requires experiment evidence")


def _acknowledge(repo: Path, current: dict, lesson: dict) -> None:
    acknowledgements = {row["path"]: row for row in lesson.get("related_review", [])}
    for match in related(repo, current, lesson):
        row = acknowledgements.get(match["path"], {})
        require(row.get("sha256") == match["sha256"] and row.get("reason")
                and row.get("relationship") in ("same", "extends", "different"),
                f"Agent must review retrieved experience: {match['path']}")
    for name, row in acknowledgements.items():
        require(current["files"].get(name) == row["sha256"], "Related experience changed")
    if lesson["decision"] == "new":
        require(all(row["relationship"] == "different" for row in acknowledgements.values()),
                "Existing or extendable experience must use reuse or update")


def _document(lesson: dict) -> str:
    sections = (("Trigger", "trigger"), ("Mechanism", "mechanism"), ("Reusable rule", "claim"),
                ("Applicability", "applicability"), ("Limits and counterexamples", "limits"))
    lines = [f"# {lesson['title']}", "", f"Evidence qualification: {lesson['qualification']}", ""]
    for heading, key in sections:
        require(isinstance(lesson.get(key), str) and lesson[key].strip(), f"Missing lesson {key}")
        lines += [f"## {heading}", "", lesson[key], ""]
    return "\n".join(lines)


def _edit(repo: Path, current: dict, change: dict, proposed: dict) -> None:
    name = change["path"]
    path = contained(repo, name)
    require(name.startswith(".agent/") and path.suffix in SUFFIXES
            and any(name.startswith(root + "/") for root in current["roots"]),
            "Learning destination must belong to a discovered .agent code area")
    require(name not in proposed, f"Two lessons edit the same file; combine their patch first: {name}")
    require(file_hash(path) == change["before_sha256"], f"Destination changed since review: {name}")
    content = change["content"]
    require(isinstance(content, str) and content.endswith("\n"), "Replacement content must end with a newline")
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    additions = "\n".join(line[1:] for line in difflib.ndiff(before.splitlines(), content.splitlines())
                          if line.startswith("+ "))
    _portable(additions)
    if path.suffix == ".py":
        compile(content, name, "exec")
    proposed[name] = content


def _plan_lesson(repo: Path, run: Path, current: dict, lesson: dict, registry: dict, proposed: dict) -> dict:
    key = lesson["id"]
    require(re.fullmatch(r"[a-z][a-z0-9-]{2,79}", key) is not None, "Use a stable, portable lesson ID")
    decision = lesson["decision"]
    require(decision in ("new", "update", "reuse", "defer", "reject"), "Invalid lesson decision")
    require(bool(lesson.get("reason")), "Every consolidation decision needs a reason")
    result = {"id": key, "decision": decision, "reason": lesson["reason"], "files": []}
    result.update(title=lesson.get("title", key), claim=lesson.get("claim", ""),
                  qualification=lesson.get("qualification", "not promoted"), evidence=lesson.get("evidence", []))
    if decision in ("defer", "reject"):
        require(not lesson.get("changes"), "Deferred or rejected lessons cannot modify the repository")
        return result
    _portable(json.dumps({name: lesson.get(name) for name in
                          ("id", "title", "claim", "trigger", "mechanism", "applicability", "limits")}))
    evidence = _evidence(run, lesson)
    fingerprint = digest(" ".join(re.findall(r"\w+", lesson["claim"].lower())))
    prior = registry["lessons"].get(key)
    duplicate = next((row for row in registry["lessons"].values() if row["fingerprint"] == fingerprint), None)
    if duplicate and decision == "new":
        require(not lesson.get("changes"), "Existing experience with a proposed code improvement must use update")
        require(all((repo / name).is_file() for name in duplicate["files"]),
                "Registered experience is missing; repair it rather than silently reusing it")
        return {**result, "decision": "reuse", "reason": "Equivalent registered experience already exists",
                "files": duplicate["files"]}
    _acknowledge(repo, current, lesson)
    if decision == "reuse":
        require(not lesson.get("changes"), "Reused experience must not create another copy")
        matches = [row["path"] for row in lesson.get("related_review", []) if row["relationship"] == "same"]
        require(prior is not None or matches, "Reuse requires an existing verified repository location")
        return {**result, "files": prior["files"] if prior else matches}
    require(lesson.get("qualification") in ("validated", "negative"),
            "Unqualified screens and untried hypotheses must be deferred")
    owner = f".agent/skills/{lesson['owner']}/SKILL.md"
    require(owner in current["skills"], "Owner Skill is absent from the current architecture")
    require(prior is None or decision == "update", "Existing lesson ID must be improved with update")
    if decision == "update":
        require(prior is not None or any(row["relationship"] in ("same", "extends")
                                        for row in lesson.get("related_review", [])),
                "Update must identify the existing experience")
        require(bool(lesson.get("changes")), "Update requires a concrete improvement to the existing implementation")
    changed = []
    for change in lesson.get("changes", []):
        _edit(repo, current, change, proposed)
        changed.append(change["path"])
    if decision == "new":
        name = f".agent/skills/{lesson['owner']}/references/experience/{key}.md"
        require(not (repo / name).exists(), "Lesson destination already exists; review it for update or reuse")
        proposed[name] = _document(lesson)
        _portable(proposed[name])
        changed.append(name)
        link = f"- [{lesson['title']}](references/experience/{key}.md)\n"
        entry = proposed.get(owner, (repo / owner).read_text(encoding="utf-8"))
        if "## Consolidated experience\n" not in entry:
            entry = entry.rstrip() + "\n\n## Consolidated experience\n\n"
        proposed[owner] = entry + link
        changed.append(owner)
    code = [name for name in changed if not name.endswith(".md")]
    if code:
        tests = lesson.get("regression_tests", [])
        require(bool(tests) and bool(lesson.get("validation")),
                "Executable changes require regression tests and commands")
        require(all(name.startswith((".agent/tests/", "tests/")) and (name in proposed or name in current["files"])
                    for name in tests), "Regression tests must exist in the repository or this patch")
    record = {"title": lesson["title"], "claim": lesson["claim"], "fingerprint": fingerprint,
              "owner": lesson["owner"], "form": lesson.get("form", "procedure"),
              "qualification": lesson["qualification"],
              "files": sorted(set(changed + (prior["files"] if prior else []))),
              "evidence_sha256": sorted(set(evidence + (prior["evidence_sha256"] if prior else []))),
              "revision": prior["revision"] + 1 if prior else 1}
    registry["lessons"][key] = record
    return {**result, "files": record["files"], "form": record["form"], "revision": record["revision"]}


def _validate(repo: Path, run: Path, review: dict) -> list[dict]:
    receipts = []
    for lesson in review["lessons"]:
        if lesson["decision"] not in ("new", "update"):
            continue
        for index, check in enumerate(lesson.get("validation", [])):
            argv = check["argv"]
            require(isinstance(argv, list) and argv and all(isinstance(arg, str) for arg in argv),
                    "Use validation argv")
            timeout = check.get("timeout_seconds", 120)
            require(isinstance(timeout, int) and 0 < timeout <= 300, "Validation timeout must be 1-300 seconds")
            command = [sys.executable, *argv[1:]] if argv[0] in ("python", "python3") else argv
            done = subprocess.run(command, cwd=repo, capture_output=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, check=False, shell=False)
            path = run / "rsi" / f"validation-{lesson['id']}-{index}.log"
            path.write_text(done.stdout + "\n" + done.stderr, encoding="utf-8")
            receipts.append({"lesson": lesson["id"], "argv": argv, "exit_code": done.returncode,
                             "log": path.relative_to(run).as_posix(), "sha256": file_hash(path)})
            save(run / "rsi/validation.json", {"checks": receipts})
            require(done.returncode == 0, f"Learning regression failed: {lesson['id']}; repository changes rolled back")
    return receipts


def _protected(repo: Path, run: Path, proposed: dict) -> None:
    """Do not rewrite measured source or frozen validators to consolidate experience."""
    paths = []
    auto = run / "run.json"
    if auto.exists():
        config = read(auto)
        paths.extend(config.get("target_files", []))
        paths.extend(arg for key in ("gate_cmd", "benchmark_cmd") for arg in config.get(key, [])
                     if (repo / arg).is_file())
    riser = run / "config.json"
    if riser.exists():
        paths.extend(read(riser).get("validator_files", []))
    controller = run / "controller.json"
    if controller.exists():
        for trial in read(controller)["attempts"]:
            plan = read(run / "trials" / trial / "context.json")["plan"]
            paths.extend(str(Path(plan["workspace"]) / name) for name in plan["source_paths"])
    for name in proposed:
        target = (repo / name).resolve()
        require(not any(target == (repo / old).resolve() or target.is_relative_to((repo / old).resolve())
                        for old in paths),
                "Learning cannot rewrite measured source or frozen gates; use a separate checkout")


def _apply(repo: Path, run: Path, review: dict, proposed: dict, expected: dict) -> list[dict]:
    _protected(repo, run, proposed)
    require(all(file_hash(repo / name) == expected.get(name) for name in proposed),
            "A destination changed while planning consolidation")
    originals = {name: (repo / name).read_bytes() if (repo / name).exists() else None for name in proposed}
    save(run / "rsi/validation.json", {"checks": []})
    try:
        for name, content in proposed.items():
            path = contained(repo, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        receipts = _validate(repo, run, review)
        require(all(file_hash(repo / name) == hashlib.sha256(content.encode()).hexdigest()
                    for name, content in proposed.items()), "Validation modified a proposed repository artifact")
        return receipts
    except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt):
        for name, content in originals.items():
            path = repo / name
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise


def _report(run: Path, result: dict) -> dict:
    save(run / "rsi/report.json", result)
    lines = ["# RSI improvement report", "", f"Mode: **{result['mode']}**", "",
             f"Consolidation state: **{result['state']}**", "", result.get("reason", ""), "",
             "Performance acceptance is a separate result; this report describes repository learning.", "",
             "| Experience | Decision | Form | Repository location | Reason |",
             "|---|---|---|---|---|"]
    for row in result.get("lessons", []):
        cells = [row["title"] + ": " + row["claim"], row["decision"], row.get("form", "existing / none"),
                 ", ".join(row["files"]), row["reason"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    if result["complete"] and not result.get("lessons"):
        lines += ["", "All experiments were reviewed; no new reusable experience required repository changes."]
    lines += ["", f"Reviewed experiments: {result.get('reviewed_experiments', 0)}", "",
              "See rsi/report.json for content hashes and validation receipts. No remote publication is performed.", ""]
    (run / "RSI_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    rows = []
    for row in result.get("lessons", []):
        locations = "<br>".join(
            f"<a href='{html.escape((Path(result['repository']) / name).as_uri(), quote=True)}'>{html.escape(name)}</a>"
            for name in row["files"])
        cells = [html.escape(row["title"] + ": " + row["claim"]), html.escape(row["decision"]),
                 html.escape(row.get("form", "existing / none")), locations, html.escape(row["reason"])]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>RSI improvement report</title>"
            "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#183044}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ced8e0;padding:12px;text-align:left}"
            "td{overflow-wrap:anywhere}th{background:#edf4f7}</style><h1>RSI improvement report</h1>"
            f"<p>Mode: {html.escape(result['mode'])} · State: {html.escape(result['state'])}</p>"
            f"<p>{html.escape(result.get('reason', ''))}</p>"
            "<p>Performance acceptance is separate. Repository learning does not imply a performance pass.</p>"
            "<table><tr><th>Experience</th><th>Decision</th><th>Form</th>"
            "<th>Repository location</th><th>Reason</th></tr>"
            f"{''.join(rows)}</table><p>Reviewed experiments: {result.get('reviewed_experiments', 0)}</p>"
            "<p><a href='rsi/report.json'>Structured report and validation evidence</a></p></html>")
    (run / "RSI_REPORT.html").write_text(page, encoding="utf-8")
    return result


def synchronize(repo: Path, run: Path, phase: str = "iteration", mode: str = "acceptance",
                require_clean_targets: bool = False) -> dict:
    """Reconcile experience without publishing or modifying an in-flight experiment.

    Args:
        repo: Current repository checkout receiving portable improvements.
        run: Experiment directory containing raw evidence and the Agent review.
        phase: Iteration boundary or final acceptance boundary.
        mode: Frozen acceptance-only or continuous consolidation policy.
        require_clean_targets: Reject pre-existing edits in paths that a caller will commit automatically.
    Returns:
        A report with complete=false until review coverage and all updates pass.
    """
    require(phase in ("iteration", "acceptance"), "Unknown consolidation boundary")
    initialize(run, mode)
    result = {"schema": "hp-rsi-learning-report/v1", "mode": mode, "phase": phase,
              "complete": False, "state": "REVIEW_REQUIRED", "lessons": [], "changed_files": {},
              "repository": str(repo.resolve()),
              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with exclusive(repo):
            if phase == "iteration" and mode == "acceptance":
                observed = events(run)
                save(run / "rsi/pending-experiments.json", {"experiments": observed})
                return _report(run, {**result, "state": "QUEUED", "queued_experiments": len(observed),
                                     "reason": "Evidence queued; corpus review and changes run at acceptance"})
            task = inspect(repo, run)
            review_path = run / "rsi/review.json"
            if not review_path.exists():
                return _report(run, {**result, "reason": "Agent must complete rsi/review.json before final delivery"})
            review = read(review_path)
            _review_coverage(review, task["experiments"])
            result["reviewed_experiments"] = len(task["experiments"])
            receipt_path = run / "rsi/applied.json"
            previous = read(receipt_path) if receipt_path.exists() else None
            if previous and previous["review_digest"] == digest(review):
                require(all(file_hash(contained(repo, name)) == checksum
                            for name, checksum in previous["changed_files"].items()),
                        "Consolidated files changed after review")
                return _report(run, {**result, **previous, "phase": phase, "state": "COMPLETE", "complete": True})
            require(review["inventory_digest"] == task["inventory_digest"],
                    "Repository changed; refresh the architecture review")
            current = read(run / "rsi/architecture.json")
            registry_path = contained(repo, REGISTRY)
            registry = (read(registry_path) if registry_path.exists()
                        else {"schema": "hp-rsi-experience/v1", "lessons": {}})
            proposed = {}
            result["lessons"] = [_plan_lesson(repo, run, current, lesson, registry, proposed)
                                 for lesson in review["lessons"]]
            if proposed:
                proposed[REGISTRY] = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
                if require_clean_targets:
                    dirty = subprocess.run(["git", "status", "--porcelain", "--", *proposed], cwd=repo,
                                           capture_output=True, text=True, check=False)
                    require(dirty.returncode == 0 and not dirty.stdout.strip(),
                            "Learning destinations contain pre-existing edits; preserve or commit them separately")
            active_review = {**review, "lessons": [lesson for lesson, resolution in
                                                  zip(review["lessons"], result["lessons"])
                                                  if resolution["decision"] in ("new", "update")]}
            receipts = _apply(repo, run, active_review, proposed, current["files"])
            result.update(state="COMPLETE", complete=True, review_digest=digest(review), validation=receipts,
                          changed_files={name: file_hash(repo / name) for name in proposed},
                          reason="Experience reconciled; deployment-specific and unqualified items remain local")
            save(receipt_path, result)
            return _report(run, result)
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        validation = run / "rsi/validation.json"
        result["validation"] = read(validation)["checks"] if validation.exists() else []
        return _report(run, {**result, "state": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> int:
    """Run the shared learning tool from either Harness or an external adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "inspect", "sync"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--mode", choices=MODES, default="acceptance")
    parser.add_argument("--phase", choices=("iteration", "acceptance"), default="iteration")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    if args.command == "init":
        result = initialize(run, args.mode)
    else:
        require(args.repo is not None, "--repo is required")
        result = (inspect(args.repo.resolve(), run) if args.command == "inspect" else
                  synchronize(args.repo.resolve(), run, args.phase, args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if args.command == "sync" and args.phase == "acceptance" and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
