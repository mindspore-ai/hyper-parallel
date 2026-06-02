#!/usr/bin/env python3
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
"""gate-doctor CLI: diagnose / autofix / retest / check-pr.

Read-only subcommands (diagnose) and write subcommands (retest, check-pr)
operate against the GitCode API; autofix is a driver that emits a JSON plan
the calling LLM consumes to edit files, then delegates the commit step to the
sibling `autogit` skill.

Environment:
    GITCODE_TOKEN   PRIVATE-TOKEN value (same one autogit uses)

Subcommands:
    diagnose <pr>       Print a JSON report of the latest pipeline failure.
    retest <pr>         Post `/retest` on the PR.
    check-pr <pr>       Post `/check-pr` on the PR.
    autofix <pr>        Drive the diagnose -> propose-plan workflow. The
                        actual file edits + autogit commit + retest are
                        performed by the caller (the AI agent), since they require
                        code judgement; this command only emits the plan.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional

DEFAULT_OWNER = "mindspore"
DEFAULT_REPO = "hyper-parallel"
GITCODE_API = "https://gitcode.com/api/v5"
JENKINS_BASE = "https://build.mindspore.cn"
# Repo -> Jenkins gate job name. Verified jobs as of 2026-05-16:
#   mindspore/mindspore      -> MindSpore_Atomgit_Gate
#   mindspore/hyper-parallel -> Hyper-Parallel_Atomgit_Gate
# Add new entries here when the skill is adopted for another repo.
_JENKINS_JOBS = {
    ("mindspore", "mindspore"):      "MindSpore_Atomgit_Gate",
    ("mindspore", "hyper-parallel"): "Hyper-Parallel_Atomgit_Gate",
}


def jenkins_job_for(owner: str, repo: str, override: Optional[str] = None) -> str:
    """Resolve the Jenkins job name for a (owner, repo)."""
    if override:
        return override
    job = _JENKINS_JOBS.get((owner, repo))
    if job:
        return job
    raise GateDoctorError(
        f"No Jenkins gate job mapped for {owner}/{repo}. "
        f"Pass --jenkins-job=<name> or add an entry to _JENKINS_JOBS in "
        f"gate_doctor.py."
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class GateDoctorError(RuntimeError):
    """User-facing error message; printed and exits 1."""


# ---------------------------------------------------------------------------
# PR ref parsing
# ---------------------------------------------------------------------------
_PR_URL_RE = re.compile(
    r"^https?://(?:[^/]+/)+([^/]+)/([^/]+)/(?:pulls|pull|merge_requests)/(\d+)/?"
)


def parse_pr_ref(ref: str) -> tuple[str, str, int]:
    """Parse a PR reference and return (owner, repo, number).

    Accepted forms:
      - ``647``                                  -> default (mindspore/hyper-parallel)
      - ``#647``                                 -> default (mindspore/hyper-parallel)
      - ``mindspore/mindspore#92567``            -> explicit owner/repo
      - ``mindspore/hyper-parallel#647``         -> explicit owner/repo
      - ``https://gitcode.com/mindspore/mindspore/pull/92567``        -> explicit
      - ``https://gitcode.com/mindspore/hyper-parallel/pull/647``     -> explicit

    A bare number always falls back to the default repo (hyper-parallel,
    because that's where this skill lives). To target a different repo,
    pass the full URL or the ``owner/repo#N`` form.
    """
    ref = ref.strip().lstrip("#")
    m = _PR_URL_RE.match(ref)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    if "#" in ref:
        repo_part, num = ref.split("#", 1)
        owner, repo = repo_part.split("/", 1)
        return owner, repo, int(num)
    if ref.isdigit():
        return DEFAULT_OWNER, DEFAULT_REPO, int(ref)
    raise GateDoctorError(f"Cannot parse PR reference: {ref!r}")


# ---------------------------------------------------------------------------
# GitCode API helpers (no external deps; urllib is enough)
# ---------------------------------------------------------------------------
def _request(method: str, url: str, token: str,
             body: Optional[dict] = None, timeout: int = 20) -> tuple[int, dict | list | str]:
    """Send one HTTP request to the GitCode API and decode the response."""
    headers = {
        "PRIVATE-TOKEN": token,
        "Accept": "application/json",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise GateDoctorError(f"Cannot reach {url}: {e.reason}") from e
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def get_pr_comments(owner: str, repo: str, pr: int, token: str,
                    max_pages: int = 10) -> list[dict]:
    """Fetch *all* PR comments by paginating, then sort newest-first.

    GitCode's `?per_page=100` cap means PRs with >100 comments need pagination;
    we also can't rely on the server's default ordering being chronological, so
    we re-sort by `created_at` descending after collection.
    """
    collected: list[dict] = []
    for page in range(1, max_pages + 1):
        status, data = _request(
            "GET",
            f"{GITCODE_API}/repos/{owner}/{repo}/pulls/{pr}/comments"
            f"?per_page=100&page={page}",
            token,
        )
        if status != 200:
            raise GateDoctorError(f"comments fetch page {page} failed (HTTP {status}): {data}")
        if not isinstance(data, list) or not data:
            break
        collected.extend(data)
        if len(data) < 100:
            break
    # Newest-first
    collected.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return collected


def get_pr_labels(owner: str, repo: str, pr: int, token: str) -> list[str]:
    """Return the list of label names currently attached to the PR.

    Used by bypass-build detection to distinguish a real lint-only green
    (label ``ci-pipeline-passed`` present, ~50-90s duration) from a true
    bypass build (no label, ~17s duration, ran nothing).
    """
    status, data = _request(
        "GET",
        f"{GITCODE_API}/repos/{owner}/{repo}/pulls/{pr}",
        token,
    )
    if status != 200 or not isinstance(data, dict):
        return []
    out: list[str] = []
    for lab in (data.get("labels") or []):
        if isinstance(lab, dict):
            name = lab.get("name") or lab.get("title")
            if name:
                out.append(name)
        elif isinstance(lab, str):
            out.append(lab)
    return out


def post_pr_comment(owner: str, repo: str, pr: int, body: str, token: str) -> dict:
    status, data = _request(
        "POST",
        f"{GITCODE_API}/repos/{owner}/{repo}/pulls/{pr}/comments",
        token,
        body={"body": body},
    )
    if status not in (200, 201):
        raise GateDoctorError(f"comment POST failed (HTTP {status}): {data}")
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Comment-stream parsing: locate latest pipeline & its Jenkins URL
# ---------------------------------------------------------------------------
_PIPE_LINK_RE = re.compile(
    r"\((https?://\S+?/job/[^/]+/(\d+)/[^)]*)\)"
)
# micro-compass build result rows look like:
#   `|  | Check_Pylint | **FAILURE** | [3903](https://build.mindspore.cn/.../3903...) |`
# The FAILURE/SUCCESS token is usually wrapped in `**`, sometimes preceded by
# `:x:` on the project summary row (which we filter out by requiring the
# stage column to match `Check_XXX`).
_BUILD_TABLE_RE = re.compile(
    r"\|\s*(Check_[A-Za-z0-9_]+)\s*\|\s*[^|]*?(FAILURE|SUCCESS)[^|]*?\|\s*\[(\d+)\]\(([^)]+)\)\s*\|"
)
_CHECK_PR_FAIL_RE = re.compile(r"当前/check-pr未通过|/check-pr.*未通过")
# Bot emits this when /retest is posted but the PR still lacks pr-check-pass.
# The Jenkins build that follows runs for ~10-20s and exits SUCCESS without
# doing any real work — that's a bypass, not a real green.
_GATE_BYPASS_RE = re.compile(r"门禁触发失败.*pr-check-pass|请通过/check-pr获取pr-check-pass")
# Threshold below which we treat a "SUCCESS" build as a bypass build.
_BYPASS_DURATION_MS = 60_000


@dataclass
class StageStatus:
    name: str
    status: str   # FAILURE / SUCCESS
    pipeline: int
    url: str


@dataclass
class PipelineFailureSummary:
    pipeline_number: Optional[int]
    pipeline_url: Optional[str]
    failed_stages: list[StageStatus] = field(default_factory=list)
    check_pr_failure: Optional[str] = None
    check_pr_failure_ts: Optional[str] = None
    gate_trigger_blocked: Optional[str] = None  # "门禁触发失败 ..." text if present
    latest_comment_id: Optional[int] = None


def parse_latest_failure(comments: list[dict]) -> PipelineFailureSummary:
    """Walk PR comments newest-first; collect three independent signals.

    Signals (all kept even if they co-occur — that's the bug-fix point):

      1. Newest build-result table -> ``pipeline_number`` /
         ``failed_stages``. A bypass build that finishes in ~20s still
         shows up here as SUCCESS; the caller decides what to do with
         that (see ``_BYPASS_DURATION_MS``).
      2. Newest ``/check-pr`` failure text -> ``check_pr_failure`` /
         ``check_pr_failure_ts``. Independent of the build verdict.
      3. Newest "门禁触发失败 ... pr-check-pass" notice ->
         ``gate_trigger_blocked``. Bot emits this when ``/retest`` is
         rejected because the PR still lacks the ``pr-check-pass`` label.
         Co-occurs with signal #2.

    Iteration stops once all three are filled.
    """
    summary = PipelineFailureSummary(pipeline_number=None, pipeline_url=None)
    seen_build_table = False
    seen_check_pr_fail = False
    seen_gate_block = False

    for c in comments:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        ts = c.get("created_at", "")

        if not seen_build_table:
            stages_in_comment: list[StageStatus] = []
            for m in _BUILD_TABLE_RE.finditer(body):
                name, status, pipe_num, url = m.groups()
                stages_in_comment.append(StageStatus(
                    name=name, status=status, pipeline=int(pipe_num), url=url
                ))
            if stages_in_comment:
                failed = [s for s in stages_in_comment if s.status == "FAILURE"]
                summary.pipeline_number = stages_in_comment[0].pipeline
                summary.pipeline_url = stages_in_comment[0].url
                summary.latest_comment_id = c.get("id")
                if failed:
                    summary.failed_stages = failed
                seen_build_table = True

        if not seen_check_pr_fail and _CHECK_PR_FAIL_RE.search(body):
            summary.check_pr_failure = body
            summary.check_pr_failure_ts = ts
            seen_check_pr_fail = True

        if not seen_gate_block and _GATE_BYPASS_RE.search(body):
            summary.gate_trigger_blocked = body
            seen_gate_block = True

        if seen_build_table and seen_check_pr_fail and seen_gate_block:
            break

    return summary


# ---------------------------------------------------------------------------
# Jenkins console log parsing
# ---------------------------------------------------------------------------
def find_latest_pr_build(pr_number: int, job: str,
                         scan_depth: int = 60) -> Optional[dict]:
    """Find the newest non-ABORTED Jenkins build that belongs to this PR.

    Queries the Jenkins job's builds API and filters builds whose
    `description` contains "Pull Request #<pr>". Returns the build dict
    (with keys ``number``, ``result``, ``building``, ``description``,
    ``url``, ``timestamp``) for the *most recent* such build, OR None if no
    build is associated with the PR.

    ABORTED builds are skipped: when the user spams `/retest`, intermediate
    builds get aborted and the *previous* fully-completed result is what
    actually represents the gate verdict.
    """
    url = (f"{JENKINS_BASE}/job/{job}/api/json"
           f"?tree=builds[number,result,building,description,timestamp,duration,url]"
           f"{{0,{scan_depth}}}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as e:
        raise GateDoctorError(f"Jenkins builds API failed: {e}") from e

    needle = f"Pull Request #{pr_number}"
    candidates = []
    for b in data.get("builds", []):
        desc = b.get("description") or ""
        if needle not in desc:
            continue
        if b.get("result") == "ABORTED":
            continue
        if b.get("building"):
            # In progress; remember but only return if nothing else newer.
            candidates.append(b)
            continue
        candidates.append(b)
    if not candidates:
        return None
    # Newest-first by build number
    candidates.sort(key=lambda b: int(b.get("number") or 0), reverse=True)
    return candidates[0]


def fetch_jenkins_console(pipeline_url_or_num: str | int,
                          default_job: Optional[str] = None) -> str:
    """Fetch consoleText for a build.

    Accepts either a build number (preferred, requires ``default_job``) or
    any Jenkins URL containing the job name and build number. Handles both
    classic ``/job/<job>/<n>/...`` and BlueOcean
    ``/blue/organizations/jenkins/<job>/detail/<job>/<n>`` forms.
    """
    build_num: Optional[int] = None
    job = default_job
    if isinstance(pipeline_url_or_num, int):
        if not job:
            raise GateDoctorError("fetch_jenkins_console(int) requires default_job")
        build_num = pipeline_url_or_num
    else:
        s = pipeline_url_or_num
        m = (re.search(r"/job/([^/]+)/(\d+)(?:/|$)", s)
             or re.search(r"/blue/organizations/jenkins/([^/]+)/detail/[^/]+/(\d+)(?:/|$)", s))
        if not m:
            raise GateDoctorError(f"Cannot derive consoleText URL from: {pipeline_url_or_num}")
        job, build_num = m.group(1), int(m.group(2))
    url = f"{JENKINS_BASE}/job/{job}/{build_num}/consoleText"
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise GateDoctorError(f"Jenkins log fetch failed: {e}") from e


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


_PYLINT_LINE_RE = re.compile(
    r"^(?P<file>\S+?\.py):(?P<line>\d+):\s*\[(?P<rule>[A-Z]\d{4})\(([^)]+)\),\s*([^\]]*)\]\s*(?P<msg>.+)$"
)
_DT_DESIGN_LINE_RE = re.compile(
    r"^\[PYTHON\] Function '(?P<func>[^']+)' is missing: (?P<missing>.+)$"
)


@dataclass
class FailureEntry:
    stage: str
    rule: str            # e.g. "W1510", "DT_Design"
    file: str
    line: Optional[int]
    function: Optional[str]
    message: str


def parse_console_failures(console: str) -> list[FailureEntry]:
    """Parse a Jenkins console log into structured ``FailureEntry`` records."""
    text = _strip_ansi(console)
    entries: list[FailureEntry] = []

    for raw in text.splitlines():
        line = raw.rstrip()

        # strip Jenkins timestamp prefix '[2026-05-16T02:18:23.087Z] '
        line = re.sub(r"^\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*", "", line)

        if m := _DT_DESIGN_LINE_RE.match(line):
            entries.append(FailureEntry(
                stage="Check_DT_Design",
                rule="DT_Design",
                file="",  # not in the log line; caller resolves
                line=None,
                function=m.group("func"),
                message=f"missing docstring fields: {m.group('missing')}",
            ))
            continue

        if m := _PYLINT_LINE_RE.match(line):
            entries.append(FailureEntry(
                stage="Check_Pylint",
                rule=m.group("rule"),
                file=m.group("file"),
                line=int(m.group("line")),
                function=None,
                message=m.group("msg").strip(),
            ))
            continue

    return entries


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def _token() -> str:
    tok = os.environ.get("GITCODE_TOKEN", "").strip()
    if not tok:
        raise GateDoctorError("GITCODE_TOKEN not set. export GITCODE_TOKEN=<token>")
    return tok


def cmd_retest(args) -> int:
    owner, repo, pr = parse_pr_ref(args.pr)
    post_pr_comment(owner, repo, pr, "/retest", _token())
    print(f"/retest posted on {owner}/{repo}#{pr}")
    print(f"https://gitcode.com/{owner}/{repo}/pulls/{pr}")
    return 0


def cmd_check_pr(args) -> int:
    owner, repo, pr = parse_pr_ref(args.pr)
    post_pr_comment(owner, repo, pr, "/check-pr", _token())
    print(f"/check-pr posted on {owner}/{repo}#{pr}")
    return 0


def _build_report_from_jenkins(owner: str, repo: str, pr: int,
                               jenkins_build: dict, job: str,
                               labels: Optional[list[str]] = None) -> dict:
    """Pull Jenkins console for a build dict and turn it into a diagnose report.

    ``labels`` is the current set of PR labels. When present, bypass-build
    detection AND-s with ``ci-pipeline-passed not in labels`` — the bot only
    applies that label to real green builds, so a labelled fast SUCCESS is a
    legitimate lint-only run, not a bypass.
    """
    labels = labels or []
    build_num = int(jenkins_build["number"])
    build_url = jenkins_build.get("url") or f"{JENKINS_BASE}/job/{job}/{build_num}/"
    result = jenkins_build.get("result") or "BUILDING"
    console = fetch_jenkins_console(build_num, default_job=job)
    parsed = parse_console_failures(console)

    # Discover which stages failed by scanning for "Execute project of Check_X is failed"
    # or "[ERROR] ... is failed" markers. Also include any stage that produced parsed entries.
    stages_failed = set()
    stage_re = re.compile(r"Execute project of (\S+) is failed")
    for raw in console.splitlines():
        s = _strip_ansi(raw)
        m = stage_re.search(s)
        if m:
            stages_failed.add(m.group(1))
    for f in parsed:
        stages_failed.add(f.stage)

    # Smoke / test failures: pull pytest "FAILED ..." lines
    test_failures: list[dict] = []
    for raw in console.splitlines():
        s = _strip_ansi(raw)
        s = re.sub(r"^\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*", "", s)
        m = re.match(r"FAILED\s+(\S+?)::(\S+?)(?:\s*-\s*(.*))?$", s.strip())
        if m:
            test_failures.append({
                "file": m.group(1),
                "test": m.group(2),
                "summary": (m.group(3) or "").strip(),
            })

    # Excerpt: last ~200 lines that match common failure markers
    excerpt = []
    pat = re.compile(r"\[(ERROR|FAILURE)\]|FAILED\s+\S+::|FAILURE\b|"
                     r"\[PYTHON\] Function|pylint.*\[[WE]\d{4}|AssertionError|"
                     r"currentBuild\.result\s*=\s*FAILURE")
    for raw in console.splitlines():
        s = _strip_ansi(raw)
        if pat.search(s):
            excerpt.append(s)

    duration_ms = int(jenkins_build.get("duration") or 0)
    # A bypass build is a fake green: someone posted /retest while
    # pr-check-pass was missing, Jenkins ran a ~17-second no-op build that
    # returns SUCCESS but proves nothing. The bot does NOT apply the
    # ci-pipeline-passed label to those, so when the label IS present the
    # build is a legitimate (often lint-only) green even if duration < 60s.
    is_bypass_build = (
        result == "SUCCESS"
        and 0 < duration_ms < _BYPASS_DURATION_MS
        and not parsed and not test_failures and not stages_failed
        and "ci-pipeline-passed" not in labels
    )
    return {
        "pr": {"owner": owner, "repo": repo, "number": pr,
               "url": f"https://gitcode.com/{owner}/{repo}/pulls/{pr}"},
        "pipeline_number": build_num,
        "pipeline_url": build_url,
        "pipeline_result": result,
        "pipeline_duration_ms": duration_ms,
        "is_bypass_build": is_bypass_build,
        "labels": labels,
        "failed_stages": sorted(stages_failed),
        "failures": [asdict(f) for f in parsed],
        "test_failures": test_failures,
        # Filled in by cmd_diagnose from comment stream (parse_latest_failure).
        "check_pr_failure": None,
        "gate_trigger_blocked": None,
        # gate_ok aggregates pipeline-side + /check-pr-side verdicts. Set by
        # cmd_diagnose after the comment stream is parsed.
        "gate_ok": None,
        "raw_log_excerpt": "\n".join(excerpt[-200:]),
    }


def _diagnose_finished_build(owner, repo, pr, build, job) -> int:
    """Emit the report + gate_ok verdict for a completed Jenkins build."""
    # Fetch PR labels up front so bypass-build detection can AND on
    # whether the bot has applied `ci-pipeline-passed` (the
    # label-authoritative real-green signal).
    labels = get_pr_labels(owner, repo, pr, _token())
    report = _build_report_from_jenkins(owner, repo, pr, build, job,
                                        labels=labels)
    # The bot's /check-pr verdict is independent of the Jenkins build —
    # a "SUCCESS" build can be a 17-second bypass when the PR still
    # lacks the pr-check-pass label. Always attach the comment-side signals.
    comments = get_pr_comments(owner, repo, pr, _token())
    summary = parse_latest_failure(comments)
    if summary.check_pr_failure:
        report["check_pr_failure"] = summary.check_pr_failure
    if summary.gate_trigger_blocked:
        report["gate_trigger_blocked"] = summary.gate_trigger_blocked
    # Aggregate verdict: green when the bot has applied BOTH terminal
    # labels (`pr-check-pass` AND `ci-pipeline-passed`) on the PR.
    # Label-side truth supersedes the comment-stream-derived
    # `check_pr_failure` text (which can lag stale across builds) and
    # the script's `is_bypass_build` heuristic (which can false-
    # positive on lint-only sub-60s pipelines).
    labels_now = report.get("labels") or []
    has_both_labels = (
        "pr-check-pass" in labels_now and "ci-pipeline-passed" in labels_now
    )
    report["gate_ok"] = has_both_labels or (
        report["pipeline_result"] == "SUCCESS"
        and not report["is_bypass_build"]
        and report["check_pr_failure"] is None
        and report["gate_trigger_blocked"] is None
    )
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    # exit 0 even when pipeline failed — diagnose is informational
    return 0


def _diagnose_pending(owner, repo, pr, build, job) -> int:
    """Emit the report when no build is registered yet or one is still building."""
    comments = get_pr_comments(owner, repo, pr, _token())
    summary = parse_latest_failure(comments)
    # When the newest Jenkins build is still in progress, the comment-derived
    # `summary.pipeline_number` is a previous (stale) build. Override the
    # reported pipeline_number with the building one so the caller doesn't
    # act on stale failure data.
    if build and build.get("building"):
        pipe_num = int(build["number"])
        pipe_url = build.get("url")
        pipe_res = "BUILDING"
        stale_pipe = summary.pipeline_number
    else:
        pipe_num = summary.pipeline_number
        pipe_url = summary.pipeline_url
        pipe_res = None
        stale_pipe = None
    report = {
        "pr": {"owner": owner, "repo": repo, "number": pr,
               "url": f"https://gitcode.com/{owner}/{repo}/pulls/{pr}"},
        "pipeline_number": pipe_num,
        "pipeline_url": pipe_url,
        "pipeline_result": pipe_res,
        "previous_pipeline_number": stale_pipe,
        "failed_stages": [s.name for s in summary.failed_stages] if not pipe_res else [],
        "failures": [],
        "test_failures": [],
        "check_pr_failure": summary.check_pr_failure,
        "gate_trigger_blocked": summary.gate_trigger_blocked,
        "raw_log_excerpt": "",
    }
    if pipe_res == "BUILDING":
        report["note"] = (f"Build #{pipe_num} is currently running for "
                          f"{owner}/{repo}#{pr}. Re-run diagnose once it "
                          f"completes for failure detail.")
    if summary.pipeline_url:
        console = fetch_jenkins_console(
            summary.pipeline_number or summary.pipeline_url, default_job=job)
        failures = parse_console_failures(console)
        report["failures"] = [asdict(f) for f in failures]
    # gate_ok aggregate (same definition as the Jenkins-direct path).
    # In the fallback path we don't have build duration, so we can't detect
    # a bypass build; we only mark gate_ok = True when comments show no
    # failed stages AND no /check-pr fail AND no trigger-blocked notice.
    report["gate_ok"] = (
        not report["failed_stages"]
        and report["check_pr_failure"] is None
        and report["gate_trigger_blocked"] is None
        and pipe_res != "BUILDING"
    )
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if (summary.failed_stages or summary.check_pr_failure
                 or summary.gate_trigger_blocked or summary.pipeline_number) else 1


def cmd_diagnose(args) -> int:
    """Emit a JSON report of the PR's latest pipeline + ``/check-pr`` state."""
    owner, repo, pr = parse_pr_ref(args.pr)

    # Source of truth #1: Jenkins API — newest non-ABORTED build for this PR.
    job = jenkins_job_for(owner, repo, getattr(args, "jenkins_job", None))
    build = find_latest_pr_build(pr, job=job)
    if build and not build.get("building"):
        return _diagnose_finished_build(owner, repo, pr, build, job)
    # Source of truth #2: only PR comments (no build registered yet, or building).
    return _diagnose_pending(owner, repo, pr, build, job)


def cmd_autofix(args) -> int:
    """Emit a structured plan; the LLM caller executes the patch+commit step."""
    owner, repo, pr = parse_pr_ref(args.pr)

    plan: dict = {
        "pr": {"owner": owner, "repo": repo, "number": pr,
               "url": f"https://gitcode.com/{owner}/{repo}/pulls/{pr}"},
        "diagnose_only": False,
        "actions": [],
    }

    # Prefer Jenkins-direct latest build over PR comments.
    job = jenkins_job_for(owner, repo, getattr(args, "jenkins_job", None))
    build = find_latest_pr_build(pr, job=job)
    if build and not build.get("building"):
        labels = get_pr_labels(owner, repo, pr, _token())
        report = _build_report_from_jenkins(owner, repo, pr, build, job,
                                            labels=labels)
        plan.update(report)
        if report["pipeline_result"] == "SUCCESS":
            plan["actions"] = ["Pipeline already green - nothing to do."]
        else:
            plan["actions"] = [
                "Read references/common-failures.md for each stage in failed_stages.",
                ("For each entry in `failures` and `test_failures`, run the "
                 "3-question triage from references/workflow.md before assuming "
                 "the failure is unrelated. Out-of-diff failures CAN be "
                 "PR-induced via submodule pins, env vars, or site-packages."),
                "Edit only the files truly responsible.",
                "Run pylint --rcfile=ms_pylintrc on touched .py files; ensure 10.00/10.",
                (f"`autogit commit -m \"fix(...): satisfy "
                 f"{', '.join(plan['failed_stages']) or 'gate'}\"`"),
                f"`gate_doctor.py retest {pr}`",
            ]
        json.dump(plan, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    comments = get_pr_comments(owner, repo, pr, _token())
    summary = parse_latest_failure(comments)
    if summary.pipeline_url:
        console = fetch_jenkins_console(
            summary.pipeline_number or summary.pipeline_url, default_job=job)
        failures = parse_console_failures(console)
        plan["pipeline_number"] = summary.pipeline_number
        plan["pipeline_url"] = summary.pipeline_url
        plan["failed_stages"] = sorted({f.stage for f in failures}
                                       or {s.name for s in summary.failed_stages})
        plan["failures"] = [asdict(f) for f in failures]
        plan["actions"] = [
            "Read references/common-failures.md for each unique stage above.",
            "Edit the offending files using the Read+Edit tools.",
            "Run pylint --rcfile=ms_pylintrc on touched .py files; ensure 10.00/10.",
            f"`autogit commit -m \"fix(...): satisfy {', '.join(plan['failed_stages'])}\"`",
            f"`gate_doctor.py retest {pr}`",
        ]
    elif summary.check_pr_failure:
        plan["check_pr_failure"] = summary.check_pr_failure
        plan["failed_stages"] = ["/check-pr"]
        plan["actions"] = [
            "Update PR description to satisfy the failing check-list / template item.",
            "Use `autogit update <pr> --title --body` to push the new description.",
            f"`gate_doctor.py check-pr {pr}`",
        ]
    else:
        plan["diagnose_only"] = True
        plan["note"] = "No failure detected. Pipeline may not have run yet."
        plan["actions"] = [f"`gate_doctor.py retest {pr}` if you want to trigger a fresh run."]

    json.dump(plan, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0



# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all four ``gate_doctor`` subcommands."""
    p = argparse.ArgumentParser(prog="gate_doctor",
                                description=__doc__.splitlines()[0] if __doc__ else "")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("retest", cmd_retest),
                     ("check-pr", cmd_check_pr),
                     ("diagnose", cmd_diagnose),
                     ("autofix", cmd_autofix)):
        sp = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else "")
        sp.add_argument(
            "pr",
            help=("PR reference. A bare number / '#N' resolves to the default "
                  "repo (mindspore/hyper-parallel). Use full URL "
                  "'https://gitcode.com/<owner>/<repo>/pull/<N>' or "
                  "'<owner>/<repo>#<N>' to target another repo "
                  "(e.g. mindspore/mindspore).")
        )
        if name in ("diagnose", "autofix"):
            sp.add_argument(
                "--jenkins-job",
                default=None,
                help=("Override the Jenkins gate job. By default the script "
                      "picks the right one from the (owner, repo) pair. "
                      "Pass this when the skill is used in a repo that isn't "
                      "in the built-in mapping.")
            )
        sp.set_defaults(func=fn)
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except GateDoctorError as e:
        print(f"gate-doctor: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
