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
"""Command-line front of the bring-up asset benchmark.

    bench.py snapshot  --repo-root R --ref SHA --dest DIR [--inject PATCH]
    bench.py suite     --assets-root DIR --catalog YAML [--paths YAML] --out JSON
    bench.py score     --trial-dir DIR --out JSON
    bench.py calibrate --scores JSON... --out JSON
    bench.py compare   --arm-a JSON... --arm-b JSON... --resolution JSON --out JSON
    bench.py diff      --card-a JSON --card-b JSON [--injection NAME] --out JSON
    bench.py mask      --snapshot DIR --family NAME --quarantine DIR --out JSON

Every subcommand writes one JSON artifact and prints a one-line summary;
exit 0 means the artifact was produced, not that the result is favorable —
reading the verdict is the operator's job, the file is the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pylint: disable=sys-path-mutation
import bench_lib  # pylint: disable=wrong-import-position


def _write(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")


def _load_scores(paths: list[str]) -> list[bench_lib.TrialScore]:
    scores = []
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("schema", None)
        scores.append(bench_lib.TrialScore(**raw))
    return scores


def _cmd_snapshot(args: argparse.Namespace) -> str:
    inject = Path(args.inject) if args.inject else None
    bench_lib.snapshot_assets(Path(args.repo_root), args.ref, Path(args.dest), inject)
    suffix = f" + injection {inject.name}" if inject else ""
    return f"snapshot {args.ref} -> {args.dest}{suffix}"


def _cmd_suite(args: argparse.Namespace) -> str:
    models = bench_lib.load_catalog(Path(args.catalog))
    dirs = bench_lib.resolve_model_dirs(
        models, Path(args.paths) if args.paths else None)
    card = bench_lib.run_suite(Path(args.assets_root), models, dirs,
                               timeout=args.timeout)
    _write(Path(args.out), card)
    return f"suite {card['ok_cells']}/{card['total_cells']} cells ok -> {args.out}"


def _cmd_score(args: argparse.Namespace) -> str:
    score = bench_lib.score_trial(Path(args.trial_dir))
    _write(Path(args.out), score.to_json())
    return (f"trial {score.model_id}: {score.gaps_total} gaps, "
            f"{score.gates_passed} gates passed -> {args.out}")


def _cmd_calibrate(args: argparse.Namespace) -> str:
    resolution = bench_lib.calibrate(_load_scores(args.scores))
    _write(Path(args.out), resolution)
    return (f"resolution n={resolution['n']} "
            f"mdd={resolution['min_detectable_gap_difference']} -> {args.out}")


def _cmd_compare(args: argparse.Namespace) -> str:
    resolution = json.loads(Path(args.resolution).read_text(encoding="utf-8"))
    verdict = bench_lib.compare_trials(_load_scores(args.arm_a),
                                       _load_scores(args.arm_b), resolution)
    _write(Path(args.out), verdict)
    return f"A/B verdict: {verdict['verdict']} -> {args.out}"


def _cmd_mask(args: argparse.Namespace) -> str:
    manifest = bench_lib.mask_model_family(
        Path(args.snapshot), args.family, Path(args.quarantine))
    _write(Path(args.out), manifest)
    return (f"masked '{manifest['family']}': {len(manifest['moved'])} entries moved, "
            f"{len(manifest['residual_mentions'])} residual mentions -> {args.out}")


def _cmd_diff(args: argparse.Namespace) -> str:
    card_a = json.loads(Path(args.card_a).read_text(encoding="utf-8"))
    card_b = json.loads(Path(args.card_b).read_text(encoding="utf-8"))
    if args.injection:
        result = bench_lib.selfcheck_injection(card_a, card_b, args.injection)
        _write(Path(args.out), result)
        status = "detected" if result["detected"] else "NOT detected (blind spot)"
        return f"injection {args.injection}: {status} -> {args.out}"
    result = bench_lib.selfcheck_aa(card_a, card_b)
    _write(Path(args.out), result)
    status = "identical" if result["identical"] else f"{len(result['flips'])} flips"
    return f"A/A: {status} -> {args.out}"


def main(argv: list[str] | None = None) -> int:
    """Dispatch one subcommand; BenchError prints as a one-line failure."""
    parser = argparse.ArgumentParser(prog="bench.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="materialize an asset snapshot")
    p_snap.add_argument("--repo-root", required=True)
    p_snap.add_argument("--ref", required=True)
    p_snap.add_argument("--dest", required=True)
    p_snap.add_argument("--inject", default=None,
                        help="self-check damage patch to apply inside the snapshot")
    p_snap.set_defaults(func=_cmd_snapshot)

    p_suite = sub.add_parser("suite", help="run the deterministic check suite")
    p_suite.add_argument("--assets-root", required=True)
    p_suite.add_argument("--catalog", required=True)
    p_suite.add_argument("--paths", default=None,
                         help=f"local id->model-dir YAML; or ${bench_lib.PATHS_ENV}")
    p_suite.add_argument("--timeout", type=int, default=900)
    p_suite.add_argument("--out", required=True)
    p_suite.set_defaults(func=_cmd_suite)

    p_score = sub.add_parser("score", help="score one clean-room trial")
    p_score.add_argument("--trial-dir", required=True)
    p_score.add_argument("--out", required=True)
    p_score.set_defaults(func=_cmd_score)

    p_cal = sub.add_parser("calibrate", help="repeat-run spread -> resolution")
    p_cal.add_argument("--scores", nargs="+", required=True)
    p_cal.add_argument("--out", required=True)
    p_cal.set_defaults(func=_cmd_calibrate)

    p_cmp = sub.add_parser("compare", help="three-valued A/B on trial scores")
    p_cmp.add_argument("--arm-a", nargs="+", required=True)
    p_cmp.add_argument("--arm-b", nargs="+", required=True)
    p_cmp.add_argument("--resolution", required=True)
    p_cmp.add_argument("--out", required=True)
    p_cmp.set_defaults(func=_cmd_compare)

    p_mask = sub.add_parser("mask", help="quarantine a model family's own "
                                         "implementation from a snapshot")
    p_mask.add_argument("--snapshot", required=True)
    p_mask.add_argument("--family", required=True)
    p_mask.add_argument("--quarantine", required=True,
                        help="destination OUTSIDE the snapshot for the moved entries")
    p_mask.add_argument("--out", required=True)
    p_mask.set_defaults(func=_cmd_mask)

    p_diff = sub.add_parser("diff", help="A/A identity or injection detection")
    p_diff.add_argument("--card-a", required=True)
    p_diff.add_argument("--card-b", required=True)
    p_diff.add_argument("--injection", default=None,
                        help="name the injection to assert detection instead of identity")
    p_diff.add_argument("--out", required=True)
    p_diff.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)
    try:
        print(args.func(args), flush=True)
    except bench_lib.BenchError as err:
        print(f"[bench] {err}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
