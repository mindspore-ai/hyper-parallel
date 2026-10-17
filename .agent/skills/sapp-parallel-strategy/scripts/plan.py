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
"""Plan recompute layer by layer for one layout: which layers can run without it within memory.

    python <this skill's folder>/scripts/plan.py --family qwen3_5_moe --cluster a3 --layers 8 --seq 8192 \\
        --global-batch 64 --micro-batch 1 --dies 64 --layout "EP=2,OP=32" --request "the request"

ND's search-config path chooses each layer's mode among those the trust record allows, against a budget stated
from the cluster's own measurements: what a die leaves a rank, less what the allocator reserved beyond
allocation at full recompute, since ND prices allocated memory and a run dies on reserved. ND writes the plan
into the trainer's yaml (``activation_checkpoint.layers``); the layout's expert shard group is set first, so the
yaml launches what was priced. Everything lands in the run folder, as with ``ask.py``.
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

import answer
import ask

# A degree of --layout and where the trainer yaml states it; DP is what the dies leave.
FIELDS = {"TP": ("accelerator", "tp_size"), "CP": ("accelerator", "cp_size"), "EP": ("accelerator", "ep_size"),
          "PP": ("accelerator", "pp_size"), "OP": ("fsdp_config", "dp_shard_size")}
OUTPUT = " - OUTPUT - "


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The request's values and the layout to plan, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ask.add_run_arguments(parser)
    parser.add_argument("--layout", required=True,
                        help='the layout to plan, for example "EP=2,OP=32"; TP, CP and PP are 1 unless given')
    args = parser.parse_args(argv)
    if not (args.family or args.yaml):
        parser.error("give --family, or --yaml with the asker's trainer yaml")
    args.recompute = ["full"]  # the plan names the layers that run without it
    return args


def layout_degrees(spec: str, dies: int) -> Dict[str, int]:
    """``EP=2,OP=32`` as every degree of the layout, DP being what the dies leave."""
    degrees = {"TP": 1, "CP": 1, "PP": 1}
    for item in spec.split(","):
        name, _, value = item.partition("=")
        name = "TP" if name.strip().upper() == "MP" else name.strip().upper()
        if name not in FIELDS:
            raise SystemExit(f"--layout names {name!r}; it takes TP, CP, EP, PP and OP")
        degrees[name] = int(value)
    if "EP" not in degrees or "OP" not in degrees:
        raise SystemExit("--layout needs EP and OP")
    degrees["DP"] = dies // (degrees["TP"] * degrees["CP"] * degrees["PP"])
    return degrees


def _fix_layout(doc: Dict[str, Any], degrees: Dict[str, int]) -> None:
    """State the layout in the yaml, with the expert shard group ND prices under ``expert_shard: group``."""
    for name, (section, field) in FIELDS.items():
        doc.setdefault(section, {})[field] = degrees[name]
    if (doc.get("context") or {}).get("expert_shard") == "group":
        doc["fsdp_config"]["edp_shard_size"] = degrees["DP"] * degrees["CP"] * degrees["TP"] // degrees["EP"]


def search_config(run: Path, doc: Dict[str, Any], degrees: Dict[str, int], cluster: Dict[str, Any],
                  trust: Dict[str, Any]) -> Dict[str, Any]:
    """The search config: the layout fixed, recompute chosen per layer, the budget the cluster's reserve leaves."""
    dies = int(doc["context"]["device_num"])
    per_node = int(cluster.get("dies_per_node", 16))
    modes = [mode for mode in ("off", "full") if mode not in (trust.get("exclude_modes") or {})]
    return {
        "train_yaml": (run / "run.yaml").as_posix(),
        "cluster": {"num_nodes": max(1, dies // per_node), "cards_per_node": per_node,
                    "device_type": cluster["device"].lower(), "device_memory_gb": cluster["usable_gib"]},
        "parallelism": {"dp": "auto", "fsdp": degrees["OP"], "ep": degrees["EP"], "cp": degrees["CP"],
                        "tp": degrees["TP"], "pp": degrees["PP"]},
        "constraint": {"global_batch_size": int(doc["training"]["global_batch_size"]),
                       "memory_limit_gb": round(cluster["usable_gib"] - cluster["reserve_at_full_gib"], 2)},
        "recompute": "per_layer",
        "recompute_modes": modes,
    }


def plan_lines(lines: Sequence[str]) -> List[str]:
    """ND's statement of the plan: each layer's mode, the trainer's form of it and the strategy's line."""
    out: List[str] = []
    inside = False
    for raw in lines:
        text = raw.split(OUTPUT, 1)[1].strip() if OUTPUT in raw else raw.strip()
        if text.startswith("Recompute of the best configuration"):
            inside = True
            continue
        if inside and (not text or not text.startswith("layer")):
            inside = False
        if inside or text.startswith(("As the trainer runs it", "Optimal strategy")):
            out.append(text)
    return out


def _report(args: argparse.Namespace, run: Path, facts: Dict[str, Any], lines: Sequence[str]) -> List[str]:
    """The answer: status, how far a plan was checked, the budget and why, ND's plan, notices, files."""
    cluster, trust, versions, config = facts["cluster"], facts["trust"], facts["versions"], facts["config"]
    per_layer = trust.get("per_layer") or {"status": "Never checked", "note": "No plan of this family was timed."}
    notices = [line.strip() for line in lines if any(word in line for word in ask.NOTICES)]
    limit = config["constraint"]["memory_limit_gb"]
    planned = plan_lines(lines)
    return [
        f"Status: {trust['status']}. {trust['may_say']}",
        f"Plans per layer: {per_layer['status']}. {per_layer['note']}",
        f"Priced by ND {versions['nd_commit'][:8]}, Transformers {versions['transformers']}, cluster {args.cluster}, "
        f"in {facts['seconds']:.0f} s.",
        *ask.version_notes(trust, versions),
        "",
        f"Layout planned: {args.layout} on {config['cluster']['num_nodes'] * config['cluster']['cards_per_node']} "
        f"dies, data parallel what is left.",
        f"Budget stated to ND: {cluster['usable_gib']} GiB a die leaves a rank, less "
        f"{cluster['reserve_at_full_gib']} GiB the allocator reserved beyond allocation at full recompute: "
        f"{limit} GiB. {cluster.get('reserve_note', '')}".strip(),
        per_layer.get("allocator_note", ""),
        "",
        "The plan, as ND states it:" if planned else "ND printed no plan: read nd.log.",
        *[f"- {line}" for line in planned],
        "",
        "ND's notices, verbatim:" if notices else "ND printed no notice.",
        *[f"- {line}" for line in notices],
        "",
        *facts["launches"],
        f"Run kept in {run}",
    ]


def _command(run: Path, doc: Dict[str, Any], cluster: Dict[str, Any]) -> List[str]:
    """The ND command that plans the run ``run/run.yaml`` describes under ``run/search.yaml``."""
    return [sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", "-f", "hyper_v2",
            "-y", str(run / "run.yaml"), "-s", str(run / "search.yaml"), "-A", cluster["device"],
            "-d", str(doc["context"]["device_num"]), "-b", str(doc["training"]["global_batch_size"]),
            "-o", str(run / "nd_out"), "-v", "2"] + (["-c", str(cluster["ratios"])] if cluster["ratios"] else [])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Describe the run with the layout fixed, plan its recompute per layer with ND, keep it, print it."""
    args = _arguments(argv)
    repo = ask.repo_root()
    trust = answer.trust_entry(args.family or "*", args.cluster)
    cluster = ask.cluster_record(args.cluster, trust, None)
    if "usable_gib" not in cluster or "reserve_at_full_gib" not in cluster:
        sys.stdout.write(f"Cluster {args.cluster} has no measured reserve, so no budget can be stated for a plan.\n")
        return 1
    doc = ask.describe(args)
    degrees = layout_degrees(args.layout, int(doc["context"]["device_num"]))
    _fix_layout(doc, degrees)
    run = ask.run_folder(args, repo, doc)
    config = search_config(run, doc, degrees, cluster, trust)
    (run / "search.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = _command(run, doc, cluster)
    (run / "command.txt").write_text(f"cd {repo}\n" + " ".join(command) + "\n", encoding="utf-8")
    code, seconds, lines = ask.price(command, repo, run, args.timeout)
    if code != 0 or not (run / "nd_out" / "resolved.yaml").is_file():
        sys.stdout.write(f"ND failed (exit {code}) after {seconds:.0f} s; the last lines of {run / 'nd.log'}:\n")
        sys.stdout.write("\n".join(lines[-25:]) + "\n")
        return 1
    shutil.copyfile(run / "nd_out" / "resolved.yaml", run / "launch.yaml")
    facts = {"cluster": cluster, "trust": trust, "versions": ask.library_versions(repo, cluster["ratios"]),
             "seconds": seconds, "config": config,
             "launches": [f"Trainer yaml for this plan: {run / 'launch.yaml'}; set its tokenizer and data paths "
                          "on the cluster."]}
    report = _report(args, run, facts, lines)
    (run / "answer.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
