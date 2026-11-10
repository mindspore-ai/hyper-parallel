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
#!/usr/bin/env python3
"""
MoE Facts Extractor — Main Entry Point

Extracts 15 fixed artifacts from MoE training logs/profiles/checkpoints.
Pure fact extraction — no diagnosis, no recommendations.

Usage:
    python extract_moe_facts.py \
        --log-dir /path/to/logs \
        --output-dir ./moe_facts
"""

import argparse
import csv
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import sub-modules
from _moe_load_analysis import LoadAnalyzer
from _moe_comm_analysis import CommunicationAnalyzer
from _moe_routing_check import RoutingValidator


class MoEFactsExtractor:
    """Main orchestrator for MoE facts extraction"""

    def __init__(self, log_dir: Optional[Path], profile_dir: Optional[Path],
                 checkpoint_dir: Optional[Path], output_dir: Path,
                 total_steps: Optional[int] = None, ranks: Optional[list] = None):
        self.log_dir = log_dir
        self.profile_dir = profile_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = Path(output_dir)

        # Initialize sub-analyzers
        self.load_analyzer = LoadAnalyzer(steps=total_steps, ranks=ranks)
        self.comm_analyzer = CommunicationAnalyzer(steps=total_steps, ranks=ranks)
        self.routing_validator = RoutingValidator(total_steps=total_steps)

        # Artifacts to generate
        self.artifacts = []

    def discover_inputs(self) -> Dict[str, Any]:
        """Phase 1: Discover and validate input sources"""
        print("[Phase 1] Discovering input sources...")

        sources = {
            "log_files": [],
            "profile_files": [],
            "checkpoint_files": [],
        }

        # Discover log files
        if self.log_dir and self.log_dir.exists():
            log_files = list(self.log_dir.rglob("*.log"))
            sources["log_files"] = [str(f) for f in log_files]
            print(f"  Found {len(log_files)} log files")

        # Discover profiler output
        if self.profile_dir and self.profile_dir.exists():
            profile_files = list(self.profile_dir.rglob("*.json")) + \
                           list(self.profile_dir.rglob("*.db"))
            sources["profile_files"] = [str(f) for f in profile_files]
            print(f"  Found {len(profile_files)} profiler files")

        # Discover checkpoints
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            ckpt_files = list(self.checkpoint_dir.rglob("*.pt")) + list(self.checkpoint_dir.rglob("*.pth")) + list(self.checkpoint_dir.rglob("*.json"))
            sources["checkpoint_files"] = [str(f) for f in ckpt_files]
            print(f"  Found {len(ckpt_files)} checkpoint files")

        # Validate at least one source
        total_files = sum(len(v) for v in sources.values())
        if total_files == 0:
            raise ValueError("No input sources found. Provide at least one of: "
                           "--log-dir, --profile-dir, --checkpoint-dir")

        print(f"  Total input files: {total_files}")
        return sources

    def extract_facts(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Extract MoE facts from all sources"""
        print("\n[Phase 2] Extracting MoE facts...")

        facts = {
            "load_distribution": {},
            "communication": {},
            "correctness": {},
            "stragglers": {},
        }

        # Extract load distribution
        if sources["log_files"]:
            print("  Analyzing expert load distribution...")
            facts["load_distribution"] = self.load_analyzer.analyze(
                sources["log_files"]
            )

        # Extract communication metrics
        if sources["log_files"] or sources["profile_files"]:
            print("  Analyzing All-to-All communication...")
            facts["communication"] = self.comm_analyzer.analyze(
                log_files=sources["log_files"],
                profile_files=sources["profile_files"],
            )

        # Validate routing correctness
        if sources["checkpoint_files"]:
            print("  Validating routing integrity...")
            facts["correctness"] = self.routing_validator.validate(
                sources["checkpoint_files"]
            )

        facts["input_sources"] = sources
        return facts

    def generate_artifacts(self, facts: Dict[str, Any], sources: Dict[str, Any]):
        """Phase 3: Generate 15 fixed output artifacts"""
        print("\n[Phase 3] Generating output artifacts...")

        self.artifacts = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1-3. Load distribution
        self._generate_load_artifacts(facts["load_distribution"])

        # 6-8. Communication
        self._generate_comm_artifacts(facts["communication"])

        # 9-11. Correctness
        self._generate_correctness_artifacts(facts["correctness"])

        # 12-13. Stragglers
        self._generate_straggler_artifacts(facts["stragglers"])

        # 14-15. Raw data
        self._generate_raw_data(facts, sources)

        # Write the index last so it contains the complete artifact list.
        self._generate_index(facts, sources)

        print(f"  Generated {len(self.artifacts)} artifacts")

    def _generate_index(self, facts: Dict, sources: Dict):
        """Generate moe_facts.json and moe_facts.md"""
        # JSON index
        index_names = ["moe_facts.json", "moe_facts.md"] + list(self.artifacts)
        index = {
            "extraction_time": self._get_timestamp(),
            "input_sources": sources,
            "summary": self._compute_summary(facts),
            "evidence_status": self._evidence_status(facts),
            "artifacts": index_names,
        }

        index_path = self.output_dir / "moe_facts.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        self.artifacts.append("moe_facts.json")

        # Markdown summary
        md_content = self._generate_markdown_summary(facts)
        md_path = self.output_dir / "moe_facts.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        self.artifacts.append("moe_facts.md")

    def _generate_load_artifacts(self, load_data: Dict):
        """Generate load distribution CSVs"""
        load_data = load_data or {}

        # 3. expert_load_per_layer.csv
        self._write_csv("expert_load_per_layer.csv",
                       ["layer_id", "expert_id", "mean_tokens", "std_tokens",
                        "min_tokens", "max_tokens", "utilization_ratio"],
                       load_data.get("per_layer", []))

        # 4. expert_load_timeline.csv
        self._write_csv("expert_load_timeline.csv",
                       ["step", "layer_id", "expert_id", "tokens_assigned",
                        "routing_score_mean"],
                       load_data.get("timeline", []))

        # 5. token_distribution_summary.csv
        self._write_csv("token_distribution_summary.csv",
                       ["layer_id", "total_tokens", "num_experts",
                        "mean_per_expert", "std_per_expert",
                        "load_imbalance_factor", "gini_coefficient"],
                       load_data.get("summary", []))

    def _generate_comm_artifacts(self, comm_data: Dict):
        """Generate communication CSVs"""
        comm_data = comm_data or {}

        # 6. alltoall_timing.csv
        self._write_csv("alltoall_timing.csv",
                       ["step", "layer_id", "direction", "duration_ms",
                        "volume_mb", "bandwidth_gbps", "ep_size", "rank"],
                       comm_data.get("timing", []))

        # 7. alltoall_volume.csv
        self._write_csv("alltoall_volume.csv",
                       ["layer_id", "total_dispatch_mb", "total_combine_mb",
                        "tokens_exchanged", "avg_token_size_bytes"],
                       comm_data.get("volume", []))

        # 8. communication_breakdown.csv
        self._write_csv("communication_breakdown.csv",
                       ["step", "compute_ms", "alltoall_ms", "other_comm_ms",
                        "alltoall_ratio", "total_step_time_ms", "rank"],
                       comm_data.get("breakdown", []))

    def _generate_correctness_artifacts(self, correctness_data: Dict):
        """Generate correctness check CSVs"""
        correctness_data = correctness_data or {}

        # 9. routing_integrity.csv
        self._write_csv("routing_integrity.csv",
                       ["layer_id", "step", "input_tokens", "output_tokens",
                        "tokens_matched", "integrity_status"],
                       correctness_data.get("routing", []))

        # 10. gradient_flow_check.csv
        self._write_csv("gradient_flow_check.csv",
                       ["layer_id", "expert_id", "param_name", "has_grad",
                        "grad_norm", "grad_zero_ratio"],
                       correctness_data.get("gradients", []))

        # 11. expert_coverage.csv
        self._write_csv("expert_coverage.csv",
                       ["layer_id", "expert_id", "steps_activated",
                        "activation_ratio", "zero_assignment_count"],
                       correctness_data.get("coverage", []))

    def _generate_straggler_artifacts(self, straggler_data: Dict):
        """Generate straggler detection CSVs"""
        straggler_data = straggler_data or {}

        # 12. rank_timing_distribution.csv
        self._write_csv("rank_timing_distribution.csv",
                       ["rank", "mean_step_time_ms", "std_ms", "min_ms",
                        "max_ms", "straggler_score"],
                       straggler_data.get("timing", []))

        # 13. straggler_analysis.csv
        self._write_csv("straggler_analysis.csv",
                       ["rank", "is_straggler", "slowdown_ratio",
                        "suspected_cause", "expert_load_variance"],
                       straggler_data.get("analysis", []))

    def _generate_raw_data(self, facts: Dict, sources: Dict):
        """Generate raw data files"""
        # 14. raw_metrics.json
        raw_path = self.output_dir / "raw_metrics.json"
        with open(raw_path, 'w') as f:
            json.dump(facts, f, indent=2)
        self.artifacts.append("raw_metrics.json")

        # 15. profiler_metadata.json
        metadata = {
            "extraction_time": self._get_timestamp(),
            "input_sources": sources,
            "extractor_version": "1.0.0",
            "model_config": facts.get("model_config", {}),
            "hardware": facts.get("hardware", {}),
        }
        meta_path = self.output_dir / "profiler_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.artifacts.append("profiler_metadata.json")

    def _write_csv(self, filename: str, headers: List[str], rows: List):
        """Helper to write CSV file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        self.artifacts.append(filename)

    def _compute_summary(self, facts: Dict) -> Dict:
        """Keep the optimizer handoff fields stable and missing evidence null."""
        load = facts.get("load_distribution", {})
        comm = facts.get("communication", {})
        correctness = facts.get("correctness", {})
        imbalance = [float(row[5]) for row in load.get("summary", []) if row[5] is not None]
        ratios = [float(row[4]) for row in comm.get("breakdown", []) if row[4] is not None]
        slowdowns = [float(row[2]) for row in facts.get("stragglers", {}).get("analysis", []) if row[2] is not None]
        return {"steps_analyzed": load.get("num_steps"), "layers_analyzed": load.get("num_layers"),
                "experts_per_layer": load.get("num_experts"), "ranks_analyzed": comm.get("num_ranks"),
                "mean_load_imbalance_factor": sum(imbalance) / len(imbalance) if imbalance else None,
                "mean_alltoall_ratio": sum(ratios) / len(ratios) if ratios else None,
                "max_straggler_slowdown": max(slowdowns) if slowdowns else None,
                "routing_integrity": correctness.get("routing_status", "UNVERIFIED"),
                "gradient_flow": correctness.get("gradient_status", "UNVERIFIED")}

    def _generate_markdown_summary(self, facts: Dict) -> str:
        """Generate human-readable markdown summary"""
        summary = self._compute_summary(facts)

        md = f"""# MoE Training Facts Summary

**Extraction Time:** {self._get_timestamp()}

## Input Sources

- Log files: {len(facts.get('input_sources', {}).get('log_files', []))}
- Profiler files: {len(facts.get('input_sources', {}).get('profile_files', []))}
- Checkpoint files: {len(facts.get('input_sources', {}).get('checkpoint_files', []))}

## Analysis Scope

- Steps analyzed: {summary.get('steps_analyzed', 'N/A')}
- Layers analyzed: {summary.get('layers_analyzed', 'N/A')}
- Experts per layer: {summary.get('experts_per_layer', 'N/A')}
- Ranks analyzed: {summary.get('ranks_analyzed', 'N/A')}

## Key Metrics

- Mean load imbalance factor: {summary.get('mean_load_imbalance_factor')}
- Mean All-to-All overhead ratio: {summary.get('mean_alltoall_ratio')}
- Routing integrity: {facts.get('correctness', {}).get('routing_status', 'N/A')}
- Gradient flow: {facts.get('correctness', {}).get('gradient_status', 'N/A')}

## Generated Artifacts

{chr(10).join(f'- {a}' for a in ['moe_facts.json', 'moe_facts.md'] + [name for name in self.artifacts if name != 'moe_facts.json'])}

---

**Note:** This is pure fact extraction. For interpretation and recommendations,
consult domain experts or use the moe-optimizer skill. Null/empty CSV cells mean unavailable.
Communication ratios use summed operation durations and may include overlap.
Straggler attribution is unavailable without aligned per-rank observations.
"""
        return md

    def _get_timestamp(self) -> str:
        """Get ISO timestamp"""
        return datetime.now().isoformat()

    def validate_output(self):
        """Phase 4: Validate all artifacts generated"""
        print("\n[Phase 4] Validating output...")

        expected_count = 15
        actual_count = len(self.artifacts)

        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected exactly {expected_count} artifacts, generated {actual_count}"
            )
        print(f"  ✓ All {expected_count} artifacts generated")

        # Check file existence
        for artifact in self.artifacts:
            filepath = self.output_dir / artifact
            if not filepath.exists():
                raise FileNotFoundError(f"Artifact not found: {filepath}")

        index = json.loads((self.output_dir / "moe_facts.json").read_text())
        if len(set(self.artifacts)) != 15 or set(index["artifacts"]) != set(self.artifacts):
            raise ValueError("Artifact index is incomplete or duplicated")
        print("  ✓ All artifacts validated")

    def _evidence_status(self, facts: Dict) -> str:
        summary = self._compute_summary(facts)
        complete = all(summary.get(key) is not None for key in (
            "mean_load_imbalance_factor", "mean_alltoall_ratio", "max_straggler_slowdown"))
        complete = complete and summary["routing_integrity"] == "PASS" and summary["gradient_flow"] == "PASS"
        return "COMPLETE" if complete else "PARTIAL"

    def print_summary(self, facts: Dict):
        """Print structured JSON summary to stdout"""
        summary = self._compute_summary(facts)

        output = {
            "status": "EXTRACTED",
            "evidence_status": self._evidence_status(facts),
            "output_dir": str(self.output_dir),
            "artifacts_generated": len(self.artifacts),
            "summary": summary,
        }

        print("\n" + "="*60)
        print(json.dumps(output, indent=2))
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MoE training facts from logs/profiles/checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From logs only
  python extract_moe_facts.py --log-dir /path/to/logs --output-dir ./moe_analysis

  # From logs + profiler
  python extract_moe_facts.py \\
    --log-dir /path/to/logs \\
    --profile-dir /path/to/profiler_output \\
    --output-dir ./moe_analysis

  # Full analysis
  python extract_moe_facts.py \\
    --log-dir /path/to/logs \\
    --profile-dir /path/to/profiler_output \\
    --checkpoint-dir /path/to/checkpoint \\
    --output-dir ./moe_analysis
        """
    )

    parser.add_argument("--log-dir", type=Path,
                       help="Directory containing training logs (*.log)")
    parser.add_argument("--profile-dir", type=Path,
                       help="Profiler output directory (optional)")
    parser.add_argument("--checkpoint-dir", type=Path,
                       help="Model checkpoint directory (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("./moe_facts_output"),
                       help="Output directory for artifacts (default: ./moe_facts_output)")
    parser.add_argument("--steps", type=int,
                       help="Number of steps to analyze (default: all)")
    parser.add_argument("--ranks", type=str,
                       help="Comma-separated rank IDs (default: all)")

    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    # Validate inputs
    if not any([args.log_dir, args.profile_dir, args.checkpoint_dir]):
        parser.error("Must provide at least one of: --log-dir, --profile-dir, --checkpoint-dir")

    try:
        # Initialize extractor
        extractor = MoEFactsExtractor(
            log_dir=args.log_dir,
            profile_dir=args.profile_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            total_steps=args.steps,
            ranks=[int(rank) for rank in args.ranks.split(",")] if args.ranks else None,
        )

        # Execute extraction pipeline
        sources = extractor.discover_inputs()
        facts = extractor.extract_facts(sources)
        extractor.generate_artifacts(facts, sources)
        extractor.validate_output()
        extractor.print_summary(facts)

        return 0

    except Exception as e:
        error_output = {
            "status": "FAILED",
            "error": str(e),
            "input_sources_checked": [
                str(args.log_dir) if args.log_dir else None,
                str(args.profile_dir) if args.profile_dir else None,
                str(args.checkpoint_dir) if args.checkpoint_dir else None,
            ],
            "suggestion": "Ensure HP_MOE_PROFILING=1 during training for detailed metrics"
        }

        print("\n" + "="*60, file=sys.stderr)
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        print("="*60, file=sys.stderr)

        return 1


if __name__ == "__main__":
    sys.exit(main())
