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
MoE Routing Validator - Main Entry Point

Validates MoE routing correctness through three core checks:
1. Token integrity (conservation)
2. Routing correctness (expert-major order)
3. Gradient flow (all experts receive gradients)
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import sub-modules
from _token_integrity import TokenIntegrityChecker
from _gradient_flow import GradientFlowChecker
from _expert_coverage import ExpertCoverageChecker
from _routing_evidence import check_routing, load_capture


class RoutingValidator:
    """Main orchestrator for MoE routing validation"""

    def __init__(self, checkpoint_dir: Optional[Path], log_dir: Optional[Path],
                 output_dir: Path, strict: bool = False):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.output_dir = Path(output_dir)
        self.strict = strict

        # Initialize checkers
        self.token_checker = TokenIntegrityChecker()
        self.gradient_checker = GradientFlowChecker()
        self.coverage_checker = ExpertCoverageChecker()

        self.validation_id = f"routing_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def discover_inputs(self) -> Dict[str, Any]:
        """Phase 1: Discover and validate input sources"""
        print("[Phase 1] Discovering input sources...")

        sources = {
            "checkpoint_files": [],
            "log_files": [],
        }

        # Discover checkpoint files
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            ckpt_files = list(self.checkpoint_dir.rglob("*.pt")) + \
                        list(self.checkpoint_dir.rglob("*.pth")) + list(self.checkpoint_dir.rglob("*.json"))
            sources["checkpoint_files"] = [str(f) for f in ckpt_files]
            print(f"  Found {len(ckpt_files)} checkpoint files")

        # Discover log files
        if self.log_dir and self.log_dir.exists():
            log_files = list(self.log_dir.rglob("*.log"))
            sources["log_files"] = [str(f) for f in log_files]
            print(f"  Found {len(log_files)} log files")

        # Validate at least one source
        if not sources["checkpoint_files"] and not sources["log_files"]:
            raise ValueError("No input sources found. Provide --checkpoint-dir or --log-dir")

        return sources

    def check_token_integrity(self, sources: Dict) -> Dict[str, Any]:
        """Phase 2: Token integrity check"""
        print("\n[Phase 2] Checking token integrity...")

        if not sources["log_files"]:
            print("  SKIP: No log files for token integrity check")
            return {
                "status": "SKIP",
                "reason": "No log files available"
            }

        try:
            result = self.token_checker.check(sources["log_files"])

            if result["status"] == "PASS":
                print(f"  ✓ PASS: {result['details']['conservation_ratio']:.3f} conservation ratio")
            else:
                print(f"  {result['status']}: {result.get('reason', result.get('error', ''))}")

            return result

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            return {
                "status": "FAIL",
                "error": str(e)
            }

    def check_routing_correctness(self, sources: Dict) -> Dict[str, Any]:
        """Phase 3: Routing correctness check"""
        print("\n[Phase 3] Checking routing correctness...")

        if not sources["checkpoint_files"]:
            print("  SKIP: No checkpoint for routing correctness check")
            return {
                "status": "SKIP",
                "reason": "No checkpoint available"
            }

        results = []
        for filename in sources["checkpoint_files"]:
            try:
                results.append(check_routing(load_capture(filename)))
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                results.append({"status": "UNVERIFIED", "reason": str(exc)})
        statuses = [result["status"] for result in results]
        status = "FAIL" if "FAIL" in statuses else (
            "PASS" if statuses and all(item == "PASS" for item in statuses) else "UNVERIFIED")
        return {"status": status, "details": {"checkpoint_results": results}}

    def check_gradient_flow(self, sources: Dict) -> Dict[str, Any]:
        """Phase 4: Gradient flow check"""
        print("\n[Phase 4] Checking gradient flow...")

        if not sources["checkpoint_files"]:
            print("  SKIP: No checkpoint for gradient flow check")
            return {
                "status": "SKIP",
                "reason": "No checkpoint available"
            }

        results = [self.gradient_checker.check(filename) for filename in sources["checkpoint_files"]]
        statuses = [result["status"] for result in results]
        status = "FAIL" if "FAIL" in statuses else (
            "PASS" if statuses and all(item == "PASS" for item in statuses) else "UNVERIFIED")
        return {"status": status, "details": {"checkpoint_results": results}}

    def check_expert_coverage(self, sources: Dict) -> Dict[str, Any]:
        """Additional: Expert coverage check"""
        print("\n[Additional] Checking expert coverage...")

        if not sources["log_files"]:
            print("  SKIP: No log files for coverage check")
            return {
                "status": "SKIP",
                "reason": "No log files available"
            }

        try:
            result = self.coverage_checker.check(sources["log_files"])

            if result["status"] == "PASS":
                print(f"  ✓ PASS: All experts activated")
            elif result["status"] == "WARN":
                dead_count = len(result["details"].get("dead_experts", []))
                print(f"  ⚠ WARN: {dead_count} dead experts detected")

            return result

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            return {
                "status": "WARN",
                "warning": str(e)
            }

    def generate_report(self, checks: Dict[str, Any], sources: Dict) -> Dict:
        """Phase 5: Generate validation report"""
        print("\n[Phase 5] Generating validation report...")

        # Determine overall status
        statuses = [checks[key]["status"] for key in checks if key in checks]

        required_names = ("token_integrity", "routing_correctness", "gradient_flow")
        required_statuses = [checks.get(name, {}).get("status", "UNVERIFIED") for name in required_names]

        if any(s == "FAIL" for s in statuses):
            overall_status = "FAIL"
            failed = [k for k, v in checks.items() if v["status"] == "FAIL"]
            summary = f"Failed checks: {', '.join(failed)}"
        elif any(s in ("UNVERIFIED", "SKIP") for s in required_statuses):
            overall_status = "FAIL" if self.strict else "UNVERIFIED"
            summary = "Required checks are incomplete; no correctness PASS is issued"
        elif any(s == "WARN" for s in statuses):
            overall_status = "FAIL" if self.strict else "WARN"
            summary = "Some checks have warnings"
        elif all(s == "PASS" for s in required_statuses):
            overall_status = "PASS"
            summary = "All checks passed"
        else:
            overall_status = "UNKNOWN"
            summary = "Validation incomplete"

        # Build report
        report = {
            "validation_id": self.validation_id,
            "timestamp": datetime.now().isoformat(),
            "input_sources": {
                "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
                "log_dir": str(self.log_dir) if self.log_dir else None,
                "checkpoint_files_count": len(sources.get("checkpoint_files", [])),
                "log_files_count": len(sources.get("log_files", []))
            },
            "checks": checks,
            "overall_status": overall_status,
            "summary": summary,
            "strict_mode": self.strict
        }

        # Save JSON report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Report saved: {report_file}")

        # Generate markdown summary
        markdown = self._generate_markdown_report(report)
        md_file = self.output_dir / "validation_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"  ✓ Summary saved: {md_file}")

        return report

    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate human-readable markdown summary"""
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "SKIP": "⏭️",
            "UNVERIFIED": "❓"
        }

        overall_emoji = status_emoji.get(report["overall_status"], "❓")

        md = f"""# MoE Routing Validation Report

**Status:** {overall_emoji} {report['overall_status']}
**Validation ID:** {report['validation_id']}
**Timestamp:** {report['timestamp']}

## Input Sources

- Checkpoint directory: {report['input_sources']['checkpoint_dir'] or 'N/A'}
- Log directory: {report['input_sources']['log_dir'] or 'N/A'}
- Checkpoint files: {report['input_sources']['checkpoint_files_count']}
- Log files: {report['input_sources']['log_files_count']}

## Check Results

"""

        for check_name, check_result in report["checks"].items():
            status = check_result["status"]
            emoji = status_emoji.get(status, "❓")
            title = check_name.replace("_", " ").title()

            md += f"### {emoji} {title}: {status}\n"

            if "details" in check_result:
                for key, value in check_result["details"].items():
                    md += f"- {key.replace('_', ' ').title()}: {value}\n"

            if "error" in check_result:
                md += f"- **Error:** {check_result['error']}\n"

            if "warning" in check_result:
                md += f"- **Warning:** {check_result['warning']}\n"

            md += "\n"

        md += f"""## Summary

{report['summary']}

---

**Strict Mode:** {"Enabled" if report['strict_mode'] else "Disabled"}
"""

        return md

    def print_summary(self, report: Dict):
        """Print structured summary to stdout"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "WARN": "⚠",
            "SKIP": "⏭",
            "UNVERIFIED": "?"
        }

        print(f"\nOverall Status: {status_symbol.get(report['overall_status'], '?')} {report['overall_status']}")
        print(f"Summary: {report['summary']}")

        print(f"\nCheck Details:")
        for check_name, check_result in report["checks"].items():
            status = check_result["status"]
            symbol = status_symbol.get(status, "?")
            print(f"  {symbol} {check_name}: {status}")

        print(f"\nReport Location: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate MoE routing correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From checkpoint
  python validate_routing.py --checkpoint-dir /path/to/checkpoint --output-dir ./validation

  # From logs
  python validate_routing.py --log-dir /path/to/logs --output-dir ./validation

  # Both (recommended)
  python validate_routing.py \\
    --checkpoint-dir /path/to/checkpoint \\
    --log-dir /path/to/logs \\
    --output-dir ./validation
        """
    )

    parser.add_argument("--checkpoint-dir", type=Path,
                       help="Model checkpoint directory")
    parser.add_argument("--log-dir", type=Path,
                       help="Training logs directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./routing_validation"),
                       help="Output directory (default: ./routing_validation)")
    parser.add_argument("--strict", action="store_true",
                       help="Treat warnings as failures")

    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint_dir and not args.log_dir:
        parser.error("Must provide at least one of: --checkpoint-dir, --log-dir")

    try:
        # Initialize validator
        validator = RoutingValidator(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            strict=args.strict
        )

        # Execute validation pipeline
        sources = validator.discover_inputs()

        checks = {}
        checks["token_integrity"] = validator.check_token_integrity(sources)
        checks["routing_correctness"] = validator.check_routing_correctness(sources)
        checks["gradient_flow"] = validator.check_gradient_flow(sources)
        checks["expert_coverage"] = validator.check_expert_coverage(sources)

        report = validator.generate_report(checks, sources)
        validator.print_summary(report)

        # Exit code based on overall status
        if report["overall_status"] == "PASS":
            return 0
        elif report["overall_status"] == "WARN":
            return 0 if not args.strict else 1
        else:
            return 1

    except Exception as e:
        print(f"\n✗ Validation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
