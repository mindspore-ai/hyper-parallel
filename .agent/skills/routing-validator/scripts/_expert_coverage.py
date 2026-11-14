"""
Expert Coverage Checker

Checks if all experts are being activated during training.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict


class ExpertCoverageChecker:
    """Check expert activation coverage"""

    def __init__(self):
        self.patterns = {
            'tokens_per_expert': re.compile(
                r'tokens_per_expert=tensor\(\[([\d,\s]+)\]\)'
            ),
        }

    def check(self, log_files: List[str]) -> Dict[str, Any]:
        """Main check entry point"""
        # Parse logs
        coverage_data = self._parse_logs(log_files)

        if not coverage_data:
            return {
                "status": "SKIP",
                "reason": "No expert activation data found in logs"
            }

        # Analyze coverage
        result = self._analyze_coverage(coverage_data)

        return result

    def _parse_logs(self, log_files: List[str]) -> Dict[int, List[int]]:
        """Parse expert activations from logs"""
        # expert_id -> [tokens in each step]
        expert_activations = defaultdict(list)

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = self.patterns['tokens_per_expert'].search(line)
                        if match:
                            tokens_str = match.group(1)
                            tokens = [int(t.strip()) for t in tokens_str.split(',')]

                            for expert_id, token_count in enumerate(tokens):
                                expert_activations[expert_id].append(token_count)
            except Exception as e:
                print(f"  Warning: Failed to parse {log_file}: {e}")
                continue

        return dict(expert_activations)

    def _analyze_coverage(self, coverage_data: Dict[int, List[int]]) -> Dict[str, Any]:
        """Analyze expert coverage"""
        if not coverage_data:
            return {
                "status": "SKIP",
                "reason": "No coverage data available"
            }

        num_experts = len(coverage_data)
        total_steps = len(next(iter(coverage_data.values())))

        # Find dead experts (never activated)
        dead_experts = []
        for expert_id, activations in coverage_data.items():
            if all(tokens == 0 for tokens in activations):
                dead_experts.append(expert_id)

        # Find rarely-used experts (< 10% activation rate)
        rarely_used = []
        for expert_id, activations in coverage_data.items():
            activation_rate = sum(1 for t in activations if t > 0) / len(activations)
            if 0 < activation_rate < 0.1:
                rarely_used.append(expert_id)

        # Determine status
        if dead_experts:
            status = "WARN"
            warning = f"{len(dead_experts)} dead experts detected (never activated)"
        elif rarely_used:
            status = "WARN"
            warning = f"{len(rarely_used)} rarely-used experts (< 10% activation rate)"
        else:
            status = "PASS"
            warning = None

        result = {
            "status": status,
            "details": {
                "experts_checked": num_experts,
                "steps_analyzed": total_steps,
                "dead_experts": dead_experts,
                "rarely_used_experts": rarely_used,
                "all_active": len(dead_experts) == 0
            }
        }

        if warning:
            result["warning"] = warning

        return result
