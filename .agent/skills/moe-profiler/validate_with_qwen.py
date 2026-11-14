#!/usr/bin/env python3
"""
MoE Profiler Validation Script for Qwen3-30B-A3B

Simulates MoE training logs and validates moe-profiler extraction.
"""

import os
import sys
import json
import tempfile
import random
from pathlib import Path
import numpy as np

# Add scripts to path
SKILL_DIR = Path(__file__).parent
sys.path.insert(0, str(SKILL_DIR / "scripts"))


def generate_mock_moe_logs(log_dir: Path, model_config: dict):
    """Generate mock MoE training logs with realistic patterns"""

    num_layers = model_config["num_hidden_layers"]
    num_experts = model_config["num_experts"]
    num_steps = 50

    log_file = log_dir / "rank0.log"

    with open(log_file, 'w') as f:
        f.write(f"[INFO] Starting Qwen3-30B-A3B training\n")
        f.write(f"[INFO] Model: {num_layers} layers, {num_experts} experts\n")
        f.write(f"[INFO] MoE config: top_k=8, ep_size=8\n\n")

        for step in range(1, num_steps + 1):
            f.write(f"[Step {step}/{num_steps}]\n")

            # Simulate load imbalance pattern (realistic distribution)
            # Pattern: 20% experts get 60% of tokens (heavy-tail)
            base_tokens = 512

            for layer in range(min(4, num_layers)):  # Sample first 4 layers
                # Generate tokens_per_expert with imbalance
                tokens_per_expert = []
                for exp_id in range(num_experts):
                    if exp_id < int(num_experts * 0.2):
                        # Hot experts (20%)
                        tokens = int(np.random.normal(base_tokens * 1.8, 50))
                    elif exp_id > int(num_experts * 0.7):
                        # Cold experts (30%)
                        tokens = int(np.random.normal(base_tokens * 0.5, 30))
                    else:
                        # Normal experts (50%)
                        tokens = int(np.random.normal(base_tokens, 40))

                    tokens = max(0, tokens)
                    tokens_per_expert.append(tokens)

                # Write tokens_per_expert
                tokens_str = ','.join(map(str, tokens_per_expert))
                f.write(f"  Layer {layer} tokens_per_expert=tensor([{tokens_str}])\n")

                # Simulate All-to-All timing
                # dispatch time (varies with load imbalance)
                dispatch_time = 8.5 + np.random.normal(0, 1.2)
                dispatch_volume = 4.2
                f.write(f"  [MoE] Layer {layer} dispatch: {dispatch_time:.2f}ms, {dispatch_volume:.2f}MB\n")

                # combine time (usually similar to dispatch)
                combine_time = 8.2 + np.random.normal(0, 1.0)
                f.write(f"  [MoE] Layer {layer} combine: {combine_time:.2f}ms, {dispatch_volume:.2f}MB\n")

            # Step time
            step_time = 175.0 + np.random.normal(0, 10)
            f.write(f"  step_time={step_time:.2f}ms\n")
            f.write(f"  loss=2.{300-step:03d}\n\n")

    print(f"✓ Generated mock log: {log_file}")
    return log_file


def test_load_analyzer(log_file: Path):
    """Test load distribution analyzer"""
    from _moe_load_analysis import LoadAnalyzer

    print("\n[Test 1] Load Distribution Analyzer")
    print("=" * 60)

    analyzer = LoadAnalyzer()
    result = analyzer.analyze([str(log_file)])

    # Validate results
    assert result["num_steps"] > 0, "Should extract steps"
    assert result["num_layers"] > 0, "Should extract layers"
    assert result["num_experts"] > 0, "Should extract experts"

    print(f"  ✓ Steps analyzed: {result['num_steps']}")
    print(f"  ✓ Layers analyzed: {result['num_layers']}")
    print(f"  ✓ Experts per layer: {result['num_experts']}")

    # Check per_layer stats
    per_layer = result.get("per_layer", [])
    assert len(per_layer) > 0, "Should have per-layer stats"
    print(f"  ✓ Per-layer records: {len(per_layer)}")

    # Check summary
    summary = result.get("summary", [])
    assert len(summary) > 0, "Should have summary stats"

    for row in summary[:2]:  # Show first 2 layers
        layer_id, total_tokens, num_experts, mean, std, imbalance, gini = row
        print(f"  Layer {layer_id}: imbalance={imbalance}, gini={gini}")

    return result


def test_comm_analyzer(log_file: Path):
    """Test communication analyzer"""
    from _moe_comm_analysis import CommunicationAnalyzer

    print("\n[Test 2] Communication Analyzer")
    print("=" * 60)

    analyzer = CommunicationAnalyzer()
    result = analyzer.analyze([str(log_file)], [])

    # Validate results
    timing = result.get("timing", [])
    assert len(timing) > 0, "Should have timing data"
    print(f"  ✓ Timing records: {len(timing)}")

    # Show sample
    for row in timing[:3]:
        step, layer, direction, duration, volume, bandwidth, ep_size = row
        print(f"  Step {step}, Layer {layer} {direction}: {duration}ms, {volume}MB, {bandwidth} Gbps")

    # Check breakdown
    breakdown = result.get("breakdown", [])
    assert len(breakdown) > 0, "Should have breakdown data"
    print(f"  ✓ Breakdown records: {len(breakdown)}")

    # Show sample breakdown
    for row in breakdown[:3]:
        step, compute, alltoall, other, ratio, total = row
        print(f"  Step {step}: AllToAll ratio = {ratio} ({alltoall}ms / {total}ms)")

    return result


def test_routing_validator():
    """Test routing validator (without real checkpoint)"""
    # Disable torch_npu auto-loading
    import os
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

    from _moe_routing_check import RoutingValidator

    print("\n[Test 3] Routing Validator")
    print("=" * 60)

    validator = RoutingValidator()

    # Test with no checkpoint (should return SKIPPED)
    result = validator.validate([])

    assert result["routing_status"] == "SKIPPED", "Should skip without checkpoint"
    assert result["gradient_status"] == "SKIPPED", "Should skip without checkpoint"
    print("  ✓ Correctly handles missing checkpoint")

    return result


def test_full_extraction(log_dir: Path, output_dir: Path):
    """Test full extraction pipeline"""
    from extract_moe_facts import MoEFactsExtractor

    print("\n[Test 4] Full Extraction Pipeline")
    print("=" * 60)

    extractor = MoEFactsExtractor(
        log_dir=log_dir,
        profile_dir=None,
        checkpoint_dir=None,
        output_dir=output_dir,
    )

    # Phase 1: Discover inputs
    sources = extractor.discover_inputs()
    assert len(sources["log_files"]) > 0, "Should find log files"
    print(f"  ✓ Phase 1: Discovered {len(sources['log_files'])} log files")

    # Phase 2: Extract facts
    facts = extractor.extract_facts(sources)
    assert "load_distribution" in facts, "Should extract load data"
    assert "communication" in facts, "Should extract comm data"
    print(f"  ✓ Phase 2: Extracted facts")

    # Phase 3: Generate artifacts
    extractor.generate_artifacts(facts, sources)
    print(f"  ✓ Phase 3: Generated {len(extractor.artifacts)} artifacts")

    # Phase 4: Validate output
    extractor.validate_output()
    print(f"  ✓ Phase 4: All artifacts validated")

    # Check key artifacts exist
    key_artifacts = [
        "moe_facts.json",
        "moe_facts.md",
        "expert_load_per_layer.csv",
        "token_distribution_summary.csv",
        "alltoall_timing.csv",
        "communication_breakdown.csv",
    ]

    for artifact in key_artifacts:
        artifact_path = output_dir / artifact
        assert artifact_path.exists(), f"Missing artifact: {artifact}"

    print(f"  ✓ All key artifacts present")

    # Read and display summary
    with open(output_dir / "moe_facts.json") as f:
        index = json.load(f)

    summary = index.get("summary", {})
    print(f"\n  Summary Metrics:")
    print(f"    - Steps analyzed: {summary.get('steps_analyzed', 0)}")
    print(f"    - Layers analyzed: {summary.get('layers_analyzed', 0)}")
    print(f"    - Experts per layer: {summary.get('experts_per_layer', 0)}")
    print(f"    - Mean load imbalance: {summary.get('mean_load_imbalance_factor', 0):.2f}")
    print(f"    - Mean AllToAll ratio: {summary.get('mean_alltoall_ratio', 0):.3f}")

    return facts


def main():
    print("=" * 60)
    print("MoE Profiler Validation for Qwen3-30B-A3B")
    print("=" * 60)

    # Load model config
    model_dir = os.environ.get("MOE_VALIDATION_MODEL_DIR")
    if not model_dir:
        print("Set MOE_VALIDATION_MODEL_DIR to a local Qwen3-30B-A3B directory")
        return 1
    model_config_path = Path(model_dir) / "config.json"
    if not model_config_path.exists():
        print(f"❌ Model config not found: {model_config_path}")
        return 1

    with open(model_config_path) as f:
        model_config = json.load(f)

    print(f"\nModel Configuration:")
    print(f"  - Architecture: {model_config.get('architectures', ['Unknown'])[0]}")
    print(f"  - Layers: {model_config['num_hidden_layers']}")
    print(f"  - Experts: {model_config['num_experts']}")
    print(f"  - Top-K: {model_config['num_experts_per_tok']}")

    # Create temp directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        log_dir = tmpdir / "logs"
        output_dir = tmpdir / "moe_facts_output"

        log_dir.mkdir()
        output_dir.mkdir()

        print(f"\nTemp directories:")
        print(f"  - Log dir: {log_dir}")
        print(f"  - Output dir: {output_dir}")

        # Generate mock logs
        print("\n" + "=" * 60)
        print("Phase 1: Generate Mock Training Logs")
        print("=" * 60)
        log_file = generate_mock_moe_logs(log_dir, model_config)

        # Run tests
        try:
            load_result = test_load_analyzer(log_file)
            comm_result = test_comm_analyzer(log_file)
            routing_result = test_routing_validator()
            facts = test_full_extraction(log_dir, output_dir)

            print("\n" + "=" * 60)
            print("✅ All Tests Passed!")
            print("=" * 60)

            # Show output files
            print("\nGenerated Artifacts:")
            for artifact in sorted(output_dir.iterdir()):
                size = artifact.stat().st_size
                print(f"  - {artifact.name} ({size} bytes)")

            return 0

        except AssertionError as e:
            print(f"\n❌ Test Failed: {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
