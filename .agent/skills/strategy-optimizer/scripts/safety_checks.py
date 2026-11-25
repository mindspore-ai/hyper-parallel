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
Safety checks for strategy validation.

Ensures safe execution of distributed validation benchmarks:
- Device availability
- Resource isolation
- Time estimation
- User confirmation
"""
import subprocess
import sys
import time
from typing import Dict, Any, Tuple, Optional


def check_device_availability(device_type: str = "npu", required_count: int = 8) -> Tuple[bool, str]:
    """
    Check if required devices are available.

    Args:
        device_type: "npu" or "cuda"
        required_count: Number of devices required

    Returns:
        (success, message)
    """
    if device_type not in ("npu", "cuda"):
        return False, f"Unknown device type: {device_type}"
    # Runtime queries respect visible-device masks and count logical dies,
    # unlike the physical-board rows printed by npu-smi on A3.
    query = "import torch; "
    if device_type == "npu":
        query += "import torch_npu; print(torch.npu.device_count())"
    else:
        query += "print(torch.cuda.device_count())"
    cmd = [sys.executable, "-c", query]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return False, f"{device_type.upper()} runtime query failed: {result.stderr.strip()}"

        available = int(result.stdout.strip().splitlines()[-1])

        if available < required_count:
            return False, f"Only {available} devices available, need {required_count}"

        return True, f"✓ {available} {device_type.upper()} devices available"

    except subprocess.TimeoutExpired:
        return False, f"{device_type} runtime query timeout"
    except FileNotFoundError:
        return False, f"Runtime executable not found: {sys.executable}"
    except Exception as e:
        return False, f"Device check failed: {e}"


def estimate_runtime(
    num_steps: int = 50,
    avg_step_time_sec: float = 0.7,
    warmup_steps: int = 10,
) -> Dict[str, Any]:
    """
    Estimate validation runtime.

    Args:
        num_steps: Number of validation steps
        avg_step_time_sec: Average step time in seconds
        warmup_steps: Warmup steps (usually slower)

    Returns:
        Dict with time estimates
    """
    warmup_time = warmup_steps * avg_step_time_sec * 1.5  # Warmup slower
    main_time = (num_steps - warmup_steps) * avg_step_time_sec
    total_time = warmup_time + main_time + 60  # +1 min for init/cleanup

    return {
        "total_seconds": total_time,
        "total_minutes": total_time / 60,
        "estimated_completion": time.time() + total_time,
        "breakdown": {
            "warmup": warmup_time,
            "training": main_time,
            "overhead": 60,
        }
    }


def check_resource_isolation() -> Tuple[bool, str]:
    """
    Check if validation will interfere with production workloads.

    Returns:
        (safe, message)
    """
    # Check for running training jobs
    try:
        # Check for pytorch processes
        result = subprocess.run(
            ["pgrep", "-f", "torch.distributed"],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            return False, f"⚠️  {len(pids)} torch.distributed processes running (potential conflict)"

        return True, "✓ No conflicting processes detected"

    except Exception as e:
        # If check fails, be conservative
        return False, f"⚠️  Cannot verify isolation: {e}"


def request_user_confirmation(
    config_summary: Dict[str, Any],
    time_estimate: Dict[str, Any],
    first_run: bool = False,
) -> bool:
    """
    Request user confirmation before running validation.

    Args:
        config_summary: Summary of validation config
        time_estimate: Runtime estimate
        first_run: Whether this is first validation run

    Returns:
        True if user confirms
    """
    print("\n" + "="*70)
    print("VALIDATION SAFETY CHECK")
    print("="*70)

    print("\nConfiguration:")
    print(f"  Strategy: {config_summary.get('strategy', 'N/A')}")
    print(f"  Devices: {config_summary.get('world_size', 'N/A')}")
    print(f"  Steps: {config_summary.get('steps', 50)}")

    print(f"\nEstimated runtime:")
    print(f"  Total: ~{time_estimate['total_minutes']:.1f} minutes")
    print(f"  Breakdown:")
    print(f"    - Warmup: {time_estimate['breakdown']['warmup']:.0f}s")
    print(f"    - Training: {time_estimate['breakdown']['training']:.0f}s")
    print(f"    - Overhead: {time_estimate['breakdown']['overhead']:.0f}s")

    if first_run:
        print("\n⚠️  First validation run - approval required")

    print("\n" + "="*70)

    # In automated mode, return True (assume pre-approved)
    if "--auto-approve" in sys.argv:
        print("✓ Auto-approved (--auto-approve flag)")
        return True

    # Interactive confirmation
    try:
        response = input("Proceed with validation? (yes/no): ").strip().lower()
        return response in ["yes", "y"]
    except (EOFError, KeyboardInterrupt):
        print("\n✗ Cancelled by user")
        return False


def run_safety_checks(
    device_type: str = "npu",
    world_size: int = 8,
    config_summary: Optional[Dict[str, Any]] = None,
    first_run: bool = False,
) -> Tuple[bool, str]:
    """
    Run all safety checks before validation.

    Args:
        device_type: "npu" or "cuda"
        world_size: Number of devices required
        config_summary: Summary of validation config
        first_run: Whether this is first run

    Returns:
        (safe_to_proceed, message)
    """
    print("\n" + "="*70)
    print("PRE-VALIDATION SAFETY CHECKS")
    print("="*70)

    checks_passed = []
    checks_failed = []

    # 1. Device availability
    print("\n1. Checking device availability...")
    success, msg = check_device_availability(device_type, world_size)
    print(f"   {msg}")
    if success:
        checks_passed.append("device_availability")
    else:
        checks_failed.append(("device_availability", msg))

    # 2. Resource isolation
    print("\n2. Checking resource isolation...")
    success, msg = check_resource_isolation()
    print(f"   {msg}")
    if success:
        checks_passed.append("resource_isolation")
    else:
        # Warning only, not fatal
        print(f"   (continuing anyway)")

    # 3. Runtime estimation
    print("\n3. Estimating runtime...")
    time_est = estimate_runtime()
    print(f"   ✓ Estimated: ~{time_est['total_minutes']:.1f} minutes")
    checks_passed.append("runtime_estimate")

    # 4. Summary
    print("\n" + "="*70)
    print(f"Checks: {len(checks_passed)} passed, {len(checks_failed)} failed")
    print("="*70)

    if checks_failed:
        print("\n✗ Safety checks failed:")
        for check, reason in checks_failed:
            print(f"  - {check}: {reason}")
        return False, "Safety checks failed"

    # 5. User confirmation (if needed)
    if first_run or "--confirm" in sys.argv:
        if config_summary is None:
            config_summary = {"world_size": world_size, "steps": 50}

        if not request_user_confirmation(config_summary, time_est, first_run):
            return False, "User declined"

    print("\n✓ All safety checks passed")
    return True, "Safe to proceed"


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run safety checks")
    parser.add_argument("--device-type", choices=["npu", "cuda"], default="npu")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--auto-approve", action="store_true", help="Skip confirmation")
    parser.add_argument("--confirm", action="store_true", help="Force confirmation")

    args = parser.parse_args()

    config_summary = {
        "strategy": "TP=1, CP=1, EP=4, DP=2",
        "world_size": args.world_size,
        "steps": 50,
    }

    safe, message = run_safety_checks(
        device_type=args.device_type,
        world_size=args.world_size,
        config_summary=config_summary,
        first_run=False,
    )

    if safe:
        print("\n✅ Ready to run validation")
        sys.exit(0)
    else:
        print(f"\n❌ Cannot proceed: {message}")
        sys.exit(1)
