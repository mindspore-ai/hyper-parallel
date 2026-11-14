# Implementation summary

[The optimizer](scripts/run_optimization.py) runs repository PSA, writes runnable Trainer configurations,
launches sequential distributed benchmarks, refines constraints from failures, and retains each round's evidence.
Planning has an explicit `PLANNED` status. A production config requires an eligible measurement.
See [SKILL.md](SKILL.md) for supported arguments and [orchestration](workflows/orchestration.md) for limits.
