# <Model / scenario> performance tuning summary

> Record each technique, its target bottleneck, before/after breakdown, and keep/revert
> decision. Include the final configuration's breakdown and optimization trajectory.
> Attach commands, commits, and environments to all numbers. Use consecutive A/B runs
> on the same nodes. Mark unmeasured values as `[NOT MEASURED]`.

| Field | Value |
|---|---|
| Model / scenario | [FILL] |
| Parallel configuration | [FILL] (TP/DP/EP/CP/PP, sequence length, device count) |
| Baseline | [FILL] (commit / configuration / environment) |
| Nodes | [FILL] (same nodes, consecutive measurements) |

## Experiment record (one row per experiment, chronological order)

| # | Technique | Target bottleneck | Before | After | keep/revert | Command/commit |
|---|------|---------|----|----|-------------|-------------|
| 1 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

## Optimization trajectory

> Show the primary metric across retained commits using a table or chart.

```
[FILL] baseline → ... → best
```

## Best-configuration performance breakdown

> Use the `profiling` skill's output to show compute (Cube/Vector/FlashAttention),
> exposed communication (TP/DP/EP/CP/PP), device idle time, and optimizer updates.

| Component | Share | Duration | Notes |
|---|---|---|---|
| Compute | [NOT MEASURED] | [NOT MEASURED] | [FILL] |
| Exposed communication | [NOT MEASURED] | [NOT MEASURED] | [FILL] |
| Device idle (`Free`) | [NOT MEASURED] | [NOT MEASURED] | [FILL] |
| Optimizer update | [NOT MEASURED] | [NOT MEASURED] | [FILL] |

## Fixes and additions

- **Bugs fixed:** [FILL]
- **Features or techniques added:** [FILL]
- **Open issues and follow-up:** [FILL]
