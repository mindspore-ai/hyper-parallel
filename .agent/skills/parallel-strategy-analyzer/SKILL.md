---
name: parallel-strategy-analyzer
description: >
  Recommend DP/FSDP/TP/PP/EP/CP strategy mixes from model + hardware constraints,
  with memory/comm/compute/bubble estimates. Use for strategy planning, scale-out,
  OOM, or comparing configs. Not for implementing platform/op code (platform-dev /
  dist-op-dev) or running training jobs.
---

# Parallel Strategy Analyzer

**Single source of truth** for formulas and decision trees in this directory.
The agent `.agent/agents/parallel-strategy-analyzer.md` is a thin proxy only.

## When to use

- Choose or rescale parallel strategies; debug OOM; compare configs; estimate comm/bubble

Inputs / examples / full output shape / limits:
[references/io-contract.md](references/io-contract.md).

## Analysis flow (load workflow before each phase)

```text
1 Collect info     → workflows/01-collect-model-info.md
2 Global baseline  → workflows/02-global-baseline.md   (memory + FLOPs, no parallel)
3 Strategy search  → workflows/03-strategy-search.md
4 Cost analysis    → workflows/04-cost-analysis.md     (TP/CP/EP/DP + PP bubble)
5–7 Shard mem / MFU / score + report → workflows/05-scoring-output.md
```

Do **Phase 2 before** cost/memory-after-shard so “fits in memory” is not preferred
over strategies with prohibitive communication.

## Tables & formulas (on demand)

- [references/known-models.md](references/known-models.md)
- [references/known-hardware.md](references/known-hardware.md)
- [references/memory-estimation.md](references/memory-estimation.md)
- [references/strategy-rules.md](references/strategy-rules.md)

## Output (short)

Emit: baseline → recommendation + DeviceMesh code → post-shard mem → comm →
bubble → opts → top-3 alternatives. Details in `io-contract.md`.
