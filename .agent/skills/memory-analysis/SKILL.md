---
name: memory-analysis
description: >
  Analyze NPU training memory from allocation history and snapshots.
  Attribute active blocks to parameters, gradients, optimizer state,
  activations, and allocation call sites. Inspect reserved memory,
  fragmentation, growth across steps, and whether a snapshot covers the
  measured peak. Use for device-memory analysis, memory leaks, OOM
  diagnosis, or allocation breakdowns. Produces reproducible JSON and HTML.
---

# Memory Analysis

Record allocation history, capture a snapshot at the relevant point in
training, and determine which allocations occupy device memory. The same
snapshot must produce the same rule-based breakdown.

**This file is the index.** Load the snapshot-format guide and
classification rules when their details are needed.

> **Validation scope.** Fields, stack frames, and accounting identities were
> checked on **torch_npu 2.9.0.post2 + CANN 9.0.0 / Ascend 910**. Rule
> version **v2** was reconciled against a workload with known allocations;
> see [Breakdown dimensions](references/breakdown-dimensions.md).

## Subcommands

```bash
python .agent/skills/memory-analysis/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| Collect snapshots | `collect_snapshot.py --run-dir <d> --launch "<cmd>" [--steps 1-3]` | Launch the workload and verify a counter sidecar for every dump. |
| Instrument training | `mem_hook.start()` / `step()` / `dump(tag)` | Record in-process allocations and dump at any relevant point. |
| Generate JSON and HTML | `breakdown.py --snapshot <f> [--compare <f2>] [--json r.json]` | Parse, classify, attribute, and validate the snapshot. Exit codes are 0 for all checks passing, 2 for a failed check, and 1 for a missing file. |

For a standalone training script:

```python
import sys
sys.path.insert(0, ".agent/skills/memory-analysis/scripts")
import mem_hook

mem_hook.start()  # Read MEM_* settings exported by collect_snapshot.
for step, batch in enumerate(loader):
    out = model(batch)
    out.register_hook(lambda g: mem_hook.dump("midbwd") and None)
    loss(out).backward()
    opt.step()
    opt.zero_grad()
    mem_hook.step()  # Dump steps selected by the configured window.
mem_hook.stop()
```

The autograd hook captures activations during backward; a post-step
snapshot can miss them.

## References

| When | Read |
|---|---|
| Interpret snapshot fields, stack frames, capture behavior, and identities | [Snapshot format](references/snapshot-format.md) |
| Apply memory categories, call-site attribution, and consistency checks | [Breakdown dimensions](references/breakdown-dimensions.md) |

## Report

`templates/memory-breakdown.html` contains six sections: totals and
fragmentation, allocation categories, largest active blocks and call sites,
growth across snapshots, consistency checks, and findings. The totals
include the gap between sampled allocation and the recorded peak.

## Supported analysis

- **Totals and fragmentation.** Report reserved, allocated, and active
  memory, together with reserved-but-unused capacity.
- **Allocation categories.** Identify parameters, gradients, optimizer
  state, activations, other user tensors, and unclassified blocks using
  the documented stack rules.
- **Call-site attribution.** Attribute large blocks to the user-code line
  that allocated them. Native `AccumulateGrad` frames identify gradients
  without Python frames.
- **Growth across steps.** Compare two snapshots with `--compare` to
  investigate persistent growth versus one-time allocations.
- **Peak coverage.** Compare sampled allocation with
  `max_memory_allocated`. A reference post-step snapshot was 30 MiB
  below the peak and contained no activations.

### Not yet implemented

- Module/layer aggregation. Current attribution uses file and line.
- Recomputation-savings estimates, which require module-level activation
  accounting.
- Reconstructing peak composition from `device_traces`. The current report
  only shows the sampled composition and its gap from the peak.

## Hard rules

- **A snapshot is not necessarily the peak.** Include
  `max_memory_allocated`; a gap above 2% fails the peak-coverage check.
  Capture during backward to inspect live activations.
- **Do not attribute by `frames[0]`.** Inner frames belong to the capture
  mechanism. Scan frames using the documented attribution order.
- **Detect native gradient frames.** `AccumulateGrad` can be the only
  signal because gradient allocations have no Python frame.
- **Capture multiple steps.** The default is steps 1-3. A single snapshot
  cannot distinguish persistent growth from a one-time allocation.
- **Respect consistency failures.** Segment/block identities and process
  counters provide positive evidence that the parser is correct.
- **Record provenance.** Include the workload command, commit, and
  environment.
- **Measure the performance cost of memory optimizations.** This skill
  localizes allocations; `perf-playbook` selects an intervention. A
  reduction in memory alone does not establish an end-to-end benefit.
- Mark unavailable attribution **pending validation** and unparsed input
  **not parsed**. Do not invent measurements.
