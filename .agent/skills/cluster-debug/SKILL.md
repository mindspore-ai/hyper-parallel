---
name: cluster-debug
description: >
  Read-only cluster failure diagnosis for multi-rank training: gather each
  worker/rank's training log, plog and py-spy stacks, find the first real
  error across ranks, and tell apart HCCL timeout, cluster hang, partial-rank
  exit, collective-sequence mismatch, wrong communication group, and
  NPU/node/link/scheduler faults. Use for 集群调试 / 多卡挂死 / HCCL 超时 /
  部分 rank 退出 / 集群首错定位 / cluster debug. Diagnosis only - any code
  fix routes to the debug skill and is verified single-node first.
---

# Cluster Debug

Diagnose a multi-rank failure from the evidence the cluster already produced.
**Read-only by design**: this skill collects and reasons, it does not change
cluster state, and it never fixes code — a fix routes to `debug` and is
verified on one card / one node before going wide.

**This file is the index + the hard rules.** Fault-class routing loads from
references.

## Hard rules

- **Read-only.** Collect logs, read state. Never kill other users' processes,
  never reset devices, never change a shared node's configuration. An
  occupied node belongs to its owner; wait or move.
- **Never fix code here.** A code fix routes to `debug`, is verified at the
  smallest scope (single card / single node) first, and only then re-runs at
  cluster scale.
- **First real error, not first loud error.** Ranks spray errors as they
  tear down; the earliest *causal* error on the earliest rank is the target,
  not the loudest or the last.
- **Occupancy before launching anything.** Any re-run scans occupancy first
  (process table of *all* users + HBM + `/proc/*/fd` davinci holders); a node
  with another user's live task is off limits.
- **Evidence is reproducible**: every finding carries the rank, the file, the
  timestamp and the command that collected it.

## Required inputs

1. **Job identity**: node list, world size, parallel layout (TP/DP/EP/CP/PP),
   the launch command and commit.
2. **Symptom**: hang / timeout / crash / partial exit, and when it appeared
   (step, wall time).
3. **Access**: which nodes are readable, and whether the job is still live
   (live job → py-spy stacks are available; dead job → logs only).

## Collection (read-only)

| Evidence | Where | Notes |
|---|---|---|
| per-rank training log | launcher output dir, one file per rank | the first error's rank and step |
| plog | Ascend plog dir on each node | device-side errors; see triage for how to read |
| py-spy stacks | live process only | where each rank is stuck (hang diagnosis) |
| npu-smi / occupancy | each node | device health, who else is on the node |
| env / config dump | job dir | parallel layout, env vars actually in effect |

Collect from **every** rank, not just rank 0: a partial-rank fault is
invisible in rank 0's log.

## Diagnosis flow

1. **Timeline across ranks**: order every rank's first error by timestamp.
   The earliest is the candidate cause; later ones are usually teardown.
2. **Classify the fault** → [references/fault-classes.md](references/fault-classes.md)
   (HCCL timeout, hang, partial exit, collective mismatch, wrong group,
   device/node/link/scheduler fault).
3. **Separate waiting from failing**: a rank blocked in a collective is a
   *victim* of whichever rank never arrived; find who did not arrive.
4. **Confirm with a falsifiable check** where possible (re-run same layout
   with the suspect feature off, on free nodes) rather than concluding from
   log correlation alone.

## Output

Report: the fault class, the first real error (rank / file / timestamp /
message), why the other ranks' errors are downstream, the evidence list, and
the recommended next step — which is either a `debug`-skill fix verified
single-node, a configuration/layout change, or an infrastructure escalation.
