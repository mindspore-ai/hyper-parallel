# Distributed Failure Classes and Evidence

Classify the symptom, then collect evidence for that failure class.
Distinguish a rank waiting on communication from the rank that first failed:
a collective timeout often occurs downstream of a missing participant.

## HCCL timeouts

| Evidence | Interpretation |
|---|---|
| HCCL timeout or execution-timeout message | A rank timed out waiting for collective progress. |
| First-error timestamps across ranks | The earliest error often identifies a participant that exited or failed before the timeout. |
| First execution of a new shape | A long initial AOT compilation can exceed an execution timeout. Inspect kernel-cache timestamps and CPU activity before treating it as a kernel failure. Repeat with an appropriate timeout if compilation is confirmed. |

A timeout is a symptom. Identify the missing participant and why it did not
reach the collective.

## Distributed hangs without an error

| Evidence | Interpretation |
|---|---|
| Per-rank py-spy stacks | Check where each rank is waiting. Matching collective call sites can indicate a missing participant; different call sites can indicate inconsistent collective order. |
| Device plog | Check for device-side errors not propagated to the application. |
| Rank liveness | An exited rank can leave the remaining ranks waiting. |

## Partial rank exits

| Evidence | Interpretation |
|---|---|
| Tail of the exited rank's log | Look for OOM, device errors, or assertions preceding communication failure. |
| Communication errors on surviving ranks | Often downstream symptoms. |
| Truncated torchrun output | Read complete per-rank logs before concluding that no traceback exists. |

## Inconsistent collective order

| Evidence | Interpretation |
|---|---|
| Ranks stopped in different collective calls | Rank-local data may have caused control-flow divergence. |
| Different operation types or sequence numbers | Some ranks may have executed extra or missing calls. |

Inspect whether data-dependent branch conditions agree across every rank
participating in the group.

## Process-group mismatch

| Evidence | Interpretation |
|---|---|
| Group membership differs from the configured TP/DP/EP/CP/PP layout | Group construction or arguments are inconsistent. |
| Ranks use different groups for the same operation | Check group-creation order and conditional group creation. |

## Device, node, network, and scheduler failures

| Evidence | Interpretation |
|---|---|
| plog contains `PrintCoreInfo` and a decoded `errorStr:` | Evidence of a device-core execution failure. |
| Only notification-related `Report*Error` messages | Can indicate waiting for another participant; insufficient alone to establish a device fault. |
| Abnormal npu-smi health or a down link | Investigate device or network health. |
| Occupied ports or EI networking errors | Another active job may own device-network resources. Inspect processes across users; wait or select another node rather than terminating unrelated work. |

## Shared-filesystem and concurrent-build failures

These failures may not reproduce on a single device.

| Evidence | Interpretation and action |
|---|---|
| `ld: final close failed: Input/output error` | Multiple ranks may be compiling the same operator library on shared storage. Prebuild serially on one rank before the distributed launch. |
| Different imported code or an apparently inactive patch | Stale shared `__pycache__` files can load old bytecode. Clean only the owned experiment checkout's cache and use `PYTHONDONTWRITEBYTECODE=1`; verify the imported source path and active-path markers. |

## False passes

A run can finish successfully without executing the intended patch.
One observed multi-device run loaded stale bytecode and validated the old
implementation instead.

| Claim to verify | Evidence |
|---|---|
| The intended source revision ran | Source/import-path checks, controlled bytecode handling, and an explicit marker for the changed path. |
| The intervention was active | Positive evidence such as `engaged=1`; the absence of a fallback warning is insufficient. |
| The claimed scale and configuration ran | Recorded runtime configuration and rank inventory. |

## Evidence requirements

- Attach rank, file, timestamp, and collection command to each finding.
- Order errors across ranks by timestamp and separate the initiating error
  from teardown failures.
- Diagnose read-only. Do not terminate unrelated jobs, reset shared
  devices, or change shared-node configuration.
- Route code changes to `debug`; validate locally before repeating at
  cluster scale.
