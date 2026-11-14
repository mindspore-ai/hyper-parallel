# Migration Deliverables

Model integration must provide the artifacts needed for another developer
to run the model independently: checkpoint conversion where needed,
training recipes, launchers, tests, and a migration report.

## Checkpoint conversion

Convert upstream Hugging Face weights into a format the framework can load.

| Requirement | Check |
|---|---|
| Complete name mapping | Account for every upstream tensor. Explain missing or additional tensors rather than silently dropping them. |
| Correct sharding | Match TP/EP partitioning and make the converted artifacts loadable at the target parallel degrees. |
| Shape and dtype validation | Check each converted tensor's shape and dtype explicitly. A successful load alone is insufficient. |
| Verifiable values | At minimum, compare sampled converted tensor values with the upstream tensors. |
| Preserve source weights | Read upstream weights and write separate conversion outputs. |

The conversion script must fail explicitly on missing tensors, unexpected
tensors, or shape mismatches and identify the affected tensors.

## Training recipe

| Requirement | Check |
|---|---|
| Configure component behavior through fields | Reserve environment variables for infrastructure. |
| Expose parallel degrees | Declare TP, DP, EP, CP, PP, and sequence length in the recipe. |
| Provide runnable defaults | Defaults must run on the declared hardware rather than serve as placeholders. |

## Launchers

| Requirement | Check |
|---|---|
| Single-node and multi-node launchers | Make the sources of ranks, addresses, and ports explicit. |
| Fail early on missing inputs | Identify missing weights, data, or configuration before training starts. |
| Configurable environment | Avoid personal paths or accounts; keep cluster-specific settings at the caller boundary. |

## Tests

| Requirement | Check |
|---|---|
| CPU-compatible unit tests | Put structural and contract tests in `tests/ut`. |
| Portable unit-test dependencies | CI can copy `tests/` separately. Do not assume a fixed directory depth to files outside the installed package or test tree. |
| Distributed system tests | Use existing system-test launchers and declare the required device count. |

## Migration report

Use `templates/migration-report.md` to record scope, adaptations, artifacts,
functional/numerical/performance validation, constraints, and changes.
Numerical acceptance follows `.agent/rules/precision-acceptance.md`.
Reference performance and memory reports produced by `profiling` and
`memory-analysis`.

## Evidence requirements

- Attach the command, commit, and environment to every measurement.
  Mark unavailable measurements [not measured].
- Successful loading or execution alone does not prove correctness.
  Preserve verifiable checkpoint-conversion and distributed-adaptation
  evidence.
