# Supported extraction inputs and missing evidence

The extractor produces exactly 15 indexed artifacts even for partial evidence. Empty CSVs contain headers.
`moe_facts.json` and stdout include `evidence_status`; extraction success is not a correctness PASS.
Missing metrics are null. Straggler attribution has no implemented aligned-rank estimator yet, so those
CSVs remain empty, `max_straggler_slowdown` stays null and the full evidence status is PARTIAL.

## Logs

```text
rank=0 step=1 world_size=8 ep_size=2 hidden_size=2048 dtype_bytes=2
Layer 3 input_tokens=2 top_k=2 tokens_per_expert=[2,2]
[MoE] Layer 3 dispatch: 2.0ms, 4.0MB
[MoE] Layer 3 combine: 3.0ms, 4.0MB
step_time=0.1s other_comm=1.0ms
```

Step/rank context comes from explicit lines (rank can also come from `rank0.log`).
Sparse layer IDs are supported. `--steps` selects the first N observed step IDs; `--ranks` selects
observed ranks and excludes unknown-rank logs/events. These filters apply to log/profile observations;
all supplied checkpoint shards/captures are checked. No default world size, hidden size, EP or step count is inferred.
Routing scores remain unavailable; load counts do not imply those scores.

## Profiler JSON / SQLite

```json
{"ops":[{"name":"AllToAll","duration_us":2000,"bytes":4000000,
"rank":0,"step":1,"layer_id":3,"direction":"dispatch"}]}
```

JSON requires `ops` as a list of mappings. Durations carry explicit `duration_ns`, `duration_us` or
`duration_ms` units; bytes are optional. SQLite is read-only: tables must expose `name`, `op_name` or `type`
and an explicit duration column, with optional matching JSON fields. Unsupported native profiler schemas
produce warnings and no fabricated timings. Profile operations are used only if the logs contain no communication
operations, avoiding duplicate counting. Logs must supply step timing for a communication/step ratio.

Volumes use decimal MB; bandwidth uses Gbit/s. Summed operation duration / step time can exceed 1 under
overlap and is not uncovered communication. Compute time remains null. Estimated token payload counts require
explicit hidden size and dtype bytes and are labelled as estimates.

## Routing and gradients

The shared [routing evidence contract](../../routing-validator/references/evidence-contract.md) applies.
Real activation history can be supplied as `expert_activation_history`, a list of mappings with
`layer_id`, `step` and `tokens_per_expert`; it is never replaced by a synthetic 50-step history.
