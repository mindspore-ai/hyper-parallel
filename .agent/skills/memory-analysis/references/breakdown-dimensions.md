# Memory Breakdown and Classification Rules

Rule version **v2** uses fields and stack frames captured on torch_npu
2.9.0.post2. See [Snapshot format](snapshot-format.md) for field origins
and measured examples.

## Memory totals

| Quantity | Source | Meaning |
|---|---|---|
| Reserved | Sum of `segment.total_size` | Total device memory reserved by the allocator |
| Allocated | Sum of `segment.allocated_size` | Memory in use at capture time |
| Active | Sum of `segment.active_size` | Active memory, including allocations pending release |
| Fragmentation | Reserved minus allocated | Reserved-but-unused memory; inspect allocator behavior if it exceeds 10% of reserved memory |

Read totals directly from segment fields. Separately sum blocks for
consistency checks rather than using the same computation for both sides
of an identity.

`max_memory_allocated` is stored by `mem_hook` in the
`<snapshot>.stats.json` sidecar. It is not a snapshot field. The report
checks and displays the difference between sampled allocation and peak.

## Allocation classification

Apply these stack rules in order.

| Order | Category | Rule | Reference observation |
|---|---|---|---|
| 1 | Optimizer state | A Python frame path contains `/optim/` | AdamW uses `torch/optim/adam.py:177/181 _init_group`; matching only `adamw` would miss it. |
| 2 | Parameters | A Python frame names `_module.py` and function `convert` or `to` | `module.to(device)` reaches `torch_npu/utils/_module.py:76 convert`. |
| 3 | Gradients | A native frame contains `AccumulateGrad` | The captured gradient allocation has no Python frame. |
| 4 | Activations | The innermost user-code function contains `forward` | Capture during backward while activations are live. |
| 5 | Other user tensors | A user-code frame exists but no earlier rule matches | For example, an input batch. |
| 6 | Unclassified | No usable attribution frame | Report the missing evidence without guessing. |

### Reconciliation against a known workload

The reference workload used three BF16 `Linear(2048, 2048, bias=False)`
layers, AdamW, and a BF16 input with shape 512 by 2048.

| Category | Parsed value | Independent accounting | Result |
|---|---|---|---|
| Parameters | 24.0 MiB | 3 x 2048 squared x 2 bytes | Match |
| Gradients | 24.0 MiB | Same BF16 element count as parameters | Match |
| Optimizer state | 48.0 MiB | 2 states x 3 parameters x 8 MiB | Match |
| Activations during backward | 12.0 MiB | No independent byte formula recorded | A later sample reported 2.0 MiB |
| Other user tensors | 2.0 MiB | 512 x 2048 x 2 bytes for the input | Match |
| Total | 112.0 MiB | `allocated_size` of 112.01 MiB | Match at the displayed precision |

In this AdamW configuration, `exp_avg` and `exp_avg_sq` use
`zeros_like(param)`. BF16 parameters therefore have BF16 states of 8 MiB
each, not FP32 states of 16 MiB each. Do not generalize another optimizer's
state dtype to this capture.

## Call-site attribution

Stack frames are ordered from innermost to outermost.

1. Skip capture-internal frames such as `unwind`,
   `CapturedTraceback`, and unresolved `??` frames.
2. Prefer the innermost `.py` frame outside `site-packages` and
   `dist-packages`.
3. Otherwise use the innermost framework Python frame.
4. Otherwise use a native symbol, skipping allocator plumbing with
   `at::` and `c10::` prefixes.

This avoids attributing every block to the capture mechanism or to
`at::_ops::empty_strided::redispatch`. See
[Stack-frame format](snapshot-format.md#stack-frame-format).

## Consistency checks

| Check | Basis | Required evidence |
|---|---|---|
| Per segment, all block sizes sum to `total_size` and active block sizes sum to `allocated_size` | Allocator consistency | Snapshot |
| All active block sizes sum to all segment `allocated_size` values | Global accounting | Snapshot |
| Segment `total_size` sums to `memory_reserved()` | Process counter | Sidecar |
| Segment `allocated_size` sums to `memory_allocated()` | Process counter | Sidecar |
| Sampled allocation is within 2% of the peak | `max_memory_allocated` | Sidecar |

The first four checks validate accounting. The last prevents interpreting
a non-peak sample as peak composition. A failed check produces exit code 2
and is highlighted in the report.

## Capture requirements

- Capture steps 1-3 by default. Later peaks can require additional samples.
- Capture during backward to observe live activations.
  `mem_hook.dump(tag)` can be called from an autograd hook.
- For long runs, capture when a memory or performance anomaly occurs rather
  than waiting until the job ends.
- Use `perf-playbook` for recomputation, offload, and allocator changes.
  Measure both the memory effect and the end-to-end performance effect.

## Pending validation

- Peak-composition reconstruction by replaying `device_traces`.
- Module/layer aggregation of allocation call sites.
