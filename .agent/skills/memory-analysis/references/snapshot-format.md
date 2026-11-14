# Memory Snapshot Format and Capture Notes

The fields and behavior below were observed with `torch_npu.npu.memory`
on **torch_npu 2.9.0.post2 + CANN 9.0.0 / Ascend 910**.

## API

| Call | Purpose |
|---|---|
| `_record_memory_history(enabled="all", context="all", stacks="all", max_entries=N)` | Record allocation history; `stacks="all"` captures Python frames for attribution. |
| `_snapshot(device=None)` | Return an in-process snapshot dictionary. |
| `_dump_snapshot(filename=...)` | Write a pickle snapshot. |
| `_record_memory_history(enabled=None)` | Stop recording. |

The captured implementation signature is
`_record_memory_history_impl(enabled='all', context='all', stacks='all',
max_entries=sys.maxsize, device=None)`.

### Capture pitfalls

1. In the validated version, `_dump_snapshot` creates a file with mode
   `0200`, which is write-only even for its owner. `mem_hook.dump()`
   changes it to `0600`. For manual dumps, use `chmod 600 <file>`
   before parsing.
2. `_dump_snapshot` also creates `PROF_00000N_*` and `FRAMEWORK/`
   directories in the working directory.
3. The recorded environment emitted
   `AclrtGetMemUsageInfo failed, ret:207000` at ERROR level while still
   producing valid snapshots. Verify the actual artifacts and counters
   rather than treating that message alone as capture failure.

## Snapshot structure

```python
{"segments": [...], "device_traces": [...]}
```

### Segment fields

Recorded fields are `active_size`, `address`, `allocated_size`,
`blocks`, `device`, `frames`, `is_expandable`, `requested_size`,
`segment_pool_id`, `segment_type`, `stream`, and `total_size`.

| Field | Meaning |
|---|---|
| `total_size` | Bytes reserved from the device for the segment |
| `allocated_size` | Bytes in use in the segment |
| `active_size` | Active bytes including pending frees; equality with `allocated_size` indicates no pending release |
| `segment_type` | `large` or `small` |
| `frames` | Empty in the reference capture; use block frames for attribution |

### Block fields

Recorded fields are `address`, `frames`, `requested_size`, `size`,
and `state`. Observed states include `active_allocated` and `inactive`.
The parser also treats `active_pending_free` as active by its prefix.

### Accounting identities

The snapshot alone supports:

```text
Per segment: sum(all block.size) == segment.total_size
Per segment: sum(active block.size) == segment.allocated_size
```

The process counters in `<snapshot>.stats.json` support:

```text
sum(segment.total_size)     == memory_reserved()
sum(segment.allocated_size) == memory_allocated()
```

All four identities had zero drift in the reference capture.
`breakdown.py` checks them and exits with code 2 on failure.

## Stack-frame format

The captured stack is innermost-first. Its initial frames belong to the
capture mechanism itself:

```text
[0] torch_npu::unwind::unwind()
[1] torch_npu::CapturedTraceback::gather(...)
[2..] ??                      # Unresolved native frames
...
[26] first Python frame       # Example position in a 62-frame stack
...
[61] python3 / __libc_start_main
```

- The first Python-frame index is not fixed. Scan the stack.
- Python frames are also innermost-first, for example
  `torch_npu/utils/_module.py:76 convert`,
  `torch/nn/modules/module.py:957 _apply`, then `train.py:31 main`.
- Prefer the innermost user Python frame outside `site-packages` and
  `dist-packages`, then a framework Python frame, then a native symbol.
- Skip allocator-plumbing native symbols with `at::` or `c10::`
  prefixes. `torch::autograd::AccumulateGrad::apply` is a useful fallback;
  `at::_ops::empty_strided::redispatch` is not.

### Gradients can have no Python frames

The reference gradient allocations contained no `.py` frames because
the C++ autograd engine created them. The useful signal was
`torch::autograd::AccumulateGrad::apply`. Classification based only on
Python frames would miss the gradient category.

## device_traces

Each device has a list of events with `action`, `addr`, `size`,
`stream`, and `frames`. Actions include `segment_alloc` and `alloc`.
These traces could reconstruct peak-time composition, but the current
`breakdown.py` does not replay them. That accounting remains pending
validation.

## A snapshot is not necessarily the peak

One reference run recorded:

| Capture point | Allocated memory |
|---|---|
| After `optimizer.step()` | 100.01 MiB, with no live activations |
| During backward, inside an autograd hook | 112.01 MiB |
| `max_memory_allocated()` | 130.01 MiB |

The post-step sample misses live activations. Even the backward sample is
18 MiB below the peak. `mem_hook.dump(tag)` supports capture at arbitrary
points and writes the peak counter into the sidecar. `breakdown.py`
fails its peak-coverage check when the gap exceeds 2% and labels the report
as sampled composition rather than peak composition.
