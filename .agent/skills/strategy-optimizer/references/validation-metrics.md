# Benchmark metrics

The Trainer emits distinct step records containing `tokens/s=`, `step_time=...s`, and
`memory/device_max_allocated_gb=`. The scheduler requires all requested steps, rejects nonfinite/nonpositive values,
and ignores explicitly identified nonzero-rank duplicates. It recognizes `step N` and tqdm `N/total` records.

`throughput_tokens_per_sec` and `step_time_ms` are means after excluding up to ten warmup steps
(20% of the observed steps). `memory_peak_gb` is the maximum over the full run.
`total_steps`, `measured_steps` and `warmup_steps` retain the sample counts.
A zero exit without these observations is a failed validation.
