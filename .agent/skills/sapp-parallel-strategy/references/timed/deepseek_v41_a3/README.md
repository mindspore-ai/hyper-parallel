# DeepSeek-V4.1 on the A3, the round of 8 October 2026

Seven layouts of the validation crop, three runs each with the profiler off, timed by
`examples/training_demo/deepseek_v41/repeat_deepseek_v41_runs.sh` on one node of 16 dies: 4096 tokens a
sequence, global batch 16, micro-batch 1, recompute off, Muon, six steps a run with steps 3 to 6 kept.

- `runs.csv`: one row a run, its mean step (ms) over the kept steps and its peaks. The trainer reports the
  peaks as `max_memory_allocated() / 1024 ** 3`, so they are GiB.
- `steps.csv`: one row a kept step, from the trainer's own clock.

`scripts/launch.py` leaves these layouts out of a new round, apart from the one it times again to set the new
round beside this one, and `scripts/readback.py` reads them with every round it is given.
