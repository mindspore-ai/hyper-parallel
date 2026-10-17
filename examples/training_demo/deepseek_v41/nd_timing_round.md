# Timing a round of DeepSeek-V4.1 layouts

A round answers one question: of the ways this crop can be laid out on a host of sixteen dies, which
one runs a step fastest? The cost model ranks every layout off the device in minutes; a round measures
the few worth measuring, so the ranking can be checked rather than believed.

Four tools do it, in this order. None of them ranks a layout against the cost model on the node: the
reading and the prediction are kept apart on purpose.

## 1. Before the window: check the node

```bash
./preflight_deepseek_v41_window.sh /path/to/DeepSeek-V4.1-Flash
```

Every check is a read. Nothing is launched and no device is allocated, so it is safe to run while
someone else holds the node. Run it the day before the window and again in its first minute. Exit
status is 0 only when every check at FAIL level passed. The warnings do not stop a round, they change
what it can be used for, so read them.

## 2. Pick the points

```bash
python nd_v41_ladder.py points
```

Prints the points no round has measured, as the `POINTS` string the next step takes, and the command
that runs them. Nothing is typed by hand on the node.

## 3. Measure

```bash
POINTS="4:1 4:2 8:1" REPEATS=3 ./repeat_deepseek_v41_runs.sh /path/to/DeepSeek-V4.1-Flash
```

Each point is launched `REPEATS` times with no profiler, and every step of every run is written as a
row, with the summary derived from those same rows. Useful knobs:

| Variable | What it does |
|----------|--------------|
| `POINTS` | the points to run, as `<expert degree>:<shard width>`, space separated |
| `REPEATS` | passes over the points, so each point gets a spread rather than one number |
| `RUN_ROOT` | where the rows land. Launch again with the same one and it skips the points already recorded |
| `LAUNCHER` | the trainer yaml every run reads. The default is `nd_train_deepseek_v41_timing.yaml` |
| `PROFILE` | turns the profiler on, which measures composition and never an order |
| `TRAIN_ITERS`, `SKIP_STEPS` | steps a run takes, and the warmup steps left out of the mean |

The default launcher is the one the cost model prices: it trains what the released launcher trains,
with checkpointing off, the grouped expert path off, and no drafter overrides. A round that reads
another yaml is timing another job, so say which yaml wherever the round is reported.

## 4. Read the round back

```bash
python nd_v41_ladder.py summary <RUN_ROOT> --out round.txt
```

Prints what the round measured: each point's mean step and spread, the ladder by expert degree at each
shard width, the order of shard widths inside each degree, and the points that have not run. Standard
library only, so it runs under any Python 3 on a node with nothing installed.

## What the crop is

```bash
python nd_describe_deepseek_v41.py spec /path/to/DeepSeek-V4.1-Flash --divisor 8 --experts 16
```

Writes what the launcher's factory builds as facts the cost model can read: the layers at their
released indices, their parts and their widths. `verify` compares that description against the
tensors the factory actually builds, which is the check that the two agree before any layout is
priced. Run it whenever the recipe changes.

`nd_deepseek_v41_crop.yaml` is that description kept as a file, beside the run it mirrors, for a
checkout where Transformers cannot resolve the released checkpoint. Its own header carries the three
commands that describe the crop, check the description and rank the layouts from it.

## Asking for a layout instead

The sapp-parallel-strategy skill wraps the cost model, this launcher and these tools, so an assistant can
answer "which layout" and then time the layouts worth timing. From the root of a checkout:

```bash
python .agent/skills/sapp-parallel-strategy/install.py
```
