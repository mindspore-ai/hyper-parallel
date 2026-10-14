# How far each answer can be trusted

ND ranks layouts well where its ranking was checked against runs timed on the same cluster, and is
unproven elsewhere. `trust.json` beside this file records that per template and cluster (`*` matches
any), and `scripts/ask.py` applies it to every answer, so the language model never has to remember it.

| Field | What it decides |
|---|---|
| `status`, `may_say` | The line at the top of every answer: quote it, do not soften it |
| `calibrated` | Whether the cluster's ratios turn ND's score into milliseconds for this family. The A3 ratios were fitted on Qwen3.5-MoE; on another family the answer gives relative scores |
| `region` | The degree values the timed runs covered. A pick outside it is flagged, and the best layout inside it is named as the launch |
| `exclude_modes` | Recompute modes a search that leaves the mode open does not try, with the reason the answer prints |
| `per_layer` | How far a plan of recompute per layer (`scripts/plan.py`) was checked, and the allocator it holds for |
| `checked_with` | The library versions the region was checked with. ND's census follows the installed Transformers (5.17's delta rule keeps 42% less in a Qwen3.5 linear-attention layer than 5.14.1's), so another version gets a warning |

What the records hold today:

- **Qwen3.5-MoE on the A3, back-tested.** At CP 1, TP 1, PP 1 with full recompute, ND's order matched
  the job's own step time (rank correlation +0.91 and +0.89 on two sets of timed layouts, its pick within
  0.54% of the fastest). At CP 2 it was not: with recompute off it priced those layouts 4.8 to 18.2% fast
  and could not order them, and its pick there cost 7.3%. Selective recompute is left out of open
  searches: ND prices it 37 to 45%
  below its measured step. A gain's size is never quoted from ND: it said 14.4% for a recompute plan
  that measured 6.0%.
- **DeepSeek-V4.1 on the A3, timed and not validated.** ND prices the expert degree backwards (order
  -0.39 over seven timed layouts, with the census as without), so the answer says what fits and
  recommends no expert degree. What fits is now measured: through the census, ND's memory came within
  -2.6% to +1.1% of each of the seven layouts' peaks, where the crop described by hand reads 60.6% low.
- **Any other family or cluster, never timed.** A ranking labelled not validated, after `scripts/check.py`
  has checked what ND prices for each layer kind; on a cluster with no calibration, relative scores only.

To change a record, measure first: a status moves only with timed runs on our own cluster, and the
record names the runs and the ND commit that priced them.

## Timing layouts, and the rounds already timed

`timers.json` says how a template's layouts are timed on a cluster: the script that times them, the launcher
yaml it reads, the degrees it varies and those it fixes, and the seconds a launch took. `scripts/launch.py` reads
it to plan a round and refuses to write a command when the launcher would run another job than the one ND
priced. `timed/` keeps the rounds already timed: a new round leaves their layouts out, apart from ND's best of
them and the fastest, which it times again first, and `scripts/readback.py` reads them with every new round.

- **DeepSeek-V4.1 on the A3:** `examples/training_demo/deepseek_v41/repeat_deepseek_v41_runs.sh` on one node of
  16 dies, expert degree and FSDP shard width varied, TP, CP and PP at 1; one round kept, seven layouts of the
  25 the script can launch, three runs each.
