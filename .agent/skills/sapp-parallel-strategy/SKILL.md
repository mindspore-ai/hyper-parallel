---
name: sapp-parallel-strategy
description: >
  Recommend how to lay a job out on Ascend dies (data parallel, FSDP shard group, TP, CP, PP, EP and
  the recompute mode), say what fits in memory, and compare named layouts, by pricing every valid
  layout with ND, HyperParallel's device-free cost model. Use for "which layout", "which parallel
  strategy", "will it fit", "which of these is faster". Not for writing parallel code, debugging a
  run or kernel work.
---

# SAPP parallel strategy

You answer one question: how should this job be laid out on these dies? ND prices every valid layout
of a job in a few minutes on a laptop, with no device. Your work is to turn the request into ND's
inputs, run ND once through `scripts/ask.py`, and explain what comes back. **Every number you give
comes from the scripts' output**: never compute, round away or estimate one yourself.

## Install

The skill sits in the repository at `.agent/skills/sapp-parallel-strategy/`, where no assistant looks by itself.
One command puts it where they do look, from the root of a checkout:

```bash
python .agent/skills/sapp-parallel-strategy/install.py
```

It links the skill into `~/.agents/skills` for Codex, which asks with `$sapp-parallel-strategy`, and into
`.claude/skills` for Claude Code, which asks with `/sapp-parallel-strategy`. Then it says what the checkout
can price and prints a first command to try. `--check` reports without changing anything, `--for`
picks one assistant, `--scope` moves between the home folder and the checkout, and `--copy` copies
the folder where links cannot be made, leaving the path of the checkout beside it.

Any other assistant: tell it to read and follow `.agent/skills/sapp-parallel-strategy/SKILL.md`.

Work in a checkout with the Python environment that has the repository's dependencies (torch,
Transformers, PyYAML). ND runs on the CPU, with no device. Where `hyper_parallel/models/deepseek_v41`
is in the checkout, the V4.1 crop is priced with a census of its layers; where it is not, `ask.py`
prices the description instead and says so. `install.py` reports which of the two applies.

## Answer a request

A question about the fastest layout runs as a loop: check the model (step 5), rank the layouts (steps 1 to 4),
time the layouts worth timing (step 8), then read the runs back and pick (step 9). Steps 8 and 9 are what make
an answer trustworthy where ND's order was never checked against timed runs.

1. **Read the request.** Collect the model (a template name, or the asker's trainer yaml), the layers,
   the tokens a sequence, the global batch, the micro-batch, the recompute mode if stated, the total
   dies, the cluster and a memory budget if stated. Where a value is missing, use the template's and
   say so. Templates: `templates/qwen3_5_moe.yaml` (Qwen3.5-35B-A3B and its layer crops) and
   `templates/deepseek_v41.yaml` (the DeepSeek-V4.1 validation crop as its launcher builds it, 40
   layers, which a request cannot change; ND measures its memory with the census). In a checkout without
   the V4.1 model, `ask.py` prices the crop described by hand instead, by formulas, and says so.
2. **Check what can be trusted.** `references/trust.json` says, for each template on each cluster, how
   far ND was checked against timed runs; `references/status.md` explains it. `ask.py` applies it: it
   prints the status line, uses the cluster's calibration only where it holds for the family, leaves
   out of an open search the recompute modes ND misprices, and checks the pick against the region
   that was timed.
   - No template and no yaml from the asker: ask for their trainer yaml; nothing can be priced.
   - A yaml of a family with no record: check the model (step 5) before ranking, and say the
     ranking is not validated.
3. **Run ND once**, through the script beside this file. It finds the checkout it belongs to and runs ND
   there, so it works from any folder, also through a link to this skill:

   ```bash
   python <this skill's folder>/scripts/ask.py --family qwen3_5_moe --cluster a3 \
       --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 --recompute full \
       --request "<the request, in the asker's words>"
   ```

   `--recompute` takes the mode the run uses (`off`, `selective`, `full`), several to compare them,
   or `auto` when the request leaves it open (the default): every mode the family's record does not
   exclude, and the answer says which were left out and why. With the asker's yaml, use
   `--yaml <file>` and keep `--family` when the model belongs to a listed family. Never run ND by hand:
   `ask.py` passes the recompute mode, the cluster's calibration and its budget, which ND otherwise
   prices as full recompute, drops, or reads as 64 GB.
4. **Answer from the output** (also saved as `answer.md` in the run folder), in this order: the
   layout to launch, the next two, why (the table of the best layout with another value of each
   degree, read out in words from its part moves), the status line, ND's notices verbatim, and the
   run folder. When several layouts sit within the cluster's re-measure spread, the firm answer is
   what they share: say one round cannot order them, and offer to time the two that differ most.
   **When the answer says ND's pick lies outside the region that was timed, the layout to launch is
   the best one inside it**, which the answer names; give ND's pick as the candidate worth timing, with
   the reason the answer quotes. With no region at all, recommend nothing ND alone decides: say what
   fits, and which layouts are worth timing.
5. **Check the model**, for a model ND has not been checked on, or when asked what ND prices of it:
   `python <this skill's folder>/scripts/check.py` with `--family` or `--yaml` as for `ask.py`. It sets
   the parameters ND prices for each part of a layer beside what the model's own layers hold (DeepSeek-V4.1
   through its own tool, against the crop's tensors, and the blocks no ND field carries; any other model
   through ND's verify mode, FLOPs too), and keeps the check in a run folder. Report every part that
   differs; a part ND prices at zero is a part the ranking ignores.
6. **Named candidates**: `python <this skill's folder>/scripts/answer.py <run>/nd_rank.csv
   --calibrated --compare "EP=2,OP=32;EP=1,OP=64"` gives each one's place, step and peak. A candidate
   with no row did not fit the budget or is outside ND's search; say which you cannot tell apart.
7. **Which layers can skip recompute**, for a layout already chosen: `python <this skill's folder>/scripts/plan.py`
   with the same run flags plus `--layout "EP=2,OP=32"` (TP, CP and PP are 1 unless given). It states the
   budget from the cluster's measured reserve, since ND prices allocated memory and a run dies on reserved,
   lets ND choose each layer's mode, and writes the plan into `launch.yaml` (`activation_checkpoint.layers`).
   Give the plan, the budget and how it was derived, and the allocator note it prints; never the gain.
8. **Time the layouts worth timing**, when the asker wants the fastest layout and the status line says ND's order
   was not checked, or a near tie needs a round: `python <this skill's folder>/scripts/launch.py <run folder>`,
   the run folder of step 3. It chooses the layouts by the trust record's rule, never by hand: where ND's order
   was never checked, every layout that fits, that the template's timer launches and that no round has timed;
   where it was, the layout to launch and those one round cannot tell from it. It times again first the layouts
   earlier rounds measured, ND's best of them and the fastest, so the new round can be set beside them. It writes
   a trainer yaml a layout and one command for the cluster (`launch/` in the run folder). Give the layouts and
   why, the launches and the time it states, and the command verbatim: a person runs it on the cluster and
   brings back `runs.csv` and `steps.csv`. `--layouts "EP=2,OP=4;EP=4,OP=1"` times named layouts instead.
9. **Read the runs back and pick**, once a round has run: `python <this skill's folder>/scripts/readback.py
   <run folder> <round folder>`; with no round folder it reads the rounds kept with this skill. Answer in this
   order: the pick and whether one more round could change it, the table, ND's order over the timed layouts and
   what following ND costs, ND's memory against the measured peaks, failed launches, what the plan still has to
   time, and the trainer yaml of the pick (`readback/` in the run folder). The pick is the measured one.

ND's degrees, as the answer names them: DP data parallel, TP tensor parallel, CP context parallel,
PP pipeline stages (VPP virtual stages), EP expert parallel, OP the FSDP shard group of the
non-expert parameters, MB micro-batches a step, MBS sequences a micro-batch.

The run folder also holds the trainer yaml to launch: `launch.yaml` for the layout the answer recommends,
and `launch_nd_pick.yaml` for ND's own pick when that differs or nothing validates it. They are the
request's yaml with the degrees written where the trainer reads them, the expert shard group included
(`edp_shard_size`, the whole expert group, as ND priced it and as every timed run was launched), and
without ND's `context` section, which the trainer refuses. Tell the asker to set the tokenizer and data
paths for their cluster; a virtual pipeline degree, when one is picked, is not written.

## Rules

- Every figure in the answer is one the scripts printed. No differences, ratios or memory sums of your own.
- One ND run a question; run again only when the request changes, and say why.
- Never change ND, a template or a calibration to make an answer come out differently.
- Once layouts are timed, the layout to launch is the fastest measured, never ND's pick: ND's order is reported
  as a check on ND. Never choose which layouts to time yourself; `launch.py` applies the rule.
- Quote the order, never the size of a gain: the status line says how far each ranking was checked.
- Pass ND's notices on verbatim. A part it says it does not price is a part the ranking ignores.
- Name the run folder: it holds the request, the yaml, the command, the versions, ND's log and
  ranking, so the answer can be checked and run again.
