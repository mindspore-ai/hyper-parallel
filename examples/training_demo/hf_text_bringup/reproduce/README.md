# Reproducing the recorded bring-ups

Six HF-native models were brought up on the generic kit in this directory's
parent, stage by stage, following the `hf-model-integration` skill's stage
machine (S0 survey through S5 acceptance). This directory lets someone else
rerun every stage and compare against what was recorded:

| file | what it is |
|---|---|
| `reproduce.sh` | one command per model and stage, with the options each recorded run used |
| `parity_check.py` | the S2 same-process parity, built through the path training uses |
| `adamw_recipe.py` | the AdamW-only recipe for the S3 attribution cell (a `_target_` cannot be overridden) |
| `reference_results.md` | what each stage recorded, and which records predate fixes |

Read "Before you compare" below first: two recorded result sets came from
code paths that have since been fixed or retired.

## What you need

- **Hardware as recorded:** Ascend NPU nodes with 8 dies each and 64 GB of
  HBM per die. The 32B model's two-node cells need two such nodes, shared
  storage mounted at the same path on both, and password-less ssh between
  them. Host RAM matters for the cells that load released weights with
  `training.init_device=cpu`: every rank materializes the full model on the
  host first (the 1 TB hosts used were never the limit).
- **Software as recorded:** CANN 9.0.0, Python 3.10, torch 2.9.0 with
  torch_npu 2.9.0.post2, and transformers 5.13.0. If your base environment
  carries an older transformers, put 5.13.0 in an overlay directory and pass
  it as `EXTRA_PYTHONPATH`; transformers 5.13 pins tokenizers <= 0.23.0 and
  huggingface_hub < 2.0. The image-text model also needs torchvision and
  Pillow in the same environment.
- **This repository** at a commit that contains the non-persistent buffer fix
  (`fix(models): keep constructed non-persistent buffers across to_empty`),
  or later.
- **Weights**, each in `$MODEL_ROOT/<directory>` where the directory is the
  last component of the Hub id:

  | model key | Hub id | parameters |
  |---|---|---|
  | `glm4-9b` | `THUDM/GLM-4-9B-0414` | 9.4B dense |
  | `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M` | 0.36B dense, tied embeddings |
  | `dsv2-lite` | `deepseek-ai/DeepSeek-V2-Lite` | 15.7B MoE with MLA |
  | `qwen25-32b` | `Qwen/Qwen2.5-32B` | 32.5B dense |
  | `qwen25-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 3.8B image-text |
  | `qwen35-9b` | `Qwen/Qwen3.5-9B` | 9B hybrid linear attention |

  The recorded runs did not note Hub revisions. Note the revision you
  download (`huggingface-cli download <id> --local-dir <dir>` prints it), so
  that a later difference can be told apart from a changed checkpoint.

## Running a stage

```bash
bash examples/training_demo/hf_text_bringup/reproduce/reproduce.sh list
DRYRUN=1 MODEL_ROOT=/data/models bash examples/training_demo/hf_text_bringup/reproduce/reproduce.sh qwen35-9b s3
MODEL_ROOT=/data/models CANN_SETENV=/path/to/set_env.sh PYTHON_BIN=/path/to/python \
    bash examples/training_demo/hf_text_bringup/reproduce/reproduce.sh qwen35-9b s3
```

`DRYRUN=1` prints every command and touches nothing; read it before a real
run. Outputs go to `$OUT/<model>/<stage>/` (default `<repo>/output/reproduce`):
one log per cell, plus `parity.json`, `judge.txt`, `breakdown.html` or
`untestable.txt` where the stage makes them.

| stage | what runs | compare with the record |
|---|---|---|
| `s1` | released-layout pin and tied-save prediction on the host | exit 0, entry count |
| `s2a` | parity on one die: parameters, buffers and logits bitwise | `PARITY PASS` |
| `s2b` | 30-step training smoke | loss start and end, step time, peak memory |
| `s3` | equivalence matrix (recompute, rerun, cross-layout) and verdicts | `judge.txt` |
| `s4` | 12-step profile, steps 5-8 traced, then the step breakdown | six consistency checks, the three shares |
| `s5` | long run sized by model scale | step-time median and tail, memory, loss trend |

Per-model extras: `smollm2-360m s3-adamw` repeats the cross-layout cell under
AdamW alone; `qwen25-32b s2b-full` and `s4-full` run the full 64-layer depth,
which needs host-side construction and recompute.

Run the stages in order; a later one assumes the earlier ones passed. The
two-node cells need `NODES="host1 host2"`.

## Shared nodes

Single-node cells run `.agent/skills/hf-model-integration/scripts/node_guard.sh pre`
on the node before launching and `post` after; the two-node launcher guards
every node itself, all-or-nothing. A busy node stops the stage with exit 3.
Back off and probe again later; do not queue behind someone else's job.

## S3 evidence and verdicts

By default the S3 cells are judged from the trainer log, as recorded: six
printed significant digits and no check that both arms consumed the same
inputs. `S3_EVIDENCE=structured` switches every cell to the runtime's
structured evidence (`model_integration.mode=runtime`: full-precision loss
and gradient norms with a per-step input hash), which the skill now
prescribes. That mode has not yet been run on device with this kit; if a cell
stops at build with model-integration findings, keep those findings and fall
back to the default.

Each verdict is one judge call, so a cell that could not run costs only the
verdicts that need it. Recompute and rerun pairs must match exactly.
Cross-layout pairs must match exactly for the first three steps; the judge
reports where they part and how fast the gap grows. A pair that differs at
step one is a defect. Divergence after a few steps is accumulated reduction
order, and its onset depends on the model. A cell that cannot fit is
recorded in `untestable.txt` with the reason, not skipped silently.

## Before you compare

1. **Runs that loaded released weights on more than one rank predate a fix.**
   Until the buffer fix, the pretrained path built on meta and called
   `to_empty`, and then restored only what the checkpoint held. It left
   every non-persistent buffer as uninitialized memory, RoPE `inv_freq`
   included. Affected records: every S3 cell of `smollm2-360m`,
   `qwen25-32b` and `qwen35-9b`, and every training stage of `qwen25-vl-3b`.
   Absolute losses will not match them. Both arms of each recorded
   comparison held the same buffer contents, as their identical early steps
   show, so "the layouts agree" still holds. The onset steps and growth
   figures were measured on a model with corrupted positional encoding and
   need measuring again. Runs from random initialization were not affected:
   the S2b, S4 and S5 stages of the text models, and every `glm4-9b` and
   `dsv2-lite` stage.
2. **GLM-4 was recorded on a pilot recipe that has since been retired:** a
   family registration and `examples/training_demo/glm4/`, kept under the
   `archive/glm4-bringup` tag. The driver runs it through the generic kit,
   which is the intended route, but that route has not yet run GLM-4 on
   device. To reproduce the recorded numbers exactly, check out the tag and
   use its `run_glm4_text.sh`.
3. **DSV2-Lite's recorded S3 used the earlier protocol:** random
   initialization on a 13-layer crop, judged by a statistical band. The
   current criterion replaced that protocol, and the driver runs the
   current one (released weights, a corpus of identical samples). The
   record therefore has no direct counterpart. Your run is the first under
   the current protocol.
4. **The recorded S2a parities exercised other paths.** The single-process
   parities ran before deferred loading reached this code base, so they
   compared Transformers' own loading against itself. The image-text parity
   also used `force_hf=True`. `parity_check.py` builds through the path the
   S3 runs use, which is why it can see the buffer defect and they could
   not. Its `--image` option, used for the image-text model, has not been
   run yet.
5. **The code base moved.** The recorded runs used the bring-up branch on
   an upstream checkout of 2026-10-20. This repository has since merged
   later upstream work, including Hugging Face loading through DCP. Expect
   step times to move slightly; a large difference is worth a report.
6. **The long runs never saved a checkpoint**, as recorded. The S5 records
   cover step-loop stability only.
