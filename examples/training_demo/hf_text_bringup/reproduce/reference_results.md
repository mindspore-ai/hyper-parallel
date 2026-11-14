# Recorded results, per model and stage

What each stage recorded when it was first run, on Ascend nodes with 8 dies
of 64 GB, CANN 9.0.0, torch 2.9.0 and transformers 5.13.0. Read `README.md`
"Before you compare" first.

- **Defect** marks a record made with released weights loaded on more than
  one rank before the non-persistent buffer fix (README item 1). Its
  absolute numbers will not reproduce.
- **"identical"** in an S3 row is identity at the trainer log's printed
  precision (six significant digits, inputs unchecked) unless the row says
  otherwise. No recorded S3 cell used the structured runtime evidence.
- Step times are min / median / p99.5 over the run unless stated.
- Loss values are on the kit's synthetic corpus. They show the run is
  learning; they are not a quality claim.

## glm4-9b — GLM-4-9B-0414 (pilot recipe, `archive/glm4-bringup`)

| stage | recorded | notes |
|---|---|---|
| S1 | layout pin PASS, 523 released entries, untied `lm_head` | |
| S2a | logits bitwise equal (max diff 0.0); short greedy generation coherent | single process, Transformers loading on both sides |
| S2b | 40 layers, dp8, recompute full, bf16 init: loss 12.75 → 3.12 over 30 steps, 2.56 s/step, 22.2 GB peak | fp32 init does not fit a 64 GB die at 9B |
| S3 | 20-layer crop, 10 steps: recompute off/full identical; rerun identical; dp8 vs dp4 band 1.7e-4 at the last step | earlier protocol (random init) |
| S4 | profiler invariant within tolerance (80.1 µs drift at 2.576 s/step) | |
| S5 | 1000/1000 steps, loss 12.68 → ~0.01, 2.56–2.57 s/step, memory flat at 22.16 GB | |

## smollm2-360m — SmolLM2-360M

| stage | recorded | notes |
|---|---|---|
| S1 | layout pin PASS, tied `lm_head` recognised as an alias | |
| S2a | HP vs Transformers bitwise (0.0) | single process, Transformers loading on both sides |
| S2b | fp32 init, dp8: loss 11.06 → 10.06 over 30 steps, ~0.45 s/step, 15.65 GB | |
| S3 | recompute off/full identical; rerun identical; dp8 vs dp4 band 8.66e-3 | earlier protocol (random init) |
| S3 (released weights, identical samples) | first 3–4 steps identical across dp8/dp4, then apart. Muon+AdamW: onset step 4 at 2.4e-3, 4.1e-2 at step 10. AdamW alone (`s3-adamw`): onset step 5 at 5e-5, 4.6e-4 at step 10 | **Defect** |
| S4 | compute 38.2 % / uncovered communication 29.4 % / free 32.4 %; six checks pass | small model, eager attention, micro-batch 1 |
| S5 | 1000/1000 steps; loss 11.03 → 3.74 (100) → 0.084 (500) → 0.044 (1000); 0.404 / 0.459 / 0.542 s; 15.652 GB flat | no checkpoint saved |

## dsv2-lite — DeepSeek-V2-Lite

| stage | recorded | notes |
|---|---|---|
| S1 | layout pin PASS: 4,992 released per-expert weights reconciled with 52 fused expert tensors | |
| S2a | bitwise PASS | single process, Transformers loading on both sides |
| S2b | bf16 init, dp8: loss 11.74 → 6.94 over 30 steps, ~2.0 s/step, 45.4 GB peak | |
| S3 | 13-layer crop: recompute off/full and rerun identical; dp8 vs dp4 differed at step 1 (1.9e-2); root cause was initialization following the mesh plus different packing | earlier protocol; that finding led to the current one |
| S4 | six checks pass first time; MoE routing falls back to AI_CPU: Sort 82.5 ms, IndexPutV3 24.9 ms, Identity 22 ms; AI_CPU ahead of Cube (2.72 M µs against 1.82 M µs over the traced window) | upstream issue #1272 |
| S5 | 1000/1000 steps, loss 11.71 → 0.060, median 1.97 s, p99.5 2.15 s; one 852 s stall at step 457 that recovered with no error logged | no checkpoint saved |

## qwen25-32b — Qwen2.5-32B

| stage | recorded | notes |
|---|---|---|
| S1 | full-shape layout pin PASS, 771/771 | |
| S2b | 16-layer crop, two nodes, HSDP 2×8: loss 13.32 → 0.81 over 30 steps, 2.1 s/step, 43.2 GB | full depth blocked on device-side init (#1245) |
| S2b full depth (`s2b-full`) | 64 layers, two nodes dp16, host-side init, recompute full: 10/10 steps, loss 13.0845 → 8.35583, ~8.96 s/step, 37.41 GB | the init ceiling is not the model's ceiling |
| S3 | 16-layer crop: recompute off/full identical over 10 steps; rerun identical; one node dp8 vs two nodes dp16 identical for the first 3 steps, then apart from step 7 (7.267e-5), 1.841e-4 at step 10; dp4 untestable (memory) | **Defect** |
| S4 | 16-layer crop, two nodes dp16: 6/6 checks after the profiler-tolerance fix | |
| S4 full depth (`s4-full`) | compute 85.0 % / uncovered communication 14.4 % / free 0.6 % | |
| S5 | 16-layer crop, two nodes dp16: 1000/1000 steps, 2.05 / 2.16 / 2.28 s, 39.10 GB | no checkpoint saved |

## qwen25-vl-3b — Qwen2.5-VL-3B-Instruct

| stage | recorded | notes |
|---|---|---|
| S1 | 824/824 after prefix-rename reconciliation | |
| S2a | bitwise (0.0), synthetic-image forward and generation correct | built with `force_hf=True`, not the training path |
| S2b | dp8: loss 14.76 → 2.06 over 30 steps, ~8.2 s/step, 6.5 GB | **Defect** |
| S3 | recompute off/full identical; rerun identical; dp8 vs dp4 band 4.4e-2 | **Defect**; earlier protocol |
| S4 | compute 2.1 % / uncovered communication 78.5 % / free 19.4 %, 8.0–8.3 s/step; AI_CPU 21.37 M µs over the traced window, from indexing in the packing path (Index, IndexPut, PadV3, UniqueConsecutive) | made before the buffer fix, but the shares do not depend on buffer values; upstream issue #1268 |
| S5 | 27,000 samples, 1000/1000 steps, loss 14.68 → 0.097, 7.95 / 8.26 / 8.86 s, 6.47 GB flat | **Defect**; no checkpoint saved |

## qwen35-9b — Qwen3.5-9B

| stage | recorded | notes |
|---|---|---|
| S1 | 760/760 built entries match; 15 released `mtp.*` tensors labelled as a declared-ignored subtree (Transformers implements no MTP module) | |
| S2a | text path bitwise (0.0), generation coherent | single process; the product API built the text-only class (upstream issue #1269) |
| S2b | bf16 init, recompute full: 30/30 steps, loss 13.5171 → 2.77388, 4.99 → 4.03 s/step, 24.54 GB. Recompute off: one step, then out of memory at 56.84 GB | needs the `log_` distributed-op registration, which this repository carries |
| S3 | rerun identical over 10 steps; dp8 vs dp4 identical for 2 steps, apart from step 3 (8.096e-4), 3.640e-2 at step 10; recompute off untestable (out of memory at step 1) | **Defect** |
| S4 | compute 72.4 % / uncovered communication 7.2 % / free 20.4 %; invariant drift 33.2 µs against a 4,515 µs tolerance; no operator fallback | |
| S5 | 1000/1000 steps, loss 13.4791 → 0.00129 (500) → 0.000296, 4.00 / 4.06 / 4.44 s, 26.79 GB | no checkpoint saved |
