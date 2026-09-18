# Benchmark Artifact Contract

Reuse the benchmark framework's native outputs and add Hyper metadata without maintaining a second task database.

## Required Artifacts

```text
<run>/
├── run_manifest.json
├── predictions/                 # framework-native prediction files
├── samples.jsonl                # normalized evidence index or references
├── metrics.json
└── summary.md
```

`run_manifest.json` records framework and Hyper revisions, checkpoint/tokenizer/processor fingerprints, task and dataset
revision, split, prompt/few-shot/generation/scoring/judge config, precision, device topology, sample range, and expected
sample count.

Each sample record contains:

- stable sample ID and task/category;
- input reference or digest and preprocessing metadata;
- raw prediction or candidate scores;
- extracted answer, reference answer when disclosure is allowed, and score;
- result owner, status, retry count, and explicit failure reason;
- framework-native prediction location.

## Completeness

- Every expected sample has exactly one terminal record: completed, failed, or explicitly skipped by the pinned protocol.
- Only the declared result owner writes the prediction and contributes metric counts.
- Report completed, failed, skipped, parsed, and judged counts separately. State the metric denominator policy.
- An interrupted run is `incomplete`; scores from the completed subset are diagnostic and are not the full-task score.
- Rebuild aggregate metrics from terminal sample records when validating resume. Do not add old totals to rerun totals.

## Resume And Reuse

Resume only when the manifest fingerprint matches. A new model/checkpoint, prompt, tokenizer/processor, media transform, or
generation config requires new inference. A scoring or judge-only change may reuse predictions when the framework supports
independent evaluation and the report records the new scoring fingerprint.

## Comparison

Compare native/reference and Hyper runs by sample ID. Require exact expected-sample coverage and no duplicates before
reporting a score delta. Retain disagreement records, candidate margins or raw text, extraction path, and judge/fallback
state so aggregate agreement can be explained.
