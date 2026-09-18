---
name: benchmark-evaluation
description: >
  Design, implement, run, or review HyperParallel capability evaluation with
  OpenCompass for LLMs and VLMEvalKit for VLMs. Covers pinned protocols,
  Hyper execution adapters, sample ownership, resumable predictions, metrics,
  and native-vs-Hyper parity. Use for benchmark、MMLU、OpenCompass、VLMEvalKit.
---

# Benchmark Evaluation

Use external benchmark frameworks for task semantics and HyperParallel for model execution. Do not duplicate dataset,
prompt, answer extraction, judge, or aggregation logic inside HyperParallel.

Load `.agent/rules/model-development-validation.md`. Use `accuracy-validation` first when the execution path has not passed
numerical parity.

## Routing

- LLM knowledge/reasoning/scoring tasks: OpenCompass. Read [opencompass-demo.md](references/opencompass-demo.md).
- VLM image/video generation-style tasks: VLMEvalKit. Read [vlmevalkit-demo.md](references/vlmevalkit-demo.md).
- Shared run artifacts and completeness checks: [artifact-contract.md](references/artifact-contract.md).

## Required Inputs

- benchmark framework revision and environment lock;
- model/checkpoint and tokenizer/processor revision;
- task/dataset revision, split, prompt/few-shot protocol, and scoring mode;
- native/reference model adapter and proposed Hyper adapter;
- device topology, precision, generation/PPL parameters, and result directory;
- judge model/config when the task requires one.

Ask only when the benchmark protocol or comparison reference is ambiguous.

## Workflow

1. Pin framework, task config, dataset, model assets, and the requested protocol. Save their identifiers before inference.
2. Run a small native/reference subset and save expanded inputs, tokenization/media metadata, raw predictions or candidate
   scores, extracted answers, and metrics.
3. Run the same subset through the Hyper adapter. Verify processed inputs before comparing outputs.
4. Confirm outer benchmark scheduling does not treat TP/CP/EP/PP ranks as independent model replicas. One persistent Hyper
   execution group receives a request; one owner returns and writes the result.
5. Compare sample coverage, order, prediction agreement, failures, and metrics. Use the benchmark tolerance in
   `accuracy-validation/references/tolerance-policy.md` for deterministic classification.
6. Only after subset parity, run the complete task. Preserve prediction reuse only when model, input protocol, and
   preprocessing fingerprints match.
7. Validate artifacts against [artifact-contract.md](references/artifact-contract.md) and publish a support matrix for the
   exact model/backend/dtype/topology combination.

## Boundaries

- OpenCompass owns datasets, retrievers, prompt templates, inferencers, evaluators, and summaries. The Hyper adapter owns
  `get_token_len`, sequence scoring/PPL semantics, generation, and conversion to the framework return contract.
- VLMEvalKit owns media preparation, dataset prompts, prediction persistence, answer extraction, judges, and reports. The
  Hyper adapter owns structured-message processing, model processor calls, parallel execution, and generated text.
- A standard text-generation API cannot satisfy OpenCompass PPL tasks unless it exposes the required token-level scoring.
- Training forward support does not prove autoregressive generation, KV-cache, stop-token, multi-image, or video support.
- Framework fallback behavior, especially judge fallback, must be visible in the report.

## Output

- State pinned versions, model/task protocol, adapter boundary, command, topology, and artifact directory.
- Report sample coverage, duplicates, failures, prediction agreement, score delta, and metric delta.
- Separate native/reference results from Hyper results and inference from evaluation.
- Mark an adapter described only by a demo contract as PROPOSED, never IMPLEMENTED.

## Read On Demand

- [artifact-contract.md](references/artifact-contract.md): run manifest, sample records, metrics, completeness, and resume.
- [opencompass-demo.md](references/opencompass-demo.md): MMLU PPL-first mechanism demo.
- [vlmevalkit-demo.md](references/vlmevalkit-demo.md): image multiple-choice generation mechanism demo.
- `examples/evaluation/`: user-facing demo walkthroughs.
