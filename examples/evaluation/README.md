# Capability evaluation demos

These examples define how external benchmark frameworks call HyperParallel. OpenCompass MMLU has a first single-process
PPL adapter; VLMEvalKit remains an integration contract and does not claim current production support.

| Demo | Purpose |
| --- | --- |
| [OpenCompass](opencompass_demo.md) | Implemented single-process MMLU PPL adapter and runnable config |
| [VLMEvalKit](vlmevalkit_demo.md) | VLM generation and framework-owned answer extraction/scoring |

Both demos follow the same boundary:

- the benchmark framework owns datasets, prompts, prediction files, answer processing, metrics, and reports;
- a thin adapter converts framework requests into a persistent HyperParallel execution session;
- model-parallel ranks cooperate on one logical request and only one result owner returns or writes it;
- a pinned native/reference adapter runs first, followed by Hyper single-device and supported distributed topologies;
- every complete run saves a manifest, sample-level records, metrics, and a human-readable summary.

Agent workflow and artifact requirements live in
`.agent/skills/benchmark-evaluation/SKILL.md`.
