# Hyper-RL Design Goals and Principles

**English** | [简体中文](design.zh.md)

## Goals and Scope

Long-term goal: **Support everything from basic reinforcement learning to multi-turn agents, multimodal learning, and large-scale asynchronous training with a compact, extensible core.**

The framework targets **online reinforcement learning for LLMs/VLMs**, covering basic policy optimization and learning through agent interaction. Simplicity, extensibility, and native agentic support are design goals. Agentic capabilities evolve independently while sharing core interfaces with basic RL; every development stage need not add agentic features. The current implementation builds on synchronous Ascend NPU training, covering sampling, rewards, policy updates, publication, evaluation, and recovery. The first rayless fully asynchronous GRPO implementation uses a fixed single-node disjoint deployment. Independent colocated asynchronous GRPO reuses its sampling contract and alternates generation and updates by phase; a short two-device flow has passed, while learning quality and long-run stability remain unvalidated. Multimodal and larger-scale training remain future directions. The runtime architecture document defines support and validation boundaries.

The main repository maintains general training capabilities and representative recipes within this scope. Business logic and proprietary integrations can extend downstream projects or forks. Exhaustive coverage of algorithms, models, backends, and deployment combinations is not a goal. See the [README](../README.md#supported-capabilities) for current support; this document defines long-term tradeoffs, not implemented features.

## Infrastructure Choices

| Component | Responsibility |
| --- | --- |
| HyperParallel / HyperAutoModel | Model loading, training, distributed parallelism, and optimizers |
| vLLM | Inference execution, request scheduling, and sampling services |
| Rayless runtime | torchrun training ranks, spawn sampling, bounded standard-library IPC |
| Hyper-RL | Tasks and rewards, learning algorithms, training orchestration, sample contracts, and policy publication integration |

The main repository maintains a small set of representative recipes covering distinct technical boundaries, not every combination of model, backend, and topology. Infrastructure choices are not an exhaustive dependency list; the [runtime image documentation](../docker/README.md) owns the underlying environment.

Other backends, such as SGLang, may be adapted in downstream forks and are outside official maintenance. Independently validate tokens, logprobs, action masks, policy identity, weight updates, cache handling, and failure propagation; correctness claims from the vLLM path do not transfer automatically.

## Design Principles

A unified task interface should support single-turn tasks, multi-turn environments, and programmatic agents. Policy actions and environment observations remain distinct and enter training through the same trajectory contract.

1. **Explicit orchestration and separate responsibilities.** Each execution mode has a clear top-level orchestrator: SyncTrainer today, potentially separate orchestration for asynchronous mode. Algorithms own data requirements and mathematics; tasks own environments and rewards. Users control task programs, while the framework provides generation, experience collection, and learning. Trainer must not embed task names or customer logic.
2. **Explicit data and state boundaries.** Components exchange data through explicit contracts, with one owner per mutable state. Every trainable action segment traces to its effective generation context, original tokens, policy identity, and probability semantics; rewards trace to scoring rules and environment versions. Unpublished policies must not be visible to new sampling. Synchronous mode consumes one published version; asynchronous mode must define grouping, consumption, sample lag, and correction/rejection conditions. Correct parameter publication does not establish correct sample and learning semantics.
3. **Local extensions.** Integrating data, rewards, or environments must not change training infrastructure; algorithm extensions compatible with existing roles must not duplicate Trainers. Training mechanism changes may modify core implementations directly, without constructing a generic plugin system in advance.
4. **Limited abstractions and unified configuration.** Abstractions must remove actual duplication or protect necessary contracts, not add compatibility layers for hypothetical backends. Parse and validate configuration centrally at entry; launch scripts must not reimplement training decisions.
5. **Reuse foundational capabilities.** Infrastructure should provide models, parallelism, and inference where possible. Synchronous, asynchronous, and multimodal execution should share task interfaces, sample semantics, and algorithm components rather than duplicate the training stack. RL-specific adaptations belong at their corresponding boundaries.
6. **Measure efficiency through learning outcomes.** Distinguish task failure from environment failure, and training rewards from evaluation results. Compare training time and resource costs at equal quality; throughput and utilization help locate bottlenecks.
7. **Readability and reliability constrain simplification.** Keep traceable call paths and clear names; do not reduce apparent code size by compressing syntax or moving logic elsewhere. Sample validation, error propagation, necessary logging, and recovery are foundational capabilities.

See [Architecture](architecture.md) for interfaces, state ownership, and failure semantics. See the [Hyper-RL development rules](../../../.agent/rules/hyper-rl.md) for workflows and documentation maintenance, and [feature navigation](../../../docs/rl-navigation.md) for feature-to-implementation/test mappings.

## Feature Admission and Downstream Maintenance

New core capabilities should solve shared needs, fit existing component responsibilities, and include appropriate tests and end-to-end validation. Business rewards, proprietary tools, customer deployments, and experimental mechanisms belong first in examples or downstream projects. The main repository is not a general agent application platform or continuous-action control framework. A second implementation alone does not justify a new abstraction.

Keep user-facing task entry points and key contracts explicit. Provide migration guidance for necessary incompatible changes. Avoid unnecessary interface renaming or directory moves to reduce the cost of synchronizing general fixes into downstream forks.

Evaluate the design through reproducible development and training tasks:

- Follow documentation for data preparation, training, evaluation, and restart recovery, without launch scripts hiding overrides of training decisions.
- Replace rewards, add tools, or integrate environments without modifying training infrastructure; algorithms compatible with existing roles must not duplicate Trainer.
- Publish representative task evaluation results and the actual modification scope so usability and extension cost can be assessed.
- When adding asynchronous execution, modalities, or parallel combinations, validate sample alignment, policy versions, failure propagation, and recovery. Results include learning quality, models, hardware, runtime configuration, and cost to reach target quality.

Representative models, stage objectives, and quantitative acceptance criteria are maintained in [support and planning](../README.md#支持范围). Current support remains documented in the README and corresponding product documents, without duplication here.
