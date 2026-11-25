# HyperParallel agent workflows

The [repository guide](../AGENTS.md) defines the training framework's conventions.
This catalog describes the agent workflows, standalone tooling, and examples
in this directory. Model bring-up and performance experiments use the
training interfaces available in the selected checkout.

### Skills

| Skill | Entry point |
|---|---|
| **add-unit-test** | [Unit test authoring](skills/add-unit-test/SKILL.md) |
| **ascend-profiling-timeline** | [Interactive Ascend timelines and layer SVG reports](skills/ascend-profiling-timeline/SKILL.md) |
| **autogit** | [Commit and review workflow](skills/autogit/SKILL.md) |
| **autoresearch** | [Bounded experiment loop and experience consolidation](skills/autoresearch/SKILL.md) |
| **bringup-benchmark** | [Bring-up asset evaluation](skills/bringup-benchmark/SKILL.md) |
| **cluster-debug** | [Multi-rank failure diagnosis](skills/cluster-debug/SKILL.md) |
| **code-review** | [Distributed code review](skills/code-review/SKILL.md) |
| **debug** | [Single-node failure diagnosis](skills/debug/SKILL.md) |
| **dist-op-analysis** | [Distributed operator analysis](skills/dist-op-analysis/SKILL.md) |
| **dist-op-dev** | [Distributed operator development](skills/dist-op-dev/SKILL.md) |
| **feature-doc** | [Feature and model documentation](skills/feature-doc/SKILL.md) |
| **formal-verification** | [Training contracts and formal verification](skills/formal-verification/SKILL.md) |
| **gate-doctor** | [CI gate diagnosis](skills/gate-doctor/SKILL.md) |
| **hf-model-integration** | [Model adaptation and migration](skills/hf-model-integration/SKILL.md) |
| **hf-model-precision-validation** | [Model parity and numerical acceptance](skills/hf-model-precision-validation/SKILL.md) |
| **knowledge-retrieval** | [Repository evidence retrieval](skills/knowledge-retrieval/SKILL.md) |
| **memory-analysis** | [Device memory attribution](skills/memory-analysis/SKILL.md) |
| **mfu-riser** | [Evidence-based training performance optimization](skills/mfu-riser/SKILL.md) |
| **moe-optimizer** | [MoE performance diagnosis and candidate selection](skills/moe-optimizer/SKILL.md) |
| **moe-profiler** | [MoE expert load and communication profiling](skills/moe-profiler/SKILL.md) |
| **parallel-strategy-analyzer** | [Parallel layout analysis](skills/parallel-strategy-analyzer/SKILL.md) |
| **perf-playbook** | [Training performance diagnosis and tuning methods](skills/perf-playbook/SKILL.md) |
| **precision** | [Tensor dump comparison and numerical diagnosis](skills/precision/SKILL.md) |
| **profiling** | [Ascend trace capture and performance breakdown](skills/profiling/SKILL.md) |
| **routing-validator** | [MoE routing and gradient validation](skills/routing-validator/SKILL.md) |
| **sapp-parallel-strategy** | [Symbolic parallel strategy search and ranking](skills/sapp-parallel-strategy/SKILL.md) |
| **strategy-optimizer** | [Measured parallel strategy optimization](skills/strategy-optimizer/SKILL.md) |
| **strategy-validator** | [Parallel strategy benchmarking](skills/strategy-validator/SKILL.md) |

### Commands

[Command definitions](commands/) map slash commands to skill entry points.
For shell and Python usage, follow each skill's own instructions.

### Agents

| Agent | Scope |
|---|---|
| **activation-dev** | [Activation checkpointing and swap](agents/activation-dev.md) |
| **code-reviewer** | [Distributed code review](agents/code-reviewer.md) |
| **code-verifier** | [Implementation and test verification](agents/code-verifier.md) |
| **dtensor-dev-expert** | [DTensor and layout](agents/dtensor-dev-expert.md) |
| **ep-dev-expert** | [Expert parallelism and routing](agents/ep-dev-expert.md) |
| **formal-verifier** | [Training correctness](agents/formal-verifier.md) |
| **fsdp-dev-expert** | [Sharding, gradient reduction, and memory](agents/fsdp-dev-expert.md) |
| **llamafactory-hp** | [LlamaFactory integration](agents/llamafactory-hp.md) |
| **moe-load-balancer** | [MoE load distribution](agents/moe-load-balancer.md) |
| **parallel-strategy-analyzer** | [Parallel layout analysis](agents/parallel-strategy-analyzer.md) |
| **pipeline-dev-expert** | [Pipeline schedules and microbatches](agents/pipeline-dev-expert.md) |
| **planner** | [Implementation planning](agents/planner.md) |
| **sapp-parallel-strategy** | [SAPP ND search](agents/sapp-parallel-strategy.md) |
| **simple-code-reviewer** | [Quick code review](agents/simple-code-reviewer.md) |
| **tensor-dev-expert** | [Tensor parallelism and mesh](agents/tensor-dev-expert.md) |

### Rules and local validation

Refer to the [scope-specific rules](rules/) when editing agent tooling.
Run the existing catalog and Markdown link checks from the repository root.
Both checks use the Python standard library.

```bash
python .agent/scripts/check_agents_catalog.py
python .agent/skills/strategy-optimizer/scripts/check_skill_architecture.py --links-root .agent
```

These checks validate catalog entries and local link targets, not optimizer
behavior or training correctness. Add behavioral regression tests to the
repository's [unit test suite](../tests/ut/) using the
[unit test authoring workflow](skills/add-unit-test/SKILL.md); do not add
per-skill validators that duplicate these checks or report fixed simulated
optimization results as validation.

The [experience registry](learning/registry.json) records reusable procedures
and their owning skills. Examples under [examples](examples/) use public
framework interfaces. Performance and numerical receipts come from the
commands configured for each experiment.
