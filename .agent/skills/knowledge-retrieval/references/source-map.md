# Repository Knowledge Map

Start with rules and skills, then documentation, implementation, tests, and
official external references. Use the most authoritative source available
for the question.

## Repository sources

| Question | Location |
|---|---|
| Project conventions and skill/agent/rule catalog | `AGENTS.md`, then its linked entries |
| Code style, distributed, testing, and numerical acceptance constraints | `.agent/rules/**` |
| Commit, review, experiment, diagnosis, and documentation workflows | `.agent/skills/*/SKILL.md` and their `references/` |
| Feature behavior, interfaces, and constraints | `docs/**`, including `docs/guide/**`, `docs/api/**`, and `docs/feature_status.md` |
| DTensor, TP, FSDP, PP, EP, CP, activation, and checkpoint implementation | `hyper_parallel/core/**` |
| Public assembly of distributed strategies | `hyper_parallel/distributed/**`, including `expert_parallel/`, `context_parallel/`, and `activation_checkpoint.py` |
| Automatic parallelization, strategy search, and offload planning | `hyper_parallel/auto_parallel/**`, including `sapp_nd`, `sapp_ppb`, `hyper_offload`, and `config_adapter` |
| Model-family adapters | `hyper_parallel/models/<family>/adapter/**` |
| Shared components and operator wrappers | `hyper_parallel/components/**` |
| Trainer and configuration fields | `hyper_parallel/trainer/**` |
| Datasets, tokenization, batching, and distributed data loading | `hyper_parallel/data/**` |
| Graph compilation and passes | `hyper_parallel/compile/**`, including `passes/`, `tracer/`, and `pass_plan.py` |
| Process-group construction | `hyper_parallel/collectives/cc.py` |
| External framework integration | `hyper_parallel/integration/**`, currently LlamaFactory |
| Generation, KV cache, and sampling | `hyper_parallel/infer/**` |
| General tools outside a particular parallel implementation | `hyper_parallel/tools/**` |
| Reinforcement learning | Start with `.agent/rules/hyper-rl.md`, the dedicated RL entry point |
| Contracts and expected behavior | `tests/ut/**` and distributed system tests under `tests/**` |
| Runnable recipes and launchers | `examples/**` |

Keep the map aligned with the actual tree. In particular, public
`distributed/**` assembly and `auto_parallel/**` search are separate from
the core execution implementation. Missing a directory can lead to
inventing an API that already exists elsewhere.

## Official external sources

Use external sources when the repository does not answer the question.

| Question | Source |
|---|---|
| Framework documentation | Official HyperParallel / MindSpore project documentation |
| Operator or CANN behavior | Official Ascend documentation and operator repositories |
| Released model architecture | Upstream model repository, including Hugging Face configuration |

## Common lookup paths

| Question | Search path |
|---|---|
| How is an option configured? | Find its field in `hyper_parallel/trainer/config/**`, then inspect runnable recipes. |
| Is a feature supported? | Check `docs/feature_status.md` and relevant documentation, then verify the implementation. |
| What does an error mean? | Search the exact message in skill references and then at its raise site. |
| What is the module contract? | Read the corresponding unit-test names and assertions. |
| Why is it implemented this way? | Read docstrings and the owning skill's design references. |

## Evidence requirements

- If documentation conflicts with implementation, report the conflict and
  use the implementation to describe current behavior.
- A matching line is a lead. Read the complete definition or test before
  answering.
- State when a capability cannot be found; do not invent an API.
