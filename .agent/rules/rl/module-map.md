# hyperparallel-RL Module Map

Use this stable index to locate subsystem ownership. Paths are relative to `hyper_parallel/rl/`; interface and runtime semantics live in the linked product docs.

| Area | Code | Detail |
| --- | --- | --- |
| User entry | — | [中文 README](../../../hyper_parallel/rl/README.md) · [English README](../../../hyper_parallel/rl/README.en.md) |
| Design | — | [Design goals and principles](../../../hyper_parallel/rl/docs/design.md) |
| Delivery plan | — | [Milestones and acceptance](../../../hyper_parallel/rl/docs/TODO.md) |
| Config | `rl/config.py` | [Architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Trainer | `rl/trainer.py`, `rl/evaluation.py` | [Architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Dataset / Agentic | `rl/dataset/`, `rl/agentic/` | [Architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Agentic harnesses | `rl/agentic/codex/`, `rl/agentic/deepseek/`, `rl/agentic/core/program_runner.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Algorithm / Policy | `rl/algorithm/`, `rl/roles/policy/`, `rl/roles/model.py` | [Architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| MoE models | `rl/roles/rollout/vllm_moe.py`, `rl/roles/rollout/vllm_qwen3_moe.py`, `rl/roles/rollout/vllm_deepseek_v3.py` | [MoE models](../../../hyper_parallel/rl/docs/moe_models.md) |
| Rollout | `rl/roles/rollout/` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Weight sync | `rl/roles/weight_sync/` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency | `rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Runtime | `examples/scripts/` | [Runtime image](../../../hyper_parallel/rl/docs/hyper_rl_runtime_image.md) |
| Utils | `rl/utils/` | — |
| Main-project impact | Outside `hyper_parallel/rl/` | [Public module changes](../../../hyper_parallel/rl/docs/public_module_changes.md) |
