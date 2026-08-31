# Workflow 1: Scope Analysis

## Goal

Confirm the change stays inside `hyper_parallel/rl/` and identify exactly which
subsystems, files, tests and docs it touches.

## Steps

### 1.1 Read the design and the module map

- `.agent/rules/hyper-rl-workflow.md` (RL context + constraints, auto-applied)
- [../references/module-map.md](../references/module-map.md)

### 1.2 Classify the change

| Change touches | Subsystem(s) | CPU gate | NPU smoke |
| --- | --- | --- | --- |
| Config keys, model identity | `config.py` | `rl_tests/test_config.py` | both |
| Training loop, lifecycle, checkpoint | `trainer.py` | `rl_tests/test_trainer_orchestration.py` | ordinary TP |
| GRPO / loss / math | `algorithm/` | `rl_tests/test_algorithm_registry.py` | both |
| Dataset / agentic | `dataset/`, `agentic/` | `rl_tests/test_experience_preparer.py`, `test_agentic_runner.py`, `test_data_source.py` | ordinary TP |
| Rollout | `roles/rollout/` | `rl_tests/test_rollout_topology.py`, `test_vllm_runtime.py` | ordinary TP |
| Weight sync | `roles/weight_sync/` | `rl_tests/test_direct_reshard.py`, `test_hccl_weight_sync.py`, `test_checkpoint_manager.py` | consistency |
| Consistency | `consistency/` | `rl_tests/test_master_qwen3_contracts.py` | consistency |

### 1.3 Boundary gate

- Any file outside `hyper_parallel/rl/` → **stop**, tell the user to open a
  separate upstream task.
- Any rejected-by-design item (`topology`, rank-local server, 2nd Router,
  multi-node, off-policy) → **stop**, surface in the design before implementing.

### 1.4 Output

A one-line scope summary + the exact test files + the NPU launcher to use. This
is the input to the design the human approves.
