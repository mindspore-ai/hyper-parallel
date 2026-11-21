#!/usr/bin/env bash

fail() {
    echo "$*" >&2
    exit 2
}


configure_determinism() {
    if [[ "${DETERMINISTIC:-0}" == 1 ]]; then
        export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
        export HCCL_DETERMINISTIC=true CLOSE_MATMUL_K_SHIFT=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
        overrides+=(
            "++actor_rollout_ref.actor.engine.full_determinism=true"
            "++actor_rollout_ref.ref.engine.full_determinism=true"
            "actor_rollout_ref.rollout.full_determinism=true"
        )
    else
        unset HCCL_DETERMINISTIC CLOSE_MATMUL_K_SHIFT NVTE_ALLOW_NONDETERMINISTIC_ALGO
    fi
}

configure_debug() {
    local layers
    case "${DEBUG:-0}" in
        0) ;;
        1)
            layers=${DEBUG_LAYERS:-3}
            overrides+=(
                "data.max_response_length=1024"
                "trainer.test_freq=-1"
                "trainer.val_before_train=false"
                "++actor_rollout_ref.model.override_config.num_hidden_layers=$layers"
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.num_hidden_layers=$layers"
                "trainer.total_training_steps=${DEBUG_STEPS:-1}"
            )
            ;;
        *) fail "Unsupported DEBUG: $DEBUG (expected 0 or 1)" ;;
    esac
}

configure_rollout() {
    local steps
    [[ "${ROLLOUT_SKIP:-0}" != 1 ]] || export ROLLOUT_MODE=replay
    case "${ROLLOUT_MODE:-off}" in
        off) ;;
        replay|record)
            steps=${ROLLOUT_SKIP_STEPS:-${TRAINING_STEPS:-${DEBUG_STEPS:-1}}}
            [[ "$steps" =~ ^[1-9][0-9]*$ ]] || fail "Invalid rollout cache steps: $steps"
            [[ -n "${ROLLOUT_CACHE_DIR:-}" ]] || fail "Set ROLLOUT_CACHE_DIR to a model/layer-specific cache"
            overrides+=(
                "skip.rollout_tq.steps=[$(seq -s, 1 "$steps")]"
                "skip.rollout_tq.enable=true"
                "skip.rollout_tq.action=cache"
            )
            ;;
        *) fail "Invalid ROLLOUT_MODE: $ROLLOUT_MODE" ;;
    esac
}

launch_training() {
    cd "$HP_SOURCE_ROOT"
    printf 'HyperParallel source: %s\n' "$HP_SOURCE_ROOT"
    printf 'HyperParallel branch: %s\n' "$(git symbolic-ref --quiet --short HEAD || printf 'detached HEAD')"
    git log -1 --format='HyperParallel commit: %H%nHyper commit date: %cI%nHyper commit subject: %s'
    python -m verl.trainer.main_ppo \
        --config-path "$HP_SOURCE_ROOT/examples/verl" \
        --config-name "${VERL_CONFIG_NAME:-hyper_parallel}" \
        "${overrides[@]}" "$@"
}

# Configure the environment before applying patches or importing VeRL.
set -eo pipefail

export HP_SOURCE_ROOT="${HP_SOURCE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONPATH="$HP_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/cache/data/dapo-math-17k/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-/cache/data/aime-2024/test_unique.parquet}"
export MODEL_PATH="${MODEL_PATH:-/cache/data/Qwen3-30B-A3B-Base}"
export HYPER_PARALLEL_RECIPE="${HYPER_PARALLEL_RECIPE:-$HP_SOURCE_ROOT/hyper_parallel/models/qwen3_moe/recipes/train.yaml}"
export ROLLOUT_CACHE_DIR="${ROLLOUT_CACHE_DIR:-}"
export VERL_USE_EXTERNAL_MODULES="hyper_parallel.integration.verl.patch${VERL_USE_EXTERNAL_MODULES:+,$VERL_USE_EXTERNAL_MODULES}"

overrides=()
configure_determinism
configure_debug
configure_rollout
launch_training "$@"
