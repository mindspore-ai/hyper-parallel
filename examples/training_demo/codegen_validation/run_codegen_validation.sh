#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# End-to-end codegen dual-arm validation for the supported model matrix.
#
# Runs, per model, on a single 8-NPU host:
#
#   1. (Part 1) The repository-recommended training YAML with `codegen: true`
#      freshly generates the codegen artifact bundle (no pre-existing bundle).
#   2. (Part 2) The generated modeling file boots and drives a 10-step
#      training.
#   3. (Part 3) The native arm (identical YAML minus `codegen: true`) runs the
#      same 10 steps; per-step `training/total_loss` must align within
#      LOSS_TOL under seed 42 + full determinism.
#
# Usage (see README.md in this directory for the full contract):
#   bash examples/training_demo/codegen_validation/run_codegen_validation.sh
#
# Overridable environment variables:
#   PYTHON TORCHRUN NPROC TREE_ROOT VAL_ROOT QWEN_CONFIG DSV41_CONFIG
#   KIMI_CONFIG QVL_CONFIG RUN_QWEN RUN_DSV41 RUN_KIMI RUN_QVL LOSS_TOL
#   CLEAN_GENERATED
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------- config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python}"
TORCHRUN="${TORCHRUN:-/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/torchrun}"
NPROC="${NPROC:-8}"
TREE_ROOT="${TREE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VAL_ROOT="${VAL_ROOT:-/home/lxy/codegen_reval/val}"
QWEN_CONFIG="${QWEN_CONFIG:-/home/lxy/qwen3_30b_a3b_4L}"
DSV41_CONFIG="${DSV41_CONFIG:-/home/lxy/DeepSeek-V4.1-Flash}"
KIMI_CONFIG="${KIMI_CONFIG:-/home/lxy/probe_ckpts/kimi_k26}"
QVL_CONFIG="${QVL_CONFIG:-/home/lxy/probe_ckpts/qwen3_vl_moe}"
# Optional local dir already holding the Qwen2.5 tokenizer files; when empty
# the flow downloads them from the hub via QVL_HF_ENDPOINT (offline hosts
# should pre-seed this or QVL_TOKENIZER_SOURCE).
QVL_TOKENIZER_SOURCE="${QVL_TOKENIZER_SOURCE:-}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_DSV41="${RUN_DSV41:-1}"
RUN_KIMI="${RUN_KIMI:-0}"
RUN_QVL="${RUN_QVL:-0}"
LOSS_TOL="${LOSS_TOL:-0.0005}"
CLEAN_GENERATED="${CLEAN_GENERATED:-1}"
TRAIN_ITERS=10

# --------------------------------------------------------------- helpers ----
PASS_LIST=""
FAIL_LIST=""
record_pass() { PASS_LIST="${PASS_LIST}$1"$'\n'; echo "[PASS] $1"; }
record_fail() { FAIL_LIST="${FAIL_LIST}$1"$'\n'; echo "[FAIL] $1"; }
die() { echo "FATAL: $*" >&2; exit 2; }

# extract_losses <log> -> "step loss" lines for steps 1..TRAIN_ITERS
extract_losses() {
  grep -oE "step=[0-9]+ epoch=0 .*training/total_loss=[0-9.]+" "$1" \
    | sed -E "s/.*step=([0-9]+).*training\/total_loss=([0-9.]+)/\1 \2/" \
    | sort -n | uniq | head -n "$TRAIN_ITERS"
}

# run_arm <yaml> <log> <module> : run one training arm under torchrun.
run_arm() {
  local yaml="$1" log="$2" module="$3"
  echo "---- running $yaml ----"
  (cd "$TREE_ROOT" && ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
    --module "$module" "$yaml" > "$log" 2>&1)
}

# check_codegen_criteria <log>: the three mandatory success lines.
check_codegen_criteria() {
  local log="$1"
  grep -q "codegen: generating artifact" "$log" || return 1
  grep -q "codegen: imported generated modeling file" "$log" || return 1
  grep -q "codegen: sharding applied by the generated module" "$log" || return 1
  return 0
}

# render_arms <template> <native_yaml> <codegen_yaml> <arm_title_re>
# Renders placeholders, derives the codegen arm, and asserts the two arms
# differ only by the header comment and the `codegen: true` field.
render_arms() {
  local template="$1" native_yaml="$2" codegen_yaml="$3" sed_expr="$4"
  sed -e "s|__VAL_ROOT__|$VAL_ROOT|g" -e "$sed_expr" "$template" > "$native_yaml"
  sed '1,6s/^# \(.*\)native arm\(.*\)$/# \1codegen arm\2/' "$native_yaml" \
    | { cat; echo; echo "# codegen control field: the only difference from the native arm."; \
        echo "codegen: true"; } > "$codegen_yaml"
  local diff_lines
  diff_lines="$(diff "$native_yaml" "$codegen_yaml" | grep -cE '^[<>]' || true)"
  if [ "$diff_lines" -ne 5 ]; then
    die "arm diff sanity failed for $native_yaml: expected 5 diff lines (1c1 + 3 appended), got $diff_lines"
  fi
}

# validate_model <tag> <native_yaml> <codegen_yaml> <model_type> <module>: run both
# arms and check all three parts. Assumes the caller already prepared data +
# rendered YAMLs. The codegen bundle is model-anchored:
# <TREE_ROOT>/hyper_parallel/models/<model_type>/generated.
validate_model() {
  local tag="$1" native_yaml="$2" codegen_yaml="$3" model_type="$4" module="$5"
  local gen_dir log_cg log_nat
  gen_dir="$TREE_ROOT/hyper_parallel/models/$model_type/generated"
  log_cg="$VAL_ROOT/logs/${tag}_codegen_train.log"
  log_nat="$VAL_ROOT/logs/${tag}_native_train.log"
  mkdir -p "$VAL_ROOT/logs"

  # Part 1 preflight: force a fresh artifact (no pre-existing bundle).
  if [ "$CLEAN_GENERATED" = "1" ] && [ -d "$gen_dir" ]; then
    rm -rf "$gen_dir"
  fi
  if [ -d "$gen_dir" ] && [ -n "$(ls -A "$gen_dir" 2>/dev/null)" ]; then
    die "preflight failed: $gen_dir is not empty (set CLEAN_GENERATED=1)"
  fi

  echo "================ $tag: codegen arm ================"
  if ! run_arm "$codegen_yaml" "$log_cg" "$module"; then
    record_fail "$tag/codegen-arm-run"
    tail -30 "$log_cg" || true
    return 1
  fi
  record_pass "$tag/codegen-arm-run (rc=0)"

  if check_codegen_criteria "$log_cg"; then
    record_pass "$tag/part1-fresh-generate (generating/imported/sharding lines present)"
  else
    record_fail "$tag/part1-fresh-generate (missing codegen criteria lines, see $log_cg)"
  fi
  if ls "$gen_dir"/modeling_*_gen_npu.py >/dev/null 2>&1 \
     && [ -f "$gen_dir/codegen_meta.json" ]; then
    record_pass "$tag/part1-artifact-bundle (modeling_*_gen_npu.py + codegen_meta.json)"
  else
    record_fail "$tag/part1-artifact-bundle (incomplete bundle in $gen_dir)"
  fi

  local cg_losses nat_losses
  cg_losses="$(extract_losses "$log_cg")"
  if [ -n "$cg_losses" ] && [ "$(echo "$cg_losses" | grep -c .)" -ge "$TRAIN_ITERS" ]; then
    record_pass "$tag/part2-10-step-training (loss progression complete)"
  else
    record_fail "$tag/part2-10-step-training (incomplete loss lines, see $log_cg)"
  fi

  echo "================ $tag: native arm ================"
  if ! run_arm "$native_yaml" "$log_nat" "$module"; then
    record_fail "$tag/native-arm-run"
    tail -30 "$log_nat" || true
    return 1
  fi
  record_pass "$tag/native-arm-run (rc=0)"
  if [ "$(grep -cE "codegen: (generating|imported)" "$log_nat")" -eq 0 ]; then
    record_pass "$tag/native-arm-no-codegen"
  else
    record_fail "$tag/native-arm-no-codegen (unexpected codegen lines)"
  fi
  nat_losses="$(extract_losses "$log_nat")"
  if [ -n "$nat_losses" ] && [ "$(echo "$nat_losses" | grep -c .)" -ge "$TRAIN_ITERS" ]; then
    record_pass "$tag/native-10-step-training"
  else
    record_fail "$tag/native-10-step-training (incomplete loss lines, see $log_nat)"
  fi

  echo "================ $tag: loss alignment ================"
  echo "step | native      | codegen     | abs_diff"
  local aligned=1
  while read -r nat_step nat_loss cg_step cg_loss; do
    local abs_diff
    abs_diff="$("$PYTHON" -c "print(abs($nat_loss - $cg_loss))")"
    echo "$nat_step | $nat_loss | $cg_loss | $abs_diff"
    if [ "$nat_step" != "$cg_step" ] \
       || [ "$("$PYTHON" -c "print(1 if abs($nat_loss - $cg_loss) > $LOSS_TOL else 0)")" = "1" ]; then
      aligned=0
    fi
  done < <(paste <(echo "$nat_losses") <(echo "$cg_losses"))
  if [ "$aligned" = "1" ]; then
    record_pass "$tag/part3-loss-alignment (tol=$LOSS_TOL)"
  else
    record_fail "$tag/part3-loss-alignment (tol=$LOSS_TOL)"
  fi
  echo "$tag losses (native):"; echo "$nat_losses"
}

# ------------------------------------------------------------ 0. preflight ---
echo "== [0] environment preflight =="
[ -x "$PYTHON" ] || die "PYTHON not found: $PYTHON"
[ -x "$TORCHRUN" ] || die "TORCHRUN not found: $TORCHRUN"
if [ "$RUN_QWEN" = "1" ]; then [ -d "$QWEN_CONFIG" ] || die "QWEN_CONFIG not found: $QWEN_CONFIG"; fi
if [ "$RUN_DSV41" = "1" ]; then [ -d "$DSV41_CONFIG" ] || die "DSV41_CONFIG not found: $DSV41_CONFIG"; fi
if [ "$RUN_KIMI" = "1" ]; then [ -d "$KIMI_CONFIG" ] || die "KIMI_CONFIG not found: $KIMI_CONFIG"; fi
if [ "$RUN_QVL" = "1" ]; then [ -d "$QVL_CONFIG" ] || die "QVL_CONFIG not found: $QVL_CONFIG"; fi
# The import check runs from TREE_ROOT so it mirrors how torchrun picks up
# the checkout (cwd on sys.path), regardless of any stale editable install.
IMPORT_PATH="$(cd "$TREE_ROOT" && "$PYTHON" -c "import hyper_parallel, os; print(os.path.realpath(hyper_parallel.__file__))")"
case "$IMPORT_PATH" in
  "$(cd "$TREE_ROOT" && pwd)"*) echo "hyper_parallel imports from $IMPORT_PATH" ;;
  *) die "hyper_parallel must import from TREE_ROOT=$TREE_ROOT (got $IMPORT_PATH)" ;;
esac
record_pass "preflight/env"

# ------------------------------------------------------- 1. data assets ------
echo "== [1] data assets (idempotent) =="
mkdir -p "$VAL_ROOT/data"
if [ "$RUN_QWEN" = "1" ] && [ ! -s "$VAL_ROOT/data/parallel_online.jsonl" ]; then
  (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.qwen3_moe.prepare_parallel_data \
    --output-dir "$VAL_ROOT/data" --num-samples 128 --seq-length 128)
fi
if [ "$RUN_DSV41" = "1" ]; then
  if [ ! -s "$VAL_ROOT/data/online_1k.jsonl" ]; then
    (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_online_data \
      --output "$VAL_ROOT/data/online_1k.jsonl" --num-samples 128 --sequence-length 1024)
  fi
  if [ ! -s "$VAL_ROOT/data/engram_depth_preserving_d8.json" ]; then
    (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_assets \
      --model-dir "$DSV41_CONFIG" \
      --output "$VAL_ROOT/data/engram_depth_preserving_d8.json" \
      --bucket-base 4096 --parameter-divisor 8)
  fi
fi
if [ "$RUN_KIMI" = "1" ] && [ ! -s "$VAL_ROOT/data/kimi_k26/data.json" ]; then
  (cd "$TREE_ROOT" && "$PYTHON" -m hyper_parallel.models.kimi_k26.recipes.prepare_kimi_vlm_data \
    --output-dir "$VAL_ROOT/data/kimi_k26" --num-samples 128 --image-size 112)
fi
if [ "$RUN_KIMI" = "1" ]; then
  (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.codegen_validation.kimi_k26_assets \
    --source "$KIMI_CONFIG" --output "$VAL_ROOT/data/kimi_k26/model_config" \
    --data "$VAL_ROOT/data/kimi_k26/data.json")
fi
if [ "$RUN_QVL" = "1" ] && [ ! -s "$VAL_ROOT/data/parallel_offline_text_document.idx" ]; then
  (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.qwen3_moe.prepare_parallel_data \
    --output-dir "$VAL_ROOT/data" --num-samples 128 --seq-length 128)
fi
if [ "$RUN_QVL" = "1" ]; then
  QVL_TOK_ARGS=(--output "$VAL_ROOT/assets/qwen3_tok")
  if [ -n "$QVL_TOKENIZER_SOURCE" ]; then
    QVL_TOK_ARGS+=(--source "$QVL_TOKENIZER_SOURCE")
  else
    QVL_TOK_ARGS+=(--endpoint "${QVL_HF_ENDPOINT:-https://hf-mirror.com}")
  fi
  (cd "$TREE_ROOT" && "$PYTHON" -m examples.training_demo.codegen_validation.qwen3_vl_moe_assets \
    "${QVL_TOK_ARGS[@]}")
fi
record_pass "data/assets"

# ------------------------------------------------------ 2. render arm YAMLs --
echo "== [2] render dual-arm YAMLs =="
mkdir -p "$VAL_ROOT/qwen3_moe" "$VAL_ROOT/deepseek_v41" "$VAL_ROOT/kimi_k26" "$VAL_ROOT/qwen3_vl_moe"
if [ "$RUN_QWEN" = "1" ]; then
  render_arms "$SCRIPT_DIR/qwen3_moe_native.yaml" \
    "$VAL_ROOT/qwen3_moe/train_native.yaml" "$VAL_ROOT/qwen3_moe/train_codegen.yaml" \
    "s|__QWEN_CONFIG__|$QWEN_CONFIG|g"
fi
if [ "$RUN_DSV41" = "1" ]; then
  render_arms "$SCRIPT_DIR/deepseek_v41_native.yaml" \
    "$VAL_ROOT/deepseek_v41/train_native.yaml" "$VAL_ROOT/deepseek_v41/train_codegen.yaml" \
    "s|__DSV41_CONFIG__|$DSV41_CONFIG|g"
fi
if [ "$RUN_KIMI" = "1" ]; then
  render_arms "$SCRIPT_DIR/kimi_k26_native.yaml" \
    "$VAL_ROOT/kimi_k26/train_native.yaml" "$VAL_ROOT/kimi_k26/train_codegen.yaml" \
    "s|__KIMI_CONFIG__|$VAL_ROOT/data/kimi_k26/model_config|g"
fi
if [ "$RUN_QVL" = "1" ]; then
  render_arms "$SCRIPT_DIR/qwen3_vl_moe_native.yaml" \
    "$VAL_ROOT/qwen3_vl_moe/train_native.yaml" "$VAL_ROOT/qwen3_vl_moe/train_codegen.yaml" \
    "s|__QVL_CONFIG__|$QVL_CONFIG|g"
fi
record_pass "yaml/render+diff-sanity"

# ------------------------------------------------------------- 3-5. validate --
if [ "$RUN_QWEN" = "1" ]; then
  validate_model qwen3_moe \
    "$VAL_ROOT/qwen3_moe/train_native.yaml" "$VAL_ROOT/qwen3_moe/train_codegen.yaml" \
    qwen3_moe examples.training_demo.train_text || true
fi
if [ "$RUN_DSV41" = "1" ]; then
  validate_model deepseek_v41 \
    "$VAL_ROOT/deepseek_v41/train_native.yaml" "$VAL_ROOT/deepseek_v41/train_codegen.yaml" \
    deepseek_v41 examples.training_demo.train_text || true
fi
if [ "$RUN_KIMI" = "1" ]; then
  validate_model kimi_k26 \
    "$VAL_ROOT/kimi_k26/train_native.yaml" "$VAL_ROOT/kimi_k26/train_codegen.yaml" \
    kimi_k26 scripts.train_vl || true
fi
if [ "$RUN_QVL" = "1" ]; then
  validate_model qwen3_vl_moe \
    "$VAL_ROOT/qwen3_vl_moe/train_native.yaml" "$VAL_ROOT/qwen3_vl_moe/train_codegen.yaml" \
    qwen3_vl_moe examples.training_demo.train_text || true
fi

# ---------------------------------------------------------------- 6. report --
echo ""
echo "==================== VALIDATION REPORT ===================="
while IFS= read -r item; do
  if [ -n "$item" ]; then echo "  PASS  $item"; fi
done <<< "$PASS_LIST"
while IFS= read -r item; do
  if [ -n "$item" ]; then echo "  FAIL  $item"; fi
done <<< "$FAIL_LIST"
echo "==========================================================="
if [ -n "$FAIL_LIST" ]; then
  echo "RESULT: FAILED"
  exit 1
fi
echo "RESULT: PASSED"
