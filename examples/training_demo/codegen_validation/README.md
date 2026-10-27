# Codegen Dual-Arm Validation (Qwen3-MoE + DeepSeek-V4.1 + Kimi-K2.6 + Qwen3-VL-MoE)

This directory freezes the end-to-end validation flow for the codegen `gen`
backend on a single 8-NPU host (Ascend 910B3). It re-runs, from a clean
checkout, the three acceptance parts for both models:

| Part | What is verified | Mandatory evidence |
| ---- | ---------------- | ------------------ |
| 1 | The repository-recommended training YAML with `codegen: true` **freshly generates** the artifact bundle (no pre-existing bundle) | Log lines `codegen: generating artifact ...`, `codegen: imported generated modeling file ...`, `codegen: sharding applied by the generated module ...`; `generated/` contains `modeling_*_gen_npu.py` + `codegen_meta.json` |
| 2 | The generated modeling file **boots and trains** | Training completes 10 steps with a complete loss progression (`training/total_loss` per step) |
| 3 | Codegen arm vs native arm **loss alignment** under fixed determinism | Both arms run the identical YAML except `codegen: true`; per-step `training/total_loss` abs diff <= `LOSS_TOL` (default 5e-4) |

Both arms use `seed: 42` and `enable_full_determinism: true`, so the loss
sequence is reproducible run-to-run on the same host.

## Layout

```text
examples/training_demo/codegen_validation/
├── README.md                    # this document
├── run_codegen_validation.sh    # the frozen end-to-end flow
├── kimi_k26_assets.py           # cropped-config + offline processor assets (kimi_k26)
├── qwen3_vl_moe_assets.py       # tokenizer assets (qwen3_vl_moe)
├── qwen3_moe_native.yaml        # native-arm template (from qwen3_moe/train_parallel_online.yaml)
├── deepseek_v41_native.yaml     # native-arm template (from deepseek_v41/train_deepseek_v41_online.yaml, 8-card adapted)
├── kimi_k26_native.yaml         # native-arm template (from kimi_k26/recipes/train.yaml, cropped)
└── qwen3_vl_moe_native.yaml     # native-arm template (from the Qwen3-VL-MoE codegen probe YAML, tp2/ep2/fsdp2+edp2)
```

The templates carry `__QWEN_CONFIG__`, `__DSV41_CONFIG__`, `__KIMI_CONFIG__`,
`__QVL_CONFIG__` and `__VAL_ROOT__` placeholders. The script renders them into
the validation workspace and derives the codegen arm by appending
`codegen: true` — a diff sanity check asserts the two arms differ by that
field (plus the header comment) only. Codegen artifacts land in
`<TREE_ROOT>/hyper_parallel/models/<model_type>/generated`, which is why each
model gets its own subdirectory under the validation root.

## Prerequisites

- One host with 8 free NPUs (the flow pins `ASCEND_RT_VISIBLE_DEVICES=0-7`).
- A checkout of this repository at the commit under validation, with
  `pip install -e .` done from that checkout (the script asserts
  `hyper_parallel` imports from `TREE_ROOT`).
- Cropped model assets on the host:
  - Qwen3-MoE cropped config (config + tokenizer), default
    `/home/lxy/qwen3_30b_a3b_4L`
  - DeepSeek-V4.1-Flash config (config + tokenizer), default
    `/home/lxy/DeepSeek-V4.1-Flash`
  - Kimi-K2.6 cropped checkpoint config, default `/home/lxy/probe_ckpts/kimi_k26`
  - Qwen3-VL-MoE cropped config, default `/home/lxy/probe_ckpts/qwen3_vl_moe`
- Validation data assets are generated idempotently by the script into
  `$VAL_ROOT/data/` (128 samples; seq 128 for Qwen, seq 1024 + scaled-Engram
  JSON for DeepSeek-V4.1; Kimi-K2.6 uses `hyper_parallel.models.kimi_k26.recipes`
  offline assets; Qwen3-VL-MoE reuses the Qwen3-MoE offline indexed dataset
  plus the Qwen2.5 tokenizer fetched into `$VAL_ROOT/assets/qwen3_tok`).

## Usage

```bash
bash examples/training_demo/codegen_validation/run_codegen_validation.sh
```

Defaults target the 8-NPU validation host. Override via environment
variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `PYTHON` | `/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python` | Python interpreter of the target env |
| `TORCHRUN` | same env's `torchrun` | Launcher binary (must support `--standalone`) |
| `NPROC` | `8` | Processes per node |
| `TREE_ROOT` | repository root (derived from script location) | Code tree under validation |
| `VAL_ROOT` | `/home/lxy/codegen_reval/val` | Validation workspace (YAMLs, data, logs) |
| `QWEN_CONFIG` | `/home/lxy/qwen3_30b_a3b_4L` | Cropped Qwen3-MoE asset dir |
| `DSV41_CONFIG` | `/home/lxy/DeepSeek-V4.1-Flash` | DeepSeek-V4.1-Flash asset dir |
| `KIMI_CONFIG` | `/home/lxy/probe_ckpts/kimi_k26` | Kimi-K2.6 cropped asset dir |
| `QVL_CONFIG` | `/home/lxy/probe_ckpts/qwen3_vl_moe` | Qwen3-VL-MoE cropped config dir |
| `QVL_TOKENIZER_SOURCE` | *(empty)* | Local dir with the Qwen2.5 tokenizer files; when empty the flow downloads them from `QVL_HF_ENDPOINT` (default `https://hf-mirror.com`) |
| `RUN_QWEN` / `RUN_DSV41` | `1` | Select which models to validate |
| `RUN_KIMI` / `RUN_QVL` | `0` | Select which models to validate |
| `LOSS_TOL` | `0.0005` | Per-step loss abs-diff tolerance (part 3) |
| `CLEAN_GENERATED` | `1` | Remove a pre-existing `generated/` before the codegen arm so the artifact is always freshly generated |

Examples:

```bash
# DeepSeek-V4.1 only, keep any existing artifacts for inspection
RUN_QWEN=0 CLEAN_GENERATED=0 \
  bash examples/training_demo/codegen_validation/run_codegen_validation.sh

# Custom workspace and assets
VAL_ROOT=/data/val_qwen QWEN_CONFIG=/data/models/qwen3_30b_a3b_4L \
  RUN_DSV41=0 \
  bash examples/training_demo/codegen_validation/run_codegen_validation.sh
```

## Outputs

- `$VAL_ROOT/logs/<model>_codegen_train.log` / `<model>_native_train.log` —
  full training logs (both codegen criteria and loss lines live here).
- `$TREE_ROOT/hyper_parallel/models/<model_type>/generated/` — the codegen
  artifact bundle (model-anchored, one per model type; `CLEAN_GENERATED=1`
  removes it before the run to force a fresh generation — this touches
  tracked files in a checkout, so run the flow in a disposable worktree).
- Terminal summary: one `PASS`/`FAIL` line per check and a final
  `RESULT: PASSED|FAILED`; the exit code mirrors the result.

## Interpretation

- The codegen arm must show all three criteria lines in order; a
  `reusing` line instead of `generating` means the preflight failed to
  force a fresh generation (check `CLEAN_GENERATED`).
- The native arm must contain **no** `codegen:` lines — it proves the two
  arms are genuinely independent paths.
- Part 3 failures with part 1/2 passing usually indicate a numerical
  divergence between the generated modeling file and the source modeling
  path; compare `generated/modeling_*_gen_npu.py.diff` against the source
  module to narrow it down.

## Reference run

Recorded on 2026-10-22/23, branch `shawny999/codegen_model_bundles` @
`ad60aa2f`, 8x Ascend 910B3 (both models: all checks PASS, loss abs diff 0):

- qwen3_moe (tp2/cp2/ep2/fsdp2+edp2): 10-step `training/total_loss`
  12.3141 -> 12.2515, identical across arms.
- deepseek_v41 (ep8/fsdp8): 10-step `training/total_loss` 11.9626 ->
  11.6481, identical across arms.

Reference runs for kimi_k26 and qwen3_vl_moe are recorded in
`tests/torch/codegen/kimi_k26/TEST_REPORT.md` and
`tests/torch/codegen/qwen3_vl_moe/TEST_REPORT.md` respectively.
