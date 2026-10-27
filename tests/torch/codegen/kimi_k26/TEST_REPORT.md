# Kimi-K2.6 Codegen 双臂精度对比测试报告

- **测试日期:** 2026-10-24
- **结论:** **PASSED**（11/11 检查通过，双臂 10 步 `training/total_loss` 逐步 `abs_diff = 0.0`，容差 5e-4）
- **分支:** `kimi_k26_codegen`（源分支 `codegen_model_bundles`）

---

## 1. 测试版本

| Item | Value |
| ---- | ----- |
| HyperParallel | branch `kimi_k26_codegen`（本分支；源分支 `codegen_model_bundles`） |
| 验证入口 | `examples/training_demo/codegen_validation/run_codegen_validation.sh`（冻结脚本） |
| pytest 用例 | `tests/torch/codegen/kimi_k26/test_kimi_k26_codegen_dual_arm.py` |
| 修复记录 | `hyper_parallel/models/kimi_k26/generated/KIMI_K26_CODEGEN_FIXES.md`（fix #1–#9） |

冻结脚本在 codegen 臂运行前会**强制删除已存在的 generated bundle**（`CLEAN_GENERATED=1`），保证“从零生成”可复现。

## 2. 测试环境

| Item | Value |
| ---- | ----- |
| 主机 | 8.92.7.129（单节点，8 卡） |
| NPU | 8x Ascend 910B3（npu-smi 26.0.rc1） |
| OS | openEuler 22.03 LTS-SP1（kernel 5.10.0-136.108.0.188.oe2203sp1.aarch64） |
| CPU / 内存 | Kunpeng-920 aarch64 / ~2 TB |
| Python 环境 | `/home/lxy/miniforge3/envs/hyper_restructure_py311` |

## 3. 测试依赖

| Package | Version |
| ------- | ------- |
| Python | 3.11.16 |
| torch | 2.12.1+cu130 |
| torch_npu | 2.12.0 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |
| numpy | 1.26.4 |
| safetensors | 0.8.0 |
| pillow | 12.3.0 |
| datasets | 5.0.1 |
| transformer_engine | 0.4.0 |

运行前要求 `pip install -e .`（editable install 指向本 checkout），脚本 preflight 会校验 `import hyper_parallel` 的路径必须来自 `TREE_ROOT`。

## 4. 测试用例

pytest 用例：`test_kimi_k26_codegen_dual_arm_precision`（`@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")`），驱动冻结脚本 `run_codegen_validation.sh`（`RUN_KIMI=1`）。

### 4.1 验证场景（kimi_k26，8 NPU）

| Item | Value |
| ---- | ----- |
| 并行几何 | `tp=1, cp=1, ep=2, pp=1, dp_shard=4, edp_shard=1` |
| 裁剪模型 | `num_hidden_layers=2`，`n_routed_experts=8`，冻结 vision tower（`freeze_patterns: ["model.vision_tower*"]`） |
| 训练入口 | `torchrun --standalone --nproc_per_node=8 --module scripts.train_vl <yaml>` |
| 确定性 | `seed: 42`，`enable_full_determinism: true` |
| 数据 | 离线合成 128 样本（`prepare_kimi_vlm_data`，112x112 图像） |
| 步数/精度 | `train_iters: 10`，bfloat16，`loss_aggregation: token_weighted` |

### 4.2 冻结脚本的 11 项检查

| # | 检查项 | 判定 |
| - | ------ | ---- |
| 1 | preflight/env（PYTHON/TORCHRUN/KIMI_CONFIG 存在 + `hyper_parallel` 从 TREE_ROOT 导入） | PASS |
| 2 | data/assets（合成数据 + 裁剪 model_config 生成） | PASS |
| 3 | yaml/render+diff-sanity（两臂仅差 header 注释与 `codegen: true`） | PASS |
| 4 | kimi_k26/codegen-arm-run（torchrun rc=0） | PASS |
| 5 | kimi_k26/part1-fresh-generate（三条 codegen 判据日志齐全） | PASS |
| 6 | kimi_k26/part1-artifact-bundle（`modeling_*_gen_npu.py` + `codegen_meta.json`） | PASS |
| 7 | kimi_k26/part2-10-step-training（loss 序列完整 10 步） | PASS |
| 8 | kimi_k26/native-arm-run（rc=0 且**无** codegen 日志行） | PASS |
| 9 | kimi_k26/native-10-step-training | PASS |
| 10 | kimi_k26/part3-loss-alignment（逐步 `abs_diff <= 5e-4`） | PASS |
| 11 | 汇总 `RESULT: PASSED`（无 FAIL） | PASS |

### 4.3 codegen 执行判据（codegen 臂日志实测）

```text
codegen: generating artifact .../hyper_parallel/models/kimi_k26/generated (signature 5883174ce5b4)
codegen: rendered bundle .../generated (modeling modeling_Kimi_K25_gen_npu.py, 37 plan boundaries)
codegen: imported generated modeling file .../modeling_Kimi_K25_gen_npu.py
codegen: sharding applied by the generated module; source_shard_info keys=72
```

### 4.4 双臂 loss 对齐结果（实测）

```text
step | native      | codegen     | abs_diff
   1 | 6.08196    | 6.08196    | 0.0
   2 | 6.08159    | 6.08159    | 0.0
   3 | 6.06782    | 6.06782    | 0.0
   4 | 6.06213    | 6.06213    | 0.0
   5 | 6.03831    | 6.03831    | 0.0
   6 | 6.04553    | 6.04553    | 0.0
   7 | 6.03696    | 6.03696    | 0.0
   8 | 6.03675    | 6.03675    | 0.0
   9 | 6.02938    | 6.02938    | 0.0
  10 | 6.02681    | 6.02681    | 0.0
```

> **假阳性更正说明:** 早期（fix #1–#4 后）报告的 `max_diff = 0.0` 为假阳性——fix #5 未落地时 `codegen: true` 被 `Target.build()` 静默丢弃，两臂实际均为 native。本报告的通过以三条 codegen 判据日志 + 逐行对齐为证。

## 5. 测试步骤（完整复现）

### 5.1 前置条件

1. 8x Ascend 910B3 单节点，环境按 §3 安装，`pip install -e .` 指向本 checkout。
2. Kimi-K2.6 checkpoint 的 config 目录（含 `config.json` 等）位于主机某路径，例如 `/home/lxy/probe_ckpts/kimi_k26`（脚本变量 `KIMI_CONFIG`）。**无需权重文件**——本验证走 `from_config` 随机初始化。

### 5.2 运行冻结脚本（推荐，与实测一致）

```bash
cd /path/to/hyper-parallel   # TREE_ROOT

export PYTHON=/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python
export TORCHRUN=/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/torchrun
export KIMI_CONFIG=/home/lxy/probe_ckpts/kimi_k26
export VAL_ROOT=/home/lxy/codegen_model_bundles/val_e2e   # 工作目录（日志/数据/YAML 输出）
export RUN_KIMI=1 RUN_QWEN=0 RUN_DSV41=0
export HF_HUB_OFFLINE=1

bash examples/training_demo/codegen_validation/run_codegen_validation.sh
```

预期终端输出（节选）：

```text
==================== VALIDATION REPORT ====================
  PASS  preflight/env
  PASS  data/assets
  ...
  PASS  kimi_k26/part3-loss-alignment (tol=0.0005)
===========================================================
RESULT: PASSED
```

### 5.3 pytest 入口（同一流程的自动化包装）

```bash
cd /path/to/hyper-parallel
KIMI_CONFIG=/home/lxy/probe_ckpts/kimi_k26 \
  pytest -vs tests/torch/codegen/kimi_k26/test_kimi_k26_codegen_dual_arm.py
```

未设置 `KIMI_CONFIG` 或平台不符时用例会打印 `SKIP: ...` 而非失败。

### 5.4 产物位置

| 产物 | 路径 |
| ---- | ---- |
| 汇总报告/日志 | `$VAL_ROOT/run.log`（本次实测 `rc=0`） |
| codegen 臂训练日志 | `$VAL_ROOT/logs/kimi_k26_codegen_train.log` |
| native 臂训练日志 | `$VAL_ROOT/logs/kimi_k26_native_train.log` |
| 渲染后的双臂 YAML | `$VAL_ROOT/kimi_k26/train_codegen.yaml` / `train_native.yaml` |
| 合成数据 | `$VAL_ROOT/data/kimi_k26/data.json` + `model_config/` |
| 代码生成 bundle | `hyper_parallel/models/kimi_k26/generated/`（modeling + meta + `__init__.py`） |

### 5.5 失败时的排查路径

1. `RESULT: FAILED` → 汇总区 FAIL 项即首个失败点；`tail -30` 该臂日志可看 traceback。
2. `preflight failed: $gen_dir is not empty` → 确认 `CLEAN_GENERATED=1`（默认）。
3. codegen 臂日志无 `codegen:` 判据行 → 见 FIXES.md fix #5（Target.build 签名过滤）。
4. dtype mismatch（conv2d bf16 vs float）→ 见 FIXES.md fix #9。
5. 完整 bug → 修复链见 `hyper_parallel/models/kimi_k26/generated/KIMI_K26_CODEGEN_FIXES.md`。

## 6. 修改文件清单（相对源分支）

| 类型 | 文件 | 说明 |
| ---- | ---- | ---- |
| core 修复 | `hyper_parallel/codegen/loader.py` | fix #6 架构名规范化；fix #9 dtype 转发 |
| core 修复 | `hyper_parallel/codegen/check/preflight.py` | fix #7 tied-weights 豁免 |
| core 修复 | `hyper_parallel/core/fully_shard/hsdp_param.py` | fix #3 `@torch.no_grad` |
| core 修复 | `hyper_parallel/distributed/_builder/fsdp_adapter.py` | fix #4 冻结单元不参与 prefetch |
| kimi 修复 | `hyper_parallel/models/kimi_k26/adapter/validation/cropped_model.py` | fix #2 MoE group 裁剪；fix #5 codegen 透传；fix #8 init patch 工厂 + loader 包装 |
| 新增 assets | `examples/training_demo/codegen_validation/kimi_k26_assets.py` | 离线 tokenizer/config 资产（fix #1） |
| 新增模板 | `examples/training_demo/codegen_validation/kimi_k26_native.yaml` | 双臂 YAML 模板（native 臂） |
| 修改脚本 | `examples/training_demo/codegen_validation/run_codegen_validation.sh` | 增加 kimi_k26 臂（RUN_KIMI） |
| 新增文档 | `hyper_parallel/models/kimi_k26/generated/KIMI_K26_CODEGEN_FIXES.md` | fix #1–#9 全记录 |
| 新增用例 | `tests/torch/codegen/kimi_k26/`（含 `__init__.py`） | pytest ST 用例 |
| 新增报告 | `tests/torch/codegen/kimi_k26/TEST_REPORT.md` | 本报告 |
| bundle 更新 | `hyper_parallel/models/*/generated/`（12 个模型） | codegen 再生成产物（含 kimi_k26） |
