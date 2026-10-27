# Qwen3-VL-MoE Codegen 双臂精度对比测试报告

- **测试日期:** 2026-10-26
- **结论:** **PASSED**（11/11 检查通过，双臂 10 步 `training/total_loss` 逐步 `abs_diff = 0.0`，容差 5e-4）
- **分支:** `models_codegen`（源分支 `kimi_k26_codegen`）

---

## 1. 测试版本

| Item | Value |
| ---- | ----- |
| HyperParallel | branch `models_codegen`（本分支；源分支 `kimi_k26_codegen`，fix 提交 `b751ffbc` + `856ae7c1`） |
| 验证入口 | `examples/training_demo/codegen_validation/run_codegen_validation.sh`（冻结脚本，`RUN_QVL=1`） |
| pytest 用例 | `tests/torch/codegen/qwen3_vl_moe/test_qwen3_vl_moe_codegen_dual_arm.py` |
| 模型加载 | `HyperAutoModelForImageTextToText.from_config` + 嵌套 `transformers.AutoConfig.from_pretrained` |

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

pytest 用例：`test_qwen3_vl_moe_codegen_dual_arm_precision`（`@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")`），驱动冻结脚本 `run_codegen_validation.sh`（`RUN_QVL=1`）。

### 4.1 验证场景（qwen3_vl_moe，8 NPU）

| Item | Value |
| ---- | ----- |
| 并行几何 | `tp=2, cp=1, ep=2, pp=1, dp_shard=2, edp_shard=2` |
| 模型 | Qwen3-VL-MoE 裁剪配置（`/home/lxy/probe_ckpts/qwen3_vl_moe`，config-only，`from_config` 随机初始化） |
| 训练入口 | `torchrun --standalone --nproc_per_node=8 --module examples.training_demo.train_text <yaml>`（纯文本数据，视觉塔不参与 forward） |
| 确定性 | `seed: 42`，`enable_full_determinism: true` |
| 数据 | `build_indexed_text_dataset`，离线 indexed 128 样本 × seq 128（复用 `qwen3_moe/prepare_parallel_data` 资产） |
| tokenizer | Qwen2.5（151665 tokens，与模型 vocab 151936 同族），`tokenizer_type: pretokenized` |
| 步数/精度 | `train_iters: 10`，bfloat16，`loss_aggregation: token_weighted` |
| EP 计算 | `qwen3_vl_moe_ep_compute_fn`（`use_grouped_gemm: true`），视觉塔全量 `tp: replicate` |

### 4.2 冻结脚本的 11 项检查

| # | 检查项 | 判定 |
| - | ------ | ---- |
| 1 | preflight/env（PYTHON/TORCHRUN/QVL_CONFIG 存在 + `hyper_parallel` 从 TREE_ROOT 导入） | PASS |
| 2 | data/assets（indexed 数据 + tokenizer 资产就绪） | PASS |
| 3 | yaml/render+diff-sanity（两臂仅差 header 注释与 `codegen: true`） | PASS |
| 4 | qwen3_vl_moe/codegen-arm-run（torchrun rc=0） | PASS |
| 5 | qwen3_vl_moe/part1-fresh-generate（三条 codegen 判据日志齐全） | PASS |
| 6 | qwen3_vl_moe/part1-artifact-bundle（`modeling_*_gen_npu.py` + `codegen_meta.json`） | PASS |
| 7 | qwen3_vl_moe/part2-10-step-training（loss 序列完整 10 步） | PASS |
| 8 | qwen3_vl_moe/native-arm-run（rc=0 且**无** codegen 日志行） | PASS |
| 9 | qwen3_vl_moe/native-10-step-training | PASS |
| 10 | qwen3_vl_moe/part3-loss-alignment（逐步 `abs_diff <= 5e-4`） | PASS |
| 11 | 汇总 `RESULT: PASSED`（无 FAIL） | PASS |

### 4.3 codegen 执行判据（codegen 臂日志实测）

```text
codegen: generating artifact .../hyper_parallel/models/qwen3_vl_moe/generated (signature 8c3aa06c784a)
codegen: imported generated modeling file .../modeling_Qwen3VLMoeForConditionalGeneration_gen_npu.py
codegen: sharding applied by the generated module; source_shard_info keys=122
```

### 4.4 双臂 loss 对齐结果（实测）

```text
step | native      | codegen     | abs_diff
   1 | 12.3559    | 12.3559    | 0.0
   2 | 12.3754    | 12.3754    | 0.0
   3 | 12.351     | 12.351     | 0.0
   4 | 12.3057    | 12.3057    | 0.0
   5 | 12.3694    | 12.3694    | 0.0
   6 | 12.3236    | 12.3236    | 0.0
   7 | 12.339     | 12.339     | 0.0
   8 | 12.3588    | 12.3588    | 0.0
   9 | 12.3558    | 12.3558    | 0.0
  10 | 12.3445    | 12.3445    | 0.0
```

> **注:** 本轮（2026-10-26 集成重跑）与此前修复验证轮（12:07 轮）的 loss 序列完全一致，确认冻结脚本集成后可复现同一结果。产物 bundle 的 `signature 8c3aa06c784a` 为 129 环境实测值。

## 5. 测试步骤（完整复现）

### 5.1 前置条件

1. 8x Ascend 910B3 单节点，环境按 §3 安装，`pip install -e .` 指向本 checkout。
2. Qwen3-VL-MoE 裁剪 config 目录（仅含 `config.json`）位于主机某路径，例如 `/home/lxy/probe_ckpts/qwen3_vl_moe`（脚本变量 `QVL_CONFIG`）。**无需权重文件**——本验证走 `from_config` 随机初始化。
3. Qwen2.5 tokenizer 资产：联网主机由脚本自动下载（`QVL_HF_ENDPOINT` 默认 `https://hf-mirror.com`）；离线主机用 `QVL_TOKENIZER_SOURCE` 指向已就绪的 tokenizer 目录。

### 5.2 运行冻结脚本（推荐，与实测一致）

```bash
cd /path/to/hyper-parallel   # TREE_ROOT

export PYTHON=/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python
export TORCHRUN=/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/torchrun
export QVL_CONFIG=/home/lxy/probe_ckpts/qwen3_vl_moe
export VAL_ROOT=/home/lxy/qwen3vl_val/val_int   # 工作目录（日志/数据/YAML 输出）
export RUN_QVL=1 RUN_QWEN=0 RUN_DSV41=0 RUN_KIMI=0

bash examples/training_demo/codegen_validation/run_codegen_validation.sh
```

预期终端输出（节选）：

```text
==================== VALIDATION REPORT ====================
  PASS  preflight/env
  PASS  data/assets
  ...
  PASS  qwen3_vl_moe/part3-loss-alignment (tol=0.0005)
===========================================================
RESULT: PASSED
```

### 5.3 pytest 入口（同一流程的自动化包装）

```bash
cd /path/to/hyper-parallel
QVL_CONFIG=/home/lxy/probe_ckpts/qwen3_vl_moe \
  pytest -vs tests/torch/codegen/qwen3_vl_moe/test_qwen3_vl_moe_codegen_dual_arm.py
```

未设置 `QVL_CONFIG` 或平台不符时用例会打印 `SKIP: ...` 而非失败。

### 5.4 产物位置

| 产物 | 路径 |
| ---- | ---- |
| 汇总报告/日志 | `$VAL_ROOT/run.log`（本次实测 `rc=0`） |
| codegen 臂训练日志 | `$VAL_ROOT/logs/qwen3_vl_moe_codegen_train.log` |
| native 臂训练日志 | `$VAL_ROOT/logs/qwen3_vl_moe_native_train.log` |
| 渲染后的双臂 YAML | `$VAL_ROOT/qwen3_vl_moe/train_codegen.yaml` / `train_native.yaml` |
| 合成数据 | `$VAL_ROOT/data/parallel_offline_text_document.bin/.idx` |
| tokenizer 资产 | `$VAL_ROOT/assets/qwen3_tok/` |
| 代码生成 bundle | `hyper_parallel/models/qwen3_vl_moe/generated/`（modeling + meta + `__init__.py`） |

### 5.5 失败时的排查路径

1. `RESULT: FAILED` → 汇总区 FAIL 项即首个失败点；`tail -30` 该臂日志可看 traceback。
2. `preflight failed: $gen_dir is not empty` → 确认 `CLEAN_GENERATED=1`（默认）。
3. tokenizer 相关 → `qwen3_vl_moe_assets.py` 的 `_verify_vocab` 报错说明 bundle 不是 Qwen2.5 家族（期望 len=151665）；离线主机检查 `QVL_TOKENIZER_SOURCE` 指向。
4. codegen 臂无判据行 → 见 §7 修改清单（freeze.py 负 dim verbatim / loader.py 架构回退）。
5. 双臂 loss 发散 → 对比 `generated/modeling_*_gen_npu.py.diff` 与源模块。

## 6. 修复记录（相对上游 `codegen_model_bundles`）

| 类型 | 文件 | 说明 |
| ---- | ---- | ---- |
| codegen 修复 | `hyper_parallel/codegen/plan/freeze.py` | 负 shard dim verbatim round-trip + uneven placement fail-fast |
| codegen 修复 | `hyper_parallel/codegen/loader.py` | `architectures` 回退链（无 architectures 字段时逐级回退） |
| codegen 修复 | `.codespellrc` | SpecField 误报忽略 |
| codegen 回归测试 | `tests/ut/codegen/test_core_contracts.py` | 负 dim freeze 回归 ×2 |
| codegen 回归测试 | `tests/ut/codegen/test_spec_field_table.py` | 回归 |
| 框架修复 | `hyper_parallel/core/fully_shard/hsdp_param.py` | 冻结单元不参与 prefetch（HSDP） |
| 框架修复 | `hyper_parallel/core/fully_shard/hsdp_scheduler.py` | 同上 |
| 框架修复 | `hyper_parallel/core/fully_shard/param_group.py` | 同上 |
| 新增 assets | `examples/training_demo/codegen_validation/qwen3_vl_moe_assets.py` | tokenizer 资产（--source 离线复制 / hub 下载） |
| 新增模板 | `examples/training_demo/codegen_validation/qwen3_vl_moe_native.yaml` | 双臂 YAML 模板（native 臂） |
| 修改脚本 | `examples/training_demo/codegen_validation/run_codegen_validation.sh` | 增加 qwen3_vl_moe 臂（RUN_QVL） |
| 修改文档 | `examples/training_demo/codegen_validation/README.md` | 模型矩阵更新（含 kimi_k26） |
| 新增用例 | `tests/torch/codegen/qwen3_vl_moe/`（含 `__init__.py`） | pytest ST 用例 |
| 新增报告 | `tests/torch/codegen/qwen3_vl_moe/TEST_REPORT.md` | 本报告 |
| bundle 更新 | `hyper_parallel/models/qwen3_vl_moe/generated/` | codegen 再生成产物（129 冻结脚本重跑产出） |
