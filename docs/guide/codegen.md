# Codegen

Codegen 将原生模型构建期间产生的 replacement、sharding plan 和 plan application
记录为不可变 snapshot，再从 snapshot 生成可审计 bundle。生成产物不参与 Trainer
运行时；训练始终使用原生 HF/custom model 和原生并行化路径。

## 生成 bundle

使用普通训练 YAML，不需要也不接受 `codegen: true`：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/codegen_validation/qwen3_moe_native.yaml
```

指定输出目录：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen generate \
  --config train.yaml \
  --output /path/to/generated
```

`generate` 执行以下有限流程：

1. 使用 Trainer 的 YAML parser 解析配置和 CLI override。
2. 初始化分布式拓扑与确定性 seed。
3. 通过配置中的 model target 构建原生模型。
4. 记录原生 replacement、planner 和 apply 结果。
5. 调用 `CodegenBuildSession.finish()` 生成或复用 bundle。
6. 释放进程组并退出。

它不会创建 dataset、dataloader、optimizer、LR scheduler 或 Trainer，也不会执行
forward、backward 或 optimizer step。

CLI override 继续使用训练配置语法：

```bash
python -m hyper_parallel.codegen generate \
  --config train.yaml \
  --accelerator.tp_size=2 \
  --accelerator.ep_size=4
```

## 检查 bundle

```bash
python -m hyper_parallel.codegen check --config train.yaml
```

检查目录下所有 YAML：

```bash
python -m hyper_parallel.codegen check --all --root examples/training_demo
```

检查 Transformers 版本约束：

```bash
python -m hyper_parallel.codegen check --transformers-pin '==5.15.0'
```

## 产物

典型 bundle 包含：

- `codegen_meta.json`：snapshot、签名、source identity 和输出 hash。
- `modeling_*_gen_npu.py`：生成的可审计 modeling source。
- `modeling_*_gen_npu.py.diff`：相对原始 modeling source 的变更。
- `parallelize_gen.py`：生成侧并行信息。
- `__init__.py`：bundle 包入口。

相同 YAML、源码和原生构建事实应产生稳定 signature。签名相同会复用现有 bundle；
签名漂移会原子更新 bundle。

## 训练隔离

Trainer 不读取、导入或执行 generated bundle。损坏、缺失或过期的 bundle 不应影响
原生训练。训练进程中的以下诊断计数必须为零：

- `generated_model_import_count`
- `generated_parallelize_count`
- `offline_planner_count`

旧接口已删除：

- Trainer YAML 的 `codegen` 字段；
- AutoModel 的 `codegen`、`codegen_config`、`codegen_artifact_dir` 和
  `codegen_session` 参数；
- `ModelingBackend.GEN` / `backend="gen"`。

旧 YAML 会给出迁移错误。请移除 `codegen: true`，正常训练；需要生成产物时单独运行
codegen CLI。

## 失败语义

CLI 只有在 bundle 成功发布后才返回成功。原生模型构建失败、snapshot 不完整、rank
签名不一致或 bundle 写入失败都会返回非零退出码，不会用空 plan 或残缺文件伪造成功。
多 rank 生成使用同一 snapshot signature，并在发布完成后统一释放等待 rank。
