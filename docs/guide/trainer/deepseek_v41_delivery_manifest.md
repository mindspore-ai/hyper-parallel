# DeepSeek-V4.1-Flash × HyperParallel 交付包清单

## 交付范围

该交付包用于在 HyperParallel 基线 `8f933e7c2578934306bfba0e5a1d6672d505c3ae` 上查看或覆盖本次
DeepSeek-V4.1-Flash 接入，包含：

- DeepSeek-V4.1 model adapter、原生视觉模块和配置转换；
- mHC、Engram、共享压缩 DSA/CSA2 高性能模块；
- TP、CP、EP、FSDP 布局及多模态 FSDP 扩展；
- Online VLM transform、dataset、collator 和 get-batch；
- 4K/16 卡训练 recipe、启动脚本、handoff 和指标汇总脚本；
- 技术分析报告、100-step 宣传型报告、loss 曲线和完整指标 CSV；
- 100-step 完整日志、success marker、Engram 裁剪资产及数据清单；
- DeepSeek-V4.1-Flash 上游代码、配置和 tokenizer 快照，不含权重 shard。

上游快照来源为 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`，revision
`dba1be0a40aa45a94ad051997016db3960a90277`。

`repo_overlay/` 保留仓库相对路径，可复制到同版本 HyperParallel checkout。`evidence/` 是运行证据，
`upstream/DeepSeek-V4.1-Flash/` 是上游代码与轻量模型资产快照。

## 验证结果

- 拓扑：TP1 + CP1 + EP16 + FSDP16，16 × Ascend 910；
- 序列：4096，global batch 16；
- 100/100 step 完成，loss 和 grad norm 全部有限；
- 前 10 / 后 10 step 平均 loss：11.193749 / 4.690136；
- 稳态平均 step 时间：4.9602s；
- 单卡峰值 allocated / reserved：38.8125 / 48.502 GiB；
- DeepSeek-V4.1 聚焦 UT：21 passed。

## 未打包内容

- 48 个正式 `model-*.safetensors` 权重 shard：本地文件是 Hugging Face LFS pointer，并非实际权重；
- `mm_data/raw/*.parquet` 和导出的 640 张训练/验证图片：本地约 409 MiB，且 DocVQA 数据卡未声明
  可再分发许可证；
- Git 元数据、Python cache、pytest cache、NPU 编译 cache 和无关工作树文件。

数据来源、revision、license 和图片 SHA256 仍记录在 `evidence/data_manifest.json`；JSONL 样例保留在
`evidence/data/`，其中图片路径相对于原数据目录。

## 使用方式

查看报告：

```bash
less repo_overlay/docs/guide/trainer/deepseek_v41_flash_hyperparallel_100step_report.md
```

覆盖到相同基线的 HyperParallel checkout：

```bash
cp -a repo_overlay/. /path/to/hyper-parallel/
cd /path/to/hyper-parallel
pip install -e .
```

按 `repo_overlay/docs/guide/trainer/current_hf_model_environment.md` 准备环境，将图片数据放回 JSONL
引用的相对路径后，可用以下命令复现：

```bash
export RUN_NAME=vlm_tp1_ep16_100steps
bash examples/training_demo/run_deepseek_v41_vlm_online.sh \
    /path/to/DeepSeek-V4.1-Flash \
    /path/to/deepseek_v41_messages/train.jsonl \
    --training.train_iters=100
```

包内 `SHA256SUMS` 校验所有文件；压缩包旁的 `.sha256` 校验整个交付包。
