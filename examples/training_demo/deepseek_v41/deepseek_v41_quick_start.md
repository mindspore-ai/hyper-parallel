# DeepSeek V4.1 快速启动

在 Atlas 900 A3/A5 SuperPoD 单机 16 个逻辑 NPU die 上，使用 HyperParallel 运行
DeepSeek-V4.1 图文或纯文本预训练链路。

本文以 [VLM 配置](train_deepseek_v41_vlm_online.yaml)为主，
[纯文本配置](train_deepseek_v41_online.yaml)使用相同的文本模型、并行策略和优化器。
两者均保留 40 层文本网络，参数维度裁剪系数为 4，使用 16 个路由专家和随机初始化参数；
VLM 额外保留 32 层视觉网络。

## 1. 准备环境

以下为参考启动环境；框架和 CANN 的配套关系见
[TorchNPU 官方配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。

| 组件 | 版本 |
| --- | --- |
| HDK | 25.5.5 |
| CANN | 9.1.0 |
| Python | 3.11.14 |
| PyTorch / TorchNPU | 2.12.0 CPU wheel / 2.12.0 |
| Transformers | >=5.11.0 |

### 1.1 Python 环境

已有配套环境时直接激活；新建环境可执行：

```bash
conda create -n hyper_v41 python=3.11.14 -y
conda activate hyper_v41
git clone --branch master https://github.com/mindspore-ai/hyper-parallel.git
cd hyper-parallel
export HP_REPO_ROOT="$PWD"
export CANN_ROOT=/usr/local/Ascend/cann-9.1.0
source "${CANN_ROOT}/set_env.sh"

python -m pip install 'torch==2.12.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'torch-npu==2.12.0'
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

### 1.2 编译 Multicore

在 HyperParallel 根目录执行：

```bash
cd "${HP_REPO_ROOT}"
./build.sh --multicore on --soc-list ascend910_93 --jobs 128
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

`--jobs` 按主机 CPU 资源调整。检查日志确认 Multicore 成功；默认宽松模式下，生成 wheel
不代表每个可选组件都编译成功。参数和产物说明见[安装指南](../../../docs/installation.md)。

### 1.3 安装 Omni Ops

Omni Ops 的设备侧算子包和 Python/PTA 扩展都需要安装。详见
[自定义算子安装说明](../../../hyper_parallel/components/modules/README.md#install-custom-operators)。

```bash
export OMNI_OPS_ROOT="$(dirname "${HP_REPO_ROOT}")/omni-ops"
git clone https://gitee.com/omniai/omni-ops.git "${OMNI_OPS_ROOT}"
source "${CANN_ROOT}/set_env.sh"
cd "${OMNI_OPS_ROOT}/training/ascendc"
bash build.sh -c ascend910_93

cd output
omni_operator_package="CANN-omni_training_custom_ops--linux.$(uname -m).run"
chmod +x "${omni_operator_package}"
"./${omni_operator_package}" --quiet --install-path="${ASCEND_HOME_PATH}/opp"
source "${ASCEND_HOME_PATH}/opp/vendors/omni_training_custom_transformer/bin/set_env.bash"

# Build the PTA extension with the active Conda environment.
cd "${OMNI_OPS_ROOT}/training/ascendc/torch_ops_extension"
which python3
which pip3
bash build_and_install.sh
```

也可从 [Omni Ops v1.5.0](https://gitee.com/omniai/omni-ops/releases/tag/v1.5.0)
获取与 SoC、CANN、Python 和框架版本匹配的预编译包。

### 1.4 安装 ops-transformer

参考记录使用固定提交 `0b08075141913daeaf1e856faf7480ed393c8949`。
设备包和 Python wheel 应从同一源码构建，不能仅凭 `1.0.0` 版本号判断兼容性。
接口要求见 [DeepSeek V4.1 融合算子说明](../../../hyper_parallel/components/functional/README.md#deepseek-v41-fused-operators)。

```bash
export OPS_TRANSFORMER_ROOT="$(dirname "${HP_REPO_ROOT}")/ops-transformer"
git clone https://gitcode.com/cann/ops-transformer.git "${OPS_TRANSFORMER_ROOT}"
cd "${OPS_TRANSFORMER_ROOT}"
git checkout 0b08075141913daeaf1e856faf7480ed393c8949
source "${CANN_ROOT}/set_env.sh"

V41_OPS="lightning_indexer_v2,lightning_indexer_v2_metadata"
V41_OPS+=",sparse_flash_mla,sparse_flash_mla_metadata"
V41_OPS+=",sparse_flash_mla_grad,sparse_flash_mla_grad_metadata"
V41_OPS+=",sparse_lightning_indexer_kl_loss_grad,sparse_lightning_indexer_kl_loss_grad_metadata"
bash build.sh --pkg --soc=ascend910_93 --ops="${V41_OPS}" -j128

# Keep the native kernels in a separate vendor directory.
bash "./build_out/cann-ops-transformer-custom_linux-$(uname -m).run" \
  --install-path="${OPS_TRANSFORMER_ROOT}/install"
source "${OPS_TRANSFORMER_ROOT}/install/vendors/custom_transformer/bin/set_env.bash"

# Keep the standard Python package name: cann_ops_transformer.
bash build.sh --torch_extension
python -m pip install --no-deps ./build_out/cann_ops_transformer-1.0.0-py3-none-any.whl
```

重新打开终端后，激活 Conda，依次加载 CANN、Omni、Multicore、ops-transformer 环境，最后设置
Python 包的优先路径。
在其他环境启动前，将该行改为实际的算子环境脚本，确保其 vendor 和 Python 包与本节一致。

## 2. 准备模型和图文数据

### 2.1 模型配置与分词器

随机初始化配方只读取[官方模型仓库](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
中的配置、分词器和视觉处理配置，无需下载权重分片：

```bash
cd "${HP_REPO_ROOT}"
export MODEL_DIR="$PWD/model_from_hf/DeepSeek-V4.1-Flash"
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir "${MODEL_DIR}" \
  --include config.json 'tokenizer*' special_tokens_map.json chat_template.jinja inference/config.json
```

已有模型目录时直接设置 `MODEL_DIR`。VLM 至少需要 `config.json`、`tokenizer.json`、
`tokenizer_config.json` 和 `inference/config.json`。启动脚本按需生成裁剪后的 Engram 资产。

### 2.2 CC12M 或 fake 数据

使用准备好的 CC12M 图文 JSONL，将 `DATA_PATH` 指向文件。每行格式为：

```json
{"image": "images/000001.jpg", "caption": "A cat sitting on a chair."}
```

相对图片路径以 JSONL 所在目录为基准。当前配置为 `pretrain + image_caption`，训练时在线编码和
packing，无需转换为 `.bin/.idx`。CC12M 原始下载信息见[数据集说明](https://github.com/google-research-datasets/conceptual-12m#download)。

## 3. 启动图文训练

当前 YAML 的主要配置如下；以文件中的实际值为准：

| 项目 | 配置 |
| --- | --- |
| 模型 | 文本 40 层、视觉 32 层，参数维度裁剪系数 4，16 个路由专家 |
| 并行 | TP1 / CP4 / EP8 / PP1，dense DP4，encoder CP4 |
| FSDP | `dp_shard_size=8`，`edp_shard_size=1`；DP+CP 域为 16 个 rank |
| Batch / 序列 | global batch 8，micro batch 1，累计 2 个 microbatch，4096 token |
| 优化 | Host Engram、CSA2 融合注意力、MegaMoe/MegaMhc/MegaGate、全重算、FSDP 预取 |
| 精度 / 步数 | BF16 前向；FP32 初始化、主参数及梯度归约；默认 100 步 |

首次验证可先跑一步，并关闭历史 checkpoint 恢复：

```bash
cd "${HP_REPO_ROOT}"
RUN_NAME=quick_start NPROC_PER_NODE=16 \
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
  "${MODEL_DIR}" "${DATA_PATH}" \
  --training.train_iters=1 --checkpoint.restore_from=null --debug.check_dataset=warn
```

按 YAML 默认步数运行时，去掉 `--training.train_iters=1` 即可。

成功时退出码为 0，日志含目标训练步的 loss、grad norm 和耗时，且数值有限。
默认输出位置为：

```text
output/training_demo/deepseek_v41/run_quick_start.log
output/training_demo/deepseek_v41/quick_start.success
output/training_demo/deepseek_v41/artifacts/engram_depth_preserving_d4.json
```

## 4. 启动纯文本训练

[纯文本配置](train_deepseek_v41_online.yaml)沿用上述文本模型规格、TP1/CP4/EP8、FSDP8、
Host Engram、融合算子和优化器，关闭视觉网络与 encoder parallel。数据每行使用 `text` 字段：

```json
{"text": "A document for language-model pretraining."}
```

只传模型路径时，脚本自动生成本地测试文本；也可在模型路径后传入自己的 JSONL：

```bash
RUN_NAME=text_quick_start \
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh "${MODEL_DIR}" \
  --training.train_iters=1 --checkpoint.restore_from=null

# Use an existing text dataset.
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "${MODEL_DIR}" /path/to/text/train.jsonl
```

文本按 4096 token 切分，丢弃长文档末尾不足窗口的部分，避免短尾破坏 CSA/CP 对齐。
脚本不再接受旧的 `tp1/tp2/cp2` 模式参数，拓扑由 YAML 和末尾的 Trainer 参数指定。
默认日志为 `run_text_tp1_cp4_ep8_fsdp8.log`，checkpoint 目录为 `artifacts/text_checkpoints`，
与 VLM 的 checkpoint 分开。

## 5. 常用配置调整

### 保存、恢复与 SFT

- 保存 checkpoint：追加 `--checkpoint.save_ckpt=true --checkpoint.save_steps=10`。
- 恢复训练：使用相同模型、数据和 checkpoint 目录，设置 `--checkpoint.restore_from=LATEST`。
- 图文 SFT：准备最后一条为 assistant 的 `messages` 数据，追加
  `--dataset.training_mode=sft --dataset.data_transform.sample_format=messages`。
  当前实现只监督最后一条 assistant 回复，格式见[图文数据指南](../../../docs/guide/data/deepseek_v41_vlm_online_data_guide.md)。

### 加载官方预训练权重

如需加载权重训练，请新增如下配置：

```yaml
model:
    load_base_model: true
    # Verify all loaded shards, including CPU Host Engram, before the first forward.
    verify_hf_weights: ./output/training_demo/deepseek_v41/hf_weight_check
```

但应当注意不能直接加在当前“裁剪系数 4、16 专家”的配置上。原权重要求保留层使用官方参数维度、专家数和 Engram 哈希资产。先生成原始尺寸资产：

```bash
python -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_assets \
  --model-dir "${MODEL_DIR}" \
  --output "$HP_REPO_ROOT/output/training_demo/deepseek_v41/artifacts/engram_released.json" --released
```

将以下字段合入用于加载权重的 YAML，并准备完整 HF 权重分片：

```yaml
model:
  load_base_model: true
  verify_hf_weights: ./output/training_demo/deepseek_v41/hf_weight_check
  text_parameter_divisor: 1
  vision_parameter_divisor: 1
  num_routed_experts: 384
  engram_assets_path: ./output/training_demo/deepseek_v41/artifacts/engram_released.json
```

沿用启动脚本时，还需设置 `ASSETS_PATH` 指向 `engram_released.json`，因为 shell 会覆盖 YAML
中的资产路径。原始尺寸模型的资源需求与裁剪配方不同；HF 权重加载也不同于恢复 optimizer、RNG
和数据游标的完整断点续训。

### 关闭 CSA2 融合算子

将已有注意力替换规则中的 `use_fused_ops` 改为 `false`：

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
        use_fused_ops: false
```

此处只展示需修改的规则，其余 `plan_overrides` 保留。当前融合路径要求 TP1，并关闭 candidate pool。

### 关闭 MegaKernel 后使用普通 EP

删除 MegaMoe experts、MegaMhc 和 MegaGate 的模块替换规则，将现有 `model.layers.*.mlp.forward`
规则替换为下面的普通 EP 规则。保留 Host Engram、注意力 CP 和 VLM 的视觉编码器等其余配置：

```yaml
plan_overrides:
  - match: "model.layers.*.mlp.forward"
    when: ep
    replace:
      kind: method
      implementation:
        _target_: >-
          hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_ep_compute_fn
        use_grouped_gemm: true
        fix_router: false  # Keep learned routing for normal training.
```
