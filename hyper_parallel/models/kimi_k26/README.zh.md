# Kimi-K2.6 模型集成

[English](README.md) | **简体中文**

Kimi-K2.6 沿用 Transformers 的 `kimi_k25` 配置和源实现，模型目录身份为 `kimi_k26`。
使用原生 Transformers 配置与 processor，设置 `trust_remote_code: false`。官方发布资产中遗留的
remote-code `auto_map` 不代表需要执行远端建模代码。

## 目录与职责

```text
models/kimi_k26/
├── __init__.py                         # 轻量模型包入口
├── kimi_k26_spec.py                    # 模型身份、公共族替换、FSDP 边界、有限旧 YAML 目标
├── modeling_kimi_k26.py                # 模型原生组合；decoder 类拥有输入 CP 切窗与 chunked loss
├── dataset_adapter.py                 # 保留完整媒体输入，分发全局移位后的局部 CP 目标
├── dataset_preprocessing/
│   ├── __init__.py                     # 数据预处理包入口
│   └── transform_fn.py                 # processor 输出到 Omni eager/packing 样本的转换
├── validation/
│   ├── __init__.py                     # 验证辅助包入口
│   └── cropped_model.py                # 显式验证裁剪与冻结入口，不承载公共算法
├── generated/                         # 可选 codegen 产物
└── recipes/
    ├── train.yaml                     # 既有发布模型训练配置
    ├── run_kimi_k26_4sn.sh             # 既有多节点启动脚本
    └── prepare_kimi_vlm_data.py        # 演示图文数据构造工具
```

## 模块族与模型组合

- spec 直接选择公共 `MultiLatentAttention`、`GroupedExperts`、`SwiGLUMLP`、`RMSNorm`、
  `RotaryEmbedding`、`LearnedAxialVisionTransformer`、`SpatialPatchProjection` 和 `CausalLMHead`。
  模块拥有 source 参数接纳、局部权重转换、数学和并行实现；此模型无需重复实现私有模块族。
- 顶层模型拥有视觉输出与语言输入的架构融合、输出协议和末端 loss 组合。
  不在模型实例或视觉实例上安装 `forward`，不通过 adapter 转发公共族。
- decoder 类在完成视觉融合后对 `inputs_embeds` 切 CP 窗口，保留显式全局 `position_ids`
  的偏移与重置，并构造相应 mask。完整 Omni CP 链路仍需单独验收，不能用 CP1 数值结果替代。
- 路由和共享专家沿用源语义；Experts 拥有专家 GEMM 与 EP 通信。
  旧 benchmark EP 名称仅通过 root spec 的有限别名解析到公共 recipe。
- 不合并 Muon 的独立矩阵边界。验证从随机初始化开始，正式精度与性能用例位于仓外验证目录，
  原始 recipes 保持原样。详细结果见 [验证报告](../../../docs/model-refactor-validation-report.md)。

## recipe 使用

`recipes/train.yaml` 是**主线规模：4 超节点 / 512 卡**的配方，形状
`cp 1 × dp_shard 512（= edp 4 × ep 128）`、`GBS 512 = 512 × MBS 1`、序列长度 8192；
此前在该拓扑下使用均衡路由测得最小步时 **18.2617 s**、padded 吞吐 **228,873 tok/s**，
以及峰值 allocated **39.19 GB**（关闭 activation swap）。

```bash
# 每个节点执行一次，NODE_RANK 0..NNODES-1
NNODES=64 NODE_RANK=0 MASTER_ADDR=<rank0-ip> MASTER_PORT=6100 \
    bash hyper_parallel/models/kimi_k26/recipes/run_kimi_k26_4sn.sh /path/to/Kimi-K2.6
```

前提与说明：

- 需要 Kimi-K2.6 多模态 checkpoint（config + processor + 权重同目录）。本地/离线场景由启动脚本
  通过 dotted override 把 `model.*`、`dataset.model_assets.*` 指到该目录并置
  `local_files_only=true`；配方默认值则指向公开的 `moonshotai/Kimi-K2.6`。
- `trust_remote_code` 在 model 与 `dataset.model_assets` 两处都保持 `false`，走 transformers
  原生 `kimi_k25` 实现；VLM trainer 会把 `model.trust_remote_code` 转发给 `AutoProcessor`。
- 演示数据由 `recipes/prepare_kimi_vlm_data.py` 生成（默认 8 条，脚本内用 1024 条）；
  真实数据（COCO 等）用 `--dataset.data_path` 覆盖。
- 其它规模（1SN/2SN/3SN）需要同步调整 `accelerator.ep_size` 与 `fsdp_config` 的
  `dp_shard_size` / `edp_shard_size`，启动脚本对非 512 卡的 world size 默认拒绝启动。
