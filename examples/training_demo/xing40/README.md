# Xing40 单卡精度对齐与四卡并行复现

> 运行需要模型、公共依赖和对应修复的完整组合。模型 PR #1367 与示例 PR #1376 单独检出均不等于完整训练版本；先按第 0 节核验集成源码与导入。标准40层 recipe 不能与3层对齐配置混用。

> 本目录五份配方显式设置 `use_fused_mhc: true`，依赖融合专用 [#1409](https://github.com/mindspore-ai/hyper-parallel/pull/1409) 和已适配其接口的 Xing40 模型。公共 mHC pre/post 需要 Ascend NPU 与 Omni 自定义算子，不提供 CPU、GPU 或缺算子时的回退。此融合候选尚待设备训练验证；旧非融合 1000 步结果仅作历史记录，不能作为本候选的融合精度验收。


本目录提供中文逐步流程、Bash 启动器及五份完整 YAML。除显式启用融合 mHC 外，它们保留历史对齐任务的模型、优化器与精度设置，将仓外数据适配移为仓内示例，并迁移到当前 master 的并行配置接口。

这里的单卡续训是 **HP+PTA 读取由 MF 完整训练状态转换出的 HP native checkpoint**。仅加载模型权重、HP 自己保存后恢复，或只设相同 seed，都不能替代 MF→HP 精度对齐。

## 0. 取得完整集成版本

各特性分别评审。此融合候选需要包含 #1409、已适配模型和其他前置的完整源码；当前尚未指定或验证新的集成分支。取得经核验的源码后，记录实际提交和工作树状态，不将旧集成分支冒充融合版本。既有 PR 与历史组合见[总 issue #304](https://github.com/hf1s/hyper-parallel/issues/304)。

~~~bash
cd /path/to/hyper-parallel-xing40
# 保留实际 SHA 和工作树状态，后续不要在运行目录执行 pull。
git rev-parse HEAD
git status --short
~~~

模型接入依赖融合专用公共 mHC 与适配其 pre/post 接口的 Xing40 模型、返回总和的投影 CE、逐参数 FSDP 策略及名称匹配。单卡 Trainer、Muon 精度、普通 Chunk CE、SwiGLU、GMM、SAC 和 replica 契约各自有独立 PR。FSDP 名称扩展以前置 #1171 为基础；只有前置合入后才能把其网页差异称为纯增量。CP2 配置还需要 MLA 与 MoE CP 适配，不把模型单独存在当作该并行配置已可运行。

## 1. 选择配置，确认比较条件

| 配置 | 进程数 | 并行及分片 | GBS/LBS | 每 rank 每优化步 micro 数 | 重计算 |
|---|---:|---|---|---:|---|
| [single.yaml](alignment_configs/single.yaml) | 1 | 单卡显式混精 | 2/1 | 2 | full |
| [fsdp4.yaml](alignment_configs/fsdp4.yaml) | 4 | dense/expert FSDP4 | 4/1 | 1 | full |
| [fsdp4_cp2.yaml](alignment_configs/fsdp4_cp2.yaml) | 4 | DP2×CP2；参数沿 DP×CP 分片，共 FSDP4 | 4/1 | 2 | full |
| [fsdp4_selective.yaml](alignment_configs/fsdp4_selective.yaml) | 4 | FSDP4 | 4/1 | 1 | selective |
| [fsdp4_ep2.yaml](alignment_configs/fsdp4_ep2.yaml) | 4 | dense FSDP4；expert EDP2×EP2 | 4/1 | 1 | full |

每份配置均为 **3 层：2 Dense + 1 MoE**，hidden 3584、64 routed experts、每 token top4、序列 4096。MTP 按来源配置关闭（0 层、权重 0），MTPLOSS 为 N/A。四卡 GBS4 是 LBS1 下可整除四卡纯 FSDP 的最小 GBS，已经单独确认；不能把单卡 GBS2 与四卡 GBS4 的吞吐差直接当作并行加速收益。

固定 LR 1e-5、总日程 2500 步、Muon+AdamW、FP32 main、FP32 NS 输入/归一化、BF16 NS 迭代、epsilon 1e-8 clamp、logical matrix split、QK clip 100、融合 mHC（`use_fused_mhc: true`）、真实 router 及原精度策略均在 YAML 中明确列出。显式 attention scale 为 0.1352337788608801，优先于 PR310 新增的 YaRN 默认值。

除本候选显式启用融合 mHC 外，原配置来源为 [base.yaml](https://gitcode.com/xuxinglei/net-regression-suite/blob/main/deepseek-v3-29b-8card/base.yaml)，来源 commit 为 1dc28669e642f4cce1448456e9ad622c1285fae0、文件 SHA256 为 ccc920386d930a1b55a782575485b9cd0d0b02a3a319d2586dff9a55120d8fd4。减层、batch、步数和跨框架等价映射见[总跟踪记录](https://github.com/hf1s/hyper-parallel/issues/304)。

仓内 [40 层原始 recipe](../../../hyper_parallel/models/xing40/recipes/train.yaml) 是另一份默认接入配方，含 MTP；不能混用两者的 checkpoint 或验收结果。

## 2. 准备环境并记录版本

历史单卡对齐环境为 Linux AArch64、Python 3.12.14、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post2、CANN 9.1.0、Transformers 5.21.0、tokenizers 0.23.3、NumPy 1.26.4；MF 参考源码为正式 master 52f614ee、HP-MS r1.0.0 c69f8f70、MindSpore r2.10 2a020077。融合候选须在实际环境中验证适配模型、Omni 自定义算子及其加载，并新跑启用融合 mHC 的 MF 参考轨迹；旧非融合轨迹不能替代。历史 CANN 9.2.0/post3 四卡结果仍按其原环境记录，不与本候选混称。PyTorch 的 +cpu 是基础 wheel 的构建标识，Ascend 设备由 PTA 注册；训练实际在 NPU 执行。

按目标驱动、芯片和 Python ABI 准备官方兼容依赖，参考 [PTA 官方安装导航](https://www.hiascend.com/document/detail/zh/Pytorch/master/index/index.html)和[兼容性查询](https://www.hiascend.com/developer/download/compatibility)。不能强装其他 Python ABI 的 wheel，也不能把安装时的“最新版本”替换进来后沿用旧结果。

在已安装上述依赖的环境中，从仓库根目录执行：

~~~bash
source /path/to/venv/bin/activate
# CANN_ROOT 指向本机实际安装目录，其中包含 set_env.sh。
source "$CANN_ROOT/set_env.sh"
# 先固定匹配的 Torch/PTA wheel，再安装其余运行依赖，避免意外升级核心组件。
python -m pip install -r requirements.txt \
  -c examples/training_demo/xing40/alignment-constraints.txt
python -m pip install --no-deps -e .
python -m pip check
# 解析通用数据 recipe 会导入本地 C++ index helper；首次编译需 C++ 编译器、make 和 pybind11。
# 在可写检出中先编译，之后才可将该安装用于只读作业。
python -c 'import hyper_parallel.data.indexed.indexed_helpers'
python - <<'PY'
import sys
import importlib.metadata as metadata
import torch
import torch_npu
import hyper_parallel
print(sys.version)
print("torch:", torch.__version__, "torch_npu:", metadata.version("torch-npu"))
print("transformers:", metadata.version("transformers"), "numpy:", metadata.version("numpy"))
print("HP imported from:", hyper_parallel.__file__)
print("NPU count:", torch.npu.device_count())
PY
npu-smi info
~~~

确认导入当前检出，而不是旧 editable 目录。保留 Python/依赖、CANN 安装信息、驱动、机器型号和设备拓扑。单卡需一张空闲卡，四卡需同机四张可通信的空闲卡；启动器要求显式选择设备，不终止其他作业。

## 3. 准备真实数据和三份采样索引

需要同一前缀的 .bin/.idx，以及 MF 实际训练使用的 document_index.npy、sample_index.npy、shuffle_index.npy。INDEX_PREFIX 是三个文件名在 “-document_index.npy”等后缀之前的公共部分。不能重建 shuffle 后仍认为样本相同。

~~~bash
export DATA_PREFIX=/data/mmap_deepseekv3_datasets_text_document
export INDEX_PREFIX=/data/gpt-cache/68b9d952aee6139e19a0994442defdf5-GPTDataset-train
export RUN_ROOT=/data/xing40-runs
sha256sum "$DATA_PREFIX.bin" "$DATA_PREFIX.idx"
sha256sum "$INDEX_PREFIX"-{document_index,sample_index,shuffle_index}.npy
~~~

模板固定的原任务数据身份如下：

| 文件 | SHA256 |
|---|---|
| bin | 6b925f342504c3eae80c5bde3163ff5924ad36b262bc6a276839a6f3f817ea3c |
| idx | d535e502d62368e7a195b27a9a717543534ee1eaa9b4f9ce45f4d08d746cb5cd |
| document index | 4c1b135944d7fdec25b593a7c9436a451a1428c5b7b06c214c66a33cd2c8226e |
| sample index | 8be63e7b96ec2f2f1462cf100907e01fbb96858256fd071f45613816e375ace7 |
| shuffle index | 484d0800cccde790b43d1e91c8663cad05a39d8427fb3d1272d34b6de22f6643 |

数据不随 PR 分发。使用其他语料时，必须同时传入 dataset.data_sha256 和 dataset.index_sha256 的真实值；这是另一轮实验，不能宣称重现已有轨迹。

低层 .idx 的原始文档可以变长；采样后 input_ids、labels、loss_mask、position_ids 均须有 4096 项。[xing40_alignment_data.py](xing40_alignment_data.py) 由启动脚本仅给训练子进程添加本示例目录的 PYTHONPATH，使用独立模块名 xing40_alignment_data，避免与 RL 的 examples 包冲突。它复用仓内 IndexedDataReader，读取指定 NPY，不重建索引、不在 Dataset 叠加恢复偏移。它保留显式 padding mask，再构造 shift_labels=-100；CP2 读取同一 DP 全样本，分别取连续前后半段并保留全局 position IDs。

每 rank 启动时核验五个文件 SHA，原 bin 约 4.65 GB，启动会产生 I/O；这不是训练稳态耗时。原任务有序线程 reader 换为仓内 reader、num_workers=0、LBS1，样本等价性须用真实字段检查，不能凭读取器名称认定。

## 4. 先预览命令，再启动单卡

从仓库根目录运行，每轮选择新的输出目录。启动器拒绝覆盖已有目录：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=4
bash examples/training_demo/xing40/run_alignment.sh single \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/single-resume" \
  --resume /checkpoints/mf1500-converted-to-hp/global_step_1500 --dry-run

# 核对命令后，去掉 --dry-run 执行。
bash examples/training_demo/xing40/run_alignment.sh single \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/single-resume" \
  --resume /checkpoints/mf1500-converted-to-hp/global_step_1500
~~~

该目录必须是 **已经完成转换的 HP native DCP 完整训练状态**，含模型、主权重、Muon/AdamW slots、参数绑定与组、scheduler、global_step、loader/sampler 进度及适用的 RNG/精度状态。脚本不转换原始 MF checkpoint，不能把仅权重目录直接传入。

MF 使用融合 mHC 连续训练到 2500，在第 1500 步更新后导出、转换完整状态；不得用旧非融合 MF checkpoint 代替。该 checkpoint 完整落盘并核验后，HP 可在另一张空闲卡与 MF 剩余训练并行。HP 首步应为 1501，单卡源 consumed_samples 应为 3000。先核对恢复值和下一批实际四字段，再继续完整对照。相同 seed 不能代替状态和样本核验。

随机初始化可显式使用 --from-scratch，须另设输出目录；它不能代替 MF→HP 续训精度验收。

## 5. 建立四卡 FSDP 基线，保存第 1500 步

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# 多组作业需要预留不重叠的 HCCL 端口范围。
export HCCL_IF_BASE_PORT=52000
bash examples/training_demo/xing40/run_alignment.sh fsdp4 \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/fsdp4-baseline"
~~~

使用单机 torchrun standalone rendezvous，GBS4/LBS1，每 rank 每优化步一条全序列，连续 1–2500 步。配置采用原生同步保存，每 500 步及训练结束保存完整状态，生成 global_step_1500 和 global_step_2500。

**保存方式是新复现变体**：历史专用 worker 额外保存 step5/1500；这里使用 save_steps=500，增加 500/1000/2000 等 checkpoint 的 I/O，不改变优化器/LR 数学。不能把本命令称为历史脚本的逐字重放，性能计时不能包含额外保存开销。

只有第 1500 步 checkpoint 完整落盘、保存调用完成并核验状态/样本后，才能在另一组空闲卡开始续训；不能读取未写完的 checkpoint。

## 6. 从同一 FSDP1500 独立运行三个四卡方案

~~~bash
export FSDP1500="$RUN_ROOT/fsdp4-baseline/checkpoints/global_step_1500"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

export HCCL_IF_BASE_PORT=52100
bash examples/training_demo/xing40/run_alignment.sh fsdp4_cp2 \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/cp2-resume" --resume "$FSDP1500"

export HCCL_IF_BASE_PORT=52200
bash examples/training_demo/xing40/run_alignment.sh fsdp4_selective \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/selective-resume" --resume "$FSDP1500"

export HCCL_IF_BASE_PORT=52300
bash examples/training_demo/xing40/run_alignment.sh fsdp4_ep2 \
  "$DATA_PREFIX" "$INDEX_PREFIX" "$RUN_ROOT/ep2-resume" --resume "$FSDP1500"
~~~

以上依次执行；并行需分配互不重叠的空闲卡、端口和输出目录。训练终点仍是总步数 2500，不能重启另一条 2500 步日程。

四卡 source1500 的 consumed_samples=6000。CP2 两 rank 对同一 DP 样本分别处理连续 2048 token，每步各累积两个 micro；EP2 的全局样本数不乘 EP 倍数。恢复须核模型重分片、主权重、Muon/AdamW slots、参数组、scheduler、步数、数据进度、首批和适用的每 rank extra/RNG；退出码为零不证明语义等价。

CP 配方同时接入 MLA 和 MoE 全序列 aux 两个 wrapper，目前支持 CP2/TP1/EP1、expanded_ulysses、all_gather；这不代表任意 CP/TP/EP 混合组合已经验证。

## 7. 保存证据并统一验收口径

输出保留 input.yaml、revision.txt、worktree.patch、environment.txt、command.bash、train.log 和 checkpoints。原始 YAML 加脚本记录的 CLI 覆盖才是最终配置；应另保留解析结果、数据 SHA、硬件拓扑、设备映射、关键恢复审计和逐步数值。清理已不用于复现的重复中间产物，保留仍需恢复的 checkpoint。

单卡差值为 HP−MF，四卡为“待测配置−纯 FSDP”。按实际 step 标签选取 **1501 到 2500，包含两端，共 1000 步**，计算 abs(mean(有符号差值))，要求 ≤0.005。它不是单步差值或 MAE 的阈值。另报正/负/零步数、MAE、最大绝对差和逐步曲线，保留离群点；MTP 关闭时不填 0 冒充通过。

性能统一预热/区间（已有短测为 1501–1510 预热，1511–1540 计时），四 rank 每步耗时取最大值；报告稳态统计和波动、GBS×4096/step_seconds 的 token/s、每卡 allocated/reserved 峰值。并发干扰和保存 I/O 另记；不能将 profiler 各流时间相加当作墙钟时间。

## 8. 验证范围与历史结果

以下是历史非融合组合结果，不是当前融合候选的设备验收。历史 #1404 单卡组合 `6f979dd0` 的 1501–2500 窗口 abs(mean(HP−MF))=0.000022103786468505858；本候选需用新融合 MF 和 HP 的实测数据重新验收。

更早的一次单卡完整验收源码为 **259516b4**：从 MF1500 完整训练状态续训1501–2500，abs(mean(HP−MF))=0.0000382324457，8000输入字段摘要一致；[完整报告和逐步数据](https://github.com/xingleixu123/hyper-parallel/tree/516b07df2c77abbf701055717833004546d89faf/docs/validation/pr1367-mf-hp-single-20261030)。

更早长跑头为33096a1（单卡/FSDP/EP/selective）和1a39572（CP/TP）；历史abs(mean)分别包括单卡0.00003065288、CP−FSDP0.00000546551、EP−FSDP0.00001255035、selective−FSDP0.00000476837。它们与上述259516b4单卡都不是后续新提交的重跑；具体版本、配置与限制见[总issue](https://github.com/hf1s/hyper-parallel/issues/304)。

机器专用放行系统、MF→HP离线转换器、数据和checkpoint大文件不随示例分发；已转换的完整状态是单卡续训前提。模块验证不能代替新提交/新并行策略的完整NPU窗口验收。


必要 CPU/Gloo 检查：

~~~bash
python .agent/scripts/check_agents_catalog.py
python -m pytest -q tests/ut/models/xing40 \
  tests/ut/components/losses/test_projected_cross_entropy.py
python -m pytest -q tests/ut/examples/xing40/test_alignment_data.py
python -m pytest -q tests/torch/tensor_parallel/test_xing40_qk_clip_tp.py \
  tests/torch/expert_parallel/test_xing40_cp_aux.py
~~~

### 历史配置的入口兼容

历史resolved YAML中的 hyper_parallel.models.xing40.alignment_data 已迁至示例模块 xing40_alignment_data，数据辅助代码不属于安装包的业务API。请使用本目录五份配方及 run_alignment.sh，保持训练子进程导入环境与 command.bash 一致；不要只改YAML后直接复用旧命令。
