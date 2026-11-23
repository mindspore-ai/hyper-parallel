# 独立 Evaluation 与 Predict

[English](eval_predict.md) | **简体中文**

`examples/training_demo/run_text.py` 根据 `run_mode: train | eval | predict`
装配并执行任务。默认仍为 `train`，旧训练配置与回调执行路径保持兼容。

## 功能范围

| 模式 | 已实现 | 拓扑范围 |
|---|---|---|
| eval（P0） | 全局 token 加权 loss、纯 NLL 的 perplexity、独立数据范围、model-only 恢复 | 单卡、DP、dense FSDP/HSDP |
| predict（P1） | 文件输入、历史对话、长度分桶、批量生成、有序 JSONL、manifest | 单卡、每个 lane 持有完整模型的独立 DP |

预测的模型能力由 `ModelAdapterSpec.supports_generation` 显式声明。目前 Qwen3 和
Qwen3-MoE 原生文本模型已声明该能力。新增模型家族需要自行验证缓存和生成契约。
模型替换、量化等组合仍需分别验收，不能从训练支持情况推导生成支持情况。

独立任务拒绝训练编译、activation checkpoint 和 activation swap。
TP/CP/EP/PP/SP 尚未开放；predict 也拒绝 FSDP/HSDP 分片，跳过训练用 FSDP 包装。
生成式评估、beam search、多个候选、断点续跑和在线服务不在本阶段范围内。

## 启动

修改示例中的模型、输入与输出路径后执行：

```bash
# 显式指定 nproc_per_node=1 可运行单卡；NVIDIA 使用 training.backend=nccl。
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/eval.yaml

torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/predict.yaml
```

eval 和 predict 均不需要 optimizer。predict 也不需要 dataset、dataloader、loss_fn。
可以通过 `checkpoint.restore_from` 指定已有 DCP checkpoint，或者设置 `LATEST` 并提供
`checkpoint.checkpoint_dir`。缺失 checkpoint 会报错，不回退到初始权重。
未指定恢复时，使用模型构建入口加载的预训练权重。

## Evaluation

`evaluation.dataset`、`evaluation.dataloader` 可以覆盖顶层数据配置。
`evaluation.split` 可选 `valid`、`test`、`dataset`，最后一种用于单个未切分 Dataset。
任务不会隐式选择训练集。

- `max_samples` 在选定 Dataset 的全局索引范围上生效，然后才做 DP 分片。
- Online Mapping 将请求上限截断到自然切分大小；`max_samples: null` 遍历整个自然切分，不读取训练迭代计划。多个加权来源采用无放回采样，权重决定顺序，某个来源耗尽后继续从剩余来源取样；固定 seed 时，改变上限仍保留相同前缀。
- Iterable 来源需要 `max_samples`；其有限前缀会物化后分片。
- Indexed 构建器需要明确样本计划，使用 `max_samples` 或构建器自己的显式 split sizes。
- `max_batches` 限制前向轮数。动态 token batching 的每轮样本数由 token budget 决定。
- 不丢弃不足整批的尾部，不通过重复有效样本补齐 DP。耗尽数据的 lane 使用零权重前向保持 FSDP 轮次一致。

统计先累计分子、分母，最后归约；不平均 batch 均值或 rank 均值。
内置 loss/PPL 与样本数共用一次数值归约，归约前交换统计 schema 与错误状态。
`compute()` 可以重复读取，不重新归约。CPU/CUDA 使用 FP64 累加；NPU 使用带补偿的 FP32 累加，
避免 torch_npu 的 FP64 转换回退导致精度丢失。样本数在设备上保持整数计数，通信使用 FP32 高低位对，兼容 HCCL，
归约后在 CPU FP64 中合并。该方式保持常量统计内存，不逐 batch 同步到主机。
补偿用于减小本地累加误差；集合通信仍对每个通道执行 FP32 求和，不保证精确的 FP64 运算。
自定义指标默认执行自己的 `update`/`synchronize` 流程。`LossMetric` 子类在原样继承这两个方法时共用归约，
覆盖其中任何一个方法后保留自己的累计状态和同步逻辑。如果只需原始共享统计，自定义指标可以显式让
`supports_shared_reduction` 返回 `True`，并实现 `set_reduced_statistics`。
`num_samples` 统计含有效监督的模型样本；packing 不会把多个样本算成一个 tensor row。
原始记录被 transform 切成多个窗口时，它们分别计入模型样本数，`max_samples` 仍限定原始 Dataset 索引。

loss 模块必须声明 `loss_normalization`：`token_mean`、`sum` 或 `global_mean`。
已归约的 `global_mean` 不会再次累加。PPL 仅对 `lm_nll` 统计计算；aux/objective 不可作为 PPL。
loss 模块还可以声明 `statistic_kinds`（映射或按 loss key 取值的可调用对象）来改写某个统计的类别：
chunked causal-LM loss 返回单个局部 token 均值，当模型以非零系数把 router aux 项折入其中时，它把
`foundation` 声明为 objective，PPL 会被明确拒绝，而不是对含 aux 的值取指数。
aux 系数缺省或为零时保留纯 NLL，可以正常计算 PPL。
全 mask 的 batch 贡献零权重，不会把空交叉熵的 NaN 传播到最终报告。

## Predict 输入与输出

支持 `.txt`（每行一个非空 prompt）、`.jsonl`（每行一个对象）、`.json`（对象数组）：

```json
{"id":"plain","prompt":"介绍一下北京。"}
{"id":"chat","messages":[{"role":"system","content":"用中文简洁回答。"},{"role":"user","content":"北京有多大？"},{"role":"assistant","content":"约1.64万平方公里。"},{"role":"user","content":"人口呢？"}]}
```

`prompt` 与 `messages` 互斥。缺失 `id` 使用稳定输入序号；重复 ID、空输入或非法消息在生成前报错。
`messages` 支持 system/user/assistant/tool，保留 `tool_calls`、`tool_call_id` 等模板字段，
只生成下一条 assistant 回复。必须显式配置 `model_assets.chat_template`；模板配置属于
`model_assets`，不是 tokenizer target 的参数。普通 `prompt` 直接编码，不套聊天模板。

`batch_size` 是每个 DP lane 的请求上限。`max_batch_tokens` 约束
`batch 请求数 × 最大 prompt 长度`；单个请求超预算会报错，不静默截断。
`bucket_by_length` 可改变执行顺序，输出始终恢复原始输入顺序。

生成参数只接受 `GenerationOptions` 中的可序列化字段：
`max_new_tokens`、`temperature`、`top_k`、`top_p`、`do_sample`、`eos_token_id`、
`pad_token_id`、`repetition_penalty`、`use_cache`。
特殊 token 的解析顺序为任务显式配置、模型 generation 元数据、tokenizer；pad 缺省时可回退 EOS。
EOS 可以是多个 ID。运行时模型适配器、设备、进程组和 RNG 由代码注入，不能通过 YAML 传入。

输出每行包含：

```json
{"id":"plain","ordinal":0,"sample_idx":0,"generated_text":"北京是中国的首都。","prompt_tokens":8,"generated_tokens":9,"finish_reason":"eos"}
```

`generated_tokens` 包括停止 EOS，不包括补齐 padding。文本解码会去掉特殊 token。
`finish_reason` 来自生成循环，支持 `eos`、`length`、`stop`；配置中的 token 上限对应 `length`。
`stop` 供代码注入的 legacy stopping criteria 使用。本阶段每个请求只有 `sample_idx=0`。

DP 使用共享文件系统。每个 lane 写独立分片和完成 manifest；协调 rank 检查分片校验和、
归属、ID/序号以及完整覆盖，按原序合并后原子发布 JSONL。已存在输出默认报错，只有
`overwrite: true` 才替换。失败保留 `.<输出文件名>.<run_id>/`，不会自动续跑。
目标文件旁的 `.lock` 阻止并发 writer；进程被强制终止留下锁时，确认旧任务结束后再清理。

`<output_file>.manifest.json` 记录输入/输出校验和、模型与 checkpoint 来源、tokenizer
词表指纹、模板、解析后的生成参数、seed、batch 策略及环境版本。
JSONL 是原子提交单元；manifest 是独立 sidecar，消费者可以用 `output_sha256` 检查二者一致性。
采样可在相同配置和拓扑下复现；不承诺改变 DP 数、batch 组成后的请求级随机流保持不变。

## 分层与兼容性

`evaluation/` 与 `predict/` 不导入 `trainer/`，也不互相导入。
任务宿主负责模型、tokenizer、权重恢复、配置和拓扑门禁。
评估通过注入的 step 获取统计；Predictor 仅负责请求、batch、结果与落盘。
prefill/decode、cache、采样、停止和 padding 剥离均复用 `infer/` 的唯一生成循环。

旧 `generate(model, input_ids, generation_config, attention_mask)` 默认继续返回 Tensor。
新调用可传 `options=GenerationOptions(...)`、`runtime=GenerationRuntime(...)`、
`return_dict=True`，获取 `GenerationResult` 的真实 prompt/generated 长度与停止原因。

## 验证入口

```bash
pytest -q tests/ut/evaluation tests/ut/infer tests/ut/predict
pytest -q tests/torch/predict/test_predict_e2e.py
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_native_eval_range_and_empty_lanes
# 以下两项需要可用的 NPU/GPU，验证真实 FSDP 路径：
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_eval_matches_reference_single_card
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_eval_dp_ranks_report_identical_global_metrics
```

系统测试使用真实两层随机初始化 Qwen3/Qwen3-MoE、合成数据与真实 Torch 后端，
覆盖 HF greedy token 对齐、cache 开关、DP 非整除/空 lane、失败传播、真实 TextTrainer、
LATEST model-only 恢复与独立评估参考。小模型验收不能替代全尺寸模型、替换算子或加速器组合验收。
