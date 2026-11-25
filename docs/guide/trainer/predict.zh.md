# 离线文本预测

[English](predict.md) | **简体中文**

`examples/training_demo/run_text.py` 使用 `run_mode: predict` 执行本功能。默认仍为 `train`。

支持单卡和每个分片持有完整模型的独立 DP，无需 dataset、dataloader、loss_fn 或 optimizer。
目前原生 Qwen3 和 Qwen3-MoE 通过 `ModelAdapterSpec.supports_generation` 声明生成能力。
预测跳过训练用 FSDP 包装，拒绝 FSDP/HSDP 参数分片。
当前拒绝 TP/CP/EP/PP/SP、训练编译、activation checkpoint、activation swap、HyperOffload 和 worker recovery。

## 启动

修改 [配置示例](../../../examples/training_demo/qwen3/predict.yaml) 中的路径后执行：

```bash
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/predict.yaml
```

设置 `checkpoint.restore_from` 指定 DCP checkpoint；使用 `LATEST` 时同时配置 `checkpoint.checkpoint_dir`。
缺失 checkpoint 会报错。未指定恢复时使用模型构建入口加载的权重。Ascend 使用 `hccl`，NVIDIA 使用 `nccl`。

## Predict 输入与输出

### 输入与多轮对话

支持 `.txt`（每行一个非空 prompt）、`.jsonl`（每行一个对象）、`.json`（对象数组）：

```json
{"id":"plain","prompt":"介绍一下北京。"}
{"id":"chat","messages":[{"role":"system","content":"用中文简洁回答。"},{"role":"user","content":"北京有多大？"},{"role":"assistant","content":"约1.64万平方公里。"},{"role":"user","content":"人口呢？"}]}
```

`prompt` 与 `messages` 互斥。缺失 `id` 使用稳定输入序号；重复 ID、空输入或非法消息在生成前报错。
`messages` 支持 system/user/assistant/tool，保留 `tool_calls`、`tool_call_id` 等模板字段，
只生成下一条 assistant 回复。必须显式配置 `model_assets.chat_template`；模板配置属于
`model_assets`，不是 tokenizer target 的参数。普通 `prompt` 直接编码，不套聊天模板。

调用方可以维护完整 `messages`，每轮追加用户消息、调用 predict，再把生成文本作为 assistant 消息加入历史。
当前 `TextTrainer.predict()` 读取配置中的输入文件，返回运行摘要，没有 `predict(messages)` 形式的内存接口。
因此每轮需写输入文件、读取输出 JSONL 中的 `generated_text`，并使用新的输出路径或显式允许覆盖。
同一进程可以复用已加载的模型；会话历史、交互循环和上下文长度管理由调用方负责。

### 批量生成、分桶与 KV cache

[`batch_size`](../../../hyper_parallel/predict/batching.py) 是每个 DP lane 的请求数上限。
`bucket_by_length: true` 会在每个 lane 内，
按编码后的 prompt 长度从短到长排序，然后组批。模型输入是 `[请求数, 本批最长 prompt 长度]` 的矩形张量，
短输入在左侧补 padding，`attention_mask` 标记有效位置。最终输出仍恢复原始输入顺序。

例如同一 lane 中有四条请求，长度为 `100、900、120、880`，每批最多两条，且 token 预算充足：

| 组批方式 | 两个 batch | 总 padding |
|---|---|---|
| 输入顺序 | `(100, 900)`、`(120, 880)` | 1560 token |
| 按长度排序 | `(100, 120)`、`(880, 900)` | 40 token |

`max_batch_tokens` 约束 `batch 请求数 × 本批最大 prompt 长度`，包括 prompt padding，
不包括后续生成的 token。因此单批可能达不到 `batch_size`；单个请求超预算会报错，不静默截断。
例如 `batch_size: 4`、`max_batch_tokens: 8192`、`max_new_tokens: 256` 表示最多四条请求，
补齐后的 prompt 总量不超过 8192，每条请求最多生成 256 个新 token。

当前是静态批处理：整批生成结束后再处理下一批，提前结束的请求仍保留在当前 batch 中，
不会中途换入新请求。Continuous batching 指动态移除已完成请求、补入新请求，当前没有实现。
它解决请求调度问题；是否需要 padding 则取决于底层是否支持变长序列执行，两者不是同一能力。

predict 通过 `infer.generate` 逐 token 生成。`generation.use_cache: true` 会缓存一次生成过程中的
Key/Value，后续解码复用前缀状态；可以显式关闭。当前示例开启 cache，但不同 predict 调用之间
不共享 KV cache，多轮对话的每一轮都会对提交的完整历史重新执行 prefill。

生成参数只接受 `GenerationOptions` 中的可序列化字段：
`max_new_tokens`、`temperature`、`top_k`、`top_p`、`do_sample`、`eos_token_id`、
`pad_token_id`、`repetition_penalty`、`use_cache`。
特殊 token 的解析顺序为任务显式配置、模型 generation 元数据、tokenizer；pad 缺省时可回退 EOS。
EOS 可以是多个 ID。运行时模型适配器、设备、进程组和 RNG 由代码注入，不能通过 YAML 传入。

### 输出与 manifest

输出每行包含：

```json
{"id":"plain","ordinal":0,"sample_idx":0,"generated_text":"北京是中国的首都。","prompt_tokens":8,"generated_tokens":9,"finish_reason":"eos"}
```

`generated_tokens` 包括停止 EOS，不包括补齐 padding。文本解码会去掉特殊 token。
`finish_reason` 来自生成循环，支持 `eos`、`length`、`stop`；配置中的 token 上限对应 `length`。
`stop` 供代码注入的 legacy stopping criteria 使用。本阶段每个请求只有 `sample_idx=0`。

DP 使用共享文件系统。每个 lane 在同一个运行临时目录中写各自的分片文件和完成 manifest；协调 rank 检查分片校验和、
归属、ID/序号以及完整覆盖，按原序合并后原子发布 JSONL。已存在输出默认报错，只有
`overwrite: true` 才替换。失败保留 `.<输出文件名>.<run_id>/`，不会自动续跑。
目标文件旁的 `.lock` 阻止并发 writer；进程被强制终止留下锁时，确认旧任务结束后再清理。

`<output_file>.manifest.json` 记录输入/输出校验和、模型与 checkpoint 来源、tokenizer
词表指纹、模板、解析后的生成参数、seed、batch 策略及环境版本。
JSONL 是原子提交单元；manifest 是独立 sidecar，消费者可以用 `output_sha256` 检查二者一致性。
采样使用每个 DP lane 的 RNG；即使 seed 相同，改变 DP 数、batch 组成或执行顺序也可能改变某条请求的采样结果。

以下是设计中的 B0 扩展，当前没有实现：

| 扩展 | 含义 | 当前行为 |
|---|---|---|
| `n > 1` | 同一请求生成多个候选，以 `(input_id, sample_idx)` 标识 | 每条请求只生成一个答案，`sample_idx=0` |
| 请求级 RNG | 根据全局 seed、请求 ID、候选编号稳定派生独立随机流 | 同一个 lane 的请求共享 RNG，不保证重新组批后的请求级随机流不变 |
| 续跑 manifest | 校验任务配置，识别已可靠保存的候选，只补算未完成项 | 现有 manifest 记录来源和完整性，失败分片不会自动用于续跑 |

将来的候选级续跑也不等于恢复到某个回答已经生成一半时的 KV cache 或解码状态。

## 分层

`predict/` 不导入 Trainer。Trainer 负责模型、tokenizer、恢复、配置和拓扑检查；
`Predictor` 负责请求组批和输出，通过 `infer.generate` 复用生成循环。
旧 `generate(model, input_ids, generation_config, attention_mask)` 默认仍返回 Tensor；
新调用可传 `options=GenerationOptions(...)`、`runtime=GenerationRuntime(...)`、`return_dict=True` 获取详细结果。
`get_build_steps(run_mode)` 返回需要构建的组件名称，不实际执行构建，也不是执行顺序。

## 验证入口

```bash
pytest -q tests/ut/infer tests/ut/predict
pytest -q tests/torch/predict/test_predict_e2e.py
```

系统测试使用真实两层随机初始化 FP32 小模型和合成数据；不能据此推导全尺寸模型、BF16、
量化或替换算子组合的正确性。系统测试需要匹配的真实设备与后端，CPU 测试不能替代 NPU 验收。
