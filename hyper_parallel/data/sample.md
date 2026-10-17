# 数据格式、模型编码与监督处理总结

本文整理纯语言与多模态数据从原始记录到模型输入的处理方式，覆盖预训练、SFT、推理和偏好训练。
每个示例说明原始格式、标准输入、编码入口以及 labels/mask 的语义。示例用于说明契约，不代表所有模型
或当前 HyperParallel 已支持全部功能。数据模块当前实现以 [README.md](README.md) 和模型 adapter 为准。

框架对照基于 NeMo AutoModel、Transformers 和公开文档，记录日期为 2026-10-15。

## 1. 分类方式：先场景，再模态，处理时机另列

| 场景 | 纯语言 | 多模态 | 目标 |
| --- | --- | --- | --- |
| 预训练/继续预训练 | 文档、网页、书籍 token 流 | 图文描述、音文配对等 | 按训练任务学习有效文本或专用模态目标 |
| SFT | 指令、问答、多轮对话、工具调用、reasoning | 图片问答、视频描述、语音识别等 | 学习参考回答、调用或任务输出 |
| 推理 | 文本续写、对话生成 | 媒体输入后的文本或其他模态生成 | 不计算训练 loss，生成输出 |
| 偏好训练 | chosen/rejected、单回答反馈 | 同一媒体条件下的候选回答偏好 | 学习回答偏好，使用对应偏好目标 |

在线/离线说明处理时机；Mapping/Iterable 说明数据访问方式；packing 说明组样方式；JSONL/Parquet/.bin
说明存储方式。它们与场景和模态是不同维度，不需要重复建立整套章节。

原始文档保留 text，对话采用 messages，偏好保留 prompt/chosen/rejected，离线数据保留 token 和必要元信息。
不要为统一接口把所有输入都强行包装成 messages。

## 2. labels、mask 和 shift 的统一约定

### 2.1. 四类信息不能混用

| 信息 | 用途 | 示例 |
| --- | --- | --- |
| `input_ids` | 模型看到的 token | 提示、回答、角色标记、媒体占位 token |
| `labels` | 需要预测的 token 目标 | 不监督的目标设为 `-100`，前提是损失函数使用该 ignore index |
| `attention_mask` | 有效位置或 attention 可见性 | 常见二维 0/1；packed document ID 和四维 block mask 是其他语义 |
| `loss_mask` | 显式 loss 权重 | 0/1 或其他权重；是否传入、何时应用由训练框架决定 |

本文用 `supervision_mask` 表示目标是否参与监督，方便解释；它不是 Transformers 通用的 model.forward 参数。
对普通 token CE，常见做法是将 supervision_mask=0 的 labels 置为 -100。
提示可以不监督但仍参与 attention；媒体输入可以不计算文本 CE 但仍参与模型计算。

### 2.2. 本文默认采用 HF 常见的未 shift labels

对一段完整序列：

```text
完整 token：A B C D
HF 常见：  input_ids = [A B C D]，labels = [A B C D]
模型/loss：用位置 A 的 logits 预测 label B，依次执行 next-token shift

已 shift：input_ids = [A B C]，labels = [B C D]
下游 loss：直接按位置比较，不能再 shift
```

以下训练示例默认 labels 与原始 input_ids 同位置，模型内部 shift；supervision_mask 也按这套未 shift
目标坐标展示。首个 label 在普通内部 shift 中不参与 loss，是否有 BOS 取决于模型协议。
符号 P/A/V/E/PAD 是提示、回答、媒体占位、结束、padding token 的示意，不是真实 token ID。
角色头是否监督、EOS/EOD 是否监督，需要在具体训练配置中明确，本文 SFT 示例默认屏蔽角色头并监督结束标记。

AutoModel 的一些 Dataset/collator、HyperParallel Indexed 路径已提前 shift，必须以实际输出契约为准。

### 2.3. Padding 与边界

- Padding 通常 attention_mask=0、labels=-100、loss_mask=0。
- PAD 与 EOS 使用同一 token ID 时，应按有效长度/attention mask 区分，不能将所有同 ID 的 token 都屏蔽。
- 回答边界应由模板支持的 assistant mask 或可靠的模板解析确定。不要假定独立编码 prompt 的长度永远等于
  完整编码中的 prompt 前缀长度；BPE、模板历史重写和媒体展开均可能影响边界。
- attention 可见性和监督范围独立：labels=-100 不意味着该输入 token 对后续位置不可见。

## 3. 预训练：纯语言与多模态

### 3.1. 原始文档/网页/书籍

原始记录：

```json
{"text": "猫是一种哺乳动物。"}
```

HF 接收普通文本，不要求对话模板：

```python
text = sample["text"]
inputs = tokenizer(text, return_tensors="pt")
```

是否添加 BOS/EOS/EOD 按模型与语料处理协议决定。组样后示意：

```text
input_ids：       T1 T2 T3 E PAD
labels：          T1 T2 T3 E -100
attention_mask：   1  1  1 1  0
supervision_mask： 1  1  1 1  0
```

通常监督所有有效文本的 next-token 目标；EOD loss 可单独配置。
离线 tokenizer 可以提前写 token，训练 Dataset 再沿 token 流构建固定长度样本。

### 3.2. 图像-caption 配对

原始记录：

```json
{"image": "cat.jpg", "caption": "一只猫坐在窗边。"}
```

有两种模型接入协议，不应混为一种通用模板。

**直接图文协议：**加载图片，组织模型要求的图像占位和描述文本，再调用 processor：

```python
inputs = processor(images=image, text=formatted_caption, return_tensors="pt")
```

`formatted_caption` 按模型要求准备；不是所有 processor 都会自动插入缺失的图像占位符。

**对话协议：**若目标模型以 chat 格式训练，可以转换为 messages：

```python
messages = [
    {"role": "user", "content": [{"type": "image", "image": "cat.jpg"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "一只猫坐在窗边。"}]},
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False,
)
```

图像条件下预测 caption 的典型监督：

```text
input_ids：       [前缀/图像 V] [caption C] E
labels：          [-100 ...]   [caption C] E
attention_mask：  [有效位置均为 1；padding 为 0]
supervision_mask：[0 ...]      [1 ...]     1
```

使用 messages 是序列化选择，不自动改变训练场景。当前目标 checkout 中 DeepSeek-V4.1 的图像-caption
预训练使用模型专属编码，不能假定通用 AutoProcessorTransform 自动支持所有预训练格式。

### 3.3. 音文配对与模态生成任务

音频输入、文本目标的原始例子：

```json
{"audio": "speech.wav", "transcript": "你好"}
```

若模型支持音频对话输入，可标准化为：

```python
messages = [
    {"role": "user", "content": [{"type": "audio", "path": "speech.wav"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
]
```

Processor 编码音频与文本，训练侧监督转写文本、屏蔽输入媒体位置。该数据同样可用于任务 SFT，场景由配方定义。
`path` 等媒体字段和采样率要求应核对具体 processor。

文本到音频、文本到图像等任务必须使用模型规定的输出目标。例如音频 token CE、图像 token CE 或 diffusion
的噪声/速度目标，不能把全部媒体目标一律置为 -100，也不能统一套用文本 labels。

## 4. SFT：纯语言源格式

### 4.1. Alpaca / instruction-input-output

原始记录：

```json
{"instruction": "翻译成英文", "input": "你好", "output": "Hello"}
```

格式适配器将字段转换为 HF messages：

```python
messages = [
    {"role": "user", "content": "翻译成英文\n你好"},
    {"role": "assistant", "content": "Hello"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_dict=True, add_generation_prompt=False,
)
```

典型 answer-only 监督：

```text
input_ids：       [user/提示 P] [assistant头 H] [回答 A] E
labels：          [-100 ...]   [-100 ...]      [回答 A] E
supervision_mask：[0 ...]      [0 ...]         [1 ...]  1
attention_mask：  [所有有效输入为 1；padding 为 0]
```

问答列格式同理：

```json
{"context": "猫是哺乳动物。", "question": "猫属于哪类动物？", "answer": "哺乳动物"}
```

先按配置把 context 放入 system 或 user，把 question 放入 user，把 answer 放入 assistant；不要按列名猜测
唯一的模板。AutoModel 的 ColumnMappedTextInstructionDataset 使用 column_mapping 完成源字段映射。

### 4.2. Prompt-completion

原始记录：

```json
{"prompt": "翻译成英文：你好", "completion": "Hello"}
```

普通 completion 模型可按配方拼接并 tokenize；chat 模型可转换为：

```python
messages = [
    {"role": "user", "content": "翻译成英文：你好"},
    {"role": "assistant", "content": "Hello"},
]
```

TRL 也接受 prompt/completion 直接为 messages 列表的形式：

```json
{
  "prompt": [{"role": "user", "content": "翻译成英文：你好"}],
  "completion": [{"role": "assistant", "content": "Hello"}]
}
```

这不是把完整记录直接传给 tokenizer；Trainer 的格式准备层负责拼接、套模板并保留 completion 边界。

```text
input_ids：       [prompt P] [completion A] E
labels：          [-100 ...] [completion A] E
supervision_mask：[0 ...]    [1 ...]         1
```

completion_only_loss=false 可以选择全序列监督，但数据仍可属于 SFT；该开关不决定训练阶段。

### 4.3. 多轮 messages

原始数据已是标准对话，不需要换字段：

```json
{
  "messages": [
    {"role": "system", "content": "请简洁回答"},
    {"role": "user", "content": "猫是什么？"},
    {"role": "assistant", "content": "一种哺乳动物。"},
    {"role": "user", "content": "会游泳吗？"},
    {"role": "assistant", "content": "通常具有游泳能力。"}
  ]
}
```

取 sample.messages，交 tokenizer.apply_chat_template；监督可选：

```text
内容：             system user1 answer1 user2 answer2
所有回答监督：       0      0      1      0      1
仅最后回答监督：     0      0      0      0      1
```

未监督的历史回答仍是输入上下文。AutoModel 的 ChatDataset 默认走 assistant-only 格式化，mask_history
可保留最后一轮；不同模型对角色头、回合结束和 reasoning 的处理仍需核对模板。

### 4.4. ShareGPT

原始记录：

```json
{"conversations": [{"from": "human", "value": "猫是什么？"}, {"from": "gpt", "value": "哺乳动物"}]}
```

标准化后：

```python
messages = [
    {"role": "user", "content": "猫是什么？"},
    {"role": "assistant", "content": "哺乳动物"},
]
```

conversations/from/value 和 human/gpt 的差异由 converter 消除；其后的模板、labels 和 mask 复用多轮对话流程。
不能把工具角色直接当作普通 human/gpt 猜测转换，工具数据需要保留结构化字段。

### 4.5. 工具调用

原始业务记录：

```json
{
  "question": "北京天气如何？",
  "call": {"name": "weather", "arguments": {"city": "北京"}},
  "result": "晴",
  "answer": "北京今天晴天。"
}
```

转换为工具调用 messages 与 schema：

```python
messages = [
    {"role": "user", "content": "北京天气如何？"},
    {"role": "assistant", "tool_calls": [{
        "id": "call_0", "type": "function",
        "function": {"name": "weather", "arguments": {"city": "北京"}},
    }]},
    {"role": "tool", "tool_call_id": "call_0", "name": "weather", "content": "晴"},
    {"role": "assistant", "content": "北京今天晴天。"},
]
tools = [{"type": "function", "function": {
    "name": "weather", "description": "查询天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
inputs = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=True, return_dict=True, add_generation_prompt=False,
)
```

必须使用支持工具调用的模板；arguments 的 dict/string 形式也按模板约定核对。

```text
内容：             user  assistant调用  tool结果  assistant回答
supervision_mask：   0         1           0           1
```

调用名称和参数是目标，工具返回是上下文。只提取 content 会丢掉调用，无法形成完整训练目标。

### 4.6. Reasoning

原始记录：

```json
{"question": "1+1？", "reasoning": "将两个一相加。", "answer": "2"}
```

若模型模板支持 reasoning_content：

```python
messages = [
    {"role": "user", "content": "1+1？"},
    {"role": "assistant", "reasoning_content": "将两个一相加。", "content": "2"},
]
```

模板先渲染完整输入，训练侧再选择：

```text
内容：             提示  reasoning  final answer
训练推理过程：       0       1           1
只训练最终回答：     0       0           1
```

reasoning_content 不是所有模型的通用字段，部分模型采用其他字段或正文标记。字段未被模板渲染时，不能认为
它已进入 input_ids。模板自动插入的空 thinking 前缀也需按训练策略决定是否监督。

## 5. SFT：多模态输入

### 5.1. 图片问答

原始记录：

```json
{"image": "cat.jpg", "question": "图里是什么？", "answer": "一只猫"}
```

HF processor 接收：

```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "cat.jpg"},
        {"type": "text", "text": "图里是什么？"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "一只猫"}]},
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False,
)
```

Processor 生成 input_ids 和媒体 tensor/grid；训练侧生成监督：

```text
input_ids：       [问题 P] [图片 V] [回答 A] E
labels：          [-100]   [-100]   [回答 A] E
supervision_mask：[0 ...]  [0 ...]  [1 ...]  1
attention_mask：  [有效输入为 1，包括模型协议中的图片位置]
```

多图只是 content 中多个媒体块，但必须保留媒体顺序与 token span 的对应关系。
ShareGPT 的 <image> 加 images 列表也先转换为这个结构，不在 converter 中手工实现视觉 token 展开。

### 5.2. 视频描述

原始记录：

```json
{"video": "clip.mp4", "question": "描述动作", "answer": "有人走进房间。"}
```

在支持 path 视频输入的 processor 上可转换为：

```python
messages = [
    {"role": "user", "content": [
        {"type": "video", "path": "clip.mp4"},
        {"type": "text", "text": "描述动作"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "有人走进房间。"}]},
]
```

编码入口与图像 SFT 相同。模型 processor 负责帧采样、时间信息、grid 和视频 token 展开；训练侧通常只监督
回答及结束 token，视频和提示位置 labels=-100。不要丢掉采样 fps、frame indices 等模型所需 metadata。

### 5.3. 音频识别/问答

原始记录：

```json
{"audio": "speech.wav", "instruction": "转写音频", "answer": "你好"}
```

转换为：

```python
messages = [
    {"role": "user", "content": [
        {"type": "audio", "path": "speech.wav"},
        {"type": "text", "text": "转写音频"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
]
```

音频 tensor 作为输入，文本答案作为目标；与图像问答相同地屏蔽输入媒体和提示。音频生成则需要音频目标，
不能照搬识别任务的 mask。并非每个图文 processor 都支持 audio。

### 5.4. 模态组合

同一用户 content 可含文本、图片、视频或音频，但可接受组合由目标 processor/model 决定。
labels/mask 是任务语义，不由媒体扩展名决定。媒体输出任务还应保存专用目标、有效长度和损失权重。

## 6. 推理：纯语言与多模态

### 6.1. 文本续写

原始输入：

```json
{"prompt": "很久以前，"}
```

直接 tokenizer 编码并生成：

```python
inputs = tokenizer(sample["prompt"], return_tensors="pt")
outputs = model.generate(**inputs)
```

input_ids/attention_mask 描述输入，不提供训练 labels 或 loss_mask。
评测时可另外保存参考答案计算指标，但不应将它拼入待生成输入。

### 6.2. 文本对话

原始输入：

```json
{"question": "猫是什么？"}
```

转换为：

```python
messages = [{"role": "user", "content": "猫是什么？"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
)
outputs = model.generate(**inputs)
```

输入包含模板要求的 assistant 生成前缀，不包含参考回答；不提供 labels/loss_mask。
add_generation_prompt 是否实际添加 token 取决于模型模板。

### 6.3. 图像/视频/音频对话

原始输入：

```json
{"image": "cat.jpg", "question": "图里是什么？"}
```

转换为：

```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": "cat.jpg"},
    {"type": "text", "text": "图里是什么？"},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
)
outputs = model.generate(**inputs)
```

视频和音频用对应的媒体块替换 image，前提是 processor 支持。媒体字段需随 input_ids 一起交给模型。

```text
input_ids：      [用户/媒体/问题] [assistant生成前缀]
attention_mask：[有效输入为 1；padding 为 0]
labels：         不提供
loss_mask：      不提供
```

### 6.4. 续写已有 assistant 前缀

```python
messages = [
    {"role": "user", "content": "用 JSON 回答"},
    {"role": "assistant", "content": '{"answer": "'},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_dict=True, continue_final_message=True,
)
```

这是续写最后消息，不是新开 assistant 回合；不能同时开启 add_generation_prompt=True。
工具推理也只提供已发生的历史与 tools，由生成结果驱动后续工具执行，不使用 SFT labels。

## 7. 偏好训练：单独保留任务字段

### 7.1. 成对偏好

原始记录：

```json
{"prompt": "介绍猫", "chosen": "一种哺乳动物", "rejected": "一种植物"}
```

纯语言可保留字符串；对话模型可标准化为：

```python
record = {
    "prompt": [{"role": "user", "content": "介绍猫"}],
    "chosen": [{"role": "assistant", "content": "一种哺乳动物"}],
    "rejected": [{"role": "assistant", "content": "一种植物"}],
}
```

这是相应 Trainer 接收的记录，不是 tokenizer 直接接收的字典。Trainer/数据处理器分别编码两个候选：

```text
chosen input：    [prompt P] [chosen C] E
chosen labels：   [-100 ...] [chosen C] E
rejected input：  [prompt P] [rejected R] E
rejected labels： [-100 ...] [rejected R] E
```

通常只计算候选回答区间的 log-probability，再执行偏好损失；不是对 rejected 做普通正样本 CE。
多模态偏好同理，但两个候选必须对应相同媒体条件，各自的媒体展开和有效长度需正确保持。

### 7.2. 单回答反馈

```json
{"prompt": "介绍猫", "completion": "一种哺乳动物", "label": true}
```

保留 completion 的 token 目标/概率区间和样本级反馈。label=true 不是 token labels；具体反馈损失由所选
算法处理。不能把 true 广播成逐 token 目标。

## 8. 横向处理：离线、metadata、packing 与文件格式

### 8.1. 已编码 token

HF 未 shift 约定的示例：

```json
{"input_ids": [11, 22, 33], "attention_mask": [1, 1, 1], "labels": [-100, 22, 33]}
```

跳过 tokenize，保留或构建 labels。离线 SFT 要保存 labels/监督边界，或保留可以可靠重建它们的元信息。
仅保存普通文档 token，不能自动恢复所有 assistant、reasoning 和 tool 的监督范围。

### 8.2. 选样 metadata 与源清单

选样 metadata 示例，字段由数据系统定义：

```json
{"sample_id": 42, "text_tokens": 120, "estimated_media_tokens": 576, "packing_length": 696}
```

它用于规划成本，完整编码后才能得到真实 input_ids 和 labels。
另一类 meta 文件只记录数据文件、列映射、媒体目录、采样比例；不是模型输入，也不是离线监督文件。

### 8.3. Packing 与 shift

AutoModel VLM packing 的思路是先逐样本 shift，再拼接：

```text
样本一原序列：[P1 A1 E1] → input=[P1 A1]，labels=[A1 E1]
样本二原序列：[P2 A2 E2] → input=[P2 A2]，labels=[A2 E2]
拼接后：input=[P1 A1 P2 A2]，labels=[A1 E1 A2 E2]，边界=[0,2,4]
```

这是极简的提示与回答示例；实际多 token prompt 要将非目标 labels 同步设为 -100。
目标已对齐，不再 shift；不会让样本一的末位去预测样本二的提示首 token。
其他框架采用未 shift packed labels 时，需要明确屏蔽跨样本目标或使用损失侧的分段 shift。

Packed attention 用 cu_seq_lens、seq_lens、document IDs 或 block mask 表达边界，名称和格式取决于模型接口。
例如 `[1,1,2,2,2]` 的 document-ID attention_mask 不等于普通 0/1 有效位 mask。
Position IDs 按模型要求处理，多模态 M-RoPE 不能用普通 arange 代替。

### 8.4. 文件存储

| 形式 | 保存内容 | 不自动解决的问题 |
| --- | --- | --- |
| JSON/JSONL | 原始记录、messages 或偏好字段 | 模板与监督范围 |
| Parquet/Arrow/HF Dataset | 列式数据或 Dataset 访问接口 | 源字段映射、媒体支持和任务目标 |
| Megatron .bin/.idx | token payload、长度、地址和文档边界 | 通用离线 SFT mask 恢复 |
| 其他 token .bin | 特定 shard 协议 | 不能假定与 Megatron reader 兼容 |

## 9. 框架怎样实现统一

### 9.1. Transformers 与 TRL

Transformers 的 tokenizer/processor 接收标准文本或模型支持的 messages，并执行模型编码；它不负责识别
任意来源的 instruction/from/value 列。Chat template 和具体 processor 决定模型特殊 token 与媒体处理。

TRL 接受 text、messages、prompt/completion 和偏好等任务字段，由相应 Trainer 准备输入；SFT 的
completion_only_loss 与 assistant_only_loss 控制监督区间，不是预训练/SFT 阶段开关。
Assistant mask 依赖模板支持，媒体 mask 和专用目标也需匹配模型与任务。

### 9.2. LLaMA-Factory

```text
Alpaca / ShareGPT / OpenAI 源记录
  → align_dataset / converter
  → _prompt、_response、_system、_tools、_images、_videos、_audios
  → 按 stage 选择训练 DatasetProcessor
      pt  → PretrainDatasetProcessor
      sft → SupervisedDatasetProcessor 或 PackedSupervisedDatasetProcessor
      rm  → PairwiseDatasetProcessor
      kto → FeedbackDatasetProcessor
  → 模板、tokenizer、媒体适配 → labels → collator
```

这里的 DatasetProcessor 是训练数据处理器，不是 HF 的多模态 Processor。
框架将源格式转换与阶段监督分开，不能假定所有模型都采用 HF 默认模板或同一个媒体实现。

### 9.3. 本地 NeMo AutoModel

| 职责 | 实际组件 | 输出/行为 |
| --- | --- | --- |
| LLM 列映射 | ColumnMappedTextInstructionDataset | prompt/completion 或 messages → 公共 formatting |
| LLM 对话 | ChatDataset | 保留工具字段，构建 assistant 监督；支持 reasoning/history 等 mask |
| VLM 源适配 | convert_sharegpt_to_conversation | 统一 conversation 和媒体引用 |
| 普通 VLM 编码 | default_collate_fn / 模型 collator | processor 编码 → build_labels_from_template → shift |
| 逐样本 VLM 编码 | PreTokenizedDatasetWrapper | 在 __getitem__ 中在线加载媒体与编码，构建未 shift 单样本 labels |
| Packed VLM | packing plan + PackedDatasetWrapper | 先估算选样，读取 pack 时编码、模型位置接口、逐样本 shift、拼接 |
| 模型小范围适配 | post_tokenize_hook | 在通用流程中插入差异，避免复制全流程 |

PreTokenizedDatasetWrapper 不等同于离线 token 文件。普通 VLM collator 不生成假的一维 position IDs；packed
多模态路径可调用模型 get_rope_index，保留样本位置语义，再拼接。
LLM、VLM 与 BAGEL 有各自路径，统一的是构建职责和训练契约，不是所有模态共用一个 transform 类。

### 9.4. Qwen 与 DeepSeek 的代表性接入

Qwen3.8-27B 当前发布配置采用 qwen3_5 模型类型，预处理配置指定 Qwen3VLProcessor。
AutoModel 示例复用 transformers.AutoProcessor，并通过公共 VLM collator 构建 labels。
模型名称不能直接当作 processor 类名。

DeepSeek-V4.1 在本地 AutoModel 使用注册的 DeepseekV41Processor，提供兼容 apply_chat_template 的接口，
封装专属图片网格、patch、token span 和类型标记；标准 chat/本地图片以外的工具/thinking 等有明确限制。
统一接口不等于 processor 能力完全相同，也不要求媒体字段同名。

```text
标准 conversation
  → 检查 processor 支持的内容与参数
      ├─ Qwen：复用 HF processor
      └─ DeepSeek：模型专用、HF 接口兼容的 processor
  → 训练侧 labels、长度策略、packing/collate
  → 模型与 runtime 消费各自字段
```

## 10. 对 HyperParallel 的职责分解建议

本文不新增 YAML 配置或宣称未落地能力已支持。设计上应明确以下边界：

```text
Source：读取记录、split/blend/shuffle
  → Format adapter：列名、角色、媒体引用 → 标准样本
  → Model encoder：tokenizer/processor，复用模型已有编码规则
  → Supervision：训练模式、回答范围、reasoning/tool/media/padding 的 labels/mask
  → Packing/Collator：成本、选样、拼接、shift、边界、batch tensor
  → Parallel/Runtime：CP/TP、全局坐标、模型附加输入
```

完整契约至少记录：场景、模态、源格式、标准样本、编码入口、目标/监督范围、labels 是否 shift、padding/EOD、
packing 边界、position 语义和模型专属字段。answer_only_loss_mask 只决定监督范围，不能单独判断训练阶段。

当前目标 checkout 的 dataset.training_mode 与模型专用 pretrain/sft transform 已有实现，但支持范围仍以
具体模型为准；通用图文编码不能自动替代图像-caption 预训练协议。相关设计见：

- [数据模块总览](README.md)
- [Omni 编码接口](omni/omni_transform.py)
- [DeepSeek-V4.1 数据 transform](../models/deepseek_v41/adapter/data/transform_fn.py)
- [Dataset 配置](../trainer/config/data.py)

## 11. 外部参考

以下为接口和格式依据，不代表本文示例可不经模型配置直接执行：

- [Transformers chat templates](https://huggingface.co/docs/transformers/main/chat_templating)：训练模板、生成前缀、续写。
- [Transformers multimodal templates](https://huggingface.co/docs/transformers/main/chat_templating_multimodal)：媒体消息与 processor。
- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)：文本、对话、工具、prompt/completion 和偏好类型。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)：监督区间与预编码输入。
- [LLaMA-Factory converter](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/converter.py)：源格式归一化。
- [LLaMA-Factory loader](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/loader.py)：按训练阶段选择处理器。
- [Qwen3.8-27B 配置](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)：模型架构。
- [Qwen3.8-27B 预处理配置](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/preprocessor_config.json)：processor 类型。

