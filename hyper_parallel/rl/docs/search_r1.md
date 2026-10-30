# HotpotQA 文件搜索

See [prepare_data.py](../examples/search_r1/prepare_data.py), [training config](../examples/search_r1/configs/qwen3_4b_search_r1.yaml), and [run_search_r1.sh](../examples/search_r1/scripts/run_search_r1.sh).

在宿主机运行 `launcher.py`，由宿主机启动可信训练容器及同级、无特权 Codex 容器。
不使用 Docker-in-Docker。CLI指令文件使用候选容器内的`/codex-home`路径，不能引用训练容器的`/results`路径。
Codex 只能读取该题候选文章，通过 shell 搜索；
不挂载数据集标签、其他会话、NPU 或 Docker socket。每个会话在正式候选容器中、Codex启动前实际探测隔离边界。
训练镜像保持`hyper-rl:v0.23.0.post1-code-agent-arm64`，候选默认使用本地构建的
`hyper-rl-codex:0.152.1-arm64`。`--candidate-image`可显式选择其他经过核验的CLI镜像。
候选镜像仅包含Codex、Python和搜索工具，不包含模型推理库；
两类容器的权限和挂载保持独立，见 [镜像说明](../docker/README.md#search-r1-的独立-codex-候选容器)。

## 数据与训练

当前使用官方 `hotpotqa/hotpot_qa` 的 **Distractor** 数据，保留每题原始候选文章与句级引用。
不使用为MemAgent构造的长上下文数据。训练题从官方train的90,447题中抽取，留出题来自独立validation源；
准备程序检查跨源ID/规范化问题重叠，记录SHA256并禁止覆盖已有数据。

当前八卡两步配方沿用GSM8K已验收的Qwen3-4B训练/推理能力，搜索组合仍须独立完成NPU验收：

| 项目 | 配置 |
| --- | --- |
| 数据 | 准备6,400训练题和128留出题；两步只消费前256道训练题 |
| 每步预算 | 128题×8个episode，最多10次LLM调用/episode，包含引用修正 |
| 推理 | native vLLM TP2×DP4，0.8显存比例，256序列/副本，图执行/prefix/chunked开启 |
| 调度 | Gateway上限1024，每rank最多128个episode；宿主默认同时运行128个候选容器 |
| 容器资源 | 每候选最多2 CPU、2 GiB内存；可用`--agent-workers`改变并发容器上限 |
| 调用长度 | 每次输出最多1024 tokens，完整模型上下文最多8192 tokens |
| 训练 | DP分片8、TP1，逐微批梯度同步，生成期间卸载训练状态；训练本身不启用Actor CPU offload |
| 概率计算 | old由Actor重算，保留采样概率；Actor/old/reference动态预算均8192，selected_chunked分块512 |
| 优化器 | 每rank mini batch上限1280调用行，覆盖128个episode×10次调用，每步一次更新 |
| 奖励 | 沿用旧规则：0.70×答案F1 + 0.20×答案EM + 0.08×支持句F1 + 0.02×格式奖励；引用检查不变 |
| 验证/保存 | 两步跑通默认关闭；不将短实验称为50步或验证集验收 |

`max_prompt_length=1024`约束初始问题，不表示CLI指令、工具和历史总共只有1024 tokens。
动态预算控制普通微批分组，超预算完整调用单独成批；单样本显存容量仍需按上下文上限验收。
该配方保留8192预算；分块概率复用已有RL实现，未新增训练核心。
CPU准备重叠保持关闭：当前搜索使用自定义program及容器execution，不在原生本地CLI重叠的支持范围内。
候选容器并发与vLLM序列上限是两项资源限制，128个容器不能保证推理端达到256序列/副本。

在仓库根目录准备数据（原始文件需已下载）：

```bash
python3 hyper_parallel/rl/examples/search_r1/prepare_data.py \
  --input /absolute/raw/train-00000-of-00002.parquet \
  --additional-input /absolute/raw/train-00001-of-00002.parquet \
  --validation-input /absolute/raw/validation-00000-of-00001.parquet \
  --train-samples 6400 --test-samples 128 --seed 42 \
  --output-dir /absolute/path/to/new-prepared-hotpotqa
```

有旧留出集时增加`--preserve-test /absolute/old/test.parquet`。不可省略训练源与独立validation源：
历史准备工具的默认下载地址为validation，单源拆分不等于官方train/validation划分。

在`hyper_parallel/rl/`下，确认8卡空闲后运行：

```bash
python3 examples/search_r1/launcher.py --devices 0,1,2,3,4,5,6,7 --steps 2 \
  --model /absolute/path/to/Qwen3-4B --data /absolute/path/to/prepared-hotpotqa \
  --backend-privileged --agent-workers 128 --output /absolute/path/to/new-results
```

启动器及`scripts/run_search_r1.sh DEVICE_IDS STEPS DATA_DIRECTORY --model MODEL_DIRECTORY`支持末尾YAML覆盖。
启动时复用RL插件安装和已接受的采样器stream修补、DP协调器冷启动超时修补；依赖不匹配则失败。
本次不会覆盖已有实验目录；`container.json`记录实际训练容器ID，供宿主监控识别设备进程归属。
`--backend-privileged`只用于可信训练后端，不用于Codex候选容器。
训练HCCL端口为`62800-62900`，vLLM为`62600-62799`；设备检查与占用监控仍由实验运行环境负责。

默认验收检查目标策略版本、非零梯度、优化器更新、逐会话隔离与真实上下文。
`--strict-acceptance`保留旧的额外验收要求：严格概率一致、验证和checkpoint重载；
使用它时须相应配置hyper推理实现、consistency、验证和保存，不能把默认native短实验误标为严格一致性通过。
`--resume /原输出目录/checkpoints/step_N`要求完整checkpoint标记及相同数据哈希。

## 提示词与上下文

原始问题来自数据集的`question`，文章正文来自`context`；标准答案和supporting-facts标签不提供给候选进程。
任务模板只说明证据来源、绝对文章路径、工具示例和`ANSWER/SOURCES`格式，不强制第一次回复调用工具，
也不指定逐步搜索策略。系统指令要求依据实际读到的证据回答。奖励公式及引用检查保持不变。
启用thinking时使用vLLM已有的`reasoning_parser: qwen3`，由现有Codex协议分别传递推理与最终答案，
避免将`<think>`正文送入两行答案检查。原始生成tokens仍完整保留；不通过放宽奖励格式解决协议问题。

Search-R1设置`compact_harness_context: true`和`preserve_environment_context: true`：
过滤无关技能目录，保留Codex原始环境和权限消息，包括工作目录`/work`及shell。
候选容器的文章固定挂载在`/work/articles`；命令示例使用绝对路径，避免切换workdir后重复拼接articles。
GSM8K的精简配方仍使用默认过滤策略，不共用文件搜索任务的环境保留要求。

工具白名单仅保留`exec_command`与`write_stdin`，参数说明及历史调用/观察保留；
最后一次常规调用和引用修正显式禁用工具，Gateway允许这类请求没有工具声明。
工具返回值原样交给Codex和模型；接入层不解析shell命令、退出码或错误文本，
不追加重试指导，不因无匹配或重复命令提前中止。仅记录已见工具输出用于引用检查。

## 正确性与故障边界

- 每轮分别训练真实 `P_i + A_i`；`training-audit.json` 核对 token、mask、采样 logprob。
  不要求相邻调用前缀连续，不用重新编码后的历史替换训练输入。
- 当前native配方记录训推概率差异，不宣称逐token严格相等；严格一致性profile仍保留为显式可选路径。
- 训练 reward 和评测按完整 episode 统计，不按模型调用次数重复加权。
- 最后一个常规模型调用禁用工具，另预留一次真实引用修正调用；全部动作保留。
- 模型输出token耗尽、Gateway调用次数耗尽及明确上下文超限复用通用Codex终态判定，
  核验完成后保留原始轨迹并记零奖励，不再调用引用修正。
  容器异常退出、缺失最终消息但无完整预算证据，以及未知通信故障仍拒绝更新。
- 已确认的模型预算用尽、无效引用是任务失败，保留真实动作并记零奖励。
  没有任何采样动作则报错，不能制造空训练样本。
- `failure_origin`、`failure_reason`、`trainable` 分别记录责任、原因和训练资格。
  实际执行服务/容器故障、HTTP/网络错误、未经归因的上下文超限及未知失败中止整次更新；
  不补零奖励、不丢弃组成员、不只屏蔽 loss，避免污染 GRPO 优势。
- shell命令的非零退出、路径/参数错误和无匹配都是工具观察，由模型决定后续行动，受总预算限制。
  工具输出中的退出码不等于Codex进程或容器的退出码；后者仍按真实执行结果核验。
- Hermes 证据由固定版本 vLLM 插件记录：真实生成 token、解码文本、引擎文本、parser 输入和输出。
  证据一致的非法工具 JSON 会收到明确格式反馈并重新采样，计入总模型调用预算；不执行、不修复原命令。
  恢复成功仍按最终任务结果评分。合法原文解析失败或证据不一致则中止更新。
- 格式恢复调用也占用原有预算，因此可能用掉原计划预留的引用修正调用；不会额外扩充预算。
- `training-audit.json` 保留原始前缀不匹配数量与最多三个 token 差异样本；训练仍使用真实逐调用 token。
- `training-acceptance.json` 只有在指定优化步数、权重同步及全部会话审计通过后才产生。
  引用存在/可见不等于语义正确；短程通过也不保证任意长程训练无故障。

## 文件职责

- `workspace.py`：只导出当前题目的候选文章。
- `agent.py`：任务结果、引用检查与修正、逐调用训练轨迹。
- `container_runtime.py`：宿主交接、文章隔离探针与搜索预算反馈；调用共用工作区和Codex执行器，仅依赖标准库。
- `rl/agentic/envs/docker_workspace.py`与`rl/agentic/codex/container_execution.py`：共用容器生命周期、会话转发和CLI执行。
- `launcher.py`：宿主机训练入口、数据校验与训练验收。
- 通用 token/mask/logprob 审计位于 `rl/agentic/core/program_runner.py`。

隔离检查仍在每个正式会话启动前执行，独立冒烟入口放在测试目录。

在宿主机的 `rl/` 目录执行无模型隔离检查（不使用 NPU）：

```bash
python3 tests/st/search_r1_isolation.py \
  --output output/search_r1/search_r1_isolation_check
```

输出目录必须是新目录。该检查验证容器边界与 Codex 启动，不代替真实模型训练测试。
`candidate-runtime.json`记录实际候选镜像身份；逐会话`execution-timing.json`记录各执行阶段耗时。
历史实验详情以各输出目录的日志为准。
