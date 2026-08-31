# HyperParallel 平台层「以 torch 为主线、mindspore 打补丁」重构方案

> 状态：方案稿 v2（基于当前 `platform/` 双实现现状调研；2026-08-26 修订）
> 适用仓库分支：`code_docs_dtensor_debug_test_case_level`（master 为基线）
> 撰写日期：2026-08-25；修订日期：2026-08-26

---

## 0. 版本说明

- **v2（本稿）**：以「彻底去掉接缝 + runtime monkeypatch」为**主方案**（原备选 A 升级为主）。
- v1 主推方案「保留 `get_platform()` 接缝 + MindSpore 继承 torch」降级为**备选方案 B**。
- 变更原因：以 torch 主线为最终目标时，`get_platform()` 这个接缝本身就是「双实现」思维留下的结构；彻底去掉它，core 层直接看到 torch，是最彻底的「torch 主线」。代价是 monkeypatch 覆盖面大、调试难，因此本稿将其风险与缓解做了重点展开。
- 本版本新增：§3.4 最小示例（monkey patch 怎么实现）、§3.5「monkey patch 的分情形处理策略」。均基于 2026-08-25 的 `autograd_compat.py`（项目现存的 patch 先例）。
- **2026-08-26 修订**：
  - 删除死代码 `hyper_parallel/platform/mindspore/platform_graph.py`（`MindSporeGraphPlatform` 从未被引用）。平台层「静态图」不再保留；活跃的图下沉是**算子层** `REG_GRAPH_MODE_OP` 注册，与平台 patch 无关（见 §3.2 Step 4）。
  - §6 新增「patch 粒度原则」：patch 粒度 = core 实际消费的门面符号，而非逐个 torch 接口。
  - 新增 §3.6「用户可见的使用方式变化」，说明默认平台从 MS 翻转为 torch 后，用 MindSpore 需显式声明且要在 import 阶段生效，并列出三种 MS 初始化/声明方式（A 隐式 / B 显式 / C 显式入口函数，待定）。
  - §3.6.4 补充「用 MindSpore 必须能 import torch」：区分 torch 为软件依赖（硬性）还是运行引擎（不必），及三条规避路线。
  - 新增 §6 附录「为什么 monkey patch 屏蔽了 torch，仍需安装 torch」：阐明 patch 屏蔽的是函数实现、非模块 import。基于 2026-08-26 对 §3.4 示例的推演。
  - **2026-08-27 修订**：§6 末尾追加「彻底取消 torch 依赖」小节——结论分两半：**机制是独立新增**（`sys.modules` 拦截假 torch，不改现有 patch/core/torch 主线路径，属纯增量）；**内容不是补丁而是重建**（MS 无 torch 进程里散落的 `torch.cat/zeros/empty_like`、`isinstance(..., torch.Tensor/ModuleList/dtype)`、类定义基类 `__mro__` 同样执行且无平台判断保护，假 torch 须重建一层 MS 版 torch 表面积，含 `torch_npu`/`safetensors.torch` 等三方插件，工作量≈第二个平台实现）。故「去 torch」是本方案之外的一整块工程，pragmat 默认是维持「MS 环境装 torch」（§3.6.4 路线 2）。**注：`DTensorBase` 类继承各自框架原生类型且经 `platform` 门面消费，不属障碍（patch 可干净处理），障碍是上述散落且无平台保护的直接 torch 调用/类型判断/类基类。**
  - 新增 §3.5「monkey patch 的分情形处理策略」：按「谁有实现 + 接口是否兼容」分四类（两边有且兼容=纯改名；两边有但不兼容=适配 shim；仅 torch=patch 兜底；仅 MS=上收主线或 torch stub），并对 §6 清单逐项标注归属。
  - **2026-08-27 修订**：§3.5 情形 3 由「能力标志 + core 分支」改为「patch 兜底、core 默认零分支」——MS 缺席时用报错 stub 或退化默认兜住，仅当退化策略由调用点决定时才进 core 分支；情形 4 拆为 4a/4b——torch 无调用路径则只在 MS 实现、零残留（4a），有用户可达路径才在 torch 留显式 stub（4b）。总原则：分支=在多个合法算法间选择，patch 报错=该语义做不了。
  - **2026-08-27 修订**：新增 §1.3「覆盖范围」+ §2 范围约束——本方案只覆盖 `platform/` 双实现及经平台门面消费的 `core/` 符号；`trainer/`、`models/`、`infer/`、`data/`、`integration/`、`auto_parallel/` 等纯 torch 实现**不在范围**。相应修订 §6：衡量「假 torch 规模」只统计平台 + core 范围内符号，全库 255 处 import 仅作背景，范围外纯 torch 上层不计入。
  - **2026-08-27 修订**：§6 新增风险「并非所有 torch 实现都有完整 MS 对应 → MS 支持边界模糊」，缓解=建立能力矩阵（§3.5 四类 + `hp_capability`）作为 MS 支持范围单一事实来源，缺失项显式报错而非静默跳过。
  - **2026-08-27 修订**：新增备选方案 E「基于 MSAdapter/MindTorch 复用现成 torch 兼容层」——MSAdapter 是现成的 MS 版 torch 兼容层（`import msadapter.pytorch as torch`），正是 §6 路线 A 想「从头造」的虚拟 torch。结论：它**大幅抵扣广度成本与「去 torch」重建量**（§3.5 情形 1/2、§6 路线 A），但**碰不到核心难点**（`differentiable_*` 图式 vs 磁带式 autograd、`DTensorBase`/`_OP_DISPATCHER` 私有分发、昇腾 `custom_ops`/`REG_GRAPH_MODE_OP`），并引入语义/精度保真、版本耦合、加剧 §6 边界模糊、三方治理等新风险；不改变 torch 主线方向，属实现细节加速器，建议先做聚焦 PoC。相应在 §6 路线 A 补一句「重建成本可由 MSAdapter 抵扣，但 autograd/DTensor 核心仍需自建 + PoC」。
  - **2026-08-27 修订**：新增 §5.1「按符号增量迁移（strangler：改一个、删一个）」——把 P3/P4 主体分解为「一次只做一个符号」的独立小任务。要点：①一个符号任务的终态是**删除** `platform.X`（改全部调用方后零调用→删），转发器仅大扇出符号临时用；②删符号同时须定 core 换成 (i) 直接 `torch.*`（patch 真 torch 命名空间）还是 (ii) `hp` 薄门面（patch 自有门面），决定 MS patch 落点；③符号按种类分难度——值/函数最易（可事后重绑）、被继承的基类（`DTensorBase`）最难（MRO 在 `class DTensor(DTensorBase)` 定义时定死，须「子类定义前选对」而非事后 patch）；④标准任务模板 5 步 + MS-only 36 符号走情形 4 不套模板 + 共存期同框架不变量。建议先拿值/函数符号开路，基类根留到时序纪律立起后再做。
  - **2026-08-27 修订**：新增 §9「附：platform 符号『改一个删一个』迁移清单」——枚举 core 实际消费的约 130 个 `platform.X` 门面符号，按 §5.1.3 分六层（A 值/函数 ~90 个最易 / B 类型对象 ~8 须对象同一性 / C 被继承基类 2 个最难 / D 平台标志改能力标志 / E `differentiable_*` ~11 名字可重绑但 shim 最重 / F MS-only 走情形 4）。每层给难度、patch 方式、删除方式、§3.5 归属，并附小结表与迁移次序（A→B→E→C，D 贯穿、F 独立）。计数为 core 内引用扇出，供 §5.1 strangler 逐个取用。
  - **2026-08-27 修订**：新增 §1.4「当前做法的合理性分析：从 torch 开发者视角」——现有架构要求 core 消费的每个操作在 `platform/torch` 都有同名方法（供 MS override）。以 `TorchPlatform` 127 方法（41 个 ≤3 行）为据，分三类：(a) 纯透传（`get_rank`→`dist.get_rank()`、`empty`→`torch.empty`）、(b) 改名/词汇 shim（`get_cells_and_names`→`named_modules()`，带 MS「Cell」词汇）、(c) 真跨框架公共逻辑（`get_op_name`、`differentiable_*`、`custom_ops`）。结论：对 torch 开发者 (a)+(b) 是**双平台接缝强加的抽象税**（唯一受益方是 MS，torch 侧要付「一个操作三处改动 + MS 风味别名 + 无谓间接」），仅 (c) 值得抽象但不必以 Platform 接缝形态存在；主方案把这笔成本从 torch 主线移出、定位到 `patch.py`，让承担者与受益者一致。
  - **2026-08-28 修订**：改写 §3.4 示例——`get_rank` 因双平台均为一行、语义兼容，改为直接 patch `dist.get_rank`（真 torch 命名空间，`dist.get_rank = _ms_get_rank`），core 直接调 `dist.get_rank()` 不再留 wrapper；`all_gather_concat` 因需 group/返回值翻译仍走 `hp` 门面（`hp_comm.all_gather_concat = _ms_all_gather_concat`）。同步把 ⚠️ 取舍段从「改门面推荐 / 改 torch 本体不建议」改为「按符号性质分别选用」——语义兼容的一行原语→直接 patch torch 本体，需翻译的高层函数→走门面，对应 §5.1.2 的 (i)/(ii)。§5.1.2 表格中「(i) 直接 torch」一行的措辞同步去掉「激进/不建议」的绝对判断。
  - **2026-08-28 修订**：调整 §3 小节顺序——原 §3.6「monkey patch 的分情形处理策略（谁实现、接口是否兼容）」上移至 §3.4 之后作为**新 §3.5**（先讲「谁实现、四类情形」的总体决策，再进细节）；原 §3.5「接口不兼容的分档处理」顺延为 **§3.6**；原 §3.6「用户可见的使用方式变化」（含 3.6.1/3.6.2/3.6.4）顺延为 **§3.7**（3.7.1/3.7.2/3.7.4）。全文对这三节的交叉引用已同步更新（情形→§3.5、档位→§3.6、用户可见/时序/装 torch→§3.7.x）。
  - **2026-08-28 修订**：删除原 §3.6「接口不兼容的分档处理（档位 1/2/3/4）」——档位分类即 §3.5 四类情形（情形 2 的若干穿法 + 情形 3）的另一套命名，与 §3.5 决策规则汇总表重复；删后全文「档位」引用并入 §3.5 情形（情形 2 / 情形 3），原 §3.7「用户可见的使用方式变化」顺延为新 **§3.6**（3.6.1/3.6.2/3.6.4），交叉引用同步更新。

---

## 1. 背景与现状

`hyper_parallel/platform/` 当前的架构是 **双实现 + 单一分发器**：

- `platform/platform.py`（约 67KB）定义抽象的 `Platform` 基类，契约面高达 **100+ 个属性**（collective、`differentiable_*` 异步族、tensor 工厂、checkpoint/swap、stream/random、param 自省、custom_ops 等）。
- `platform/mindspore/platform.py`（约 83KB）与 `platform/torch/platform.py`（约 64KB）是**两套各自独立、平行实现**的具体平台类。
- 分发逻辑 `get_platform()`（`platform/platform.py:99-126`）返回全局单例，**默认 MindSpore**，可被环境变量 `HYPER_PARALLEL_PLATFORM` 覆盖，失败时 `ImportError` 回退 torch：

```python
# platform.py:123-126（当前默认）
try:
    return get_mindspore_platform()
except ImportError:
    return get_torch_platform()
```

- 整个代码库（`core/`、`trainer/`、`models/`、`collectives/`、`integration/`、`dmodule/`）几乎全部通过 `from hyper_parallel import get_platform` 拿到单例，再在模块顶层绑定 `Tensor = platform.Tensor`、`platform.get_rank()` 等使用。**`core/` 层不直接 `import torch` 或 `import mindspore`**，而是通过接缝与框架解耦。

### 1.1 两套实现真实分叉点（决定「打补丁」可行性）

| 层面 | torch | mindspore | 分叉程度 |
|---|---|---|---|
| collective 后端 | `torch.distributed`（`ProcessGroup` 句柄） | `mindspore.communication` / `mint.distributed`（**group 是字符串**） | 大 |
| `differentiable_*` 异步族 | 真实 autograd `Function` 子类（`_TorchAsyncA2AFunction`、`_TorchAsyncAllGatherFunction`、`_AsyncA2ALazyBwd`、`_TorchSyncHookFunction`、`_TorchP2PExchangeFunction`，`torch/platform.py:91-368`） | `AsyncCollectiveTensor`（`mindspore/platform.py:174`）+ `_normalize_*` 归一化 + `_MSAsync*` 族（`mindspore/platform.py:376-727`） | **最大，是 MS 多 19KB 的主因** |
| `DTensorBase` | 继承 `torch.Tensor`，走 `__torch_function__` 分发 | 继承 `ms.Tensor`，用 `_DisableMsDispatchMode` | 大 |
| recompute（平台层） | 无 | `recompute_session_ctx`/`recompute_handle`/`recompute_handle_collector_ctx`/`clear_recompute_session`（`mindspore/platform.py:1838-1854`） | 仅 MS 有 |
| 参数初始化 | 无（基类 stub） | `init_parameters`（`mindspore/platform.py:1036`）用 MS 惰性 `initializer` | 仅 MS 有 |
| `custom_ops` | 空 stub（`raise NotImplementedError`） | 20+ 个 `.cc` kernel 的 `MindSporeCustomOps` | 仅 MS 有 |
| checkpoint | `safetensors.torch` | MS `ckpt_format` | 语义不同 |
| `checkpoint_exclude_wrapper` | 无 | 有（`mindspore/platform.py:1787`） | 仅 MS 有 |
| `str_to_dtype` / `list_to_size` | 返回 `torch.dtype` / `torch.Size` | 返回 `ms.Type` / tuple | 语义不同 |

### 1.2 类结构与硬编码现状

- `MindSporePlatform(Platform)`（`mindspore/platform.py:728`）与 `TorchPlatform(Platform)`（`torch/platform.py:433`）**各自设置同类 class 属性**：

```python
# torch/platform.py:435-444
Tensor = Tensor; tensor = torch.tensor; Parameter = Parameter; Module = Module
DTensorBase = DTensorBase; PipelineStageBase = PipelineStageBase
platform_type = PlatformType.PYTORCH; tensor_dtype = torch; dtype = torch.dtype
Function = torch.autograd.Function

# mindspore/platform.py:730-739
Tensor = Tensor; tensor = Tensor; Parameter = Parameter; Module = Cell
DTensorBase = DTensorBase; PipelineStageBase = PipelineStageBase
platform_type = PlatformType.MINDSPORE; tensor_dtype = mstype; dtype = ms.Type
Function = _Function
```

- `PlatformType` 枚举已存在（`platform/platform.py:60`）。
- 硬编码类型判断：
  - `mindspore/fully_shard/scheduler.py:51`：`if not isinstance(self.platform, MindSporePlatform):`
  - `torch/fully_shard/scheduler.py:49`：`if not isinstance(self.platform, TorchPlatform):`
  - `distributed_checkpoint/offline_transform.py:457-480`：直接把 `"torch"/"mindspore"` 字符串写死并入 `HYPER_PARALLEL_PLATFORM`。

### 1.3 覆盖范围（本次重构只针对什么）

**范围：只覆盖 `hyper_parallel/platform/` 内「当前由接缝驱动的双实现功能」**——即 `platform/platform.py` 的抽象契约下、`platform/mindspore/` 与 `platform/torch/` 各自平行实现的那批能力（collective、`differentiable_*` 异步族、tensor/dtype 工厂、DTensorBase、checkpoint/swap、`fully_shard/`、`activation_checkpoint/`、`pipeline_parallel/`、`custom_ops`、`init_parameters`、recompute 等）。

**不在此次范围（排除项）**：
- **`trainer/`**（`vl_trainer.py`、`base.py`、`llm_trainer.py`、`callbacks/` 等）—— 纯 torch 实现，非平台双实现。
- **`models/`**（`qwen3_5*`、`glm5*`、`modules/` 等）—— 纯 torch 实现，非平台双实现。
- **`infer/`、`data/`、`integration/`、`auto_parallel/`、`core/multicore/`** 等 —— 纯 torch 实现，非平台双实现；不在本方案「去接缝 / 打补丁」的覆盖范围内。

**为什么必须明确此边界**：
- 本方案的「torch 主线 + MS patch」只针对**平台层双实现**（§1.1 那张分叉表的对象）。`trainer`/`models` 等纯 torch 上层**本来就只跑 torch**,不存在「双实现」问题,不被本方案的 patch 影响。
- 判断一项功能是否属于本范围:看它是否**在 `platform/mindspore/` 与 `platform/torch/` 里各有一份实现**、且被 `Platform` 契约/`get_platform()` 消费。**只有这类才需要 patch 重绑**;纯 torch 上层自然成立。
- ⚠️ 排除项在本范围之外,不代表它们在物理上不 import torch——它们照样 `import torch`、调用 `torch.cat` 等,但这些是**torch 主线的正常使用**,与「平台双实现要不要 patch」无关。

> 因此 §6 的「必须进 patch 替换表」清单、§7 关于「假 torch 要覆盖多大」的讨论,都**只统计平台层 + core 经平台门面消费的那几十个符号**,而非全库所有 `import torch` 文件。后述 §7 中引用的 `trainer/`、`infer/`、`offline_transform.py` 等纯 torch 用例,仅用于说明「全库层面有多少 torch 使用」这一背景,**不属本方案范围**,不能作为衡量 patch/假 torch 规模的依据。

### 1.4 当前做法的合理性分析：从 torch 开发者视角看「封装成 `platform/torch` 方法」是否合理

**提问角度**：现有架构要求 core 消费的每一个操作，在 `platform/torch/platform.py` 里都有一个对应方法（因为 `MindSporePlatform` 要有同名方法可 override）。站在**一个只写 torch 的开发者**立场：把这些操作封装成 `platform/torch` 的方法，对他而言合理吗？

**先看事实**（`TorchPlatform` 共 127 个方法，其中 41 个方法体 ≤3 行）。按对 torch 开发者的「自然度」分三类：

| 类别 | 典型例子（torch 侧实现） | 对 torch 开发者是否自然 |
|---|---|---|
| **(a) 纯透传** | `get_rank` → `return dist.get_rank()`；`empty` → `return torch.empty(...)`；`is_tensor` → `return isinstance(obj, Tensor)`；`zeros`/`ones`/`rand`/`randn` 同理 | **不自然**。torch 开发者会直接写 `dist.get_rank()`、`torch.empty(...)`，绝不会为此建一个平台方法。封装在此**零收益**——只是给 MS 留一个 override 点 |
| **(b) 改名/词汇 shim** | `get_cells_and_names` → `return cell.named_modules()`；`get_cell_construct`、`get_modules` 等 | **反自然**。方法名带 MindSpore 词汇（「Cell」），torch 里就叫 `named_modules()`。torch 开发者要记一套**别名**去调本该直呼的 torch API |
| **(c) 真正跨框架的公共逻辑** | `get_op_name`（22 行，归一化 `OpOverload`/`OpOverloadPacket` 命名）；`differentiable_*` 异步族；`custom_ops` | **部分合理**。这类逻辑即便在纯 torch 代码里也需要一个公共函数；但 torch 开发者会把它放进普通 util 模块，而**不是挂在一个 100+ 方法的 `Platform` 类、藏在 `get_platform()` 接缝后面** |

**结论：对 torch 开发者，绝大多数封装（(a)+(b)）是不合理的——它是「双平台接缝」强加的抽象税，而非 torch 自身需要。**

- **谁受益**：这些透传/改名方法的**唯一受益方是 MindSpore**（它需要一个同名 override 点）。torch 主线为「保持接缝对称」付出了：多一层无谓间接、一套 MS 风味的别名、以及「新加一个 torch 用法就得先去平台类补一个方法/对齐签名」的持续负担。
- **代价的具体形态**：torch 开发者想用一个尚未被封装的 torch API，不能直接调，得先在 `platform.py` 基类声明契约、在 `torch/platform.py` 写透传、在 `mindspore/platform.py` 写对应——**一个本可一行搞定的调用，变成三处改动**。这正是「双实现」思维对 torch 日常开发的直接摩擦。
- **仅 (c) 类值得保留抽象**，但即便如此，torch 开发者的自然表达是「普通函数/类」，不是「Platform 方法 + 接缝」。

**这如何支撑主方案**：把 torch 变主线、MS 改用 patch，正是**免除 torch 开发者这笔抽象税**——core 直接写 `dist.get_rank()`/`torch.empty(...)`（自然、无别名、无三处改动），而「给 MS 留 override 点」这件事被收敛到 `patch.py` 一处，只有真正分叉的 (c) 类才需要显式处理（§3.5 情形 2）。换言之，**当前做法把「MS 的 override 需求」平摊成了「torch 开发者每个操作都要封装」的普遍成本；主方案把这笔成本从 torch 主线移出、定位到 MS patch**，让承担成本的人和受益的人一致。

> 注：这不否定「接缝在双实现时代是合理设计」——它确实用一层抽象换来了两个框架的并存。本节只回答「**若目标是 torch 主线**，继续要求 torch 开发者封装一切是否合理」，答案是否定的：对 (a)/(b) 而言那是纯税，对 (c) 而言也不必以 Platform 接缝的形态存在。

---

## 2. 目标与约束

**目标**
1. torch 成为默认主线与「唯一实现」，未来新特性只在 torch 写一遍。
2. mindspore 通过「打补丁」而非平行实现来支持，消除 `platform/` 下双份代码与接缝概念。
3. 让 `core/`、`trainer/`、`models/` 等上层**直接 `import torch`**，不再经过 `get_platform()` 接缝。

**约束**
- 尽量不改动 `hyper_parallel/__init__.py` 的公共 `__all__` 导出。
- **范围约束**：本方案只覆盖 `platform/` 内的平台双实现功能（见 §1.3）；`trainer/`、`models/`、`infer/`、`data/`、`integration/`、`auto_parallel/` 等**纯 torch 实现不在本次范围**——它们本来只跑 torch，不经本方案的接缝/补丁，未被 patch 影响。目标中的「上层直接 import torch」（§目标 3）指 `core/` 及经平台消费的符号，不含这些纯 torch 上层文件的改造。

---

## 3. 主推方案：彻底去掉 `get_platform()` 接缝 + runtime monkeypatch 支持 mindspore

### 3.1 架构模型

```
core/ 等全部上层
   │  直接 import torch（不再有 platform接缝）
   ▼
torch.*  ←── hyper_parallel.platform.mindspore.patch（仅 MS 环境 import 时触发）
            │  运行时替换 torch.distributed / torch.autograd.Function / 平台方法等
            ▼
        mindspore 运行时（PyNative；Graph 模式由算子层 REG_GRAPH_MODE_OP 下沉负责）
```

- **默认路径**：core 直接写 `import torch`、`torch.distributed`、`torch.autograd.Function`，无任何中间层。
- **MindSpore 路径**：唯一的介入点是一个 **patch 模块**（如 `hyper_parallel/platform/mindspore/patch.py`），在 `HYPER_PARALLEL_PLATFORM=mindspore` 时被 import，对 `torch.*` 及平台方法做运行时替换。

### 3.2 核心改动

**Step 0 — core 层去除接缝，改为直接 import torch**

- 删除所有 `from hyper_parallel import get_platform` / `from hyper_parallel.platform import get_platform` 及 `platform = get_platform()` 绑定。
- 把模块顶层 `Tensor = platform.Tensor`、`platform.get_rank()` 等替换为 `import torch` + `torch.*` 直接调用。
- `hyper_parallel/__init__.py` 保留 `get_platform` 导出作为**兼容 shim**（返回一个 torch 主线的薄适配对象），避免破坏外部存量调用；主路径不再使用。

**Step 1 — 编写 mindspore patch 模块（核心增量）**

`hyper_parallel/platform/mindspore/patch.py`，在 MS 环境 import 时：

```python
# 伪代码示意
def enable_mindspore_patch():
    # 1) 复用现有 autograd_compat.py 的补丁，作为子集
    from .autograd_compat import enable_mindspore_backward_compat
    enable_mindspore_backward_compat()

    # 2) 替换 torch.distributed 关键原语 → mindspore.communication
    import torch.distributed as dist
    patch_dist_comm(dist)            # get_rank / all_gather / all_reduce / all_to_all / reduce_scatter ...

    # 3) 替换 torch.autograd.Function 语义 → MindSpore 可微异步族
    patch_autograd()                 # 对应原 _MSAsync* / AsyncCollectiveTensor

    # 4) 替换 torch 的 dtype / tensor 工厂 → ms.dtype / ms.ops
    patch_tensor_api()

    # 5) 接管 custom_ops（原 MindSporeCustomOps，20+ .cc kernel）
    patch_custom_ops()
```

- patch 模块**只在 `HYPER_PARALLEL_PLATFORM=mindspore` 时自动 import**（通过 `hyper_parallel/__init__.py` 的初始化钩子或环境变量开关），**默认路径绝不引入**，避免 torch 主线额外负担。

**Step 2 — 迁移 MS 特有实现进可调用 patch**

- 原 `mindspore/platform.py` 的 `differentiable_*` 异步族、`DTensorBase`、`custom_ops`、`init_parameters`、`recompute_*`、`checkpoint_exclude_wrapper` 逻辑**不删除**，改为注册到 patch 模块的替换表。
- 原 `mindspore/fully_shard/`、`mindspore/activation_checkpoint/`、`mindspore/pipeline_parallel/` 下「仅 MS 有」的功能（如 `backward.py` 30KB 的 recompute 驱动 chunked backward）同样进 patch 表，按需覆盖 torch 同名类。

**Step 3 — 消除硬编码类型判断**

- 原 `get_platform()` 接缝去掉后，`scheduler.py:51`/`:49` 的 `isinstance(self.platform, MindSporePlatform/TorchPlatform)` 不再有意义

**Step 4 — 平台层静态图为死代码，需删除**

原 `hyper_parallel/platform/mindspore/platform_graph.py` 的 `MindSporeGraphPlatform`（`platform_graph.py:21`）**未被任何路径引用**——既没接进 `get_platform()`，也无显式 import（唯一提及仅在 `.agent/skills/` 指导文档），属死代码，需要删除

### 3.3 此方案的关键前提

**monkeypatch 覆盖范围与顺序**

- 需要在 patch 中替换 `torch.distributed`、`torch.autograd.Function`、dtype/tensor 工厂、`custom_ops`、`DTensorBase` 等多个核心件。
- patch 应用**顺序敏感**：必须在任何 core 模块使用这些符号**之前**完成替换。建议在 `hyper_parallel/__init__.py` 的 import 阶段：若检测到 MS 环境，先 `import hyper_parallel.platform.mindspore.patch`，再继续后续 `import`。否则会出现「一半 torch 一半 MS」的混合状态。

**调试性退化**

- 错误栈会指向 patch 层而非真实逻辑，排查梯度/通信问题时难度上升。
- 缓解：在 patch 模块中保留**逐条替换记录**（日志/可查询状态），并提供 `HP_DISABLE_PATCH=1` 环境变量一键绕过以便对照 torch 原生行为。

**与现有接缝消费方式的冲突**

- `core/` 需要对「`platform.xxx()`」的调用进行修改，是**大面积替换**

### 3.4 最小示例：monkey patch 怎么实现

**① torch 主线 core（两个平台下这份代码完全相同）**

```python
# hyper_parallel/comm.py   <-- torch-first 主线的核心文件，唯一 import 是 torch
import torch
import torch.distributed as dist

def all_gather_concat(data, group=None, concat_dim=0):
    """differentiable all-gather + cat。这里就是 torch 原生实现。"""
    world = dist.get_world_size(group)
    gathered = [torch.empty_like(data) for _ in range(world)]
    dist.all_gather(gathered, data, group=group)
    return torch.cat(gathered, dim=concat_dim)     # 调用点直接写 dist.get_world_size()
```

- `get_rank`/`get_world_size` 这类**两平台都只有一行、语义完全兼容**的原语，core **直接在调用点写 `dist.get_rank()`**，不再包一层门面函数——包了也是零收益的间接（见 §1.4 (a) 纯透传）。
- `all_gather_concat` 这类**需要翻译**（group 形态、返回类型）的高层函数，才保留在 core 门面里，由 patch 换实现。

**② patch 模块：两种重绑各就各位**

```python
# hyper_parallel/platform/mindspore/patch.py
import mindspore as ms
import mindspore.communication as comm
from mindspore.communication import GlobalComm
import torch.distributed as dist
import hyper_parallel.comm as hp_comm          # 拿住 core 那个模块对象

_MS_PATCHED = False

def _ms_get_rank(group=None):
    return comm.get_rank() if group is None else comm.get_rank(group)

def _ms_all_gather_concat(data, group=None, concat_dim=0):
    # MS 的 group 是「字符串名」，不是 torch 的 ProcessGroup 句柄
    group_name = group if isinstance(group, str) else GlobalComm.WORLD_COMM_GROUP
    gathered = ms.mint.distributed.all_gather(data, group_name)   # MS 返回 list
    return ms.ops.cat(gathered, dim=concat_dim)

def enable() -> None:
    global _MS_PATCHED
    if _MS_PATCHED:
        return                        # 幂等，与 autograd_compat 的 _BACKWARD_COMPAT_ENABLED 同一手法
    # get_rank 一行、语义兼容 → 直接 patch torch.distributed 本体
    dist.get_rank = _ms_get_rank
    # all_gather_concat 需翻译 → 走 core 门面重绑
    hp_comm.all_gather_concat = _ms_all_gather_concat
    _MS_PATCHED = True
```

**③ 在 `__init__.py` 里，import 阶段按环境决定是否打补丁**

```python
# hyper_parallel/__init__.py
import os
if os.environ.get("HYPER_PARALLEL_PLATFORM", "").lower() == "mindspore":
    from hyper_parallel.platform.mindspore.patch import enable
    enable()          # 必须在任何 core 模块使用这些符号之前执行
```

**④ 调用效果对比**

- **`get_rank`（直接 patch torch 本体）**：core 调用点写的是 `dist.get_rank()`；patch 前它是 `torch.distributed.get_rank`（返回 torch rank），patch 后 `dist.get_rank` 已被重绑为 `_ms_get_rank`（返回 MS rank）。**core 调用点 `dist.get_rank()` 一字未改。**
- **`all_gather_concat`（门面重绑）**：patch 前 `hp_comm.all_gather_concat` 是 torch 原生实现；patch 后指向 `_ms_all_gather_concat`。

两者变的都只是「名字指向哪个函数对象」，core 源码一字未改。这就是「打 patch」与「平行实现」的根本区别。

**⚠️ 两种取舍：改 torch 本体 vs 改自己的门面——按符号性质分别选用**

- **改 torch 本体**（`dist.get_rank = _ms_get_rank`）：适用于**两平台一行代码、语义完全兼容**的简单原语（`get_rank`/`get_world_size` 等）。此时 core 直接写 `dist.get_rank()` 最自然，包门面纯属多余；直接 patch `torch.distributed` 简洁且无别名。代价是改的是全局共享的 torch 模块（任何第三方 import torch 都会看到、顺序敏感），但对这类**无翻译、无状态、幂等**的读值原语，风险足够小、收益（消灭无谓间接）明确。
- **改自己的薄门面**（`hp_comm.all_gather_concat = ...`）：适用于**需要翻译/包装**的符号（group 形态转换、返回类型适配、autograd 语义）。此时门面是承载翻译逻辑的地方，且不污染全局 torch、顺序好控。
- 判据：**语义兼容的一行原语 → 直接 patch torch 本体；需要翻译的高层函数 → 走门面。** 不是「一律不建议改 torch 本体」，而是按符号性质分别选用（对应 §5.1.2 的 (i)/(ii)）。

### 3.5 monkey patch 的分情形处理策略（谁实现、接口是否兼容）

**情形 1：torch 与 mindspore 都有实现，且接口兼容 —— 直接替换**

- **patch 责任**：纯改名，且**直接 patch torch 本体命名空间**——`dist.get_rank = _ms_get_rank`，无适配、无包装、不经 `hp` 门面。语义完全兼容的一行原语无需自建门面（包了也是零收益的间接），直接重绑 `torch.distributed` 上的名字最简洁。
- **core 改动**：**一次性迁移调用点**——把 core 里的 `platform.get_rank()` 改成 `dist.get_rank()`（即从「经平台门面」改成「直接调 torch」），并删除 `platform.get_rank`。这是 strangler「改一个删一个」的本体（§5.1.1）。**改完之后**patch 机制本身对 core 零侵入：MS 环境下 `dist.get_rank` 被重绑，core 那行 `dist.get_rank()` 前后一字不改。
- **特征**：语义一致、参数形态一致、返回类型一致。多数简单接口（`get_rank` / `get_world_size` / `get_group_rank` / `barrier` 等）属此类。
- **例**：§3.4 的 `get_rank`（`dist.get_rank = _ms_get_rank`）。

**情形 2：torch 与 mindspore 都有实现，但基础接口不兼容 —— 适配封装后语义一致的接口**

- **patch 责任**：写一层翻译/包装，让 MS 实现「看起来」是 core 调用点期望的 torch 表面。以 §3.4 的 `all_gather_concat` 为例：torch 原生版是 `dist.all_gather` + `torch.cat`；MS 版 `_ms_all_gather_concat` 里，group 要从 torch 的 `ProcessGroup` 句柄翻译成 MS 的字符串名（`GlobalComm.WORLD_COMM_GROUP`），再调 `ms.mint.distributed.all_gather` + `ms.ops.cat`。这层翻译逻辑必须有个落脚点，所以 patch 打在 core 的 `hp` 门面上——`hp_comm.all_gather_concat = _ms_all_gather_concat`，**而不是**像情形 1 那样直接 patch torch 本体。
- **core 改动**：一次性把 `platform.all_gather_concat()` 迁到 core 门面函数 `all_gather_concat()`（§3.4 ① 的 `hyper_parallel/comm.py`）并删 `platform.all_gather_concat`；此后 core 调用点仍写 `all_gather_concat(...)`，patch 重绑门面、core 一字不改（翻译逻辑全在 patch 侧，core 看到的仍是 torch 语义）。
- **特征**：两边能表达**同一语义操作**，只是「怎么驱动底层引擎」不同（`all_gather_concat` 语义两边一致，只是 group 形态与算子入口不同）。
- **工作量主体**：`differentiable_*` 异步族、collectives、`DTensorBase`、dtype/tensor 工厂均在此类。

**情形 3：torch 有实现，mindspore 没有 —— patch 报错或跳过**

先问「MS 没有这个功能，跳过它会导致什么」。**默认不在 core 加分支，而是让 patch 兜住**，分两种情况：

- **功能是语义必需的（跳过即错）→ patch 给报错 stub，core 零分支。** MS 侧把该符号替换成 `raise NotImplementedError`（或在 MS 上让它自然走失败路径）。core 两边调用**同一个符号**：torch 上正常，MS 上**在调用点显式报错**,而不是初始化时就拦（初始化报错会阻断整个 MS 建模，即便用户根本没用这个功能）。「能力标志 + 分支」在这里是纯噪音——不存在可选的替代路径供代码挑选，分支两个分支同值。
- **功能是可选优化（跳过仍正确），且退化路径全库一致 → patch 给退化默认，core 零分支。** patch 把该符号接到退化默认（no-op 上下文 / warn + skip）。例：`recompute_session_ctx` 在 MS 上接 no-op（不省显存，但数值不变、只是更吃内存），core 两边调同一个符号，MS 上静默走退化。

**例**：torch.compile / dynamo（语义必需 → MS 报错 stub）；recompute / activation checkpoint（可选优化 → MS 接退化默认）。

**情形 4：mindspore 有实现，torch 没有 —— 先问「torch 主线会不会走到它」**

torch 主线既然没有这个功能，它**压根不该出现在 torch 主线的接口契约里**。所以默认不必为它留任何东西。

- **核心示例**：`custom_ops` 的 20+ `.cc` kernel、（若确无 torch 侧调用路径的）`init_parameters` —— torch 不留东西。

#### 决策规则汇总表

| 情形 | 谁有实现 | 接口 | 处理 |
|---|---|---|---|
| 1 | 两边都有 | 兼容 | 纯改名，直接 patch torch 本体（`dist.get_rank = _ms_get_rank`） |
| 2 | 两边都有 | 不兼容 | parch对象为封装后的兼容接口，走 `hp` 门面 |
| 3 | 仅 torch | —— | 必需→MS 报错 stub；可选→MS 退化默认；仅当退化策略由调用点决定时才在 core 保留平台分支 |
| 4 | 仅 MS | —— | 只在 MS 实现，torch 零残留 |

> 本表不单列「core 改动」：**所有情形都共享同一步一次性 strangler 迁移**——core 把 `platform.X()` 的调用改成直接调 torch（情形 1）或 core 的 `hp` 门面（情形 2）/相应符号（情形 3/4），并删除 `platform.X`（§5.1.1）。

> 附：§6 清单已按本四类标注每项的归属（情形 2 是 patch 主体）。

### 3.6 用户可见的使用方式变化（相对原有实现）

**核心变化一句话**：默认平台从 MindSpore 翻转为 torch，所以「用 MindSpore 变得必须显式声明」，且声明必须发生在 `import` 阶段，不能再拖到运行时，另外环境中必须安装torch。

#### 3.6.1 新旧对比

| 维度 | 当前实现（旧接缝） | 新方案（torch 主线 + patch） |
|---|---|---|
| 默认平台 | **MindSpore**（`try: mindspore except ImportError: torch`） | **torch** |
| 用 torch | 需 `HYPER_PARALLEL_PLATFORM=torch` | **什么都不用设**（默认即 torch） |
| 用 MindSpore | **什么都不用设**（默认即 MS） | **需 `HYPER_PARALLEL_PLATFORM=mindspore`** |
| 环境变量作用时机 | 调用 `get_platform()` 时**运行时解析** | 必须在 `import hyper_parallel` **之前/随 import 时**生效（打补丁先于一切 torch import） |
| 声明方式 | 只认环境变量 | 环境变量 |
| 用 MS 是否干净 | 默认即 MS，无额外负担 | 若环境变量没设对，看到的是「torch 行为」，**不会有任何报错提示**（重点风险） |

#### 3.6.2 「环境变量作用时机」是关键

当前实现里，`__init__.py:33` 那串 core import **不碰平台选择**——`get_platform()` 直到被调用时才解析环境变量，所以环境变量可以在跑脚本的任何时刻生效，甚至运行时切换。

新方案核心代码**直接 import torch**，patch 必须**在 core 模块用到 `torch.distributed` / `torch.autograd.Function` 之前**就把符号替换掉。而 `__init__.py` 在模块加载第一行就开始 import core（`:34` 起），所以 patch 触发点**只能前置**到 import 序列最前面：

```python
# hyper_parallel/__init__.py（新方案）
import os
if os.environ.get("HYPER_PARALLEL_PLATFORM", "").lower() == "mindspore":
    from hyper_parallel.platform.mindspore.patch import enable
    enable()                       # 必须在下面所有 core import 之前

from hyper_parallel.core.shard.dfunction import DFunction   # :34
...
```

#### 3.6.3 用 MindSpore 是否必须安装 torch？—— **是，torch 必须可 import**

**结论**：新方案下，即使纯 MindSpore 环境，**torch 也必须已安装且可 `import torch`**。但 torch 只是「软件依赖」，不是「运行引擎」——patch 把 `get_rank`/`all_gather`/`empty` 等重绑到 MindSpore，实际计算走 MS 算子

---

## 4. 备选方案

### 不保留 platform 门面，core 直接调用裸符号 `xx()`，MS 打 patch 在其定义模块

**做法**：core 不再写 `platform.X()`，而是**去掉 `platform.`、直接调 `xx()`**——每个符号在某模块顶层给出默认实现（torch 版），core 的调用点就写 `xx()`，MS 支持在 import 阶段对**该符号所在模块**做 monkeypatch（`from ... import xx` 后重绑，或 patch 模块属性），把这些裸名字换成 MS 实现，取代现在「`MindSporePlatform(TorchPlatform)` 子类覆盖」的机制。

**好处**：
- **简单、patch 落点统一**：core 消费的是清晰的裸符号表（而非散落的 torch 本体），MS 只需按符号所在模块逐个重绑，不必像主方案那样为每个符号判定「(i) 直接 patch torch 本体 vs (ii) 走 hp 门面」（§5.1.2）。
- **不碰 torch 本体**：不改全局 `torch.distributed` 名字空间，避免顺序敏感、影响第三方等问题（§5.1.2 (i) 的代价）。

**坏处**：
- **不优雅、仍留一层薄名字**：符号即便是一行透传（如 `get_rank` 只是包一层 `dist.get_rank()`），也要以项目自有模块名存在、被 MS 重绑——torch 主线不够「干净」地只看到 torch，且 core 调用点拿到的不是 `torch.distributed.get_rank` 本体，而是项目封装的名字。
- **改动面与主方案相当**：既然去掉了 `platform.X` 这个中间层，core 调用点同样要做一次性迁移（strangler 改一个删一个），这块改写工作量和主方案重叠，省下的主要是「分析 patch 落点」与「改 torch 全局」的代价。

**与主方案/既有备选的关系**：等价于主方案中「走门面（§5.1.2 的 (ii)）」这条路的**更轻变体**——不另建 `hp` 门面、不 patch torch 本体，而是把符号直接铺在模块裸名字上、patch 其所在模块。它比方案 A（子类覆盖）更贴近「去接缝」，也比主方案更省「分析 + 改 torch 全局」的成本，但牺牲了「core 直接看到 torch」的纯度和主方案的统一性。

---

## 5. 分阶段落地计划

### 5.1 按符号增量迁移（strangler：改一个、删一个）

实际落地时，**可以分解成「按符号的独立小任务」，一次只做部分符号，不断迭代**。这是绞杀者（strangler fig）模式：每个任务把若干符号从接缝里抽出、core 拿 torch 版做默认、MS 用 patch 覆盖，做完即合入，任何时刻系统都可用（已迁移的走新路，未迁移的仍走接缝）。相比一次性去接缝，它每步小、可独立回归、可独立回退，且**每步都直接朝最终架构走**，不产生一次性过渡桥脚手架。

#### 5.1.1 一个符号任务的终态是「删除」，不是留转发器

以 `get_rank` 为例（core 内 **35 处调用 / 26 个文件**），一个任务的闭环：

1. core 里所有 `platform.get_rank()` 改成新写法；
2. 注册 MS patch；
3. `platform.get_rank` 此刻**零调用方 → 直接删除**。

改完 core 不再有 `platform.get_rank()`，接缝里这一项也随之消失。**默认就是「改一个、删一个」**。

#### 5.1.2 符号「种类」决定任务难度（尤其：被继承的基类）

同名符号并非同等难度，按 core 消费方式分三类：

| 符号种类 | 例子 | 能否「事后 patch」 | 难度 |
|---|---|---|---|
| **值/函数** | `get_rank`、`empty`、`cat`、`dtype` | ✅ 任何时候重绑名字即可 | 最易 |
| **被 isinstance 消费的对象** | `Tensor`、部分 `DTensorBase` | ⚠️ 要保证对象同一性 | 中 |
| **被当基类继承的类** | **`DTensorBase`** | ❌ **不能事后 patch** | 最难 |

`DTensorBase` 是最硬的一类：`class DTensor(DTensorBase)`（`core/dtensor/dtensor.py:185`）在**该行执行时**就把 `DTensor.__mro__` 定死。若 core 先绑 torch 版、之后 MS 再重绑名字，**DTensor 已继承 torch 版，改名无效**。所以基类符号必须「**子类定义之前就选对**」，而非事后 patch：

```
core/dtensor/_dtensor_base.py     ← torch 版 DTensorBase 单独放这里（默认）
core/dtensor/dtensor.py           ← from ._dtensor_base import DTensorBase; class DTensor(DTensorBase)
patch.py (MS)                     ← 必须在 dtensor.py 被 import 之前，
                                    把 _dtensor_base.DTensorBase 替换成 MS 版
```

即基类符号的 patch 必须落在「定义它的模块已加载、但用它做基类的模块尚未加载」的时序窗口内——靠 `__init__.py` 里 patch 先于 core import 的时序保证（§3.6.2）。

> **迁移顺序建议**：先拿**值/函数符号**（如 `get_rank`/`empty`）开路，跑通整条流水线（事后可重绑、回归护栏最简单）；`DTensorBase` 这类**被继承的基类**因 MRO 在子类定义时定死、须靠 import 时序「定义前选对」，放到流水线跑顺、时序纪律立起来之后再做。

#### 5.1.4 一个符号任务的标准模板

对符号 X：
1. **抽出 torch 版进 core**（从 `platform/torch/` 挪到 `core/.../_x.py`，作默认）；
2. **改全部直接调用方**：core 内所有 `platform.X` → 新写法（按 §5.1.2 选 (i)/(ii)）；扇出过大时才临时用转发器分批；
3. **MS patch 注册**：函数/值→事后重绑；基类→§5.1.3「子类定义前选对」；
4. **删除接缝项** `platform.X`（零调用方后）；
5. **跑 X 的 UT/ST，独立提交**。

> MS-only 的 36 个符号（`recompute_*`/`custom_ops`/`init_parameters` 等）不套此模板——它们无 torch 默认版，直接走 §3.5 情形 4（torch 零残留或留 stub、MS patch）。


---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| **monkeypatch 覆盖范围大**（需替换 `torch.distributed`/`autograd.Function`/dtype/`custom_ops`/`DTensorBase` 等核心件） | patch 不完整时行为错乱 | 逐件验证；patch 模块保留逐条替换记录（日志/可查询状态） |
| **补丁顺序敏感** | `import` 顺序不同 → 一半 torch 一半 MS 的混合状态 | 在 `hyper_parallel/__init__.py` 的 `import` 阶段先打 patch 再继续；|
| **调试性退化** | 错误栈指向 patch 层，排查梯度/通信问题难度上升 | 保留 patch 替换记录 + 可禁用开关，便于对照 torch 原生行为 |
| **`core/` 大面积替换**（`platform.get_rank()` → `torch.distributed.get_rank()`） | 改动量高一个量级，回归面大 | 用 `get_platform` shim 过渡，逐步替换； |
| 反转默认后 torch 路径无回归 | 默认用户受影响 | P1 即建 torch 全量 self-test 作为基线 |
| MS 特有方法（`init_parameters`、`recompute_*`、`checkpoint_exclude_wrapper`）未进 patch 表 | 功能回退 | 按 §6「必须进 patch」清单核对，逐个单测 |
| **并非所有 torch 实现都有一份完整的 MS 对应**，导致「MS 到底支持哪些功能」边界模糊 | MS 侧功能边界不清：有的 item 缺实现在 patch 里静默缺失（看似可用实则错），有的用退化默认掩盖了功能缺失，用户/维护者难以从代码推断某功能在 MS 上到底可用/是否降级/报错 | 建立**能力矩阵**：逐一声明「torch 有、MS 有/无、接口是否兼容、缺失时是报错还是退化」。矩阵随功能演进更新，作为 MS 支持范围的**单一事实来源**；缺失项显式 `NotImplementedError` 而非静默跳过，确保「模糊」从编码层面被可视化；每新增 torch 特性时先查矩阵、决定其在 MS 侧归属。 |

---

## 7. 附：本方案「必须进 patch 替换表」的分叉方法清单

以下方法在 `torch` 与 `mindspore` 中**仅一端有实际实现或语义差异显著**，是 patch 的核心边界（即上述「门面符号」上真正分叉的那几十个），必须逐个迁移进 `patch.py` 的替换表：

下面每条末尾用「情形 N」标注其 §3.5 归属：情形 2 = 适配 shim（patch 主体）；情形 4 = 只在 MS 实现、torch 零残留。

- `init_parameters`（仅 MS，`mindspore/platform.py:1036`）—— **情形 4**（MS 平台性质；若确认无 torch 侧调用路径则 torch 零残留）
- `recompute_handle_collector_ctx` / `recompute_handle` / `recompute_session_ctx` / `clear_recompute_session`（仅 MS，`mindspore/platform.py:1838-1854`）—— **情形 4**
- `checkpoint_exclude_wrapper`（仅 MS，`mindspore/platform.py:1787`）—— **情形 4**
- `custom_ops`（torch 主线为 stub，MS 为 20+ kernel 实现）—— **情形 4**
- `differentiable_all_to_all_single` / `differentiable_all_to_all_single_async` / `differentiable_all_gather_concat` / `differentiable_reduce_scatter` / `wait_async_tensor`（异步族语义完全不同）—— **情形 2**（`torch.autograd.Function` 包 MS 内核）
- `str_to_dtype` / `list_to_size`（返回类型不同）—— **情形 2**（返回类型适配）
- `DTensorBase`（extends `torch.Tensor` vs `ms.Tensor`）—— **情形 2**（类型返回/协议适配）
- 类属性：`Tensor` / `tensor` / `Parameter` / `Module` / `dtype` / `tensor_dtype` / `Function` / `platform_type`—— **情形 2**（工厂 + 类型协议适配）

---

## 8. 附录：为什么 monkey patch 屏蔽了 torch，仍需安装 torch

本节解释一个常被问到的问题：既然后台已把 torch 的实现屏蔽、换成 MindSpore，为什么环境里还必须安装 torch？

### 核心结论

**monkey patch 屏蔽的是「函数的实现」，不是「模块的 import」**——二者是两回事。patch 必须在 `import torch` 成功之后才运行，而它的对象本身来自这个 import，所以 torch 这块「基座」必须存在。

### 为什么：patch 在 import 之后运行，而不是替代 import

看 §3.4 的最小示例，`comm.py` 顶部：

```python
# hyper_parallel/comm.py
import torch                              # ← 模块加载时就执行这一行
import torch.distributed as dist

def all_gather_concat(data, group=None, concat_dim=0):
    gathered = [torch.empty_like(data) for _ in range(world)]   # torch 原生实现
    ...
```

而 patch 做的是：

```python
# patch.py
import hyper_parallel.comm as hp_comm
hp_comm.get_rank = _ms_get_rank           # 重绑「函数名」，不是重绑「import」
```

**执行顺序**：
1. 加载 `comm.py` → 执行 `import torch`。Python 必须**找到真正的 torch 包**，否则这一行直接 `ModuleNotFoundError`。
2. 加载成功后 `hp_comm` 这个模块对象才存在，才可能被 patch 拿到。
3. patch 再把 `get_rank`、`all_gather_concat` 这些**名字**指向 MS 实现。

即：**patch 是「事后改名」，前提是 torch 已被 import 成功。** 环境里没有 torch，第 1 步就挂了，根本没走到第 3 步。

### 为什么连「直接改 torch.distributed」的激进变体也绕不开

即使写 `dist.get_rank = _ms_get_rank`：
- 这个 `dist` 就是 `torch.distributed` 模块对象。
- 要拿到它，必须 `import torch.distributed` 成功，即 torch 已安装。

**你 patch 的对象本身来自 import，所以 import 必须先行成立。**

### 一个比喻

- `import torch` = 先盖好建筑、门能进得去。
- monkey patch = 进楼后，把里面的**家具**（函数实现）换成另外一批。
- 屏蔽的是「家具」，不是「建筑」。楼不在了（torch 没装），连门都进不了，更别说换家具。

所以 §3.6.4 说「torch 必须可 import，但不必真正运行」——**torch 是 patch 附着的那块基座**（提供模块对象与名字空间），真正干活的是被重绑进来的 MindSpore 算子。

### 有没有可能真的不装 torch？

技术上可以，但方案当前没这么做，且代价不小：

- **拦截 import 系统（虚拟 torch）**：在 core 任何模块 import 之前，向 `sys.modules` 塞一个**假的 torch**（用 mindspore 实现的 shim），让 `import torch` 命中这个假模块，从而不需要真实 torch。
- 但这等于从头造一个 torch 的 API 表面（`torch.Tensor`、`torch.distributed`、`torch.autograd.Function`、`torch.empty`…），工作量大，且要与真 torch 语义保持一致——这正是 §3.6.4 路线 1/3 说「不推荐 / 更接近现状」的原因。

**归根结底**：之所以必须装 torch，是因为把 `import torch` 写在 core 顶层、拿它当接口表面。只要这一行在，torch 就是硬依赖；patch 只能换掉「调用谁实现」，换不掉「import 哪个模块」。若想彻底去掉该依赖，需将 `import torch` 从 core 顶层全部移走（变懒导入，破坏主线风格），或用虚拟 torch 拦截 import——这两条都会改变主方案形态。

### 「彻底取消 torch 依赖」是追加还是大改？

**结论先行：分两半看——「机制」是可以独立新增的一块（对现有方案的代码改动是纯加法）；但「内容」不是一块补丁，而是在**范围内**重建一个「MS 版的 torch 表面」，因为剥掉接缝后 MS 无 torch 进程里，`core/` 经平台消费的那批符号与 `core/` 自身直接 `import torch` 的调用照样会执行，且没有平台判断保护。**

**先按 §1.3 圈定范围**：本方案只覆盖 `platform/` 双实现 + 经平台门面消费的 `core/` 符号。`trainer/`、`models/`、`infer/`、`data/`、`integration/`、`auto_parallel/` 是**纯 torch 实现，不在范围**——它们本来只跑 torch，不经本方案接缝/补丁。因此衡量「假 torch 得多大」，**只统计平台层 + core，而不是全库**。全库 117 文件 / 255 处 `import torch` 只是背景噪音，不是成本依据。

**范围内真正决定「假 torch 多大」的，是散落、且不受平台判断保护的 torch 使用**——这些在 MS 无 torch 进程里同样执行，假 torch 必须让它们成立：

- **core 直接算子调用**（去接缝后成为无条件执行，同一份源码在两个平台都跑）：如 `core/` 下 `torch.cat`、`torch.zeros/empty_like` 等（`core/optimizer/`、`core/pipeline_parallel/`、`core/dtensor/` 等）。没有 `if torch-major` 保护，MS 进程里它们照样跑，返回值继续被 MS 语义消费。
- **`isinstance(value, torch.Tensor)` / `isinstance(..., torch.nn.ModuleList)` / `isinstance(..., torch.dtype)` 类型判断**（`core/` 与平台层内）：假 torch 的类型对象必须能通过「是同一套类型」的判断。
- **类定义基类**：`platform/torch/dtensor.py:21` `class DTensorBase(Tensor)`——假 torch 的 `Tensor` 要能被当作基类（`__mro__` 在类定义时定死）。MS 侧 `class DTensorBase(mindspore Tensor)` 不受影响。

> 所以现行 patch 方案成立的前提，恰恰是「torch 在 MS 路径是被 import 但**大体不被执行**」（依赖接缝把两个世界隔开）。一旦 MS 无 torch，这个前提反转，MS 路径必须能被「没有真 torch」的完整堆栈支撑。

**两条去 torch 的路线，成本结构不同（成本口径 = platform + core 范围内的门面符号，而非全库）：**

| 路线 | 做法 | 机制是否独立增量 | 成本主体 |
|---|---|---|---|
| **A. 虚拟 torch（`sys.modules` 拦截）** | 在 `__init__.py` 最前向 `sys.modules` 塞一个**用 mindspore 实现、语义对齐真 torch** 的假 `torch` 模块，让 MS 无 torch 进程 `import torch` 命中假模块 | **是，独立新增一块**（新加拦截模块，不改现有 patch/core/现有 torch 主线路径） | **重建范围内 torch 表面**——为 MS 无 torch 进程会执行的 torch 用法（`core` 直接调的 `torch.cat/zeros/empty_like` 等、`torch.Tensor` 作基类+`isinstance`+`__torch_function__`、`torch.dtype`/`torch.nn`、`torch.autograd`/`torch.distributed`）各给一份**可运行**的 MS 实现。**范围外**的 `trainer/models/infer` 不用管（它们只在 torch 主线跑）。工作量仍≈补一个 MS 版 torch 门面，但**只限平台双实现功能需要的子集**，比「全库重建」小 |
| **B. core 顶层去 `import torch`** | 把 core/平台内的顶层 import 改成函数内**懒导入**（仅当 torch 被真正选中时才 import） | 否 | 这与「torch 主线、core 直接 import torch」的**核心目标直接冲突**（§2 目标 3）；且把 `torch.distributed` 塞进函数内反复触发 import，风格与其相悖（同 §3.6.4 路线 1） |

**为何路线 A 是「重建」而非「一层薄 shim」**：拦截只能控制「`torch.cat` 这个名字指向谁」，控制不了它**返回值的后续消费**（MS 张量要继续被 MS 算子/MS autograd 接收）、控制不了 `isinstance` 的类型身份、控制不了类定义时的 `__mro__`。这些都要「假 torch 本身功能齐全」——即重建一层 MS 实现的 torch 兼容表面（范围限定在平台 + core 门面符号）。

> **可用现成件抵扣重建成本**：路线 A 那份「假 torch」不必从零自造——MindSpore 官方的 **MSAdapter/MindTorch**（见方案 E）正是现成的 MS 版 torch 兼容层，可直接充当 `sys.modules["torch"]`，把「广度符号」（`torch.cat/zeros/empty_like`、`torch.Tensor` 基类与 `isinstance`、`torch.dtype`/`torch.nn`）的重建成本大幅抵扣。但**它抵扣不了本项目的核心难点**：`differentiable_*` 的 autograd shim（图式 vs 磁带式机制差异）与 `DTensorBase`/`_OP_DISPATCHER` 的私有分发集成仍需自做并做精度/语义 PoC 验证。故即便引入 MSAdapter，路线 A 也是「大幅提速但仍需自建核心层」，而非「零成本」。

**结论**：
- **对现有方案的实现**：路线 A 是**纯增量**——新加一个独立拦截块，不改现有 patch、不改 core、不改 torch 主线路径。就「会不会动到现在的代码」而言，它确实是**追加**。
- **但这个「块」仍有规格**：它不是几行 shim，而是在**范围内（平台 + core 门面符号）**重建一层 torch 表面积（算子/张量基类/autograd/distributed），其规模=「平台双实现功能所需的 torch 子集」。**范围外**的 trainer/models/infer 等纯 torch 上层**不计入**。
- 若当前交付节奏不允许这个量的重建，**pragmatic 选择是维持「MS 环境必须装 torch」**（§3.6.4 路线 2：把 torch 列为强制依赖），把「彻底去 torch」留作后续待议项——改动量/收益权衡后的合理默认。

---

## 9. 附：platform 符号「改一个删一个」迁移清单

本清单枚举 **core 实际消费的 platform 门面符号**（基于 2026-08-27 对 `hyper_parallel/core` 的全量扫描，共约 130 个 `platform.X`），按 §5.1.3 的「符号种类」分层，供 §5.1 strangler 迁移逐个取用。每层给出**难度、patch 方式、删除方式、§3.5 归属**。括号内数字为 core 内引用次数（粗略扇出）。

> 用法：每个符号一个独立任务，终态是**删除 `platform.X`**（§5.1.1）。建议迁移顺序 **Tier A → B → E → C**（易到难），Tier D 转能力标志、Tier F 走情形 4 不套此模板。

### Tier A — 值/函数符号（最易，可事后重绑，直接改一个删一个）

**特征**：无状态或纯函数，MS patch 可**任何时刻事后重绑**名字；改全部调用方后即删 `platform.X`。绝大多数属 §3.5 情形 1（兼容→纯改名）或情形 2（签名适配，如 group 句柄↔字符串）。

- **通信原语**：`get_rank`(23)、`get_world_size`(8)、`get_global_rank`(4)、`get_group_rank`(2)、`get_process_group_ranks`(2)、`get_group_local_rank`(1)、`barrier`(1)、`broadcast`(5)、`all_reduce`(5)、`all_to_all_single`(4)、`all_gather_single`(2)、`all_gather_object`(3)、`scatter`(2)、`isend`(5)、`irecv`(6)、`send_object_list`(2)、`recv_object_list`(2)、`batch_isend_irecv`(3)、`p2p_op`(2)、`p2p_exchange`(2)、`prepare_batch_p2p_group`(1)、`init_process_group`(2)、`create_group`(6)、`split_group`(4)、`get_created_group`(2)、`get_process_group_ranks`(2)
- **tensor 工厂/算子**：`empty`(9)、`empty_like`(2)、`zeros`(4)、`ones`(1)、`full`(2)、`full_like`(5)、`arange`(2)、`rand`(1)、`randn`(1)、`tensor`(2)、`from_numpy`(1)、`relu`(3)、`cat`(4)、`chunk`(1)、`cast_fp_tensor`(4)、`tensor_type_cast`(2)、`alloc_tensor_buffer`(2)、`construct_strided_slice`(1)
- **自省/工具**：`is_tensor`(9)、`is_embedding_module`(5)、`is_linear_module`(4)、`get_op_name`(10)、`tree_map`(9)、`apply_to_tensors`(3)、`get_cells_and_names`(15)、`get_cell_construct`(3)、`parameters_dict`(5)、`buffers_dict`(1)、`get_element_size`(1)、`get_tensor_storage_size`(2)、`get_param_local_data`(1)、`tensor_to_numpy`(4)、`clip_grad_norm_`(1)
- **stream/random/event/hook**：`get_current_stream`(4)、`new_stream`(1)、`get_stream_context`(4)、`new_event`(4)、`get_device_handle`(5)、`device_count`(2)、`manual_seed`(3)、`set_rng_state`(2)、`get_rng_state`(2)、`preserve_version_counter`(2)、`no_grad`(7)、`profiler_record`(6)、`register_forward_pre_hook`(7)、`register_full_backward_pre_hook`(4)、`register_full_backward_hook`(1)
- **checkpoint/state**：`load_checkpoint`(6)、`save_checkpoint`(3)、`get_model_state_dict`(1)、`set_model_state_dict`(1)、`load_into_param`(2)、`update_parameter_by_name`(1)、`search_parameter_by_name`(1)、`set_layout_into_parameter`(1)、`get_tensor_transform`(1)、`micro_batch`(1)

**处理**：patch 事后重绑（§5.1.2 (i) 直接 `torch.*` 或 (ii) `hp` 薄门面二选一）→ 改调用方 → 删 `platform.X`。

### Tier B — 类型/对象符号（中，须保证对象同一性）

**特征**：既当工厂又被 `isinstance` 判断/作类型比较，patch 后 core 各处拿到的必须是**同一个类对象**，否则 `isinstance` 失效。属情形 2。

- `Tensor`(31)、`Parameter`(20)、`Module`(20)、`dtype`(1)、`tensor_dtype`(7)、`Function`(3)、`str_to_dtype`(1)、`list_to_size`(1)

**处理**：patch 须让门面/新符号指向同一类型对象（MS 侧用 `autograd_compat` 的 `TensorPy` 表面或 DTensor 子类）；`str_to_dtype`/`list_to_size` 是返回类型适配。

### Tier C — 被继承的基类（最难，须「子类定义前选对」，不能事后 patch）

**特征**：被 `class X(Base)` 继承，`__mro__` 在**子类定义行执行时定死**（§5.1.3）。patch 必须落在「定义它的模块已加载、用它做基类的模块尚未加载」的时序窗口。

- `DTensorBase`(2) —— `class DTensor(DTensorBase)`（`core/dtensor/dtensor.py:185`）
- `PipelineStageBase`(1) —— 流水线 stage 基类

**处理**：抽到独立模块（如 `core/dtensor/_dtensor_base.py`），patch 在 core 子类模块 import 前替换；靠 `__init__.py` patch 先于 core import 的时序保证（§3.6.2）。**放到流水线跑顺后再做。**

### Tier E — `differentiable_*` autograd 主体（函数名可重绑，但 shim 最重）

**特征**：名字层面可事后重绑（同 Tier A），但**内部实现是本仓库最大的坑**——torch `autograd.Function` vs MS 图式 autograd（§1.1 最大分叉）。删 `platform.X` 容易，写等价 shim 难。

- `differentiable_all_to_all_single`(9)、`differentiable_all_to_all_single_async`(2)、`differentiable_all_gather_concat`(5)、`differentiable_reduce_scatter`(3)、`differentiable_all_reduce`(4)、`differentiable_all_to_all`(1)、`differentiable_variable_all_gather`(1)、`differentiable_sync_hook`(6)、`differentiable_async_allgather_wait`(2)、`differentiable_async_a2a_wait`(1)、`wait_grad_handle`(1)

**处理**：每个写真实 `torch.autograd.Function` 包 MS 内核（§3.4 的 MS 版示例同构），名字重绑 + 删 `platform.X` 同 Tier A，但每个都要专项前/反向精度 ST。

### Tier D — 平台标志/逃逸访问（不是「删符号」，改成能力标志或直连框架）

**特征**：不是可替换的实现，而是「查平台身份」或「拿原始框架模块」，不套「改一个删一个」。

- `platform_type`(33) —— 分支判据 → 改**运行时能力标志**（§3.2 Step 3 / §3.5 情形 3 的 `hp_capability`）
- `device_type`(1)、`device`(2) —— 设备身份查询 → 能力标志/torch 原生
- `platform`(13)、`torch`(10)、`mindspore`(5) —— 逃逸到原始框架模块 → 迁移后直接 `import torch` / MS 侧由 patch 提供

**处理**：`platform_type` 的 33 处分支按 §3.5「分支=选择、报错=做不了」原则，绝大多数应消解为「patch 兜住、core 零分支」，仅调用点级退化策略才保留 flag。

### Tier F — MS-only 符号（无 torch 默认，不套「改一个删一个」，走情形 4）

**特征**：torch 侧无实现，没有「torch 默认版可删向」；直接按 §3.5 情形 4（torch 零残留 4a / 留 stub 4b）+ MS patch。

- `init_parameters`(1)、`get_swap_optimizer`(1)、`get_symmetric_memory_handler`(1)、`get_multicore_handler`(1)、`get_symmetric_memory_handler`、`custom_ops`、`recompute_*`、`checkpoint_exclude_wrapper`（后几项见 §6）

**处理**：不建 torch 默认、不删接缝「向 torch 收敛」，而是 torch 侧按 4a/4b、MS 侧 patch 注册（§6 清单）。

### 小结

| Tier | 符号种类 | 数量级 | 能否事后 patch | 迁移次序 | §3.5 归属 |
|---|---|---|---|---|---|
| A | 值/函数 | ~90 | ✅ 事后重绑 | 1（先做，开路） | 情形 1 / 2 |
| B | 类型/对象 | ~8 | ⚠️ 须对象同一性 | 2 | 情形 2 |
| E | `differentiable_*` | ~11 | ✅ 名字可重绑，shim 最重 | 3 | 情形 2 |
| C | 被继承基类 | 2 | ❌ 须子类定义前选对 | 4（时序纪律立起后） | 情形 2/时序 |
| D | 平台标志/逃逸 | ~5 类 | —— 改能力标志/直连 | 贯穿（随分支消解） | 情形 3 / 能力标志 |
| F | MS-only | ~8+ | —— 无 torch 默认 | 独立（情形 4） | 情形 4a/4b |

> 注：计数为 core 内 `platform.X` 引用次数的粗略扇出，用于判断单任务改动面（扇出大→可用 §5.1.1 转发器分批），非精确调用点数。行号/计数以 2026-08-27 代码为准。

---

*本方案基于 2026-08-25 对 `hyper_parallel/platform/` 全量调研；所有行号以当日代码为准。*
