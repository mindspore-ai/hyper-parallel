# 模块族公共构造机制

## 先建立心智模型：一个接口，配置决定实现，两种模型入口

模型通过 `CausalAttention(config=...)` 和 `GroupedExperts(config=...)` 表达算法结构。公开模型构建入口接收作业配置，选择普通、CP/EP 等已实现的成员，再按顺序应用布局、绑定资源、进入主仓 FSDP2。forward 不解析配置。

## 例一：直接使用族构建模型

下面是可在仓库根目录直接运行的单进程用户代码。示例包拥有完整的 causal Attention、模型外置路由和 SwiGLU 专家计算；导入公开模型入口时完成族与模型 spec 的注册。使用方只构建模型并训练。该例子展示公开接入方式，使用可移植 Torch 数学；Mega/低精成员需分别实现并验收。

```python
import torch
from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.models import HyperAutoModelForCausalLM

config = DecoderConfig()
model = HyperAutoModelForCausalLM.from_config(
    config, model_factory=FamilyDecoder, torch_dtype=torch.float32,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print(tuple(model(x).shape))  # (2, 4, 8)
```

同一个模型定义在并行作业中由公开构建入口接收 `DistributedSetup`。完整可运行入口见 [train.py](../../examples/torch/module_family/train.py)：它初始化作业进程组与 HP mesh，再将配置交给 facade。模型作者只写 `self.attention = CausalAttention(config=config)`、`self.experts = GroupedExperts(config=config)`；用户无需调用注册或模型准备函数。

```bash
python -m examples.torch.module_family.train
```

四卡训练使用主仓 HyperParallel FSDP2。`train.py` 的优化器 step 沿用现有 Trainer 的 `SkipDTensorDispatch(no_skip={torch.zeros_like})` 约定，使 Adam 状态保留 DTensor 元数据；用户接入已有 Trainer 时由 Trainer 执行这一步。CP/EP 成员不额外归约参数梯度。CP2×EP2 复用四个 rank，EP 是同一 dense 域中的专家视图，作业规模为 DP2×CP2。

导入与执行的串联顺序为：公开包导入 → 族声明与模型 spec 注册 → facade 调用纯模型工厂 → 共同准备按作业配置选定成员 → 布局 → CP/EP 资源绑定 → 主仓 FSDP2 → 物化/初始化 → optimizer。注册由模块拥有者放在定义文件末尾，Python import 生命周期执行一次；用户脚本没有 `register_*()`。

工厂协议为 `factory(config, *, context)`。本例 `FamilyDecoder` 只组装模型，不读取检查点，不注册族，不创建通信组。族注册已经随公开包导入完成；模型 spec 同样由模型包内置。

模型可以先按默认成员构造，公共准备阶段再按本次作业配置创建目标成员的 meta 骨架，并接回已有 Parameter、buffer、冻结状态和别名。候选通过校验后统一安装。相同选择保留原成员；训练 forward 不切换执行类。准备完成后切换到另一个作业 context 会被拒绝。

## 例二：模型自带默认 source 整改规则

下面构造真实的普通 Attention source，再由模型 spec 选择公开族；路径 `0` 是 Sequential 的首个子模块。

```python
import torch
from examples.torch.module_family import CausalAttention, DecoderConfig
from examples.torch.module_family.attention.attention import CausalAttentionModule
from torch import nn
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest


def attention_factory(config, *, context):
    return nn.Sequential(CausalAttentionModule(config=config))


def replacements(*, model, context):
    return (ModuleReplacementRequest(
        match=("0",), target=CausalAttention, kwargs={"backend": "torch"},
    ),)


MODEL_SPEC = ModelAdapterSpec(
    architecture="Sequential", model_type="module_family_decoder",
    replacements=replacements,
)
config = DecoderConfig(architectures=("Sequential",))
model = HyperAutoModelForCausalLM.from_config(
    config, model_factory=attention_factory, model_spec=MODEL_SPEC, torch_dtype=torch.float32,
)
x = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
y = model(x)
```

构造时传 `model_spec=MODEL_SPEC`，或由模型目录中唯一的 `*_spec.py` 导出 `MODEL_SPEC` 自动发现。用户无需为模型已知的整改点重复编写 YAML。用户的显式请求优先；同一个族按参数字段合并，换成另一个 target 则整体覆盖。匹配按 Module identity 归并，同一对象的多个 FQN 保持别名关系。

`source_type/exact_type` 是可选的内部校验，不是用户必须配置的字段。新路径也不增加 `module_execution` 或 `enabled` 配置。`kernel/precision/backend` 都是 target 的具名参数；backend 使用族声明的静态默认值，不自动探测“能跑就选”的实现。

family 自动定型和 source 整改共用 `_install_module_candidates`：先检查每条旧路径仍指向原对象，再定位 root/父候选更新后的最终父对象，确认所有子模块槽位存在且没有冲突，最后统一提交。相同对象的全部别名指向同一候选，根变更通过返回值交回调用者。两个不同路径若落到同一最终槽位，只允许安装同一个候选。

source 默认请求、用户请求和旧规格统一进入 `compile_module_replacements(model, specs, defaults=..., overrides=...)`，共用一次模块身份清单。同一个 match 模式 tuple 在本次编译内只展开一次；默认层、用户层和 opaque 规格消费相同的匹配结果，再分别完成优先级与类型契约检查。所有入口直接返回 `ModuleReplacementPlan`，应用阶段消费该 plan，不重复编译或把已匹配路径再次解释成 glob。缓存随本次编译结束释放，不跨模型或作业复用。`nn.ModuleDict` 中 `a[0]` 这样的合法名字也按字面路径安装。

批量 source 整改先 prepare 所有候选，校验通过后交给同一安装器。失败时模型树保持原状。候选构造必须保持 source 只读，不申请通信资源。已有 opaque factory 兼容路径保留逐个安装行为，避免改变旧大型模型的峰值内存契约。

## 例三：同样的模型也能生成可缓存构造入口

训练入口负责模型构造与装配，不负责代码生成。codegen v2 由独立 CLI 观察原生构建、替换结果与切分计划；观察过程复用 `prepare_model_components` 和同一个安装器，只记录成功安装的元数据。

直接构造的模型如需缓存构造入口，可在准备完成后显式调用 codegen 侧的 writer：

```python
from hyper_parallel.codegen.family_model import prepare_direct_model_artifact

prepare_direct_model_artifact(
    model, attention_factory, config, model._hp_module_build_context,
    artifact_dir="./family_artifact",
)
```

该构造制品包含 `modeling_family.py` 和 `family_manifest.json`；它用于重建相同的成员选择，不承担 AST 计算内联。

```text
稳定的 model_factory + config + 有效 MODEL_SPEC
                    │
          创建最终模型并计算有效成员清单
                    │
   manifest：工厂/模型/provider/成员代码及版本、backend、配置
                    │
      缓存目录中的可导入 build_model(config, context)
                    │
           重建并核对相同 fingerprint
```

构造制品的直接调用者先执行 `prepare_model_components`；writer 要求模型已准备且属于同一个 context。缓存发布使用临时文件与原子替换，支持多个 rank 共用目录。显式目录中已有不同 fingerprint 时直接报错，不复用过期制品。普通运行允许闭包工厂；生成制品要求可稳定导入的模块级 callable。source AST 的生成与内联由独立 codegen CLI 执行。

## 代码如何实现这个原理

| 位置 | 职责 |
| --- | --- |
| `components/modules/family.py` | 兼容原注册表，增加默认成员、并行轴、backend 能力、版本和布局 provider 声明 |
| `components/modules/module_family.py` | 保存请求，选型，构造最终成员，记录与发现族身份；不创建通信资源 |
| `models/module_build.py` | 作业 context、族自动定型与权重接续、每次构建独立的 MODEL_SPEC、source 规则合并 |
| `models/replacement.py` | 一次 identity/FQN 匹配与计划合成、source 候选准备、family/source 共享槽位预检与安装 |
| `models/_transformers/auto_model.py` | 接收 model_factory/model_spec，将 config-first 与 source 入口汇入共同基础设施 |
| `codegen/family_model.py` | 生成可导入构造入口、有效计划 fingerprint 与原子缓存发布 |
| `trainer/config` | 把现有 YAML 转为中立的 request；低精度有效目标在成员选定后校验 |

族注册与具体成员类是不同概念。注册表只登记公开 callable；具体子类通过 `FamilyBuildInfo` 记录“由哪个族、哪个请求、哪个成员产生”，planner 无需猜类名。metadata 不是子 Module，不增加 state_dict 参数层级。

有效模型 spec 绑定到本次构建的根模型；显式传入 spec 不修改全局注册表，因此两个同类型模型可以使用不同 spec。HF 原生权重映射仍走原映射；普通 Torch 模型无需为了读取空映射而导入 HF 的模型实现。

实际 diff 的关键接缝是：

```diff
+model_factory, module_context = _resolve_module_construction(...)
+with init_ctx:
+    if model_factory is not None:
+        model = construct_family_model(model_factory, hf_config, context=module_context)
+    else:
+        _, model = _init_model(...)
+        attach_build_context(model, module_context)
```

context 已知时可以在参数创建前选好成员；已有模型则在公共准备阶段自动定型。后续 plan/apply、初始化、加载和 optimizer 生命周期由共同模型构建链处理。

```python
model = construct_family_model(factory, config, context=context)
model, weights_mapping = prepare_model_components(model, context)
```

`_finalize_family_modules` 先读取各实例保存的 `FamilyRequest`，用当前 context 重新解析；实现未变时保留原成员，实现变化时准备新成员并保留已有逻辑状态，最后统一更新实际 FQN 和别名。`FamilyBuildInfo` 继续标记为 direct，因此默认 source 替换不会再次接管它。`prepare_model_components` 是这个过程的唯一调用入口。`construct_family_model` 只调用工厂并附上 context；工厂返回的模型必须经过公共准备后才能用于布局或生成构造制品。模型无需逐层传递 context。

```diff
+request = _normalize_family_request(target, kwargs)
+member = resolve_family(request, context=context)
+result = _build_family_member(request, member, config=config, module=module,
+                              module_fqn=module_fqn, context=context)
```

这段接线把“请求”“选择结果”“实例”分开：同一个 target 可以用于 source 和 direct，而实例保存可发现的选择结果。完整构造代码、自动定型接线及测试 diff 见 [module-family-construction.patch](module-family-construction.patch)。patch 使用零行上下文，可在基线上通过 `git apply --unidiff-zero` 应用。

提交还修正了 lint 插件中与仓库 Torch-only 规则冲突的旧限制：原生 Torch import 被允许，MindSpore import 被拒绝；使用实际 pylint 子进程回归该行为。

## 验证与能力边界

本提交独立检出的构造与公开入口定向验证 **46 项通过**，包含 8 项配置驱动的自动定型回归：先构造后应用 CP/EP 配置、无逐层 context 传递、根族、完整状态与 meta、嵌套族和别名、批量失败原子性、逻辑权重契约与运行生命周期。共同验收还覆盖线性计算前反向、source 参数 identity 与冻结状态、backend 非法组合、factory 只调用一次、混合子树保护、真实小型 HF BERT 默认整改及 codegen、共享缓存并发发布和版本失效。

4 进程 CPU 与 4 卡 NPU 验收均覆盖先创建普通 Attention/Experts、再自动选择 CP/EP 成员的路径；后续通信、主仓 FSDP2 参数与 Adam 矩对照、DCP 恢复及下一次更新一致。

测试环境为当前 checkout 的 editable 安装，Transformers 5.11.0 与 5.22.1 分别通过上述定向用例。相关目录的扩大回归存在 11 项既有失败，已在本次收敛前提交的同一环境复现；这些失败不计为新增机制通过项。

构造机制本身没有生产 GQA/MLA/Experts 的 CP/EP/Mega/FP8 算法；能力由族显式登记的成员提供。没有登记的组合构造时即报错。物理布局、通信绑定和训练状态交接由第二个公共机制承接，见 [布局与训练状态机制](module-family-layout-state.md)。

构建中的工作归属如下：族定型和 source 整改由公共 prepare 执行一次；切换成员时，`_build_family_member` 复用已经得到的 `ResolvedFamily`，不再次解析请求；权重映射由 prepare 读取并缓存，后续入口复用结果。codegen 观察构建与原生构建共用相同的模型 spec 解析和装配入口，空 method 规则不创建额外上下文。布局覆盖复检、安装后族校验、最终 dtype/物化校验和缓存重建签名校验分别验证不同阶段的事实，继续保留。

工厂、模型类与 provider 的 fingerprint 共用 `_plain/_code_hash`，类的继承实现进入签名；只改父类算法也会使旧构造制品失效。物化与 external-state 消费方共用 `resolve_model_adapter(model)`，优先采用本次已附加的 spec（包括显式 None），避免各自重写模型身份解析。

生产目录只承载真实构建链所需逻辑。制品导入验证工具位于 `tests/common/module_family_fixtures.py`；测试直接比较 `family_plan_snapshot`，不增加专用 fingerprint API。生成文件仍调用生产的 `construct_cached_direct_model`，它负责实际重建与签名核验。

公共 prepare 直接读取本次 context 中的默认／用户规则，source 请求和规格共用选择字段校验。候选按声明式整批或旧规格单个形成批次，安装与映射收集共用同一段代码；只有实际出现权重转换才采集原 shape。签名校验覆盖必填 keyword-only、位置／关键字调用和可选位置参数省略，实际数值由对应成员验收。
