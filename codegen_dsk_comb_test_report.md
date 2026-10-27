# DeepSeek / Kimi 系列 Codegen 组合测试报告（dsk 轮）

| | |
|---|---|
| **测试分支** | `shawny999:codegen_comb_test`（本地提交 `57489b91`，同步至 129 仓库 `hyper-parallel-qwen-combtest`） |
| **测试日期** | 2026-10-24 ~ 2026-10-26 |
| **测试对象** | `hyper_parallel/models` 下 deepseek/kimi 系 6 family：`deepseek_v3` / `deepseek_v32` / `deepseek_v41` / `kimi_k26` / `kimi_k3` / `kimi_linear` |
| **用例规模** | **53 用例**（P1=11、P2=23、P3=19；manifest 总数 122 = Qwen 69 + dsk 53） |
| **总体结论** | **53/53 全部符合预期**：28 个成功生成产物（pass）+ 25 个合理报错（expected_error）；修复 kk3 sibling-host 失败 1 项根因；双环境 codegen UT 回归通过（py515 / py514 各 261 passed + 4 skipped） |

---

## 1 背景与摘要

Qwen 轮（`codegen_fix_verification_report.md`）已完成 Qwen 系列 69 用例组合测试并修复 3 项 codegen 缺陷。本轮按同一标准将覆盖面扩展到 deepseek/kimi 系列 6 个 family，目标是验证 codegen 特性对 MoE / MoM / VLM / remote-code 等更复杂模型形态的通用性与正确性。

本轮测试发现并修复 1 项阻塞级缺陷：

> **kk3 sibling-host EP 策略生成失败（HP-PLAN-004）**：`kimi_k3` 是双文件 remote-code 模型，其语言塔 `KimiSparseMoeBlock` 定义在 sibling 文件 `modeling_kimi_linear.py` 中，主文件 `modeling_kimi_k3.py` 只做 import。EP inline patch 在主文件中找不到该类定义，codegen 直接失败。

修复前后对比：

| 指标 | 修复前（kk3） | 修复后 |
|---|---|---|
| `ep2` / `tp2cp2ep2` 用例 | ❌ codegen 失败（EP 策略主机类不在主源文件） | ✅ 成功生成 |
| meta `external_state_classes` | `["KimiSparseMoeBlock"]` | `[]`（已剔除） |
| MoE 边界 `block_sparse_moe` | — | ✅ 保留编译边界（runtime skeleton 承载） |

其余 52 用例未触发任何 codegen 缺陷，52/52 首轮即符合预期，未产生新的代码修改。

---

## 2 测试环境

| 项 | 值 |
|---|---|
| 服务器 | 8.92.7.129（lxy，Atlas 900 A2 / Ascend NPU，8 卡） |
| 仓库 | `/home/lxy/hyper-parallel-qwen-combtest`（分支 `codegen_comb_test`，HEAD=`57489b91`） |
| 用例/结果目录 | `/home/lxy/qwen_combtest/{cases,results}` |
| 驱动方式 | `run_matrix.py`（复用 Qwen 轮脚本，清 generated → WORLD_SIZE 运行 → 拷产物 → git checkout 还原） |
| conda 环境 | py515（`hyper_codegen`，py3.10+transformers 5.15）/ py514（`hyper_restructure_py311`，py3.11，CP wrapper 需 einops） |
| 探针模型 | `/home/lxy/probe_ckpts`（config-only 小探针：dsv3 n_routed_experts=8、heads=128、4 层） |
| dsv41 特殊加载 | 验证裁剪构建器（`build_cropped_deepseek_v41`，config=/home/lxy/DeepSeek-V4.1-Flash + engram assets） |
| kk3/klin remote-code | `trust_remote_code: true`，auto_map 双文件建模（`modeling_kimi_k3.py` + `modeling_kimi_linear.py`） |

环境映射：dsv3 与 dsv41 用例跑 py515；dsv32 / k26 / kk3 / klin 用例跑 py514（CP wrapper 或 transformers 版本需要）。

## 3 修复清单

本轮仅 1 项修复，已提交至 `codegen_comb_test` 分支：

| 提交 | 类型 | 内容 | 修改文件 |
|---|---|---|---|
| `57489b91` | fix | defer sibling-module EP strategy hosts to runtime skeleton | `hyper_parallel/codegen/inline/{pipeline,strategy_pass,meta_plan}.py`（+512/-36）+ 新 UT `tests/ut/codegen/comb/test_sibling_host_ep_strategy.py`（9 项） |

**根因**：双文件模型（kimi_k3）的语言塔类从 sibling 模块（modeling_kimi_linear.py）import，bundle 逐字拷贝该文件；EP inline pass 只在主建模源中查找 host 类（`KimiSparseMoeBlock`），找不到即失败。

**修复方案**（三个切入点，均以“主源文件实际定义的类集合”为判据）：

1. **`pipeline.py`**：用 `build_source_index(source_text).classes` 索引主源文件类集合并下传
2. **`strategy_pass.py`**（`_split_source_hosts`）：按类拆分 inlineable / deferred——sibling host 跳过 forward extract，EP 策略由 runtime local-region skeleton（`hp_bind_compute`）在安装期承载
3. **`meta_plan.py`**（`_strip_ep_sub_boundaries`）：deferred host 保留编译边界、从 `external_state_classes` 剔除

**UT 锁定**：新增 `test_sibling_host_ep_strategy.py` 9 项断言（sibling 剔除/边界保留/skeleton 承载/主源类索引/首选用例形态），随修复同提交交入。

## 4 测试用例设计

三层标准与 Qwen 轮一致（`gen_cases_ds_kimi.py` 渲染，`cases_out/manifest.json` 合并 122 用例）：

### P1 单维退化（11）
all-1 基线 + 各轴退化（tp=1 / cp=1 / ep=1 单独退化，其余维度组合），验证退化配置可生成。dsv3/dsv32/k26 含 tp1/ep1 退化形态（P1 全部 pass）。

### P2 用户典型组合（23）
参考官方 family recipe / 已提交 codegen 产物基线：
- dsv3：ep2 / ep8 / tp2ep2（EP 经框架结构 archetype 自动推断）
- dsv32：ep2 / tp2cp2ep2（官方 train.yaml recipe）
- dsv41：ep2 / ep8（验证裁剪构建器 + 官方 validation yaml）
- k26：ep2 / tp2ep2+SP / ep2_novispec（去掉 166 条视觉 specs，验证 F3 纯 Replicate 兜底泛化性——Qwen 轮修复的第二次跨 family 验证）
- kk3：cp2 / ep2 / tp2 / tp2cp2ep2
- klin：cp2 / ep2 / tp2 / tp2cp2ep2

### P3 非法配置（19）
对齐 Qwen 轮 14 类标准，以 deepseek_v3 为框架层代表 family 补齐：
tp=0 / ep=-1 / tp3_ws8（不整除）/ ep16>experts（ep>WS dense region 守卫）/ bad_ep_factory（`local_compute_fn` 指向不存在函数）/ bad_when（`when: xyz` 非法字面量）/ match_nothing（match 命中空 + 纯 TP 守卫）/ yaml_syntax（YAML 语法错误）+ 各 family tp0/ep_neg1（dsv3/dsv32/dsv41/k26/kk3/klin 六族全覆盖）+ k26 官方配方含跨文件 norm ctor（正确拒绝）。

P1–P3 期望值设计：期望成功用例 `expect="ok"`；MoE 纯 TP（fused gate_up 设计约束）与非法配置 `expect="error"`；kk3/klin 的 CP/组合用例首轮无基线、标 `observe` 观测，实测全部成功生成。

## 5 测试步骤（复现）

```bash
# 129 服务器
cd /home/lxy/qwen_combtest
python gen_cases_ds_kimi.py          # 生成 53 个 dsk 用例 YAML（manifest 合并 122）
python run_matrix.py dsk              # 跑 dsk 矩阵（按 manifest 逐用例 WORLD_SIZE 执行）
python summarize_dsk.py               # 汇总：53/53 pass/expected_error

# 双环境 codegen UT 回归（129，仓库根）
/home/lxy/miniforge3/envs/hyper_codegen/bin/python -m pytest tests/ut/codegen -q   # py515
/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python -m pytest tests/ut/codegen -q  # py514
```

（本地侧辅助脚本在 `hyper-parallel/.sync_tmp/`：summarize_dsk / digest_errors / verify_kk3_ep2 等，经 sftp 读远端结果。）

## 6 测试结果

### 6.1 总览

| 分桶 | 数量 | 用例 |
|---|---|---|
| **pass**（成功生成产物） | **28** | dsv3: all1/tp1/ep2/ep8/tp2ep2；dsv32: all1/tp1/cp1/ep2/tp2cp2ep2；dsv41: all1/ep2/ep8；k26: all1/tp1/ep2/ep2_novispec/tp2ep2_sp；kk3: all1/cp2/ep2/tp2/tp2cp2ep2；klin: all1/cp2/ep2/tp2/tp2cp2ep2 |
| **expected_error**（合理报错） | **25** | 见 §6.3 报错形态清单 |
| unexpected_error / missing | 0 | — |

**53/53 全部符合预期。**

### 6.2 关键修复验证（kk3 sibling-host）

修复后 `dsk_p2_kk3_ep2` 产物证据（`codegen_meta.json` + 生成日志）：

- `external_state_classes = []`（修复前为 `["KimiSparseMoeBlock"]`）——sibling host 已从外部状态类剔除
- `block_sparse_moe` 边界保留为编译边界（`language_model.model.layers.*.block_sparse_moe`，48 层全覆盖）
- 两条关键日志出现：

```
INFO hyper_parallel.codegen.inline.meta_plan: codegen: host classes ['KimiSparseMoeBlock'] live in a sibling module; removed from external_state_classes — their boundaries stay compiled for the runtime local-region skeleton
INFO hyper_parallel.codegen.inline.strategy_pass: codegen: EP strategy ... host classes ['KimiSparseMoeBlock'] are defined in a sibling module, not the main modeling source; the forward extract is skipped and the runtime local-region skeleton (hp_bind_compute) carries the strategy at install time
```

- 生成日志：`codegen: rendered bundle .../kimi_k3/generated (modeling modeling_Kimi_K3_gen_npu.py, 1218 plan boundaries)`
- **修复前失败 → 修复后通过**：`dsk_p2_kk3_ep2` / `dsk_p2_kk3_tp2cp2ep2` 两用例从 codegen 失败转为成功生成（1218 个 plan boundaries）

### 6.3 P3 / 纯 TP 报错形态清单（节选）

| 用例 | 报错摘要（首行） |
|---|---|
| *_tp0（6 family） | `ValueError: accelerator.tp_size must be a positive integer, got 0` |
| *_ep_neg1（6 family） | `ValueError: accelerator.ep_size must be a positive integer, got -1` |
| dsk_p3_dsv3_tp3_ws8 | `ValueError: world_size 8 is not divisible by non-DP size 3` |
| dsk_p3_dsv3_ep16_gt_experts | `ValueError: ep_size (16) must not exceed and must divide the dense region (dp_replicate × dp_cp × tp = 8)` |
| dsk_p3_dsv3_bad_ep_factory | `ConfigResolutionError: $.plan_overrides[1].local_compute_fn._target_: target '...deepseekv3_ep_compute_fn_nonexistent' has no attribute ...` |
| dsk_p3_dsv3_bad_when | `ConfigResolutionError: $.plan_overrides[0].when: expected one of ('cp', 'ep', 'sequence_parallel', 'low_precision'), got 'xyz'` |
| dsk_p3_dsv3_match_nothing | 日志含 `match='*.no_such_submodule' skipped because condition 'ep' is inactive` 提示 + 纯 TP fused-MoE 守卫报错 |
| dsk_p3_dsv3_yaml_syntax | `ConfigResolutionError: .../dsk_p3_dsv3_yaml_syntax.yaml: invalid YAML: ... expected ',' or ']', but got '<stream end>'` |
| dsk_p2_dsv3_tp2 等纯 TP 用例（7） | `ValueError: Detected N fused MoE expert weight(s) (gate/up merged into a single parameter, chunk-split inside forward) declared with TP sharding: ...`（dsv41 为 40 个权重，报错含 3 个修复选项：启用 EP / Replicate override / 拆分 w1/w3） |
| dsk_p3_k26_norm_sink | `RuntimeError: codegen: no self.<attr> = <ctor>(...) call site found for ['DeepseekV3RMSNorm']` — 跨文件 ctor 正确拒绝 |

全部报错均可定位（含参数路径 `$.plan_overrides[0].when` 或模块 FQN），无静默失败。

### 6.4 codegen UT 回归（双环境）

| 环境 | 结果 |
|---|---|
| py515（`hyper_codegen`） | **261 passed, 4 skipped**（含 comb 新 UT：`test_sibling_host_ep_strategy.py` 9 项 + Qwen 轮 3 个新 UT 文件） |
| py514（`hyper_restructure_py311`） | **261 passed, 4 skipped** |

## 7 遗留与建议

1. **kk3/klin P2 组合用例原标 observe 实测全通过**，建议 manifest 中将 cp2/ep2/tp2/tp2cp2ep2 系用例的 expect 从 `observe` 转正为 `ok`，作为后续回归基线。
2. **dsv41 CP 用例本轮未覆盖**（DSA CP wrapper 需 omni-kernel attention 替换，129 验证机不可用，与官方 recipe NOTE 一致），待支持后补 cp2 用例。
3. **k26 官方配方含跨文件 norm ctor（`DeepseekV3RMSNorm` 定义在 deepseek_v3 源中）当前被正确拒绝**——这是 codegen patch 引擎的设计边界（跨文件 ctor 替换不可下沉），建议后续考虑支持跨文件 ctor sink 或在文档中明示该限制。
4. **dsk 系家族报错形态**均已在 P3 覆盖（tp0/ep_neg1 六族全覆盖 + 框架层 8 类）；建议将 53 用例 manifest 与 run_matrix 工具纳入 CI 定期回归（可复用 Qwen 轮 `run_matrix.py`，manifest 122 用例全量）。
5. **纯 TP fused-MoE 守卫**（7 个 expected_error 用例）是框架设计约束而非缺陷：contiguous-block TP Shard 与 fused gate/up chunk 的布局不兼容，且错误是静默精度损坏，故生成期拦截 + 3 修复选项报错是正确行为，建议在用户文档中补充该设计说明。
6. **本轮无新增分支代码修改需求**：53 用例仅 kk3 需修复（已提交 `57489b91`），其余 family 首轮即符合预期，证明 codegen 特性对 dsk 系模型形态的通用性良好。
