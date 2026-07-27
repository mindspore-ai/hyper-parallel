**What type of PR is this?**
/kind feature

----

**What does this PR do / why do we need it**:

本 PR 为分布式检查点（Distributed Checkpoint）模块实现了三项核心增强能力，以支持训练过程中的弹性拓扑切换与增量保存，满足 issue-239 的需求：

**1. 拓扑感知加载（TopologyMapper）**

新增 `TopologyMapper` 类，统一解决跨 TP/PP/EP 拓扑加载检查点的问题：
- **FQN 映射**：将加载侧目标 FQN 映射到检查点侧 FQN，支持 Pipeline 并行阶段数变化后参数名称迁移（如 `model.pp_stage_1.layers.0.weight` → `model.pp_stage_0.layers.1.weight`）
- **Chunk 重叠规划**：替代原有内嵌在 `create_read_items_for_chunk_list` 中的 reshard 逻辑，在计算 ReadItem 时正确分离 `dest_index.fqn`（目标侧）与 `storage_index.fqn`（检查点侧），并增加覆盖率校验，确保目标 chunk 完全被检查点 chunk 覆盖，避免静默产生零填充张量
- `StandardLoadPlanner` 新增 `topology_mapper` 参数，默认使用 identity mapper 保持向后兼容
- 错误信息同时报告 target_fqn 和 checkpoint_fqn，便于定位拓扑映射问题

**2. 增量保存（Incremental Checkpoint）**

在 `save` / `async_save` API 中新增 `incremental_from` 和 `changed_fqns` 参数，支持仅写入发生变化的参数：
- `FileSystemWriter` 从基线检查点加载元数据，验证未变化 FQN 的 dtype/size/chunk 兼容性
- 未变化参数继承基线存储路径（通过相对路径重定位），变化参数正常写入
- 全局计划阶段过滤掉未变化的 WriteItem，减少 I/O 开销
- 所有 rank 的 `changed_fqns` 一致性校验，防止不一致导致元数据损坏
- 增量保存时自动禁用计划缓存，避免缓存与增量逻辑冲突

**3. 元数据版本迁移（Versioning）**

新增 `versioning.py` 模块，实现检查点元数据格式版本管理：
- 元数据版本从 `"1.0"` 升级到 `"2.0"`，支持后续格式演进的渐进式迁移链
- `migrate_metadata()` 在 `load_checkpoint` 时自动调用，将旧版本元数据迁移到当前版本
- 对旧 pickle 反序列化缺少 `version` 字段的情况自动识别为 `"1.0"`
- 迁移链内置循环检测、类型校验和进度校验，防止迁移规则注册错误

**修改范围**：
- 核心模块：`api.py`、`filesystem_storage.py`、`metadata.py`、`standard_planner.py`、`__init__.py`
- 新增模块：`topology_mapper.py`、`versioning.py`
- 单元测试：新增 `test_topology_mapper.py`（528 行）、`test_versioning.py`（150 行）、`test_filesystem_storage.py`（411 行）；扩展 `test_standard_planner.py`（+371 行）、`test_api.py`（+95 行）
- 系统测试：扩展 PyTorch 侧 `dcp_save_and_load.py`（+476 行）及 MindSpore 侧 `base_shard.py`（+621 行）

----

**Which issue(s) this PR fixes**:
Fixes #239

----

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**:

**功能测试**：

1. **TopologyMapper 单元测试**（`test_topology_mapper.py`，20+ 用例）：
   - 默认 identity 映射、显式 FQN 映射、映射深拷贝隔离性
   - 非法 key/value 输入校验
   - compute_required_shards：TP4→TP2 收缩、TP2→TP4 扩展、2D TP、EP4→EP2/EP2→EP4、HSDP+EP 组合重叠、EP+TP 2D 重叠
   - PP FQN 映射 + TP/EP resharding 组合场景
   - 无交集 / 部分覆盖错误检测
   - 空输入边界条件

2. **StandardLoadPlanner 集成测试**（`test_standard_planner.py`，8 用例）：
   - 显式 TopologyMapper 委托：dest_index.fqn 为目标，storage_index.fqn 为检查点 FQN
   - 缺失 key 报告双方 FQN、shape 不匹配报告双方 FQN
   - 默认 mapper 为 identity、bytes item 使用映射 FQN
   - PP 前缀隔离 + DP 去重 + TP chunk 正确性
   - HSDP+EP 专家权重去重、replicated router 去重、组合去重

3. **Versioning 单元测试**（`test_versioning.py`，7 用例）：
   - 当前版本直通、缺失版本识别为 1.0、1.0→2.0 迁移
   - 未知/未来版本拒绝、无进展规则检测、返回类型校验

4. **FileSystemStorage 单元测试**（`test_filesystem_storage.py`，411 行）：
   - 增量保存：仅写入 changed FQN，基线路径重定位
   - 未变化 FQN 兼容性校验（dtype/size/chunk 不匹配检测）
   - changed_fqns 跨 rank 一致性校验
   - 增量检查点元数据完整性验证

5. **API 单元测试**（`test_api.py`，95 行）：
   - incremental_from / changed_fqns 参数校验
   - 增量保存不支持自定义 storage_writer 校验

6. **分布式系统测试**：
   - PyTorch 侧 `dcp_save_and_load.py`（+476 行）：全量保存/加载、增量保存/加载、TP reshard、TopologyMapper 加载
   - MindSpore 侧 `base_shard.py`（+621 行）：分片检查点保存/加载验证

**验证结果**：所有单元测试通过，覆盖 TopologyMapper、Versioning、Incremental Save、StandardPlanner 集成等核心场景。

----

**Self-checklist**:

- [ ] **设计**：PR对应的方案是否已经经过Maintainer评审，方案检视意见是否均已答复并完成方案修改
- [x] **测试**：PR中的代码是否已有UT/ST测试用例进行充分的覆盖，新增测试用例是否随本PR一并上库或已经上库
- [x] **验证**：PR描述信息中是否已包含对该PR对应的Feature、Refactor、Bugfix的预期目标达成情况的详细验证结果描述
- [x] **接口**：涉及对外接口变更，API 注释已刷新正确（详见下方接口变更说明），接口评审组织通过待确认
- [ ] **文档**：是否涉及官网文档修改，如果涉及请及时提交资料到Doc仓

### 接口变更说明

**本 PR 涉及对外接口变更**，具体如下：

| 变更类型 | 接口 | 变更内容 | 向后兼容 | docstring 状态 |
|---------|------|---------|---------|---------------|
| 新增导出 | `TopologyMapper` | 新类，支持拓扑感知加载 | 是 | 完整（class + `__init__` + `map_fqn` + `compute_required_shards`，含 Args/Returns/Raises） |
| 新增导出 | `CURRENT_CHECKPOINT_VERSION` | 模块级常量 `"2.0"` | 是 | 模块 docstring 已说明 |
| 新增导出 | `migrate_metadata()` | 版本迁移函数 | 是 | 完整（含 Args/Returns/Raises） |
| 签名扩展 | `save()` / `async_save()` | 新增 `incremental_from` + `changed_fqns` 可选参数 | 是（默认 `None`） | 已补充 |
| 签名扩展 | `FileSystemWriter.__init__()` | 新增 `incremental_from` + `changed_fqns` 可选参数 | 是（默认 `None`） | 已补充 |
| 签名扩展 | `StandardLoadPlanner.__init__()` | 新增 `topology_mapper` 可选参数 | 是（默认 `None`，内部创建 identity mapper） | 已补充 |
| 默认值变更 | `Metadata.version` | `"1.0"` → `"2.0"` | 是（`load()` 自动迁移旧版本） | 已修正为 `"2.0"` |

所有新增参数均为可选且带默认值，现有调用无需修改。`Metadata.version` 默认值变更由 `load()` 中自动迁移保证向后兼容。
