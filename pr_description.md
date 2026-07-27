/kind feature

----

**What does this PR do / why do we need it**:

本 PR 为分布式检查点模块新增三项核心能力，解决 issue-239 中提出的拓扑感知加载、增量保存与元数据版本管理需求：

1. **TopologyMapper（拓扑感知加载）**
   - 新增 `TopologyMapper` 类，支持在加载检查点时将保存拓扑下的参数分片映射到当前训练拓扑，实现跨拓扑无缝加载（如 TP4→TP2、EP2→EP4、PP 阶段重排等）。
   - 在 `StandardLoadPlanner` 中集成 `topology_mapper` 参数，加载时自动完成 FQN 映射与分片交集计算，生成正确的 `ReadItem`。

2. **Incremental Checkpoint Save（增量保存）**
   - `save()` / `async_save()` 新增 `incremental_from` + `changed_fqns` 可选参数，仅写入相对基线检查点发生变化的张量，未变化的张量在文件系统中通过索引重定位复用，大幅减少写入量和保存耗时。
   - `FileSystemWriter` 支持增量写入：校验 changed_fqns 一致性、跳过未变化张量的数据写入、将 baseline 中未变化条目的存储路径重定位到增量检查点目录。

3. **Metadata Versioning（元数据版本管理）**
   - 新增 `versioning.py` 模块，定义 `CURRENT_CHECKPOINT_VERSION = "2.0"`，提供 `migrate_metadata()` 函数实现 1.0→2.0 自动迁移。
   - 元数据默认版本从 `"1.0"` 升级到 `"2.0"`，加载时检测旧版本自动迁移，遇未来版本报错。
   - `save()` / `async_save()` 在写入元数据前自动调用版本迁移，保证落盘格式始终为最新版本。

所有 API 变更均向后兼容：新增参数均为可选且带默认值，现有调用无需修改。

**Which issue(s) this PR fixes**:

Fixes #239

----

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**：

### 单元测试（UT）— 134 passed, 0 failed

| 测试文件 | 用例数 | 覆盖内容 |
|---------|--------|---------|
| `test_topology_mapper.py` | 23 | TopologyMapper 初始化校验（7）+ compute_required_shards 计算逻辑（16），覆盖 TP/EP/PP/HSDP 拓扑变换、2D 重叠、异常输入 |
| `test_versioning.py` | 7 | 版本迁移 1.0→2.0、当前版本直通、缺失版本处理、未来版本拒绝、迁移无进展拒绝 |
| `test_filesystem_storage.py` | 12 | 原有 Reader/Writer（4）+ 增量保存（8）：基线-增量 roundtrip、未变化索引重定位、shape 不匹配拒绝、deleted FQN 不继承、ranks 不一致拒绝 |
| `test_standard_planner.py` | 20 | 原有 plan 生成/缓存（8）+ 新增 TopologyMapper 委托（4）+ 增量 plan 缓存禁用（1）+ shape/key mismatch 双 FQN 报告（2）+ bytes item mapped FQN（1）+ 其他（2） |
| `test_api.py` | 12 | 原有 save/load（8）+ 增量参数校验（3）+ 旧元数据迁移（1） |
| `test_metadata.py` | 6 | 版本字段默认值更新为 2.0、可选字段验证 |
| 其余原有测试 | 54 | layout、planner、reshard、storage、async_staging、util、convert roundtrip — 全部通过 |

### 集成测试（ST）

- `tests/torch/checkpoint/dcp_save_and_load.py` 中 4 个测试函数补充了 `barrier() + shutil.rmtree()` 清理逻辑，确保测试后无残留文件。
- `tests/mindspore/st/checkpoint/` 新增增量保存与拓扑映射的 ST 用例。

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
