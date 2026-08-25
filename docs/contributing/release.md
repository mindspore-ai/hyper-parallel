# 发布流程规范

## 版本命名

遵循语义化版本（SemVer）：`vMAJOR.MINOR.PATCH`

- **MAJOR**：不兼容的 API 变更
- **MINOR**：向后兼容的功能新增
- **PATCH**：向后兼容的 bug 修复

## 发布流程

### 1. 版本准备

```bash
# 确认所有待发布特性已合并到 master
git checkout master
git pull upstream master

# 确认版本号
# 版本号定义在 setup.py 或 __init__.py 中
```

### 2. 生成 Release Notes

```bash
# 分析自上一个 tag 到 HEAD 的变更
git log vPREV_TAG..HEAD --oneline
git diff vPREV_TAG..HEAD --stat
```

Release Notes 需包含：

- **全量特性列表**（首个正式版本需列出所有已实现特性）
- **增量变更**（标注上一个版本到当前版本的变更）
- **贡献者名单**
- **已合并 PR 列表**

- **已知限制**
- **升级指南**

### 3. 文档更新

每个版本发布前需同步更新：

| 文档类型 | 更新内容 |
|----------|----------|
| README.md / README.en.md | 特性 checkbox 更新、新特性章节 |
| docs/guide/ | 新增特性使用指南、更新已有指南 |
| docs/api/ | 新增接口说明、更新已有接口参数 |
| docs/faq.md | 新增常见问题 |
| Release Notes | 生成版本变更记录 |

### 4. 打 Tag

```bash
git tag -a vX.Y.Z -m "HyperParallel vX.Y.Z release"
git push upstream vX.Y.Z
```

### 5. 构建 Wheel

```bash
# 外部构建工程先选择 CANN 环境；统一入口完成依赖准备、native 编译和 wheel 打包
source /usr/local/Ascend/cann/set_env.sh
./build.sh --multicore all --shmem all --custom-ops on \
  --soc-list ascend910b,ascend910_93 --strict off

# 每个 Python ABI/host 架构一个 wheel，例如：
# hyper_parallel-0.1.0-cp310-cp310-linux_aarch64.whl
```

optional native 组件失败时仍保留 wheel，并在 build log 中记录稳定 reason code。native 能力缺失由
Level 0 MegaMoe ST 和全量用例拦截。正式发布由版本指定的 glibc 基线构建、全量用例和人工评审共同决定，
wheel 文件已生成不代表制品满足发布条件。native 制品的构建与运行基线为 CANN 9.1.0。

### 6. 发布后验证

HyperMegaMoe/multicore ST 必须在启动 pytest、MindSpore、Torch 或分布式 worker 前完成环境激活：

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_parallel_multicore_set_env.bash)"
pytest tests/mindspore/st/multicore/test_moe.py
```

- [ ] README 中英文版本章节一致、代码示例可运行
- [ ] 用户手册安装步骤在空环境可复现、特性文档代码示例与 examples/ 一致
- [ ] API 文档接口签名与源代码一致、参数描述与类型匹配
- [ ] Release Notes 变更分类准确、贡献者名单完整
- [ ] 所有文档 markdown lint 通过
- [ ] 交叉引用链接正确、术语一致
- [ ] wheel/PYTHONPATH 均先 source 制品内 `set_env.bash`，四种框架 import 顺序和 clean-venv smoke 通过
- [ ] 未 source 时稳定报 `HP-NATIVE-OPP-NOT-ACTIVATED`，框架已先导入时稳定报 `HP-NATIVE-OPP-ACTIVATION-TOO-LATE`
- [ ] 目标 glibc 基线门禁验证通过；910B/910C 支持矩阵验证通过，不支持的 SoC 返回明确 reason code

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.2.0 | — | 上一个发布版本 |
| v1.0.0 | 2026-06-30 | 首个正式版本，全量特性覆盖 |
