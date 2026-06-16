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
| docs/getting_started/ | 版本要求变更、新增依赖 |
| docs/faq.md | 新增常见问题 |
| Release Notes | 生成版本变更记录 |

### 4. 打 Tag

```bash
git tag -a vX.Y.Z -m "HyperParallel vX.Y.Z release"
git push upstream vX.Y.Z
```

### 5. 构建 Wheel

```bash
# MindSpore 后端
python setup.py bdist_wheel

# PyTorch + MindSpore 后端
BUILD_TORCH_EXTENSION=true python setup.py bdist_wheel

# Wheel 文件名格式：hyper_parallel-X.Y.Z+{python_tag}-{arch_tag}.whl
```

### 6. 发布后验证

- [ ] README 中英文版本章节一致、代码示例可运行
- [ ] 用户手册安装步骤在空环境可复现、特性文档代码示例与 examples/ 一致
- [ ] API 文档接口签名与源代码一致、参数描述与类型匹配
- [ ] Release Notes 变更分类准确、贡献者名单完整
- [ ] 所有文档 markdown lint 通过
- [ ] 交叉引用链接正确、术语一致

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.2.0 | — | 上一个发布版本 |
| v1.0.0 | 2026-06-30 | 首个正式版本，全量特性覆盖 |