# Skills 管理指南

## 创建 Skill

- 基于 [Skill 规范](https://github.com/anthropics/claude-code/blob/main/docs/skills.md) 编写，每个 Skill 必须包含 `SKILL.md` 入口文件。
- 推荐使用 **Skill Creator** skill 生成初始模板，确保结构和格式符合规范。

## 修改 Skill

- 优先联系 Skill 的 **owner**（上传者）提需求或讨论修改方案，避免独自大改引入问题。
- Skill 难以编写自动化测试用例，修改后务必人工验证核心流程。

## 迭代策略

- **小改动**：在原有 Skill 上修复即可。
- **大迭代**：不建议直接改原有 Skill，建议基于原有 Skill **新建一个 Skill**（如 `autogit-v2/`），保持原版本可回退。

## 安装 Skill

1. 将 Skill 仓库 clone 到本地。
2. 告知本地 Agent（Claude Code、Codex 等），Agent 会自动将 Skill 路径注册到其可识别的配置中。

```bash
# 示例：clone 后告诉 Agent
git clone <skill-repo-url> ~/.local/skills/my-skill
# Agent 会自动识别并注册
```
