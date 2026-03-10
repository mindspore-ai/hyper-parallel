#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PR title and body generation from diff analysis for AutoGit."""

import re
from typing import Dict, List, Optional, Tuple

import yaml

from models import COMMIT_TYPE_LABELS, CONVENTIONAL_RE, DiffAnalysis, FileChanges
from git_utils import run_git
from diff_analysis import analyze_diff


# ============================================================================
# Skill feature extraction
# ============================================================================

def _extract_skill_features(filepath: str) -> List[str]:
    """Extract trigger scenarios from SKILL.md as feature points.

    Args:
        filepath: Path to a SKILL.md file.

    Returns:
        List of feature description strings.
    """
    features: List[str] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as fp:
            content = fp.read()
    except OSError:
        return features

    if '---' not in content:
        return features

    parts = content.split('---')
    if len(parts) < 3:
        return features

    try:
        meta = yaml.safe_load(parts[1])
    except Exception:  # pylint: disable=broad-except
        return features

    if not meta or 'description' not in meta:
        return features

    desc = meta['description']
    if '触发场景' not in desc:
        return features

    for line in desc.split('\n'):
        line = line.strip()
        if line.startswith('- ') and '用户' in line:
            features.append(line[2:])
    return features


def _find_skill_name(files: List[str]) -> Optional[str]:
    """Find the skill name from the file list.

    Args:
        files: List of changed file paths.

    Returns:
        Skill name string, or None if not found.
    """
    for f in files:
        if 'skills/' not in f:
            continue
        parts = f.split('/')
        idx = parts.index('skills') if 'skills' in parts else -1
        if idx < 0 or idx + 1 >= len(parts):
            continue
        candidate = parts[idx + 1]
        if candidate not in ['README.md'] and not candidate.endswith('.md'):
            return candidate
    return None


# ============================================================================
# Domain inference
# ============================================================================

def _infer_skill_domain(files: List[str],
                        added_funcs: List[str]) -> Tuple[str, List[str]]:
    """Infer the functional domain and feature points for Skill-type changes.

    Args:
        files: List of changed file paths.
        added_funcs: List of added function names.

    Returns:
        Tuple of (domain_description, feature_points).
    """
    skill_name = _find_skill_name(files)
    if not skill_name:
        return "", []

    feature_domain = f"新增 Claude Code Skill: `{skill_name}`"
    feature_points: List[str] = []

    for f in files:
        if f.endswith('SKILL.md'):
            feature_points.extend(_extract_skill_features(f))

    func_features = {
        'cmd_commit': '一键提交并推送代码到远端',
        'cmd_pr': '自动创建 Pull Request',
        'cmd_status': '查看 PR 状态',
        'cmd_update': '重新生成 PR 描述',
        'analyze_diff': '智能分析代码变更',
        'generate_pr_content': '自动生成 PR 描述',
        'infer_feature': '自动推断功能领域和功能点',
    }
    for func in added_funcs:
        func_lower = func.lower()
        for key, desc in func_features.items():
            if (key in func_lower or func_lower in key) and desc not in feature_points:
                feature_points.append(desc)

    return feature_domain, feature_points


def _detect_domain(file_paths: str, files: List[str],
                   added_funcs: List[str],
                   added_classes: List[str]) -> Tuple[str, List[str]]:
    """Detect the functional domain based on file paths.

    Args:
        file_paths: Space-joined lowercase file paths for pattern matching.
        files: Original file path list.
        added_funcs: List of added function names.
        added_classes: List of added class names.

    Returns:
        Tuple of (domain_description, feature_points).
    """
    if 'skill' in file_paths:
        return _infer_skill_domain(files, added_funcs)

    if 'test' in file_paths:
        points: List[str] = []
        if added_funcs:
            points.append(f"新增 {len(added_funcs)} 个测试方法")
        return "测试用例", points

    if 'doc' in file_paths or 'readme' in file_paths:
        return "文档更新", []

    if any(x in file_paths for x in ['dist_op', 'distributed', 'parallel']):
        points = [f"新增分布式算子 `{cls}`" for cls in added_classes]
        return "分布式算子", points

    if 'core' in file_paths:
        return "核心功能", []

    if 'util' in file_paths:
        return "工具函数", []

    return "", []


def _categorize_funcs(funcs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Categorize functions by type: (cmd_funcs, api_funcs, other_funcs).

    Args:
        funcs: List of function names.

    Returns:
        Tuple of (cmd_funcs, api_funcs, other_funcs).
    """
    cmd = [f for f in funcs if f.startswith('cmd_')]
    api = [f for f in funcs if 'api' in f.lower() or 'request' in f.lower()]
    other = [f for f in funcs if f not in cmd and f not in api]
    return cmd, api, other


def _generate_default_points(added_funcs: List[str],
                             added_classes: List[str]) -> List[str]:
    """Generate default feature points based on code structure.

    Args:
        added_funcs: List of added function names.
        added_classes: List of added class names.

    Returns:
        List of feature point strings.
    """
    points = [f"新增 `{cls}` 类" for cls in added_classes[:3]]

    if not added_funcs:
        return points

    cmd_funcs, api_funcs, other_funcs = _categorize_funcs(added_funcs)
    if cmd_funcs:
        names = '`, `'.join(cmd_funcs[:3])
        points.append(f"支持 {len(cmd_funcs)} 个命令: `{names}`")
    if api_funcs:
        points.append(f"新增 {len(api_funcs)} 个 API 接口")
    if other_funcs and len(points) < 3:
        points.append(f"新增 {len(other_funcs)} 个辅助函数")
    return points


def infer_feature_purpose(analysis: DiffAnalysis) -> Tuple[str, List[str]]:
    """Infer feature domain and specific feature points from code changes.

    Args:
        analysis: DiffAnalysis result from diff parsing.

    Returns:
        Tuple of (feature_domain, feature_points).
    """
    files = analysis.files
    added_funcs = analysis.added_funcs
    added_classes = analysis.added_classes
    file_paths = ' '.join(files).lower()

    feature_domain, feature_points = _detect_domain(
        file_paths, files, added_funcs, added_classes
    )

    if not feature_domain:
        if added_classes:
            feature_domain = "功能扩展"
        elif analysis.modified_funcs:
            feature_domain = "功能优化"
        else:
            feature_domain = "代码维护"

    if not feature_points:
        feature_points = _generate_default_points(added_funcs, added_classes)

    return feature_domain, feature_points


# ============================================================================
# Conventional commit parsing
# ============================================================================

def _parse_conventional_commits(commits: List[str]) -> List[Dict[str, str]]:
    """Parse conventional commit format, extracting type/scope/subject/body.

    For non-conventional commits, type is empty and subject keeps the original message.

    Args:
        commits: List of commit SHA strings.

    Returns:
        List of dicts with keys: type, scope, subject, body.
    """
    results: List[Dict[str, str]] = []
    for sha in commits:
        raw = run_git("log", "-1", "--pretty=format:%s%n---body---%n%b", sha).stdout
        parts = raw.split("\n---body---\n", 1)
        subject_line = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        m = CONVENTIONAL_RE.match(subject_line)
        if m:
            results.append({
                'type': m.group(1),
                'scope': m.group(2) or '',
                'subject': m.group(3).strip(),
                'body': body,
            })
        else:
            results.append({
                'type': '',
                'scope': '',
                'subject': subject_line,
                'body': body,
            })
    return results


# ============================================================================
# Docstring extraction from diff
# ============================================================================

def _match_public_def(line: str) -> Optional[str]:
    """Match a public def/class definition line and return the name; None if private.

    Args:
        line: Source code line to check.

    Returns:
        Public name string, or None.
    """
    m = re.match(r'\s*(?:async\s+)?def\s+(\w+)\s*\(', line)
    if not m:
        m = re.match(r'\s*class\s+(\w+)\s*[:(]', line)
    if not m or m.group(1).startswith('_'):
        return None
    return m.group(1)


def _parse_docstring_text(lines: List[str], pos: int, quote: str) -> str:
    """Parse docstring text from line pos; return first sentence.

    Args:
        lines: Source lines.
        pos: Line index where the docstring starts.
        quote: Quote style ('\"\"\"' or \"'''\").

    Returns:
        First sentence of the docstring.
    """
    doc_text = lines[pos].strip()[len(quote):]
    if doc_text.endswith(quote):
        return doc_text[:-len(quote)].strip()
    parts = [doc_text.strip()]
    for k in range(pos + 1, min(pos + 5, len(lines))):
        line_s = lines[k].strip()
        if line_s.startswith(quote) or line_s.endswith(quote) or not line_s:
            break
        parts.append(line_s)
        if line_s.endswith('.'):
            break
    return ' '.join(p for p in parts if p)


def _find_docstring_after(lines: List[str], start: int) -> str:
    """From start line, skip to end of signature, skip blank lines, extract docstring.

    Args:
        lines: Source lines.
        start: Line index of the def/class statement.

    Returns:
        Docstring text, or empty string.
    """
    total = len(lines)
    j = start
    while j < total:
        code = lines[j].split('#')[0].rstrip()
        if code.endswith(':'):
            break
        j += 1
    j += 1

    while j < total and not lines[j].strip():
        j += 1

    if j >= total:
        return ""

    stripped = lines[j].strip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            return _parse_docstring_text(lines, j, quote)
    return ""


def _extract_docstrings_from_diff(diff_content: str) -> List[Dict[str, str]]:
    """Extract first-line docstrings of public functions/classes from diff added code.

    Skips private functions starting with ``_``; only extracts public APIs.

    Args:
        diff_content: Raw unified diff text.

    Returns:
        List of dicts with keys: name, docstring.
    """
    added_lines: List[str] = [
        line[1:] for line in diff_content.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    ]

    seen: set = set()
    results: List[Dict[str, str]] = []
    for i, line in enumerate(added_lines):
        name = _match_public_def(line)
        if not name or name in seen:
            continue
        seen.add(name)
        docstring = _find_docstring_after(added_lines, i)
        if docstring:
            results.append({'name': name, 'docstring': docstring})
    return results


# ============================================================================
# PR purpose synthesis
# ============================================================================

def _synthesize_pr_purpose(  # pylint: disable=W0613
    commit_data: List[Dict[str, str]],
    docstrings: List[Dict[str, str]],
    analysis: DiffAnalysis,
    fallback_domain: str,
    fallback_points: List[str],
) -> Tuple[str, List[str]]:
    """Synthesize PR purpose from commit intent, docstrings, and structural analysis.

    Priority:
    1. Domain: commit type + scope
    2. Feature points: commit subject -> commit body list -> docstring -> fallback

    Args:
        commit_data: Parsed conventional commit dicts.
        docstrings: Extracted public API docstrings.
        analysis: DiffAnalysis result.
        fallback_domain: Domain from structural analysis.
        fallback_points: Feature points from structural analysis.

    Returns:
        Tuple of (domain, feature_points).
    """
    domain = fallback_domain
    if commit_data:
        primary = commit_data[0]
        ctype = primary.get('type', '')
        scope = primary.get('scope', '')
        if ctype and ctype in COMMIT_TYPE_LABELS:
            label = COMMIT_TYPE_LABELS[ctype]
            domain = f"{label}（{scope}）" if scope else label

    points: List[str] = []
    seen_lower: set = set()

    def _add_point(text: str) -> None:
        key = text.strip().lower()
        if key and key not in seen_lower:
            seen_lower.add(key)
            points.append(text.strip())

    for cd in commit_data:
        subj = cd.get('subject', '')
        if subj:
            _add_point(subj)

    for cd in commit_data:
        for line in cd.get('body', '').split('\n'):
            line = line.strip()
            if line.startswith('- '):
                _add_point(line[2:])

    for ds in docstrings:
        _add_point(f"`{ds['name']}`: {ds['docstring']}")

    if not points:
        points = list(fallback_points)

    return domain, points


# ============================================================================
# Framework-agnostic sanitization
# ============================================================================

_FRAMEWORK_PATTERNS = [
    (re.compile(r'\btorch[- ]style\b', re.IGNORECASE), ''),
    (re.compile(r'\bPyTorch\b', re.IGNORECASE), ''),
    (re.compile(r'\btorch\b', re.IGNORECASE), ''),
]


def _sanitize_framework_refs(text: str) -> str:
    """Remove references to specific deep learning frameworks, keeping text framework-agnostic.

    Args:
        text: Input text to sanitize.

    Returns:
        Sanitized text with framework references removed.
    """
    for pat, repl in _FRAMEWORK_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r'  +', ' ', text).strip()
    text = re.sub(r'\s+([,.])', r'\1', text)
    return text


# ============================================================================
# Semantic summary
# ============================================================================

_MODULE_LABELS = {
    'fully_shard': '全切片并行（fully_shard）',
    'checkpoint': '分布式检查点（DCP）',
    'shard': '张量切片与分布式算子',
    'hsdp': '混合切片数据并行（HSDP）',
    'dtensor': '分布式张量（DTensor）',
    'device_mesh': '设备网格（DeviceMesh）',
    'pipeline_parallel': '流水线并行',
    'random': '分布式随机状态管理',
    'collectives': '集合通信',
    'platform': '多后端平台适配',
}


def _classify_file_module(filepath: str) -> str:
    """Classify a file into a functional module based on its path.

    Args:
        filepath: File path to classify.

    Returns:
        Module label string, or empty string if no match.
    """
    fp_lower = filepath.lower()
    for key, label in _MODULE_LABELS.items():
        if key in fp_lower:
            return label
    return ""


def _aggregate_module_additions(
    analysis: DiffAnalysis,
) -> Dict[str, Dict[str, List[str]]]:
    """Aggregate newly added classes and public methods by functional module.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        Dict mapping module labels to {'classes': [...], 'funcs': [...]}.
    """
    module_additions: Dict[str, Dict[str, List[str]]] = {}
    for filepath, changes in analysis.file_changes.items():
        module = _classify_file_module(filepath)
        if not module:
            continue
        bucket = module_additions.setdefault(module, {
            'classes': [], 'funcs': []
        })
        for cls in changes.added_classes:
            name = cls['name'] if isinstance(cls, dict) else cls
            bucket['classes'].append(name)
        for func in changes.added_funcs:
            name = func['name'] if isinstance(func, dict) else func
            if not name.startswith('_'):
                bucket['funcs'].append(name)
    return module_additions


def _format_module_line(module: str,
                        items: Dict[str, List[str]]) -> Optional[str]:
    """Generate a summary line for a single module; return None if no content.

    Args:
        module: Module label string.
        items: Dict with 'classes' and 'funcs' lists.

    Returns:
        Formatted summary line, or None.
    """
    cls_names = items['classes']
    func_names = items['funcs']
    if not cls_names and not func_names:
        return None
    parts = []
    if cls_names:
        names = ', '.join(f'`{n}`' for n in cls_names[:5])
        parts.append(f"新增 {len(cls_names)} 个类（{names}）")
    if func_names:
        names = ', '.join(f'`{n}()`' for n in func_names[:5])
        suffix = f"等 {len(func_names)} 个" if len(func_names) > 5 else ""
        parts.append(f"新增公开方法 {names}{suffix}")
    detail = '；'.join(parts)
    return f"- **{module}**：{detail}"


def _infer_usage_scenarios(
    module_additions: Dict[str, Dict[str, List[str]]]
) -> List[str]:
    """Infer usage scenarios based on involved functional modules.

    Args:
        module_additions: Module-level additions dict.

    Returns:
        List of usage scenario description strings.
    """
    scenarios: List[str] = []
    modules_lower = ' '.join(module_additions.keys()).lower()

    if 'fully_shard' in modules_lower or 'hsdp' in modules_lower:
        scenarios.append(
            "大模型训练时对模型参数进行全切片/混合切片，"
            "降低单卡显存占用"
        )
    if 'checkpoint' in modules_lower or 'dcp' in modules_lower:
        scenarios.append(
            "分布式训练中保存/加载模型检查点，"
            "支持跨不同并行拓扑的重分布"
        )
    if 'state_dict' in ' '.join(
        f for items in module_additions.values()
        for f in items['funcs']
    ).lower():
        scenarios.append(
            "获取分布式模型的完整 state_dict，"
            "用于模型导出、评估或继续训练"
        )
    if '分布式算子' in modules_lower or 'shard' in modules_lower:
        scenarios.append(
            "在切片张量上直接调用算子，"
            "框架自动处理跨设备通信与结果聚合"
        )
    if 'device_mesh' in modules_lower or '设备网格' in modules_lower:
        scenarios.append(
            "构建多维设备网格，为混合并行策略提供拓扑抽象"
        )
    if 'random' in modules_lower or '随机' in modules_lower:
        scenarios.append(
            "管理分布式训练中的随机状态，确保数据并行的可复现性"
        )

    return scenarios


def _build_semantic_summary(  # pylint: disable=W0613
    analysis: DiffAnalysis,
    feature_domain: str,
) -> List[str]:
    """Build a high-level functional summary based on overall file change semantics.

    Args:
        analysis: DiffAnalysis result.
        feature_domain: Inferred feature domain string.

    Returns:
        List of summary lines.
    """
    module_additions = _aggregate_module_additions(analysis)
    if not module_additions:
        return []

    lines: List[str] = ["**概述**："]
    for module, items in module_additions.items():
        line = _format_module_line(module, items)
        if line:
            lines.append(line)

    scenarios = _infer_usage_scenarios(module_additions)
    if scenarios:
        lines.append("")
        lines.append("**适用场景**：")
        for s in scenarios:
            lines.append(f"- {s}")

    lines.append("")
    return lines


# ============================================================================
# PR section builders
# ============================================================================

def _build_reason_section(analysis: DiffAnalysis,
                          feature_domain: str,
                          feature_points: List[str]) -> str:
    """Build the PR reason section, leading with a semantic summary.

    Args:
        analysis: DiffAnalysis result.
        feature_domain: Inferred feature domain string.
        feature_points: List of feature point strings.

    Returns:
        Formatted reason section string.
    """
    reasons: List[str] = []

    summary = _build_semantic_summary(analysis, feature_domain)
    if summary:
        reasons.extend(summary)

    if feature_domain:
        reasons.append(f"**{_sanitize_framework_refs(feature_domain)}**")

    if feature_points:
        reasons.append("")
        reasons.append("**功能点**：")
        for point in feature_points[:5]:
            reasons.append(f"- {_sanitize_framework_refs(point)}")

    tech_changes = _summarize_tech_changes(analysis)
    if tech_changes:
        reasons.append("")
        reasons.append("**技术变更**：")
        for change in tech_changes:
            reasons.append(f"- {change}")

    return "\n".join(reasons) if reasons else "- 代码优化与维护"


def _summarize_tech_changes(analysis: DiffAnalysis) -> List[str]:
    """Summarize technical changes from the analysis.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        List of technical change description strings.
    """
    changes: List[str] = []
    if analysis.added_classes:
        changes.append(f"新增 {len(analysis.added_classes)} 个类")
    if analysis.added_funcs:
        changes.append(f"新增 {len(analysis.added_funcs)} 个方法/函数")
    if analysis.modified_classes:
        changes.append(f"重构 {len(analysis.modified_classes)} 个类")
    if analysis.modified_funcs:
        changes.append(f"重构 {len(analysis.modified_funcs)} 个方法")
    if analysis.removed_funcs or analysis.removed_classes:
        cnt = len(analysis.removed_funcs) + len(analysis.removed_classes)
        changes.append(f"移除 {cnt} 个冗余定义")
    return changes


def _build_api_section(docstrings: List[Dict[str, str]]) -> List[str]:
    """Generate the core API subsection listing public functions and docstrings.

    Args:
        docstrings: List of dicts with 'name' and 'docstring' keys.

    Returns:
        List of formatted lines.
    """
    if not docstrings:
        return []
    parts = ["### 核心 API\n"]
    for ds in docstrings:
        parts.append(f"- **`{ds['name']}`** — {ds['docstring']}")
    parts.append("")
    return parts


def _generate_pr_title(analysis: DiffAnalysis, commits: List[str],
                       pr_info: Optional[Dict] = None) -> str:
    """Generate the PR title from analysis, commits, or existing PR info.

    Args:
        analysis: DiffAnalysis result.
        commits: List of commit SHA strings.
        pr_info: Optional existing PR info dict.

    Returns:
        PR title string.
    """
    if pr_info and pr_info.get('title'):
        return pr_info['title']
    if commits:
        result = run_git("log", "-1", "--pretty=format:%s", commits[0])
        return result.stdout.strip()[:70] or "Update code"

    if analysis.added_classes:
        return f"feat: add {', '.join(analysis.added_classes[:2])}"
    if analysis.added_funcs:
        return f"feat: add {', '.join(analysis.added_funcs[:2])}"
    if analysis.modified_classes:
        return f"refactor: update {', '.join(analysis.modified_classes[:2])}"
    if analysis.modified_funcs:
        return f"refactor: update {', '.join(analysis.modified_funcs[:2])}"
    if analysis.files:
        return f"refactor: update {analysis.files[0].split('/')[-1]}"
    return "Update code"


def _collect_related_issues(commits: List[str]) -> str:
    """Extract related issues from commit messages.

    Args:
        commits: List of commit SHA strings.

    Returns:
        Comma-separated issue references, or 'N/A'.
    """
    all_issues: set = set()
    for sha in commits:
        msg = run_git("log", "-1", "--pretty=format:%s%n%b", sha).stdout
        all_issues.update(re.findall(r"#\d+", msg))
    return ", ".join(sorted(all_issues)) if all_issues else "N/A"


def _build_architecture_overview(analysis: DiffAnalysis) -> List[str]:
    """Build the architecture overview section.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        List of formatted lines.
    """
    desc_parts: List[str] = []
    file_groups: Dict[str, List[str]] = {}
    for f in analysis.files:
        parts = f.split('/')
        if len(parts) >= 3:
            group_key = '/'.join(parts[:2])
        elif len(parts) >= 2:
            group_key = parts[0]
        else:
            group_key = 'root'
        file_groups.setdefault(group_key, []).append(f)

    if not file_groups:
        return desc_parts

    desc_parts.append("### 架构概览\n")
    for group, group_files in file_groups.items():
        group_adds = sum(analysis.file_stats.get(f, (0, 0))[0] for f in group_files)
        group_dels = sum(analysis.file_stats.get(f, (0, 0))[1] for f in group_files)
        desc_parts.append(
            f"**`{group}/`** — {len(group_files)} files "
            f"(+{group_adds}/-{group_dels})"
        )

        group_changes = _extract_group_changes(analysis, group_files)
        for c in group_changes[:5]:
            desc_parts.append(f"  - {c}")
        if len(group_changes) > 5:
            desc_parts.append(f"  - ...and {len(group_changes)} more changes")
        desc_parts.append("")

    return desc_parts


def _extract_group_changes(analysis: DiffAnalysis,
                           group_files: List[str]) -> List[str]:
    """Extract key changes for a file group.

    Args:
        analysis: DiffAnalysis result.
        group_files: List of file paths in the group.

    Returns:
        List of change description strings.
    """
    group_changes: List[str] = []
    for f in group_files:
        changes = analysis.file_changes.get(f, FileChanges())
        for cls in changes.added_classes:
            base_info = f"(inherits {cls['base']})" if cls.get('base') else ""
            group_changes.append(f"Add class `{cls['name']}` {base_info}")
        for func in changes.added_funcs:
            group_changes.append(f"Add `{func['name']}()`")
        for cls in changes.removed_classes:
            group_changes.append(f"Remove class `{cls}`")
        for func in changes.removed_funcs:
            group_changes.append(f"Remove `{func}()`")
    return group_changes


def _build_file_stats_section(analysis: DiffAnalysis) -> List[str]:
    """Build the file change statistics section.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        List of formatted lines.
    """
    desc_parts: List[str] = []
    desc_parts.append("### 变更统计\n")
    file_count = len(analysis.files)
    adds = analysis.additions
    dels = analysis.deletions
    desc_parts.append(f"共 **{file_count}** 个文件，**+{adds}** / **-{dels}** 行\n")

    if analysis.files:
        desc_parts.append("| 文件 | 新增 | 删除 |")
        desc_parts.append("|------|------|------|")
        for f in analysis.files[:15]:
            f_adds, f_dels = analysis.file_stats.get(f, (0, 0))
            short_name = f.split('/')[-1]
            desc_parts.append(f"| `{short_name}` | +{f_adds} | -{f_dels} |")
        if len(analysis.files) > 15:
            desc_parts.append("| ... | | |")
            remaining = len(analysis.files) - 15
            desc_parts.append(f"| *等 {remaining} 个文件* | | |")

    if analysis.modules:
        modules_str = '`, `'.join(analysis.modules[:5])
        desc_parts.append(f"\n**涉及模块**: `{modules_str}`")

    return desc_parts


def _build_affected_section(analysis: DiffAnalysis) -> str:
    """Build the potentially affected functionality section.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        Formatted affected section string.
    """
    affected: List[str] = []

    if analysis.modified_funcs:
        affected.append("**修改的方法**（可能影响调用方）:")
        for func in analysis.modified_funcs[:5]:
            affected.append(f"  - `{func}()`")

    if analysis.modified_classes:
        affected.append("**修改的类**:")
        for cls in analysis.modified_classes[:5]:
            affected.append(f"  - `{cls}`")

    if analysis.removed_funcs:
        affected.append("**删除的方法**（请确认无调用）:")
        for func in analysis.removed_funcs[:5]:
            affected.append(f"  - `{func}()`")

    if analysis.removed_classes:
        affected.append("**删除的类**:")
        for cls in analysis.removed_classes[:5]:
            affected.append(f"  - `{cls}`")

    if analysis.modules:
        modules_str = '`, `'.join(analysis.modules)
        affected.append(f"\n**涉及模块**: `{modules_str}`")

    return "\n".join(affected) if affected else "无明显影响（新增代码或内部重构）"


def _build_test_section(analysis: DiffAnalysis) -> str:
    """Build the test cases section.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        Formatted test section string.
    """
    test_files = [f for f in analysis.files if "test" in f.lower()]
    if not test_files:
        return "N/A（本次变更未涉及测试文件）"

    test_info: List[str] = []
    for tf in test_files[:5]:
        adds, dels = analysis.file_stats.get(tf, (0, 0))
        test_info.append(f"- `{tf.split('/')[-1]}` (+{adds}/-{dels})")
    return "\n".join(test_info)


def _build_skill_section(skill_files: List[str]) -> List[str]:
    """Build the Skill authoring guidelines section.

    Args:
        skill_files: List of skill-related file paths.

    Returns:
        List of formatted lines.
    """
    desc_parts: List[str] = []
    has_skill_md = any(f.endswith('SKILL.md') for f in skill_files)
    has_scripts = any('/scripts/' in f for f in skill_files)
    has_readme = any(f == 'skills/README.md' for f in skill_files)
    desc_parts.append("### Skill 创作规范\n")
    desc_parts.append("遵循 Claude Code Skill 标准目录结构：")
    if has_skill_md:
        desc_parts.append("- `SKILL.md` — Skill 元数据、触发场景、命令文档")
    if has_scripts:
        desc_parts.append("- `scripts/` — 可执行脚本，提供 CLI 入口")
    if has_readme:
        desc_parts.append("- `skills/README.md` — Skills 总览与创作指南")
    desc_parts.append("")
    return desc_parts


# ============================================================================
# Main PR content generation
# ============================================================================

def generate_pr_content(diff: str, commits: List[str],
                        pr_info: Optional[Dict] = None) -> Tuple[str, str]:
    """Generate detailed PR title and description based on diff.

    Args:
        diff: Raw unified diff text.
        commits: List of commit SHA strings.
        pr_info: Optional existing PR info dict.

    Returns:
        Tuple of (title, body).
    """
    analysis = analyze_diff(diff)

    fallback_domain, fallback_points = infer_feature_purpose(analysis)
    commit_data = _parse_conventional_commits(commits) if commits else []
    docstrings = _extract_docstrings_from_diff(diff)
    feature_domain, feature_points = _synthesize_pr_purpose(
        commit_data, docstrings, analysis, fallback_domain, fallback_points
    )

    title = _sanitize_framework_refs(
        _generate_pr_title(analysis, commits, pr_info)
    )
    related_issue = _collect_related_issues(commits) if commits else "N/A"
    reason = _build_reason_section(analysis, feature_domain, feature_points)

    desc_parts = _build_architecture_overview(analysis)

    desc_parts.extend(_build_api_section(docstrings))

    skill_files = [f for f in analysis.files if 'skills/' in f]
    if skill_files:
        desc_parts.extend(_build_skill_section(skill_files))

    desc_parts.extend(_build_file_stats_section(analysis))
    description = "\n".join(desc_parts) if desc_parts else "代码优化"

    test_cases = _build_test_section(analysis)
    affected_str = _build_affected_section(analysis)

    body = _sanitize_framework_refs(f"""## 相关的Issue
{related_issue}

## 原因（目的、解决的问题等）
{reason}

## 描述（做了什么，变更了什么）
{description}

## 测试用例（新增、改动、可能影响的功能）
{test_cases}

## 可能影响的功能
{affected_str}
""")
    return title, body
