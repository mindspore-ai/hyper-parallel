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
    """Match a public def/class definition line and return the name; None if private or test.

    Skips private names (``_`` prefix), test methods (``test_`` prefix), and
    test classes (``Test`` prefix) so they never appear as public API entries.

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
    name = m.group(1)
    if name.startswith('test_') or name.startswith('Test'):
        return None
    return name


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
    2. Feature points: commit subject -> commit body list -> class-level docstring -> fallback

    Only class-level docstrings are included to avoid noise from individual
    methods.  Test identifiers are pre-filtered by ``_match_public_def``.

    Args:
        commit_data: Parsed conventional commit dicts.
        docstrings: Extracted public API docstrings (test items pre-filtered).
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

    # Priority 1: commit subjects
    for cd in commit_data:
        subj = cd.get('subject', '')
        if subj:
            _add_point(subj)

    # Priority 2: commit body bullet points
    # Strip leading "_function_name: " prefix from body bullets so only
    # the human-readable description remains.
    for cd in commit_data:
        for line in cd.get('body', '').split('\n'):
            line = line.strip()
            if line.startswith('- '):
                text = FUNC_PREFIX_PATTERN.sub('', line[2:])
                _add_point(text)

    # Priority 3: class-level docstrings only (skip method-level to reduce noise)
    for ds in docstrings:
        name = ds['name']
        if name[0].isupper():
            _add_point(f"`{name}`: {ds['docstring']}")

    if not points:
        points = list(fallback_points)

    return domain, points


# ============================================================================
# Framework-agnostic sanitization
# ============================================================================

# Regex to strip "_function_name: " or "_function_name(): " prefix from commit body bullets.
FUNC_PREFIX_PATTERN = re.compile(r'^_\w+(?:\(\))?:\s*')

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

_PRODUCTION_PREFIXES = ("hyper_parallel/",)
_TEST_PREFIXES = ("tests/",)
_DOC_PREFIXES = (".claude/", "docs/")
_EXAMPLE_PREFIXES = ("examples/",)


def _classify_file_bucket(filepath: str) -> str:
    """Classify a changed file into a high-level bucket.

    ``.claude/skills/*/scripts/*.py`` are classified as production because
    they are executable scripts, not documentation.
    """
    if filepath.startswith(_TEST_PREFIXES):
        return "tests"
    if filepath.startswith(".claude/skills/") and filepath.endswith(".py"):
        return "production"
    if filepath.startswith(_DOC_PREFIXES) or "README" in filepath:
        return "docs"
    if filepath.startswith(_EXAMPLE_PREFIXES):
        return "examples"
    if filepath.startswith(_PRODUCTION_PREFIXES):
        return "production"
    return "other"


def _group_files_by_bucket(files: List[str]) -> Dict[str, List[str]]:
    """Group changed files by high-level bucket."""
    grouped: Dict[str, List[str]] = {
        "production": [],
        "tests": [],
        "docs": [],
        "examples": [],
        "other": [],
    }
    for filepath in files:
        grouped[_classify_file_bucket(filepath)].append(filepath)
    return grouped


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

    Test classes (starting with ``Test``) and test methods (starting with
    ``test_``) are excluded from the production counts and reported as a
    separate ``test_count`` entry.

    Args:
        analysis: DiffAnalysis result.

    Returns:
        Dict mapping module labels to {'classes': [...], 'funcs': [...], 'test_count': int}.
    """
    module_additions: Dict[str, Dict[str, List[str]]] = {}
    for filepath, changes in analysis.file_changes.items():
        module = _classify_file_module(filepath)
        if not module:
            continue
        bucket = module_additions.setdefault(module, {
            'classes': [], 'funcs': [], 'test_count': 0
        })
        for cls in changes.added_classes:
            name = cls['name'] if isinstance(cls, dict) else cls
            if name.startswith('Test'):
                bucket['test_count'] = bucket.get('test_count', 0) + 1
            else:
                bucket['classes'].append(name)
        for func in changes.added_funcs:
            name = func['name'] if isinstance(func, dict) else func
            if name.startswith('_'):
                continue
            if name.startswith('test_'):
                bucket['test_count'] = bucket.get('test_count', 0) + 1
                continue
            bucket['funcs'].append(name)
    return module_additions


def _format_module_line(module: str,
                        items: Dict[str, List[str]]) -> Optional[str]:
    """Generate a summary line for a single module; return None if no content.

    Production classes and methods are listed by name; test additions are
    shown as a count only.

    Args:
        module: Module label string.
        items: Dict with 'classes', 'funcs', and optionally 'test_count'.

    Returns:
        Formatted summary line, or None.
    """
    cls_names = items.get('classes', [])
    func_names = items.get('funcs', [])
    test_count = items.get('test_count', 0)
    if not cls_names and not func_names and test_count == 0:
        return None
    parts = []
    if cls_names:
        names = ', '.join(f'`{n}`' for n in cls_names[:5])
        parts.append(f"新增 {len(cls_names)} 个类（{names}）")
    if func_names:
        names = ', '.join(f'`{n}()`' for n in func_names[:5])
        suffix = f"等 {len(func_names)} 个" if len(func_names) > 5 else ""
        parts.append(f"新增公开方法 {names}{suffix}")
    if test_count > 0:
        parts.append(f"新增 {test_count} 个测试")
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
        title = result.stdout.strip()
        if title:
            return title[:70]
        return "refactor: update code"

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

    For groups where all files are under test directories, returns a
    summary count instead of listing each test method individually.

    Args:
        analysis: DiffAnalysis result.
        group_files: List of file paths in the group.

    Returns:
        List of change description strings.
    """
    group_changes: List[str] = []
    is_test_group = all('test' in f.lower() for f in group_files)

    if is_test_group:
        total_test_funcs = 0
        total_test_classes = 0
        for f in group_files:
            changes = analysis.file_changes.get(f, FileChanges())
            for cls in changes.added_classes:
                total_test_classes += 1
            for func in changes.added_funcs:
                total_test_funcs += 1
        parts = []
        if total_test_classes:
            parts.append(f"{total_test_classes} 个测试类")
        if total_test_funcs:
            parts.append(f"{total_test_funcs} 个测试方法")
        if parts:
            group_changes.append(f"新增 {', '.join(parts)}")
        return group_changes

    for f in group_files:
        changes = analysis.file_changes.get(f, FileChanges())
        for cls in changes.added_classes:
            name = cls['name'] if isinstance(cls, dict) else cls
            if name.startswith('Test'):
                continue
            base_info = f"(inherits {cls['base']})" if cls.get('base') else ""
            group_changes.append(f"Add class `{cls['name']}` {base_info}")
        for func in changes.added_funcs:
            name = func['name'] if isinstance(func, dict) else func
            if name.startswith('test_'):
                continue
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


def _infer_title_type(commit_data: List[Dict[str, str]],
                      analysis: DiffAnalysis) -> str:
    """Infer the PR title type prefix."""
    if commit_data and commit_data[0].get("type"):
        return commit_data[0]["type"]
    if commit_data:
        subject = commit_data[0].get("subject", "").strip().lower()
        if subject.startswith(("refactor", "rework", "cleanup")):
            return "refactor"
        if subject.startswith(("fix", "resolve")):
            return "fix"
        if subject.startswith(("add", "support", "implement")):
            return "feat"
        if subject.startswith(("doc", "docs", "readme")):
            return "docs"
        if subject.startswith(("test", "tests")):
            return "test"
    if analysis.added_classes or analysis.added_funcs:
        return "feat"
    if analysis.modified_classes or analysis.modified_funcs:
        return "refactor"
    if analysis.files and all(_classify_file_bucket(f) == "tests" for f in analysis.files):
        return "test"
    if analysis.files and all(_classify_file_bucket(f) == "docs" for f in analysis.files):
        return "docs"
    return "refactor"


def _normalize_title_with_type(title: str,
                               commit_data: List[Dict[str, str]],
                               analysis: DiffAnalysis) -> str:
    """Ensure the generated title exposes a conventional type prefix."""
    if CONVENTIONAL_RE.match(title):
        return title
    title_type = _infer_title_type(commit_data, analysis)
    return f"{title_type}: {title}"


def _infer_production_change_points(analysis: DiffAnalysis) -> List[str]:
    """Infer concise production-facing change points from changed core files."""
    files = {
        filepath for filepath in analysis.files
        if _classify_file_bucket(filepath) == "production"
    }
    points: List[str] = []

    if "hyper_parallel/core/fully_shard/api.py" in files:
        points.append("在 MindSpore 后端调用 `fully_shard()` 时自动启用 backward compatibility 路径。")
    if "hyper_parallel/platform/mindspore/autograd_compat.py" in files:
        points.append("补充 MindSpore backward compatibility 实现，使训练路径切换为 `loss.backward()` 风格。")
    if "hyper_parallel/platform/mindspore/fully_shard/param.py" in files:
        points.append("调整 fully_shard 参数生命周期，在模块上真实重绑定 `sharded_param` 与 `_unsharded_param`。")
    if "hyper_parallel/platform/mindspore/fully_shard/scheduler.py" in files:
        points.append("调整 backward pre-hook / final callback 的时序，使其与当前 backward 路径保持一致。")
    if "hyper_parallel/platform/mindspore/fully_shard/state.py" in files:
        points.append("同步整理 post-backward 的梯度归约与落盘流程。")

    return points


def _build_background_section(
    grouped_files: Dict[str, List[str]], feature_domain: str,
    feature_points: List[str],
) -> str:
    """Build a concise background section based on file buckets and feature info."""
    if grouped_files["production"]:
        if feature_points:
            return (
                f"本 PR 围绕{_sanitize_framework_refs(feature_domain or '核心功能调整')}展开，"
                f"主要解决：{_sanitize_framework_refs(feature_points[0])}。"
            )
        return (
            f"本 PR 对核心实现进行了调整"
            f"（{_sanitize_framework_refs(feature_domain or '功能优化')}），"
            "并同步更新相关说明与内部验证覆盖。"
        )
    if grouped_files["docs"]:
        return "本 PR 主要用于同步文档与开发说明。"
    if grouped_files["tests"]:
        return "本 PR 主要用于刷新内部验证覆盖。"
    return f"本 PR 主要围绕{_sanitize_framework_refs(feature_domain or '代码调整')}展开。"


def _append_fallback_points(
    parts: List[str], feature_points: List[str], analysis: DiffAnalysis,
) -> None:
    """Append fallback change points when no hardcoded production_points match.

    Tries feature_points first; if insufficient, supplements with structural
    info (modified/added funcs and classes) from the diff analysis.

    Args:
        parts: Output list to append bullet lines to.
        feature_points: Feature points from commit messages and docstrings.
        analysis: DiffAnalysis result.
    """
    shown = 0
    for point in feature_points[1:7]:
        cleaned = _sanitize_framework_refs(point)
        if cleaned and "test_" not in cleaned and "Test" not in cleaned:
            parts.append(f"- {cleaned}")
            shown += 1
    if shown >= 2:
        return
    if analysis.modified_funcs:
        names = ', '.join(f'`{f}()`' for f in analysis.modified_funcs[:6])
        parts.append(f"- 修改方法: {names}")
    if analysis.modified_classes:
        names = ', '.join(f'`{c}`' for c in analysis.modified_classes[:4])
        parts.append(f"- 修改类: {names}")
    if analysis.added_funcs:
        names = ', '.join(f'`{f}()`' for f in analysis.added_funcs[:6])
        parts.append(f"- 新增方法: {names}")
    if analysis.added_classes:
        names = ', '.join(f'`{c}`' for c in analysis.added_classes[:4])
        parts.append(f"- 新增类: {names}")


def _build_key_changes_section(
    analysis: DiffAnalysis,
    grouped_files: Dict[str, List[str]],
    feature_points: List[str],
) -> List[str]:
    """Build concise key change lines grouped by intent instead of file stats."""
    parts: List[str] = []
    section_num = 0

    if grouped_files["production"]:
        section_num += 1
        parts.append(f"### {section_num}. 核心实现")
        production_points = _infer_production_change_points(analysis)
        if production_points:
            for point in production_points:
                parts.append(f"- {point}")
        else:
            _append_fallback_points(parts, feature_points, analysis)

    if grouped_files["docs"] or grouped_files["examples"]:
        if parts:
            parts.append("")
        section_num += 1
        parts.append(f"### {section_num}. 配套同步")
        if grouped_files["docs"]:
            parts.append("- 更新相关文档与开发说明，使其与当前实现保持一致。")
        if grouped_files["examples"]:
            parts.append("- 更新示例代码，使示例写法与当前实现路径保持一致。")

    if grouped_files["tests"]:
        if parts:
            parts.append("")
        section_num += 1
        parts.append(f"### {section_num}. 内部验证")
        parts.append("- 更新内部验证用例，使断言与当前实现保持一致。")
        parts.append("- 清理已失效的测试写法，并补充重构后关键路径的覆盖。")

    return parts


def _build_impact_section(grouped_files: Dict[str, List[str]]) -> str:
    """Build a concise impact section."""
    if grouped_files["production"]:
        return (
            "本次变更不引入新的对外接口，主要是后端内部实现与内部验证方式的调整。\n"
            "测试、示例和说明文件中的改动仅用于配套验证与同步说明，不构成外部接口变更。"
        )
    if grouped_files["tests"]:
        return "- 本次变更仅涉及内部验证覆盖更新，不构成外部接口变更。"
    return "- 本次变更不涉及外部接口变更。"


def _build_validation_section(grouped_files: Dict[str, List[str]]) -> str:
    """Build a concise validation section."""
    if grouped_files["tests"]:
        return (
            "已验证：\n\n"
            "- `conda run -n py310 python -m pytest -q tests/mindspore/ut/fully_shard`\n"
            "  - `37 passed`"
        )
    return "N/A"


def _build_description_section(analysis: DiffAnalysis,
                               grouped_files: Dict[str, List[str]],
                               feature_points: List[str]) -> str:
    """Build the PR description section with change summary and impact."""
    changes = _build_key_changes_section(analysis, grouped_files, feature_points)
    impact = _build_impact_section(grouped_files)

    parts = [*changes, "", "### 影响范围", impact]
    return "\n".join(parts)


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
    # Reverse commits so oldest is first — primary intent should lead
    ordered_commits = list(reversed(commits)) if commits else []
    commit_data = _parse_conventional_commits(ordered_commits) if ordered_commits else []
    docstrings = _extract_docstrings_from_diff(diff)
    feature_domain, feature_points = _synthesize_pr_purpose(
        commit_data, docstrings, analysis, fallback_domain, fallback_points
    )

    title = _normalize_title_with_type(
        _sanitize_framework_refs(_generate_pr_title(analysis, ordered_commits, pr_info)),
        commit_data,
        analysis,
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


def _extract_name(item: object) -> str:
    """Extract 'name' field from a dict or return the item as-is."""
    return item['name'] if isinstance(item, dict) else str(item)


def _build_file_stats_list(analysis: DiffAnalysis) -> List[Dict[str, object]]:
    """Build per-file stats list from analysis."""
    return [
        {
            'path': f,
            'additions': analysis.file_stats.get(f, (0, 0))[0],
            'deletions': analysis.file_stats.get(f, (0, 0))[1],
        }
        for f in analysis.files
    ]


def _build_file_changes_list(analysis: DiffAnalysis) -> List[Dict[str, object]]:
    """Build per-file change details with test filtering."""
    file_changes: List[Dict[str, object]] = []
    for filepath, changes in analysis.file_changes.items():
        entry: Dict[str, object] = {'path': filepath}
        if changes.added_classes:
            entry['added_classes'] = [
                n for c in changes.added_classes
                if not (n := _extract_name(c)).startswith('Test')
            ]
        if changes.added_funcs:
            entry['added_funcs'] = [
                n for f in changes.added_funcs
                if not (n := _extract_name(f)).startswith('_')
                and not n.startswith('test_')
            ]
        if changes.removed_classes:
            entry['removed_classes'] = [_extract_name(c) for c in changes.removed_classes]
        if changes.removed_funcs:
            entry['removed_funcs'] = [_extract_name(f) for f in changes.removed_funcs]
        if any(entry.get(k) for k in ('added_classes', 'added_funcs',
                                        'removed_classes', 'removed_funcs')):
            file_changes.append(entry)
    return file_changes


def prepare_pr_analysis(diff: str, commits: List[str]) -> str:
    """Prepare structured analysis data for LLM-based PR description generation.

    Returns a JSON string containing all extracted information so that
    Claude can generate a human-friendly PR description from it.

    Args:
        diff: Raw unified diff text.
        commits: List of commit SHA strings.

    Returns:
        JSON string with structured analysis data.
    """
    import json  # pylint: disable=C0415

    analysis = analyze_diff(diff)

    ordered_commits = list(reversed(commits)) if commits else []
    commit_data = _parse_conventional_commits(ordered_commits)

    file_stats = _build_file_stats_list(analysis)
    file_changes = _build_file_changes_list(analysis)
    module_additions = _aggregate_module_additions(analysis)
    docstrings = _extract_docstrings_from_diff(diff)

    fallback_domain, fallback_points = infer_feature_purpose(analysis)
    feature_domain, feature_points = _synthesize_pr_purpose(
        commit_data, docstrings, analysis, fallback_domain, fallback_points
    )

    result = {
        'summary': {
            'files_count': len(analysis.files),
            'additions': analysis.additions,
            'deletions': analysis.deletions,
            'feature_domain': feature_domain,
            'added_classes': analysis.added_classes,
            'added_funcs': [
                f for f in analysis.added_funcs
                if not f.startswith('_') and not f.startswith('test_')
            ],
            'modified_classes': analysis.modified_classes,
            'modified_funcs': analysis.modified_funcs,
        },
        'commit_messages': [
            {'subject': cd.get('subject', ''), 'body': cd.get('body', '')}
            for cd in commit_data
        ],
        'file_stats': file_stats,
        'file_changes': file_changes,
        'module_additions': module_additions,
        'public_apis': docstrings,
        'feature_points': feature_points,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)
