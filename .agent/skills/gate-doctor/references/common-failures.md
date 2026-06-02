# Gate Doctor — Failure Recipe Catalogue

One section per `Check_*` stage plus `/check-pr`. Each section has the
**signature** (what the failure looks like in the Jenkins log / PR comments)
and the **fix recipe** (concrete edit steps).

When `gate_doctor.py diagnose` returns a `failed_stages` list, look up each
entry here before editing.

## Contents

- [/check-pr (description / self-checklist gate)](#check-pr--description--self-checklist-gate)
- [Check_DT_Design](#check_dt_design)
- [Check_Pylint](#check_pylint)
- [Check_Cpplint](#check_cpplint)
- [Check_Codespell](#check_codespell)
- [Check_Cmakelint](#check_cmakelint)
- [Check_Markdownlint](#check_markdownlint)
- [Check_Tab](#check_tab)
- [Check_Utf8](#check_utf8)
- [Check_Lizard](#check_lizard)
- [Check_ShellCheck](#check_shellcheck)
- [Check_ClangFormat](#check_clangformat)
- [Check_Linklint](#check_linklint)
- [Check_Cppcheck / Check_Notebooklint / Check_Rstlint / Check_Scanoss](#check_cppcheck--check_notebooklint--check_rstlint--check_scanoss)
- [When the recipe doesn't match](#when-the-recipe-doesnt-match)

---

## `/check-pr`  (description / self-checklist gate)

**Signature** (PR comment from `micro-compass`):

```
当前/check-pr未通过，原因如下:
以下Pull Request描述检查项未通过:
    模板中'Test Plan and Test Result' 信息为空，请补充对应信息。
    选项未通过检查： **设计**：...
以下issue检查项未通过:
    Pull Request未关联issue
```

**Fix**

1. Compare the PR body against `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
   in the active repo. Every `**Section**` header must be present.
2. For each missing/empty section the bot named, write a substantive
   paragraph. **Empty placeholder text counts as missing.**
3. For self-checklist items the bot named, tick the box with `[x]`. If the
   user has not actually completed the item (e.g. "设计" without an actual
   Maintainer review), explicitly call that out in chat and ask the user
   before ticking.
4. If "Pull Request未关联issue" appears, append `Fixes #<n>` to the body.
   Ask the user for the issue number — never invent one.
5. Push the updated body via
   ```bash
   python3 .agent/skills/autogit/scripts/autogit.py update <pr> \
       --title "<unchanged>" --body "$(cat /tmp/new_body.md)"
   ```
6. Re-trigger: `gate_doctor.py check-pr <pr>`.

---

## `Check_DT_Design`

**Signature**

```
[PYTHON] Function 'test_xxx' is missing: Description:, Expectation:
[FAILURE] comment issues found in test cases
[ERROR] Check DT design is failed, the actual value is 1.
```

MindSpore mandates every test function (Python + C++) carries a docstring
with three fields: `Feature:`, `Description:`, `Expectation:`.

**Fix recipe (Python)**

For each function listed:

```python
def test_xxx():
    """
    Feature: <one-line what feature is under test>
    Description: <input data, scenario, parallel layout, etc.>
    Expectation: <success criteria, e.g. "loss aligned to baseline within 0.005">
    """
    ...
```

Rules:

- Use the exact key labels `Feature:`, `Description:`, `Expectation:`
  (case-sensitive, trailing colon).
- All three keys must appear, even if just one line each.
- Don't compress to a single line — the parser scans for each key on its
  own line.

**Fix recipe (C++)**

```cpp
/// Feature: <...>
/// Description: <...>
/// Expectation: <...>
TEST_F(TestSuit, test_xxx) { ... }
```

---

## `Check_Pylint`

**Signature**

```
[INFO] Filter pylint.
<file>:<line>: [<rule>(<rule-name>), <function>] <message>
[ERROR] Filter pylint is failed, the actual value is 1.
```

Each line is a real violation. CI uses pylint 3.3.7 with the rcfile at
`https://tools.mindspore.cn/tools/check/pylint/rules/pylintrc`. Reproduce
locally:

```bash
pip install pylint==3.3.7
curl -fsSL https://tools.mindspore.cn/tools/check/pylint/rules/pylintrc \
     -o /tmp/ms_pylintrc
pylint --rcfile=/tmp/ms_pylintrc <files>
```

The score after the fix must be **10.00/10**.

**Common rule recipes**:

| Rule | Cause | Fix |
|---|---|---|
| `W1510 subprocess-run-check` | `subprocess.run(...)` without explicit `check=` | Add `check=False` (or `True` if you actually want raise-on-error). |
| `W0612 unused-variable` | A local binding that's never read | Delete the binding, OR use it, OR (last resort) prefix with `_` to mark intentional. |
| `C0116 missing-function-docstring` | Function lacks a `"""..."""` | Add a one-line docstring. `_private` helpers also need one. |
| `C0115 missing-class-docstring` | Class lacks a docstring | Add a one-line class docstring. |
| `C0301 line-too-long` | Line >100 chars (per MS rcfile) | Wrap or refactor; never use `# pylint: disable=line-too-long`. |
| `C0413 wrong-import-position` | Import after top-of-file code | Move imports to the top. |
| `E0401 import-error` | Module can't be imported in CI env | Verify it's in `requirements.txt`; for test-only deps, use `pytest.importorskip`. |
| `R0913 too-many-arguments` | Function with >5 args | Bundle related args into a dataclass, or accept the warning if the rcfile has it disabled. |
| `R0902 too-many-instance-attributes` | Class with >7 attrs | Same — refactor to nested dataclasses, OR accept if rcfile permits. |
| `W0702 bare-except` | `except:` without a type | Replace with `except Exception:` and add `# pylint: disable=broad-except` only if catching everything is genuinely needed. |
| `W0102 dangerous-default-value` | `def f(x=[])` style | Use `None` default + assign inside. |

If the rule appears in `failures[].rule` but isn't listed above, lookup at
https://pylint.readthedocs.io/en/stable/user_guide/messages/ — the lookup
URL is `.../messages/<category>/<rule-name>.html`.

---

## `Check_Cpplint`

**Signature**

```
<file>:<line>:  <message> [<category>] [<priority>]
[ERROR] Cpplint error number: <N>
```

**Fix recipe**: address each line directly. Common categories:

| Category | Fix |
|---|---|
| `whitespace/line_length` | Wrap at 120 cols. |
| `build/include_order` | C-style headers first, then C++ stdlib, then project. |
| `runtime/references` | Replace non-const reference parameter with pointer or const ref. |
| `readability/braces` | Add braces around single-line `if`/`for` bodies. |

---

## `Check_Codespell`

**Signature**

```
<file>:<line>: <misspelled> ==> <suggestion>
[ERROR] Codespell error number: <N>
```

**Fix recipe**: accept the suggested spelling unless it's a legitimate
domain term. For false positives (e.g. variable names, intentional jargon),
add the term to `.codespell-ignore` or `tools/codespell.allow` in the repo
root (depending on the repo's convention).

---

## `Check_Cmakelint`

**Signature**

```
<file>:<line>: [<category>] <message>
[ERROR] Cmakelint error number: <N>
```

**Common categories**:

| Category | Fix |
|---|---|
| `convention/filename` | Lowercase + underscores for `.cmake` files. |
| `package/consistency` | Match `find_package(<X>)` / `target_link_libraries(<X>)` capitalisation. |
| `whitespace/newline` | Ensure final newline at EOF. |

---

## `Check_Markdownlint`

**Signature**

```
<file>:<line> MD<NNN>/<rule-name>  <message>
[ERROR] Markdownlint error number: <N>
```

**Common rules**:

| Rule | Fix |
|---|---|
| `MD013 line-length` | Wrap markdown at 120 cols. Code blocks and tables are usually exempt by repo config. |
| `MD025 single-h1` | One `# Title` per file; demote duplicates to `## `. |
| `MD031 blanks-around-fences` | Blank line before and after each ```` ``` ```` fence. |
| `MD040 fenced-code-language` | Tag every fence with a language, e.g. ```` ```python ````. |
| `MD012 no-multiple-blanks` | Collapse multiple blank lines to one. |

---

## `Check_Tab`

**Signature**

```
<file>:<line>: tab character found
[ERROR] Tab error number: <N>
```

**Fix**: replace tabs with spaces. For `.py` use 4 spaces, for C/C++/cmake
use the file's existing indent width (usually 2 or 4 spaces). One-shot:

```bash
expand -t 4 <file> > <file>.tmp && mv <file>.tmp <file>
```

---

## `Check_Utf8`

**Signature**

```
<file>: non-UTF-8 byte at offset <N>
[ERROR] Utf8 error number: <N>
```

**Fix**: re-encode the file from its actual encoding (usually GBK) to UTF-8:

```bash
iconv -f GBK -t UTF-8 <file> > <file>.utf8 && mv <file>.utf8 <file>
```

Inspect first with `file <path>` to confirm the source encoding.

---

## `Check_Lizard`

**Signature**

```
<file>:<line>: <function>: cyclomatic complexity = <N> (threshold <T>)
[ERROR] Lizard error number: <N>
```

**Fix**: refactor the offending function — extract helper functions to
bring CCN under the threshold (typically 20). Don't silence the check via
config unless the function is intrinsically hard to split (rare).

---

## `Check_ShellCheck`

**Signature**

```
In <file> line <N>:
SC<NNNN>: <message>
[ERROR] ShellCheck error number: <N>
```

**Fix**: address by SC code:

| SC code | Fix |
|---|---|
| SC2086 | Quote variable expansions: `"$VAR"`. |
| SC2155 | Split `local x=$(cmd)` into `local x; x=$(cmd)`. |
| SC2068 | Use `"$@"` instead of `$@`. |
| SC2034 | Mark unused variable as `# shellcheck disable=SC2034` only if it's exported for a downstream tool. |

---

## `Check_ClangFormat`

**Signature**

```
<file>:<line>: code does not match clang-format
[ERROR] ClangFormat error number: <N>
```

**Fix**: apply clang-format using the repo's `.clang-format`:

```bash
clang-format -i <file>
git diff <file>
```

Review the diff carefully — the formatter sometimes reflows comments or
docstrings in ways that change meaning.

---

## `Check_Linklint`

**Signature**

```
<file>:<line>: dead link: <url>
[ERROR] Linklint error number: <N>
```

**Fix**: replace the dead URL with a working one. If the resource genuinely
moved, use the canonical new URL; if it's gone, remove the reference.

---

## `Check_Cppcheck` / `Check_Notebooklint` / `Check_Rstlint` / `Check_Scanoss`

**Cppcheck**: address each warning by code (e.g. `[uninitvar]`,
`[memleak]`).  Scanoss flags potential license/copyright issues —
**always** stop and ask the user; do not auto-modify.

Notebooklint / Rstlint follow the same pattern as their language siblings:
fix the offending line directly.

---

## When the recipe doesn't match

If the failure signature isn't listed here:

1. Read the Jenkins log lines around the `[ERROR]` marker (the
   `raw_log_excerpt` field of `diagnose` output has the last ~200).
2. Surface the unrecognised failure to the user — do NOT guess.
3. After the user explains the right fix, add a new section to this
   document as part of the same PR.
