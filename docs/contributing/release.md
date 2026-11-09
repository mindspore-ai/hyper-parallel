# Release Process

**English** | [简体中文](release.zh.md)

## Version Naming

Follow semantic versioning (SemVer): `vMAJOR.MINOR.PATCH`.

- **MAJOR**: Incompatible API changes.
- **MINOR**: Backward-compatible new features.
- **PATCH**: Backward-compatible bug fixes.

## Release Workflow

### 1. Prepare the Version

```bash
# Confirm all features for this release have been merged into master.
git checkout master
git pull upstream master

# Check the version number defined in setup.py or __init__.py.
```

### 2. Generate Release Notes

```bash
# Review changes from the previous tag to HEAD.
git log vPREV_TAG..HEAD --oneline
git diff vPREV_TAG..HEAD --stat
```

Release notes must include:

- **Complete feature list**: List all implemented features for the first official release.
- **Incremental changes**: Identify changes since the previous release.
- **Contributors**
- **Merged PRs**
- **Known limitations**
- **Upgrade guide**

### 3. Update Documentation

Update the following before every release:

| Documentation | Required updates |
|---------------|------------------|
| README.md / README.zh.md | Feature checkboxes and new feature sections |
| docs/guide/ | New feature guides and updates to existing guides |
| docs/api/ | New interfaces and updated parameters |
| docs/faq.md | New frequently asked questions |
| Release Notes | Version history |

English is the default language for documents without a language suffix. Keep Chinese translations in sibling `.zh.md` files and update both versions together.

### 4. Create a Tag

```bash
git tag -a vX.Y.Z -m "HyperParallel vX.Y.Z release"
git push upstream vX.Y.Z
```

### 5. Build the Wheel

```bash
# Select the CANN environment; the unified entry point prepares dependencies,
# compiles native components, and packages the wheel.
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --multicore on --strict on --jobs 24

# Build one wheel per Python ABI and host architecture, for example:
# hyper_parallel-0.1.0-cp310-cp310-linux_aarch64.whl
```

In strict mode, failure of any enabled component aborts the build without generating a wheel. With the default `--strict off` for development, optional native component failures emit warnings and packaging continues; a generated wheel therefore does not prove all optional capabilities are available. Official release acceptance combines builds against the version's specified glibc baseline, the full test suite, and human review. Native artifacts require CANN >= 9.1.0.

### 6. Validate After Release

Run each component's declared artifact, import, and system-test checks. Activate required SDK and artifact environments before starting tests.

- [ ] English and Chinese README sections match, and code examples run.
- [ ] Installation instructions reproduce in a clean environment, and feature examples match examples/.
- [ ] API signatures match source code, and parameter descriptions match types.
- [ ] Release note categories are accurate and the contributor list is complete.
- [ ] All documentation passes Markdown lint.
- [ ] Cross-references work and terminology is consistent.
- [ ] Component validation through wheel/PYTHONPATH and clean-venv smoke tests pass.
- [ ] Missing dependency, inactive environment, and ABI error diagnostics satisfy each selected component's contract.
- [ ] The target glibc baseline check passes; 910B/910C native builds and tests pass, and unsupported SoCs return explicit reason codes.

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v0.2.0 | — | Previous release |
| v1.0.0 | 2026-06-30 | First official release with full feature coverage |
