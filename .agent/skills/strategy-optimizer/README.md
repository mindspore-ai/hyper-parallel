# Strategy Optimizer

Executable search → benchmark → refinement workflow. Start with [SKILL.md](SKILL.md)
for planning and real-run commands. A real run requires `--psa-config`, `--config-template`,
a measured baseline, and accelerator dependencies; `--plan-only` generates unvalidated candidates on CPU.

See the [configuration contract](references/config-templates.md),
[orchestration](workflows/orchestration.md) and [output/status contract](references/output-format.md).
The single-node launcher produces reviewable configurations and measurements. Adoption is a separate action.
