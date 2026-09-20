# VeOmni vs HyperParallel accuracy example

This directory demonstrates the frozen-manifest accuracy workflow. The included CPU demo is synthetic: it validates the
runner and artifacts, not either framework's model accuracy.

## Run the self-contained demo

From the repository root:

```bash
python3 .agent/skills/accuracy-validation/scripts/run_accuracy_comparison.py \
  --manifest examples/accuracy/veomni_vs_hyper/run_manifest.demo.json \
  --output /tmp/hp-accuracy-demo
```

Expected result: exit code 0 and `status: PASS`. The output contains:

```text
/tmp/hp-accuracy-demo/
├── run_manifest.json
├── baseline/loss.jsonl
├── baseline/command.log
├── candidate/loss.jsonl
├── candidate/command.log
├── compare_loss.json
├── summary.json
└── report.md
```

Use `--dry-run` to validate and expand a manifest without creating artifacts or launching commands.

## Replace the demo with real runs

1. Copy `run_manifest.veomni.template.json` to an experiment-owned location.
2. Replace every `REPLACE_ME` field with immutable revisions, assets, environment evidence, and real argument arrays.
3. Make both launchers write one JSON object per optimizer step to their final argument, which resolves to
   `{baseline_loss}` or `{candidate_loss}`. Required fields are `step` and globally normalized `loss`; retaining
   `loss_sum`, `valid_tokens`, batch fingerprints, component losses, gradient norms, and update evidence is recommended.
4. Dry-run the manifest, then execute it with a new output directory.
5. Treat the result as numerical evidence only after confirming both launchers consume identical weights, samples, loss
   semantics, optimizer settings, and update boundaries.

Commands are JSON argument arrays and run with `shell=False`. The runner supports only documented `{...}` substitutions;
it does not expand shell variables, pipes, redirects, or command substitutions. Multi-card launchers such as `torchrun`
remain valid when each executable and argument is a separate JSON string.

The complete comparison contract and tolerance definitions live in
`.agent/skills/accuracy-validation/references/comparison-contract.md` and
`.agent/skills/accuracy-validation/references/tolerance-policy.md`.
