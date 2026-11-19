# Codegen validation

The Codegen PR gate uses a small offline model to compare real CLI and rank-zero
sidecar output. It runs the native API isolation, shared preparation, generated
re-entry and forward-binding tests on CPU.

For a change that affects a tracked model bundle, generate fresh CLI and
Trainer-sidecar output from the same current training recipe and overrides.
Keep one directory per model under each output root. Supply the SHA256 report
from that run and a checkout that contains the candidate bundles.

```sh
python scripts/check_codegen_bundle_parity.py \
  --repository /path/to/hyper-parallel \
  --cli-root /path/to/validation/cli \
  --sidecar-root /path/to/validation/trainer \
  --manifest /path/to/validation/parity.json
```

The checker discovers tracked models with a training recipe, verifies the
recipe and publication result for each model, and compares every regular
file against both fresh outputs and the report. Keep the modeling source,
its diff and the package entry point together. Model assets and accelerator
smoke tests run on the model validation host.
