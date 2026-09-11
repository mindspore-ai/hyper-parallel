# KDA Triton Backend

The public KDA backend names are `eager` and `triton`.

- `eager` uses Hyper-Parallel's PyTorch reference implementation and does not
  import FLA or Triton.
- `triton` uses the optional external `flash-linear-attention` package. Local
  KDA and Ulysses call FLA's public `chunk_kda` API. State-P2P uses FLA's staged
  Triton-Ascend operators around Hyper-Parallel's affine state-summary kernels.

The validated upstream revision is:

```text
fla-org/flash-linear-attention@35dceaee5408e69a555fec34cb215c93c375dabe
```

The upstream project is distributed under the MIT License:
<https://github.com/fla-org/flash-linear-attention>.

Until that revision is available as a stable package release, install it
without allowing it to replace the Torch/Torch-NPU versions selected by the
Hyper-Parallel environment:

```bash
FLA_REPOSITORY="https://github.com/fla-org/flash-linear-attention"
FLA_REVISION="35dceaee5408e69a555fec34cb215c93c375dabe"
pip install --no-deps "git+${FLA_REPOSITORY}@${FLA_REVISION}"
```

Selecting `backend="triton"` performs a lazy runtime check for FLA >= 0.6.0,
Triton-Ascend, the NPU backend registrations, and all staged KDA APIs. Missing
or incompatible dependencies raise `RuntimeError`; there is no silent eager
fallback.

The `triton/state_summary.py` file is Hyper-Parallel's own fixed-shape affine
summary implementation. It is retained because FLA's public KDA API does not
expose the P2P boundary required by Hyper-Parallel.
