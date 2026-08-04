# Platform-dev decision points & layout

Load when scoping a platform change (workflow step 1–2).

## Architecture layout

```text
platform/
├── platform.py                    # Platform base (~100+ abstract methods)
├── torch/                         # PyTorch backend
│   ├── platform.py                # TorchPlatform(Platform)
│   ├── dtensor.py                 # DTensorBase (torch.Tensor subclass)
│   ├── function_override.py       # DTensor backward hooks
│   ├── init_weights.py            # init_on_device
│   ├── group_utils.py / clip_grad.py
│   ├── activation_checkpoint/     # SAC + swap
│   ├── fully_shard/               # FSDP + HSDP (core hsdp_*.py in core/)
│   └── pipeline_parallel/
└── mindspore/                     # MindSpore backend (mirror + graph/custom_pass)
    ├── platform.py / dtensor.py / init_weights.py / parameter_init.py
    ├── platform_graph.py / custom_pass/
    ├── fully_shard/
    └── pipeline_parallel/
```

## Key decisions

| Decision | Criteria | Options | Impact |
|----------|----------|---------|--------|
| Change scope | New API vs modify existing | New abstract / Modify / Internal only | Files, compat |
| Backend priority | Which first | Torch / MindSpore / Both | Order |
| Feature parity | Both needed? | Full / one + NotImplementedError | Tests |
| Stream sync | Async? | Sync / handle / events | Correctness |
| Memory pattern | Buffers? | `resize_(0)` / reuse / alloc | Footprint |

## Example invocations

```bash
/platform-dev Add scatter() collective to Platform abstraction
/platform-dev Implement activation swap for MindSpore backend
/platform-dev Fix torch FSDP unshard prefetch
/platform-dev Add DTensorBase property for communication state
```
