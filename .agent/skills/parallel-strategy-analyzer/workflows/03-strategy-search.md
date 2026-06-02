# Phase 3: Strategy Space Search

## 3.1 Determine Long-Sequence Flag & CP Limits

Before enumerating, compute whether s² activation terms dominate:

```python
# Dynamic long-sequence detection (replaces hardcoded 32K threshold)
act_s2 = 5 * s * s * n_h          # quadratic attention terms (per sample per layer)
act_linear = 14 * s * h + 6 * s * d_ff  # linear terms
long_sequence = (s >= 8192) and (act_s2 / (act_s2 + act_linear) > 0.5)

# CP limits based on attention type (lower comm → higher cp_max)
if attn_type == "MLA":      # DeepSeek V2/V3: kv_lora_rank=512
    cp_max = 16
elif attn_type == "GQA":    # n_kv << n_h (LLaMA-2/3, Qwen3)
    cp_max = 8
else:                       # MHA: n_kv == n_h, full KV comm
    cp_max = 4
```

## 3.2 Enumerate Candidates

```python
candidates = []
cp_values = [1] if not long_sequence else [1, 2, 4, 8, 16]

for tp in [1, 2, 4, 8]:                          # TP: power of 2, ≤ n_dev
    if tp > n_dev: continue
    if n_h % tp != 0 or n_kv % tp != 0: continue

    for pp in divisors(L):                         # PP: must divide layers
        if tp * pp > N: continue

        for cp in cp_values:
            if cp > cp_max: continue               # attention-type-aware limit
            if cp > 1 and not long_sequence: continue
            if s % (cp * 2) != 0: continue

            for ep in (divisors(num_experts) if moe else [1]):
                dp = N // (tp * pp * cp * ep)
                if dp < 1: continue
                if dp * tp * pp * cp * ep != N: continue
                if B // dp < 1: continue            # at least 1 sample per DP rank

                candidates.append((dp, tp, pp, cp, ep))
```

## 3.3 Pruning Heuristics

Skip obviously bad configs early:

- If Phase 2 says DP-only is enough → only keep configs with tp=1, pp=1
- If model is not MoE → ep=1 always
- If not `long_sequence` → cp=1 always
- Prefer `tp × ep ≤ n_dev` (keep bandwidth-hungry dims intra-node)
- **MLA/GQA long-seq**: prefer configs with `cp > 1, tp=1` over `cp=1, tp > 1` (CP comm is cheap)
- **MHA long-seq**: prefer configs with `tp > 1` first, then `cp > 1` (CP comm is expensive)
