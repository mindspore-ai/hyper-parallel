# Codegen Guide

**English** | [简体中文](codegen.zh.md)

Codegen writes parallel plans and high-performance module replacements explicitly into model files. It reads training YAML, infers TP/CP/EP contracts offline, and freezes parameter layouts, boundary redistribution, forward-boundary rewrites, and module replacements into an artifact: `codegen_meta.json` plus generated modeling files. During training, all ranks load the same artifact; the runtime reads its frozen sharding plan and performs sharding.

Codegen serves two purposes: embedding a sharding plan as parameter sharding and boundary communication in generated models, and lowering model rewrites such as `replace.kind: module` into generated files. When YAML uses `replace.kind: module` to replace standard attention or MoE with fused implementations such as flash attention, grouped-GEMM experts, or CP wrappers, codegen writes the replacement entry into the model class's `__init__`, so it takes effect during construction.

## Core Concepts

| Concept | Description |
|------|------|
| Artifact bundle | Defaults to `hyper_parallel/models/<model>/generated/`, anchored to the model with one bundle per model and a directory name corresponding exactly to model_type. If `models/<model>/` does not exist, codegen logs a WARNING and creates it. Override the destination with `model.codegen_artifact_dir`. Contains generated modeling files, `.diff` files, and `codegen_meta.json` |
| Signature | First 12 SHA-256 characters over all artifact-affecting inputs: canonical spec projection, modeling-source SHA-256, Transformers version, model ID, structural configuration digest (layers/heads/experts and other shape fields), implementation digest of the entire codegen package, and inline declaration digest. Upgrading HyperParallel therefore triggers regeneration; unrelated settings such as training steps or optimizer configuration do not. |
| `codegen: true` | Enable artifact generation and preflight; with `force_hf=false`, the backend is `gen`. |
| `force_hf` | Highest-priority escape hatch that overrides `gen` with `hf`; see Important Notes below. |

## Prerequisites

- Training YAML. For offline training, point `pretrained_model_name_or_path` at a local checkpoint. Model IDs also work if the current Transformers environment and source resolver can resolve them and retrieve configuration and modeling source. Custom architectures registered through `register_custom_model` are supported, including cropped DeepSeek-V4.1.
- A YAML file parseable by `parse_training_args`: codegen records its location in `codegen_meta.json` for provenance; the path is not part of the signature.

---

## Quick Start

### 1. Enable Codegen in YAML

Start with the validated `examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml`. It matches the native reference `train_parallel_offline.yaml` exactly except for one additional top-level field:

```yaml
# train_parallel_offline_codegen.yaml: the only difference from the native variant.
codegen: true

model:
  # The demo uses from_config for a four-layer cropped Qwen3-MoE, randomly initialized without loading a checkpoint.
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  config_path: /path/to/Qwen3-30B-A3B
  num_hidden_layers: 4
  local_files_only: true
  torch_dtype: bfloat16
  attn_implementation: sdpa
```

For custom YAML on the `from_pretrained` path, **enable `codegen: true` and disable `force_hf` in model configuration**; codegen selects the `gen` backend. The `from_config` path above is unaffected by `force_hf`: `codegen: true` is sufficient. `HyperAutoModelForCausalLM.from_config` prepares an artifact from the configuration's source identity and builds from generated modeling without loading checkpoints, using random initialization. Structural fields such as cropped layer count participate in the signature, so changes trigger regeneration.

The training YAML itself is the input; no second configuration is needed. Model source and structure, parallel scale, and `plan_overrides` come from it. Only artifact-affecting inputs enter the signature; changes to optimizer settings, log levels, training steps, or YAML comments do not regenerate artifacts.

### 2. Generate Artifacts or Let the Trainer Handle Them

Training YAML triggers generation automatically. Between `_setup` and `_build_model`, the Trainer runs the codegen manager: generate missing artifacts, replace artifacts when signatures differ, or reuse matching artifacts. Reuse skips generation and costs approximately one file-validation pass.

Artifacts can also be managed manually:

```bash
# Generate or reuse artifacts; --verbose displays artifact and signature information.
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml

# Preflight integrity checks: signature, file hashes, metadata, imports, and frozen-plan structure.
python -m hyper_parallel.codegen --verbose check \
  --config examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml
```

Both `generate` and `check` are no-ops when `codegen:false`; `--verbose` displays the skip reason. `check` does not instantiate the full model. Actual parameter-coverage validation continues when training loads the generated model.

### 3. Launch Training Normally

Launch with `python -m torch.distributed.run`. Avoid relying on a `torchrun` executable from PATH, which another CLI application may shadow, causing PyTorch distributed options such as `--nproc_per_node` to be rejected. The module invocation is the safest choice:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module examples.training_demo.train_text \
  examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml \
  --model.config_path=/path/to/Qwen3-30B-A3B \
  --dataset.data_path=./output/training_demo/data/parallel_offline_text_document \
  --training.train_iters=5
```

The first launch logs generation; reuse logs `reuse artifact`:

```text
codegen: generating artifact hyper_parallel/models/<model>/generated (signature f463aefff4a4)
```

### 4. Confirm Training Actually Uses Codegen

An artifact only proves that the codegen manager ran, not that the training model came from generated modeling. To confirm the **gen backend is actually used**, verify all the following log conditions.

First, model construction must log import of the generated modeling file:

```text
codegen: imported generated modeling file .../modeling_<model>_gen_npu.py
```

This means `HyperAutoModel` selected `gen` and loaded the generated Python model from the artifact directory.

Second, parallelization must log generated-module sharding:

```text
codegen: sharding applied by the generated module; source_shard_info keys=<N>
```

This means generated modeling handles sharding rather than the native planner/applier path. The runtime reads and executes the frozen plan, preferring `hp_parallelize()` if the generated file defines it, and returns `source_shard_info` to FSDP and the training framework.

Finally, weight loading and training steps must proceed normally:

```text
Loaded <N> model tensors from ...
step=1 ... training/foundation_loss=...
```

Seeing only `generating artifact` or `reuse artifact` does not prove the generated file is used. With `force_hf=true`, artifacts may still be generated or reused while modeling falls back to HF. Confirm generated-model import, generated sharding, successful checkpoint loading, and advancing training steps together.

---

## Artifact Bundle

Artifacts use a model-anchored layout, defaulting to `hyper_parallel/models/<model>/generated/`: one bundle per model, alongside its adapter code. Directory names correspond exactly to model_type; `deepseek_v2` and `deepseek_v3` have separate artifact directories even when sharing adapter code. If `models/<model>/` does not yet exist, codegen logs a WARNING that the model has no corresponding models directory and creates it during generation:

```text
hyper_parallel/models/qwen3_moe/
├── adapter/
├── recipes/
└── generated/
    ├── __init__.py
    ├── codegen_meta.json
    ├── modeling_Qwen3_Moe_gen_npu.py
    ├── modeling_Qwen3_Moe_gen_npu.py.diff
    └── configuration_*.py / source_*.py / other sibling dependencies (optional)
```

The complete resolution priority is: explicit `model.codegen_artifact_dir` in YAML > model-anchored `hyper_parallel/models/<model>/generated/` (default) > `generated/` alongside the training YAML, only when no model identity can be resolved from configuration > `cwd/generated`. Each model should have only one codegen YAML; changing its content changes the signature, and the next generation automatically overwrites and updates `generated/`.

`model.codegen_artifact_dir` is an optional explicit override under the training YAML's `model` section, directing both generation and loading to that directory. Ordinary training does not need it. Use it for special cases, such as directing artifacts to a writable directory when hyper-parallel is installed as a wheel in read-only site-packages, or maintaining separate artifacts for multiple parallel configurations of one model:

```yaml
model:
  _target_: hyper_parallel.models._transformers.auto_model.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /path/to/Qwen3-30B-A3B
  codegen_artifact_dir: /home/lxy/my_codegen_bundles/qwen3_moe  # Optional; omit to use the default.
```

`generated/` is the bundle root. During training, the loader finds its unique `modeling_*_gen_npu.py` and registers the directory as an isolated synthetic Python package, preventing generated files from different YAML configurations from overwriting each other in `sys.modules`.

`generated/` contains machine-generated artifacts produced locally by `codegen generate` or the first training launch. The repository's two model families, `qwen3_moe/` and `deepseek_v41/`, include reference artifacts from real generation runs in their model directories for inspection. Artifacts in users' own directories are excluded from version control by default.

`__init__.py` gives `generated/` package structure. It normally contains no business logic and allows generated modeling and optional sibling dependencies to resolve relative imports within one package.

`codegen_meta.json` indexes and validates the bundle. It records the signature, YAML digest, original modeling-source digest, parallel dimensions, entry points, frozen sharding plan, module overrides, output hashes, and coverage declarations. Preflight uses it to check compatibility with the current configuration; runtime also consumes its frozen plan.

`modeling_Qwen3_Moe_gen_npu.py` is the generated model file actually imported for training. Based on the original Transformers modeling source, it adds a generation banner, lowers eligible boundary forwards into explicit boundary communication code, retaining the original body as `_forward_impl` and turning public `forward` into “input operator → original computation → output operator” as described below, and lowers `replace.kind: module` at module construction sites. The generated file no longer contains frozen-plan constants or an `hp_parallelize()` entry. Instead, `codegen_meta.json` stores the frozen plan for runtime to read and shard. During model construction, `replace.kind: module` patches take effect directly through this file's `__init__` logic.

`modeling_Qwen3_Moe_gen_npu.py.diff` is an audit file showing a unified diff against original modeling source: added imports, forward rewrites with `_forward_impl` extraction and entry/exit operators, and construction sites replaced with fused implementations.

Optional `configuration_*.py`, `source_*.py`, and other sibling Python files support remote-code or relative-import scenarios. When original modeling depends on sibling modules, codegen copies the required dependencies into the bundle so it remains importable outside its original package.

Bundles are written with an **atomic directory swap**: first write a temporary sibling directory, then replace the entire destination, preventing imports of partial artifacts.

---

## How Codegen Applies Patches

Codegen does not dynamically edit Python source during training. It writes rewritten source during generation, and training imports the already-modified file.

For an ordinary attention boundary, original Transformers modeling typically computes directly in `forward`:

```python
class Qwen3MoeAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

After generation, codegen retains the original implementation as `_forward_impl()` and lowers public `forward()` into entry operator → original computation → exit operator. This tp_collective form comes from a real artifact; its `_forward_impl` body preserves the example above verbatim:

```python
class Qwen3MoeAttention(nn.Module):
    _hp_boundary_form = "tp_collective"

    def _forward_impl(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        # [HYPER TP-COLLECTIVE] statically lowered boundary — the runtime
        # re-validates this form at install time and replaces the forward
        # with the generic engine on any mismatch.
        # hidden_states: identity ==> to_local
        hidden_states = self._hp_tp.to_local(hidden_states)
        # [HYPER TP-COLLECTIVE] original forward body (now ``_forward_impl``)
        outputs = self._forward_impl(
            hidden_states,
            position_embeddings,
            attention_mask,
            **kwargs,
        )
        # output: tp P(sum) -> R ==> all_reduce(sum)
        _hp_output_0 = self._hp_tp.all_reduce(outputs[0])
        outputs = (_hp_output_0, *outputs[1:])
        return outputs
```

Original computation remains in `_forward_impl()`; codegen inserts only parallel-boundary communication around it. Generated `# hidden_states: ...` / `# output: ...` comments describe layout transitions. For example, `P(sum) -> R ==> all_reduce(sum)` reduces TP partial sums into a complete output tensor. Runtime binds `self._hp_tp` during installation, first revalidating this form against the real mesh and frozen plan. Any mismatch replaces the entire `forward` with the generic engine; generated calls never execute before validation succeeds.

### Lowering Module Replacements into Source

For `replace.kind: module`, the native path matches and replaces modules through the Trainer's dynamic applier. Codegen instead lowers replacements into module construction sites during generation, so the generated `__init__` directly constructs high-performance implementations without rerunning replacement during model construction:

```python
# Original modeling source.
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)

# Generated source: replace the construction site with a generic fused implementation, retaining the original class as wrapper input.
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GQAAttention(
            module=Qwen3MoeAttention(config, layer_idx),
            module_fqn='',
            context=None,
            attention_interface=run_qwen3_moe_flash_attention,
        )
```

Matched FQNs remain in `codegen_meta.json` under `module_overrides`, with `covered.module_overrides=true`, for preflight verification and to tell the Trainer not to reapply those rules.

The `_target_` in YAML's `replace.implementation` supports two forms: a factory returning a wrapper, such as `replace_qwen3_moe_flash_attention` in this example, or a component class directly under `hyper_parallel.components.*`. Codegen supports both, with results matching module replacement in the native path.

In this guide's Qwen3-MoE demo configuration, the generated artifact carries RMSNorm, attention fusion, CP wrappers, MoE local compute, and related patches. Training only imports the generated modeling file; runtime reads its frozen sharding plan and applies sharding, without handing the same `replace.kind: module` rules back to the native applier. Whether other models include similar patches depends on their YAML `plan_overrides` and the static lowering capabilities currently supported by codegen.

---

## Important Notes

### 1. force_hf Overrides Codegen

Backend priority is `force_hf` > `codegen`. With `force_hf: true`, common in native/offline examples, adding `codegen:true` still resolves to `hf`. Training may run and converge normally while **never using generated modeling**. This is a particularly subtle silent failure.

The log reports:

```text
codegen is enabled but force_hf=True, so the generated modeling file will not be used
```

Disable `force_hf` when using codegen, rather than only adding `codegen:true`:

```bash
--model.force_hf=false
```

### 2. Checkpoint Reports Unloaded Model Weights

**Symptoms**

Generated modeling imports and sharding succeeds, but checkpoint loading reports:

```text
did not load <N> owned model tensors
```

Some pretrained parameters owned by the current rank were not restored correctly. Do not judge training validity from subsequent loss. Even if execution continues, treat this launch as invalid.

**Root cause**

Distinguish two cases.

An incorrect checkpoint path, model type, Transformers version, or YAML `replace.kind: module` configuration is a user configuration issue. Typically, model files and weights are already incompatible, and the native path cannot reliably load the same checkpoint either.

If the same YAML loads correctly through the native path but codegen reports missing weights, codegen likely lacks support for the current model structure or weight-conversion rules. A common cause is structural fusion or replacement in generated modeling without matching checkpoint conversion registration, preventing original parameter names from mapping to generated-model targets.

**Resolution**

Users should first check basic configuration:

1. Confirm `pretrained_model_name_or_path` points to a checkpoint matching the YAML model type.
2. Confirm HyperParallel and Transformers versions are supported by the project.
3. Regenerate artifacts; signature changes automatically replace old artifacts without manual cleanup:

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml
```

If these conditions hold, native loading works, and codegen still fails, open a HyperParallel issue with the model name, training YAML, checkpoint-loading logs, artifact `codegen_meta.json`, and a log excerpt demonstrating successful native loading of the same checkpoint.

### 3. Old Artifacts Are Reused After Configuration Changes

**Symptoms**

After changing model source, parallel strategy, or module replacement settings and restarting training, logs still show:

```text
codegen: reuse artifact ...
```

Alternatively, training imports an old generated file whose content, behavior, or results do not reflect the changes.

**Root cause**

There are usually three possibilities:

1. Changes affect runtime settings such as steps, logging, or optimizers. They do not change the generated model, so reuse is expected.
2. The model source, parallel scale, `plan_overrides`, or `replace.kind: module` changed. These changes should trigger regeneration; otherwise, the current artifact does not match expectations.
3. Files inside `generated/` were edited directly. Generated files are outputs rather than source entry points; manual edits are not stable configuration and may be rejected by checks or overwritten by regeneration.

**Resolution**

For model or parallel configuration changes, regenerate directly; changed signatures replace old artifacts automatically.
If files inside `generated/` were manually edited, delete that directory before regenerating:

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml
```

Run checks after regeneration:

```bash
python -m hyper_parallel.codegen --verbose check \
  --config examples/training_demo/qwen3_moe/train_parallel_offline_codegen.yaml
```

On the next training launch, confirm new generation or import logs:

```text
codegen: generating artifact ...
codegen: imported generated modeling file ...
```

Changes only to steps, log levels, learning rates, or other runtime settings do not require artifact cleanup or regeneration.

### 4. Generation Fails Immediately (Fail-Closed)

**Symptoms**

`codegen generate` or first-launch generation raises an error, such as a missing module construction site, missing inline declaration, or unsupported target component.

**Root cause**

Codegen does not silently fall back for unsupported model structures. If it cannot faithfully lower the YAML parallel strategy or `replace.kind: module` into generated files, generation fails immediately instead of producing a semantically incomplete artifact. This is intentional: the configuration exceeds codegen's validated structural scope.

**Resolution**

1. Check whether the targets in YAML `plan_overrides` / `replace.kind: module` belong to supported structures, using this guide's example configuration as a reference;
2. If codegen is not needed, remove `codegen: true` to return to the native Trainer path without affecting training.
3. If configuration is correct but generation still fails, open a HyperParallel issue with the model name, training YAML, and complete error message.
