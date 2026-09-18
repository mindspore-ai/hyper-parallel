# Preflight, Data, and Baseline

## Resolve Inputs Before Execution

Ask for or discover, without guessing:

1. Authoritative HF/native implementation and full pretrained checkpoint or
   cropped-from-config workflow.
2. Weight download/cache directory and permission to download there.
3. Dataset mode/source, deterministic replay evidence or generated Offline prefix,
   tokenizer, and data revision.
4. Sequence length, packing, and whether labels are already shifted.
5. Device count, topology matrix, optimizer, step count, and tolerances.

Do not put machine-specific `PYTHONPATH` or cache paths into committed examples.
Verify an editable install imports `hyper_parallel` from the active checkout.

## Pre-Execution Feasibility Gate

Complete this gate before reserving devices or launching the full matrix:

1. Resolve every case to its exact launcher world size, TP/CP/PP/EP/FSDP mesh,
   global/micro batch, recipe, replacement set, recomputation selection, and
   evidence directory. Do not validate an override with fewer ranks and infer
   that the committed launcher default is feasible.
2. Require `world_size` to match the mesh construction rules and derive the dense
   data-parallel size from the framework's actual formula. Check
   `global_batch_size % (micro_batch_size * dp_world_size) == 0`; EP does not
   implicitly change this divisor.
3. Evaluate the crop-invariant ledger against every case: semantic cardinalities
   remain intentional, all TP/CP/EP divisors are integral, and local expert/head/
   group counts are valid. A TP1 construction smoke cannot clear a TP2+ rule.
4. Instantiate the replacement-disabled reference model and the final replaced
   model, then reconcile exact parameter FQNs, final module counts, plan-glob
   matches, and required runtime fields. A partial match is not coverage.
5. Resolve activation checkpoint selection through the final builder and assert
   exact wrapper count/FQNs for `layer_count: 0`, one nonzero prefix, and an
   explicit index selection. Config text alone is not evidence that the selection
   reached the model.
6. Run one high-risk topology smoke with `model_integration.mode=runtime` and
   require structured findings, input identity, numeric probes, and observed rank
   count. For multimodal scope, use real media and additionally require modality
   fields plus vision/projector gradients.

Assign a unique rendezvous/HCCL port range and evidence directory to every launch
phase. After a failed launch, confirm all child ranks and communication endpoints
have exited before retrying; transport/bind failures in later cases are cascading
failures until this isolation is proven.

## Resource Behavior

Training must use local assets and fail explicitly when they are missing. Download
or preparation is a separate, user-authorized step.

When the user supplies only a Hugging Face repo ID or URL, resolve the exact
revision and create a local Git checkout with LFS smudging disabled before source
analysis or parity. Keep the checkout outside tracked project files, record its
commit in the evidence, and confirm that weight paths contain LFS pointers rather
than model payloads. This source-only checkout is distinct from a checkpoint
download and does not grant permission to fetch weights.

For local JSON/JSONL input, prepare Offline data with:

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path /path/to/train.jsonl \
    --output-prefix /path/to/offline/train \
    --tokenizer-name-or-path /path/to/model \
    --pack-to-seq-len 4096
```

For an authorized Hugging Face dataset download, use:

```bash
python -m hyper_parallel.data.tools.huggingface_offline \
    --dataset DATASET_ID \
    --dataset-subset SUBSET \
    --dataset-split train \
    --output-prefix /path/to/offline/train \
    --tokenizer /path/to/model \
    --pack-to-seq-len 4096
```

The framework records revisions, resolved arguments, hashes, and generated
`.bin`/`.idx` files as evidence. Do not copy those generated values into the user
validation manifest. Packed documents contain `seq_length+1` tokens; verify the
reader creates inputs and targets with exactly one causal shift.

## Mandatory Runtime Precision Checks

Validate the resolved configuration and runtime:

- top-level `model_init_dtype` is `float32` after load or initialization;
- optimizer `fp32_main_params` is true and the mixed-precision wrapper exists;
- all trainable main parameters are FP32;
- FSDP `param_dtype=bfloat16`, `reduce_dtype=float32`, absent `output_dtype`
  (resolved `None`), and `cast_forward_inputs=false`;
- actual forward inputs are compatible without FSDP input casting;
- scalar objective remains FP32 after the root FSDP forward hook;
- model/main-parameter alias or copy-back behavior matches the model parameter
  dtype;
- after backward, DTensor parameter, `main_param`, and `main_grad` agree on
  logical global shape, compatible mesh topology, and placements while their
  local shapes match the rank's actual shard;
- after optimizer-state materialization, moments retain that same distributed
  layout instead of becoming plain rank-local tensors.

## Full-Pretrained Baseline

Prove all ranks load the same checkpoint. Record checkpoint index/hash and loaded,
missing, unexpected, mismatched, and newly initialized tensors. Unexplained model
coverage gaps fail the baseline.

Compare the first loss with the authoritative HF or native implementation
using the same checkpoint, tokenizer, token IDs, labels, mask, dtype policy, and
batch. Do not assume a universal expected loss such as 2.x.

## Cropped Baseline

When the model is too large and no initial checkpoint is loaded:

1. Load the full config and apply only authorized cropping. For structural
   coverage, preserve decoder and modality-tower depths plus every distinct layer
   role; scale compatible parameter dimensions and dependent tables/buckets
   consistently instead of truncating weights. Keep semantic cardinalities unless
   equivalence is independently proven, and verify the resulting topology
   divisors against the full matrix. A shallow layer crop is smoke-only, requires
   explicit user intent, and cannot claim all-structure coverage. Verify source,
   consumer, reindex/candidate, Engram, dense/MoE, and multimodal roles rather than
   assuming an early prefix exercises them.
2. Fix initialization, data, routing, and sampler seeds.
3. Build the required FP32-initialized/main-parameter route.
4. Run an uninterrupted K+N reference and save at K.
5. Restore every candidate from the exact K checkpoint and compare the next N
   optimizer steps with the uninterrupted reference.

The checkpoint must include model, optimizer moments, distinct FP32 main
parameters when present, scheduler, global step/train state, per-rank RNG, and
dataloader/sampler cursor. Construct the scheduler for the K+N horizon before the
save; replacing it with a new N-step schedule invalidates continuation.

Keep restore modes explicit. A full-state continuation restores optimizer,
scheduler, step/epoch position, RNG, and dataloader cursor. A weights-only restore
may intentionally reset those values, but the resulting run is not continuation
evidence. Derive `start_epoch` and `start_step` from the restored Trainer state;
do not rely on stale launcher defaults.
