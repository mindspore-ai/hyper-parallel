# Bring-up Field Checklist (survey, environment, first checks)

Field-verified procedure for the survey and environment phases of a model
bring-up, and for the cheapest correctness checks that should run before any
multi-card training. Every item below was exercised on a real bring-up
(GLM-4-9B on an Ascend 910 cluster, 2026-10); items marked **(generic
pattern)** intentionally use placeholders — never record internal host
names, addresses, or credentials in this repository.

## 1. Survey before writing anything

1. **Read the modeling source at the pinned version.** Resolve the
   `transformers` version the repository actually pins (`requirements.txt`),
   and read `modeling_<family>.py` at exactly that version. A cluster conda
   env may carry an older `transformers` than the repository requires; code
   written against the wrong version's modeling file produces conclusions
   that silently do not transfer.
2. **Derive the released weight catalog first.** Download only
   `config.json` + `model.safetensors.index.json` (small) and enumerate the
   weight entries into name families (`re.sub(r"\.\d+\.", ".N.", key)`).
   This catalog decides the adapter surface and feeds the layout test
   (§5) — it requires no weight download.
3. **Identify semantics that generic modules do not cover.** Compare the
   modeling source against the generic components the repository offers
   (for example: partial *and* interleaved rotary combined; sandwich
   normalization where branch output is normalized before the residual
   add; fused `gate_up_proj` when `modules.SwiGLUMLP` adapts separate
   projections). Each uncovered semantic is a reason to start HF-native
   (§5), not a reason to extend the fused module under time pressure.
4. **Profile the cluster hardware read-only** (one command set, no cards
   occupied): chip family and per-die HBM (`npu-smi info`), intra-node die
   topology (`npu-smi info -t topo`; note direct-link pairs vs switched
   links — a TP2 layout should map to a direct-link die pair when the
   model's KV-head count caps TP at 2), per-die NIC speed
   (`hccn_tool -i <die> -speed -g`), host CPU/NUMA/memory (`lscpu`,
   `free -g`), storage mounts. **Do not copy peak-FLOPS numbers from spec
   sheets** into any mechanism estimate; measure with a microbenchmark when
   the optimization phase needs the number. Record in the migration spec how
   each number was obtained (idle-node class, the read-only commands used),
   so every hardware figure in the spec stays auditable.

## 2. Weight acquisition (generic pattern)

- **Inventory shared storage before downloading anything.** Clusters
  accumulate weight stashes (prior runs' `*/hf/` directories, shared
  `models/` trees, HF hub caches); a checkpoint that is already on the
  shared disk costs minutes to verify against its index and nothing to
  acquire. Search first, then download only what is genuinely absent —
  and verify a found copy exactly like a downloaded one (index entries
  and per-file sizes) before trusting it. Absence must be proven by a
  positive listing (`ls`/`find`), never by an empty aggregate — a `du`
  that errors quietly on a large tree reads exactly like an empty
  directory (the probe fail-open pattern again, in inventory form).
- **Verify every downloaded file against its `Content-Length`.** A silent
  connection cut leaves `curl` exiting 0 with a truncated file; a 0.25 GB
  fragment of a 4.6 GB shard "completes" otherwise. Loop
  `curl -C - --retry N` until `stat -c%s` equals the expected size, and log
  OK/FAIL per file.
- **Keep large downloads off workstations.** Use a download host with good
  mirror connectivity and a local scratch disk, then relay to the cluster's
  shared storage over the data-center network
  (`ssh -A <download-host> "scp -o ProxyJump=<jump> ... <cluster>:<path>"`),
  so bytes never transit the operator's machine. Authenticate the relay with
  agent forwarding instead of copying private keys between hosts.

## 3. Offline dependency overlay (shared cluster etiquette)

When the repository pins a newer `transformers` than any cluster env offers:

- **Never upgrade a shared conda env in place.** Install into a standalone
  directory and prepend it:
  `pip install --no-index --no-deps --target <overlay-dir> <wheels>` then
  `PYTHONPATH=<repo>:<overlay-dir>`.
- **Resolve wheels on a connected machine with explicit platform flags**
  (`--platform manylinux2014_aarch64 --python-version 310
  --only-binary=:all:`), and install **per-wheel with `--no-deps`** — the
  offline resolver dead-ends on anything outside the local wheel set.
- Pins that bite (transformers 5.13): `tokenizers>=0.22,<=0.23.0` and
  `huggingface-hub>=1.5,<2.0`; hub 2.x additionally switches its HTTP
  backend (`httpx2`), dragging in a different dependency closure. Read the
  wheel `METADATA` (`Requires-Dist`) instead of guessing closures.
- **Extend the closure per modality, not just per framework version.** An
  image-text processor pulls `torchvision` that a text-only overlay never
  needed, and the failure surfaces only when the processor is first built.
  Resolve the extra wheels before the first on-device check.
- **A download host may route pip through a proxy that is not running.**
  When resolution fails with `ProxyError ... Connection refused`, pass an
  explicit index and an empty proxy (`--index-url <mirror> --proxy ""`)
  rather than concluding the version does not exist; also check the
  platform tag actually published (`manylinux_2_28_aarch64` is not
  `manylinux2014_aarch64`).
- Two import facts on NPU hosts: `import transformers` imports `torch_npu`
  unconditionally (its NPU probe), so **source the CANN environment first**
  even for CPU-only work; for a pure-CPU probe without CANN, set
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and avoid transformers imports.
- Scope note: the repository's editable-install rule (AGENTS.md) governs
  development hosts. The overlay above and clean-room audits run on
  `PYTHONPATH` by design; assert the import path instead
  (`python -c "import hyper_parallel; print(hyper_parallel.__file__)"`).

## 4. HF-native first: registration is optional

An unregistered HF-native family builds through `HyperAutoModelForCausalLM`
exactly like a registered one — every adapter-spec consumer tolerates a
missing spec — so a fresh bring-up needs **no registration at all**:
correctness comes from the unmodified HF forward, and the generic kit under
`examples/training_demo/hf_text_bringup/` (builder, recipe, data, launcher)
runs any local HF causal-LM directory as-is. Register a family only when it
needs a provider (replacements, CP/EP wrappers, sharding rules, recompute
policy); `models/kimi_linear/adapter/registration.py` shows the minimal
form. Fusion replacements follow once they can be pinned value-for-value
against the HF forward on device.

**What the kit contains**, so a bring-up does not rebuild any of it:

| file | use it for |
|---|---|
| `build_model.py` | `from_config` construction of any HF causal LM; `load_pretrained=true` loads released weights instead (needed when several layouts must start from identical weights) |
| `train_hf_text.yaml` | the recipe: precision policy, FSDP sizes, `plan_overrides` left empty on purpose until a fusion is pinned |
| `prepare_text_data.py` | deterministic synthetic text corpus for smoke and acceptance |
| `prepare_vlm_synthetic.py` (one level up) | deterministic synthetic image-text pairs in the Omni conversation schema, for the VLM path |
| `run_hf_text.sh` | single-node launcher; site specifics come from the environment, never from edits |
| `run_hf_text_dist.sh` | **multi-node** launcher: ssh fan-out, rendezvous, all-or-nothing admission guard across every node, release verification after the run; `DRYRUN=1` prints the plan without launching |
| `judge_equivalence.py` | the S3 verdict over the runtime's structured evidence (`model_integration.mode=runtime`: exact per step, inputs checked); trainer text logs are a labelled fallback judged at six printed digits |

The multimodal path uses the repository's existing VLM facade
(`HyperAutoModelForImageTextToText`, `data.omni`, `examples/training_demo/
run_vlm.sh`) — only the data generator was missing, so do not build a
parallel kit for it.

## 5. Released-layout contract check (no weights loaded, no NPU)

Pin the released checkpoint layout with the model-agnostic script:

```bash
python .agent/skills/hf-model-integration/scripts/check_released_layout.py     --model-dir /path/to/hf-model
```

It meta-builds the architecture from `config.json` and compares the
parameter catalog against the released files — names from
`model.safetensors.index.json`, shapes from the safetensors shard headers
(read without loading tensor data) when shards are present. Exit 2 names
every missing entry, invented entry, and shape drift; easy-to-silently-lose
properties (an accidentally tied `lm_head`, dropped projection biases,
extra per-layer norms) surface as name/shape differences. Legitimately tied
weights are not false alarms: build-side alias names the release omits
(for example `lm_head.weight` under `tie_word_embeddings`) are detected via
parameter identity, reported as tied, and excluded from the mismatch set.
Fused MoE experts are likewise reconciled: a release that stores per-expert
2D projections pins cleanly against the stacked/concatenated 3D expert
tensors transformers 5.x builds (its load-time conversion) — expert counts
and shapes are checked rather than flagged as missing/invented (first hit:
a 64-expert MoE whose 4,992 per-expert weights met a 52-tensor fused build).
A third class is wholesale **prefix renames**, where the build keeps the
released tree but moves it (a text tower becoming `model.language_model.*`
under a multimodal class): the script retries on a suffix-plus-shape
pairing, which stays rename-agnostic — a genuinely absent entry or a
reshaped one still fails to pair, so the pin does not weaken. A family that
needs a stronger pin in CI can wrap the same comparison in a fixture test
(`tests/ut/tools/test_released_layout.py` shows the pattern on a tiny
model).

A fourth class is the **declared-ignored subtree**: a release legitimately
carries tensors the modelling code does not build — an extra prediction
head, a tower a task class leaves out — and the architecture says so itself
in `_keys_to_ignore_on_load_unexpected`, which is why `from_pretrained`
drops them without complaint. The pin reads that declaration and reports
those released-only entries as their own labelled class instead of "absent
from the build": calling upstream's published contract a defect costs an
operator decision and, worse, buries a real difference behind a known one.
Two properties keep it from becoming a loophole — only *released-only*
entries are reclassified, so shape drift on an entry the build has still
fails however broadly the patterns read; and the split is always printed
with its count, its prefixes and the patterns that matched, because a
declaration is a reason to *label* a difference, never a reason to stop
looking at it. First hit: a released 9B checkpoint carrying 15 `mtp.*`
tensors for a head the library declares `[r"^mtp.*"]` on, against 760 built
entries and zero invented ones.

Note what that declaration also tells you about **scope**. When the
difference is a subtree upstream does not implement at all — no module,
only an ignore rule — "integrate it" is not wiring up an existing head, it
is writing one with no reference implementation to pin against, so the
same-process bitwise parity gate cannot cover that part at all. That is a
scope change for the operator to decide, not a task to quietly start: grep
the modelling file for an actual class before estimating the work, and say
which it is.

## 5b. Tied-weight save prediction (no weights loaded, no device)

```bash
python .agent/skills/hf-model-integration/scripts/check_tied_save.py     --model-dir /path/to/hf-model
```

Weight tying makes two parameters share storage, and the sharded checkpoint
writer goes through `safetensors`, which refuses shared storage. The failure
appears only when a run actually saves, so a bring-up whose smoke and
long-run keep checkpointing off will pass every gate and still be unable to
checkpoint. Run this with the layout pin, and settle the save path before
acceptance for any model it flags.

## 6. First checks on device, cheapest first

1. **Single-die load + forward + greedy generation** with the real
   checkpoint: all weights load (entry count matches §1.2), logits finite
   with sane magnitude, and a short greedy continuation of a plain prompt
   is **coherent text** — any weight transposition, rotary mis-convention,
   or norm mis-ordering turns generation into noise immediately. This is
   the strongest single-card smoke available.
2. **Same-process bitwise parity**: build once through the repository's
   model API and once through plain `AutoModelForCausalLM`, same device,
   dtype, and attention implementation, and compare logits with
   `torch.equal`. For an HF-native registration the expected result is
   exact bitwise equality; anything less means the registration is not as
   pass-through as claimed. Persist the comparison output: on the
   zero-registration route this report is the S2→S3 handoff artifact in
   place of `integration_handoff.yaml` (see the stage machine).
   **Construct both sides at the target dtype** (`from_config(...,
   dtype=...)` or the loader's dtype argument): casting a finished model
   with `.to(dtype)` also casts non-persistent rope buffers such as
   `inv_freq`, which no checkpoint or state-dict transplant restores — a
   false bitwise FAIL that any YARN/RoPE family will hit. When weight
   shards are absent (clean-room audits), the **transplant variant** is
   canonical: build the reference once seed-fixed, load its state dict
   into the repository-API build, and still expect bitwise equality.
3. **Size the dataset for the step count before a long run.** The text
   path cycles epochs, so a small corpus sustains any number of steps; the
   multimodal packing path consumes its samples once, so a thousand steps
   needs roughly a thousand times the per-step sample count (measured on
   the smoke: 768 samples carried 30 steps, so ~26k for 1000). Check which
   behaviour applies before the acceptance run rather than discovering it
   when the loader stops early.
4. **Training smoke follows the repository convention**: `from_config`
   random initialization, initial loss near `ln(vocab_size)`, visibly
   falling within tens of iterations. Real-checkpoint fine-tuning smoke is
   optional and comes after.
5. **A "no device needed" check still imports the device stack.** Scripts
   that only meta-build and read index JSON are advertised as CPU-only, but
   the torch build auto-loads its accelerator extension, and the modelling
   library imports it again on its own (an attention-integration probe) —
   so the same script that runs fine under the device-matched environment
   dies at import under a bare one, on a missing shared library, with a
   traceback that names neither. The documented backend opt-out does not
   help, because the second import is the library's, not torch's. Run even
   the zero-device checks under the environment the runs use: source the
   device environment and use the matched interpreter.
6. **Verify a deployed framework fix by its own resolution path before
   spending a model run on it.** A one-line registration is cheap to get
   subtly wrong (a name that no lookup reaches), and a model run reports
   that as a generic child-process failure minutes later. Resolve both
   forms first and assert they land on the same implementation — seconds,
   no accelerator — then run the model. The run then measures the fix
   rather than re-measuring whether it was deployed.

## 7. Operations discipline (each item cost a real incident)

- **Remote process checks and kills must defeat self-matching**: `pkill -f
  run_x.sh` matches the ssh wrapper carrying that very string and kills the
  session itself; always use a character class (`pkill -f "[r]un_x.sh"`),
  and audit leftovers the same way.
- **Verify every cross-host file drop by size or hash.** `git show
  origin/<branch>:path` prints nothing and exits 0 when the ref is absent,
  and an empty `.py` also runs with exit 0 — two silent passes stack into a
  fake-complete deployment. Status logs must assert artifact existence.
- **Exit codes after a pipe come from the last element**: `cmd | tail;
  echo $?` reports `tail`'s status; use `${PIPESTATUS[0]}` before accusing
  the tool.
- **An inner `ssh` inside a script fed over `ssh` drinks the rest of the
  script.** Orchestration sent as `ssh host 'bash -s' <<EOF` shares its
  stdin with every `ssh` the script then runs, so the first inner call
  consumes the remaining lines and the run dies silently part-way. Pass
  `-n` to every inner `ssh`, or copy the orchestration to the host and run
  it from a file.
- Compute nodes may lack `git`/tooling on the base image (it often lives in
  a shared conda env): launchers set `PATH` explicitly and never assume the
  login shell's view.
- **`npu-smi` HBM columns are fixed-width**: a five-digit used value consumes
  the space before the slash (`36004/ 65536` vs `3158 / 65536`), so a parse
  that requires a space before the slash matches only near-idle dies and
  reports a fully occupied node as free — the failure grows with occupancy.
  Parse with optional spaces on both sides (`[0-9]+ */ *<total>`).
- **Do not key occupancy checks to known launcher names.** Scheduler-dispatched
  jobs (a custom worker spawning per-die training scripts) match neither
  `torchrun` nor `pretrain`; judge occupancy by foreign users holding device
  file descriptors or per-die HBM, never by command-line patterns alone.
- **Positive-control the idle probe before trusting it**: run it once against
  a node known to be busy and confirm it reports busy. A probe validated only
  on idle nodes can fail open — two independent broken signals (a too-narrow
  process pattern and a mis-parsed HBM column) can agree on "free" precisely
  when every die is held.
- **Probe at launch time, every launch — never at scan time.** Shared pools
  churn by the minute: a node idle at survey time can be claimed before
  submission (observed twice in one afternoon). `scripts/node_guard.sh pre`
  is the executable form — run it as the launch wrapper's first statement;
  it fails closed on unparseable telemetry, and `node_guard.sh self-test`
  keeps the positive controls (including the fixed-width busy fixture).
  Abort instead of queuing on any busy signal; when every authorized node is
  busy, back off and re-probe later rather than expanding beyond the
  authorized set. For a multi-node job, every node passes the guard or
  nothing starts. After the job, `node_guard.sh post` verifies release —
  own processes and device memory — before the node is declared free.
- **Choose the node by measured idleness, not by a priority list.** A
  hand-written order sends every job to the same machine — the one most
  likely still warm from the last job — and it encodes the occupancy of the
  day it was written: a four-node "preferred" list was wrong within a day,
  with all four occupied by other tenants while the unlisted nodes sat free.
  `scripts/pool_rank.sh` is the executable form: occupancy stays a veto
  (the guard's verdict, never a score, because sharing a die is the failure
  being avoided), then a timestamped ledger's idle fraction over a short and
  a long window, then `/proc/loadavg`, then residual device memory as the
  tiebreak. The kernel's 1/5/15-minute load windows are what make the policy
  decidable on the *first* call, before any ledger exists — so no sampler
  has to be left running, which on a shared node would itself be residue.
  The pool is passed in (`POOL_NODES` / `POOL_FILE`, with `POOL_EXCLUDE` for
  reserved hosts); host addresses never belong in the asset.
- **A ranking whose sort keys silently became text still prints a confident
  order.** Two defects, both found by running the ranker for real rather than
  only self-testing it: (1) the guard's verdict line is multi-word (`BUSY
  dies=8 maxHBM=…`), so splitting rows on whitespace shifted every later
  column one field right and the numeric sort compared `maxHBM=…` — keep the
  verdict a single token, carry free text in its own field, and parse
  tab-delimited end to end; (2) `[ cond ] && echo "$x"` as a loop's last
  statement becomes the function's exit status, so filtering out the *last*
  pool entry made a correctly resolved pool report failure. Both are the
  fail-open family: pin them with fixtures that carry multi-word text and
  with a filtered-last-entry case, and positive-control the parse by feeding
  the same fixture through the broken separator to watch the keys shift.
