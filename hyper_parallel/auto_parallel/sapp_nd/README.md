# ND: Parallelizing N Dimensions with Symbolic Estimation

ND provides symbolic search over N parallelism dimensions. It generates candidate parallel strategies, filters out-of-memory configurations with memory estimation, and ranks the remaining configurations with performance estimation.

Because both estimations are analytic, ND does not require online profiling during search. This enables exhaustive exploration, fast CPU-only search, and no dependency on the execution cluster during planning.

![ND overview](figures/nd_overview.png)

## Inputs

- Large language model type.
- Model hyperparameters.
- Parallel dimensions to explore.
- Hardware type and device count.
- Global batch size and memory budget.

### HyperParallel configurations (`-f hyper_v2`)

A HyperParallel `train.yaml` needs only the checkpoint and the training
sequence length. The dimensions are read from the Transformers config, so
nothing about the model is restated here:

```yaml
model:
  pretrained_model_name_or_path: /path/to/Qwen3-30B-A3B
  torch_dtype: bfloat16

dataset:
  data_transform:
    max_seq_len: 4096      # or the legacy data.max_seq_len
```

Without a sequence length ND costs the model's context limit, which for a
long-context model puts every candidate out of memory; it warns when it
falls back. `training`, `accelerator`, `fsdp_config`, `activation_checkpoint`
and `context` are accepted but optional: the search varies those dimensions
itself, and the device count, batch size and memory budget come from `-d`,
`-b` and `-M`. Give them only when costing one fixed strategy.

### Search configurations (`-s`)

`-s` adds what the CLI cannot express: a candidate list per dimension, pinned
degrees, and a `resolved.yaml` written back for the trainer. A search config
either restates the model itself (see `auto_parallel/examples/*_search.yaml`)
or points at a train.yaml:

```yaml
train_yaml: "./train.yaml"
cluster:
  num_nodes: 4
  cards_per_node: 16     # optional: defaults to the -A device type
  device_memory_gb: 58.0
parallelism:
  tp: 4                  # scalar -> fixed
  pp: [1, 2, 4, 8]       # list   -> search candidates
  dp: auto               # auto   -> the searcher decides
constraint:
  global_batch_size: 128
  memory_limit_gb: 58.0  # candidates above this are dropped
recompute: "full"
```

`recompute` sets how the search treats activation checkpointing. `auto`
chooses one of the trainer's modes for every layer of each candidate.
`per_layer` chooses one for each layer, and `resolved.yaml` states that
plan as `activation_checkpoint.mode` and `activation_checkpoint.layers`
(`{mode: full, layers: {3-7: off}}`) where the installed trainer declares
`activation_checkpoint.layers`. A trainer without that field refuses it, so
there `resolved.yaml` runs every layer the plan's mode, and ND prints the
plan as advice the trainer cannot run. Either way the search chooses among
off and full, plus the trainer's selective policy where the train.yaml asks
for a census; `recompute_modes: [off, full]` narrows that list. Any other
value prices every layer fully recomputed. The choice fills
`memory_limit_gb`, which is compared with the memory ND estimates a run
allocates. A run also holds what its allocator reserves beyond that, 7 to
16 GiB a rank on the Qwen3.5-MoE crop at 8192 tokens, so set the limit below
what the device leaves by that much.

Both are run the same way:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd     -y train.yaml -s search.yaml -f hyper_v2     -d 64 -b 128 -A A3 -M 58GB -o out
```

### The model spec and the run spec

Whatever the input format, ND prices a model from two typed specs:

- The model spec (`auto_parallel/_model_spec.py`, `ModelSpec`) states the
  model under any strategy: its dimensions, its experts, its layer stack
  (`layers`), its family (`arch`) and, where they differ from the family's,
  its op counts per layer kind (`ops`). A Transformers config resolves into
  one, and `model.config_overrides` states its fields offline.
- The run spec (`auto_parallel/_exec_spec.py`, `ExecSpec`) states how the
  model is trained: the parallel degrees, the pipeline layout, the batch,
  the optimizer and gradient sharding, the recompute, the precisions, the
  kernels, the sequence length and the device memory. A HyperParallel
  configuration states it in the train.yaml; `offset`, `full_rec`,
  `sel_rec`, `capacity_factor`, `use_gmm` and `seq_length` in
  `config_overrides` belong to it.

Every other field is derived from the two. What neither states comes from
the model's family, in its op profile (`auto_parallel/op_profiles/<arch>.yaml`):
its op counts per layer kind, and defaults such as its byte widths. A run
spec states recompute as ranges of layers in model order, the body layers
first and the MTP layers last, so a search can give each layer its own
option; `{option: full}` alone recomputes every layer.

A configuration that states `context.census` has each layer kind measured on
a fake layer, the layer census, which takes tens of seconds on a laptop. A
census is kept in `~/.cache/hyper_parallel/census` and read back by later
runs of the same model, length and module replacements on the same torch and
Transformers, until a file of the package it ran on changes; ND's own search
code is not one of them. `ND_CENSUS_CACHE` names another folder, or `off`
keeps none.

## Workflow

1. Construct the model.

   ND selects the model and layer types from the framework configuration by default, or from the `--model` option. Tensor shapes, data types, and model-specific features are filled from the model hyperparameters.

2. Construct the parallelism space.

   The search space is constructed from the dimensions passed with `-l`. Each dimension must satisfy its own constraints and must be compatible with the device number passed with `-d` and global batch size passed with `-b`.

3. Filter out-of-memory configurations.

   Memory filtering is delegated to `memory_estimation/`. See `memory_estimation/README.md` for the memory estimator interface and supported model details.

4. Rank configurations by performance.

   Remaining configurations are ranked by `perf_estimation/`.

![Performance estimation](figures/perf_overview.png)

## Usage

ND runs on a CPU. Besides HyperParallel's `requirements.txt`, it imports four
libraries that file does not list, for its tables, its correlations and its
plots: pandas, matplotlib, SciPy and Pillow. Install them once:

```bash
pip install pandas matplotlib scipy pillow
```

Run the ND entrypoint as a Python module from the repository root:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd \
    -y <mindformers_yaml> \
    -l DP MP PP EP MB MBS \
    -d 1024 \
    -b 2048 \
    -t 10
```

This example varies data parallelism, model or tensor parallelism, pipeline parallelism, expert parallelism, micro-batch number, and micro-batch size. The `-d` option fixes the total number of devices, `-b` fixes the global batch size, and `-t` controls how many top configurations are printed.

In HyperParallel, the ND module should be imported as:

```python
from hyper_parallel.auto_parallel.sapp_nd import nd
```

The command prints valid configurations, the subset fitting the memory budget, the top ranked strategies, and timing for search and ordering. From verbosity level 2, ND can also generate debug CSV files and plots.

![Plot example](figures/plot_example.png)

## Command Options

```text
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd
    -y YAML_CONFIG
    [-d DEVICES]
    [-b GLOBAL_BATCH_SIZE]
    [-m MODEL]
    [-l [DIMENSIONS ...]]
    [-j JOBS]
    [-v VERBOSITY]
    [-A DEVICE_TYPE]
    [-mppb | --manual_pipeline_balance]
    [-ar | --auto_recompute]
    [-ao | --auto_offload]
    [--host_link_gibps GIB_PER_S]
    [--sustained_tflops TFLOPS]
    [-t TOP_CONFIG_NUMBER]
    [-e N | --exhaustive N]
    [-ee N | --force_exhaustive N]
    [-eee N | --fforce-exhaustive N]
    [-mem MEM_FOR_PPB]
    [-o OUTPUT_DIR]
    [--real_csv REAL_CSV]
    [--ranking_csv RANKING_CSV]
    [-rp | --recompute_potentials]
```

- `-y`, `--yaml_config`: path to the framework yaml configuration file.
- `-d`, `--devices`: number of devices. If omitted, ND uses the yaml value.
- `-b`, `--global_batch_size`: global batch size. If omitted, ND uses the yaml value.
- `-m`, `--model`: model name. If omitted, ND uses the yaml value.
- `-l`, `--dimensions`: parallel dimensions to vary.
- `-j`, `--jobs`: worker processes that price the search's candidates, `1` to
  price them in the main process. The default is `ND_JOBS`, else half the
  logical cores and at most 4, and 1 above verbosity 2, whose line per
  estimate only the main process prints. The workers start before the yaml is
  parsed and rebuild the search from the command line, so a search with `-ar`,
  `--recompute`, a recompute dimension in its yaml or a multimodal model, and
  `--real_csv`, `-s` and `-V`, price in the main process, and so does a search
  of fewer than 2000 candidates, for which starting the workers costs more
  than they save. Every output is the same either way.
- `-v`, `--verbosity`: verbosity in range `[0, 6]`, default `2`. Plots and
  debug CSV files are generated from level `2`.
- `-A`, `--device_type`: device type, such as `A2` or `A3`.
- `-mppb`, `--manual_pipeline_balance`: read offset and recompute from yaml.
- `-ar`, `--auto_recompute`: give every layer of each configuration the
  fastest recompute option that fits, instead of scoring it fully recomputed.
- `-ao`, `--auto_offload`: with `-ar`, let each pipeline stage's first layers
  offload their activations to the host instead of recomputing them, over the
  device's host link (see Hardware). Priced at one chunk per stage; not with a
  search config, whose trainer runs no offload.
- `--host_link_gibps`, `--sustained_tflops`: the host link's copy bandwidth
  and the device's sustained throughput `-ao` prices with, replacing the
  device's placeholders.
- `-t`, `--top_config_number`: number of top configurations to print and plot.
- `-e N`, `--exhaustive N`: ensure at least `N` plotted configurations have
  degree greater than `1` for each requested dimension, counting the normal
  top results. Default `0`.
- `-ee N`, `--force_exhaustive N`: request at least `N` additional
  configurations per requested dimension with degree greater than `1`. When
  combined with a larger `-e`, also fill its minimum. Default `0`.
- `-eee N`, `--fforce-exhaustive N`: request `N` distinct additional
  configurations per requested dimension with degree greater than `1`;
  reserve each addition for one dimension. A positive value takes precedence
  over `-e` and `-ee`. Default `0`.
- `-mem`, `--mem_for_ppb`: memory reserved for pipeline balancing.
- `-o`, `--output-dir`: directory for the standard search's `results.pdf`,
  `debug.csv`, and `debug_mem.csv`; also controls resolved search-config
  output and real-versus-estimate output. Without it, standard search debug
  artifacts go to `nd/output/` inside the SAPP-ND package.
- `--real_csv`: instead of searching, compare ND's estimate with the
  configurations measured in a classified profiling CSV, see below. With
  `-o`, `--output-dir`, ND's real-versus-estimate plots and its estimates are
  written there.
- `--ranking_csv`: also write every configuration the search keeps, in ND's
  order, to a CSV: rank, degrees, peak memory in MB, score and the parts of
  the score. Scores keep full precision, so configurations ND cannot tell
  apart show as ties. This is what a sweep reads to profile ND's best
  configurations.
- `-rp`, `--recompute_potentials`: also report what the cost model estimates
  each further recomputed layer would cost, see below. Needs `-ar` with the
  modes each layer chooses among, and a verbosity of at least `2`. Off by
  default, and the search does none of that work without it.

### Recompute potentials

With `-ar` and the modes each layer chooses among (`--recompute`, or a
hyper_v2 yaml's `context.recompute`), `-rp` extends every configuration that
fits memory: keeping its degrees and its memory budget, it fully recomputes
one more layer, takes the fastest plan the cost model offers, and repeats
while a plan is left. Layers of one kind, pipeline stage, micro-batches in
flight and peak boundary are interchangeable, so turning up any one of them
counts as one plan.

It writes three files beside `results.pdf` in `-o`, `--output-dir`:

| File | Contents |
| --- | --- |
| `recompute_potentials.csv` | A row per step of each configuration's sequence: rank, step, degrees, the layer the step turned up, how many layers the plan fully recomputes, peak memory in MB, the score, the plan as the trainer's `activation_checkpoint` reads it, and how wide the plan is |
| `result_recompute_inc.pdf` | The retained sequences, a bar per plan split into the parts of the score, with the plan's layer table under it |
| `result_recompute_full_next.pdf` | Per configuration, the plans the cost model scores below the next strategy, plus the first plan at or above it |

Every figure is the cost model's estimate of the step, the same estimate the
search ranks strategies with. A measured run can disagree with it, so read a
step between two plans as what the cost model expects the extra recomputed
layer to cost, not as a measured or guaranteed saving.

The report prices far more plans than the search itself, so it costs a good
deal more time than a plain search: that is why it is opt-in. It applies to
the standard ND search, not to `--real_csv`, `--search-config` or `--verify`.

### Exhaustive plot selection

The three exhaustive options cover the dimensions requested with `-l`. For
each dimension `D`, a configuration qualifies when its degree for `D` is
greater than `1`. Each option accepts a non-negative integer; `0` disables
that option. They apply to the standard ND search and cannot be combined
with `--real_csv`, `-s`/`--search-config`, or `-V`/`--verify` when nonzero.

The normal plot contains the top `-t M` configurations, subject to its cutoff
at twenty times the best score. When `-t` is omitted, the console lists all
fitting configurations and the normal plot contains at most twenty. Exhaustive
additions are chosen from the remaining configurations in performance order,
after memory filtering, and can extend beyond the normal score cutoff.

Let `P` be the number of normal plotted results with `D > 1`, `E` the value
of `-e`, and `F` the value of `-ee`:

| Options | Additional qualifying results selected for `D` |
| --- | --- |
| `-e E` alone | `max(0, E - P)` |
| `-ee F` with `F >= E` | `F`; forced additions take precedence |
| `-ee F` with `F < E` | `max(F, E - P)`; force `F` additions, then fill the `-e` minimum |
| `-eee N` with `N > 0` | `N` distinct additions reserved for `D`; `-e` and `-ee` are ignored |

For example, with `-e 5 -ee 2`, a dimension present above degree one in four
normal results still gets two additions, while one present in only one
normal result gets four additions. With `-e 2 -ee 2`, every dimension gets
two additional qualifying results regardless of its count in the normal plot.

For `-e` and `-ee`, an additional configuration can satisfy several dimensions
and is plotted once. Consequently, five dimensions with `-ee 2` can yield fewer
than ten unique additions. If the top five already have DP, EP, CP and OP
greater than one, but all have MP equal to one, `-e 2` needs only two additions
with MP greater than one.

Use `-eee 2` to request ten unique additions for five dimensions. ND processes
dimensions in the order supplied to `-l`, choosing the best remaining unused
configurations for each one. Ten additions require two unused qualifying
configurations to remain for every dimension when it is processed. Each option
returns fewer additions when the candidates are exhausted; `-eee` logs a
shortfall warning, visible from verbosity `3`.

From the repository root, this plots five top results and up to two distinct
additions for each of DP, EP, MP, CP and OP:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd \
    -y train.yaml -f hyper_v2 -d 64 -t 5 \
    -l DP EP MP CP OP -eee 2 -o output/nd_perf
```

The console prints `Additional exhaustive configurations` after the top
results when `-t` is supplied, followed by the number of unique plot additions.
The plot retains performance order and draws a black vertical divider between
the normal results and the additions when both groups are present. The divider
is omitted when there are no additions. The first table row, `TOP`, shows each
configuration's one-based position in the full performance ranking of all
configurations that fit memory. The CLI tables also start with a `TOP` column
using the same global positions for both the top and additional results.
Additions retain their global ranks even when intermediate results are skipped
in the plot or additional-results table. Each column's estimated peak memory
appears in the `MEM` row below the performance bars.

Use a fresh output directory for each run: `debug.csv` and `debug_mem.csv`
append rows to existing files, while `results.pdf` is replaced. The full
ranking of configurations that fit memory remains available through
`--ranking_csv`, independently of the displayed selection. Its `rank` column
contains the same global positions as `TOP` in the CLI and plot.

## Comparing with a Profiled Run

`nd.trace_classify` turns per-rank `torch.profiler` traces recorded with `with_stack=True` into the measured-run CSV ND compares against:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify traces/rank*.pt.trace.json.gz \
    --dims DP=4,PP=1,MB=2,MBS=1,OP=4 \
    --csv real.csv --perf-parts perf.csv --detail detail.csv
```

Each profiled step is split into `comp`, one `<dim>_wait` per parallelism type and the idle rest. A wait is time blocked in communication, typed by the HyperParallel module that issued it; the CSV holds the mean over steps and ranks. `--perf-parts` writes the same step in the columns of ND's `debug.csv`, and `--detail` keeps every rank and step by call site. `--dims` takes the acronyms of `-l` except `SP`, which ND reads back as true whatever its value. The split measures host-side blocking, which is exact for host-synchronous backends such as gloo.

An **Ascend** run is read from its directory instead of a trace file, and needs neither Python frames nor host blocking:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify profiling_dp64_ep16_op2 \
    --dims DP=64,MP=1,PP=1,CP=1,EP=16,MB=1,OP=2 --csv real.csv
```

`step_trace_time.csv` gives the closed top-level split of each step, and `communication.json` apportions the exposed communication over the axes in proportion to each axis's share of HCCL elapse time. Two consequences worth knowing. Device compute is not attributed to a pass, so it lands in `UNSPLIT_COMPUTE` and leaves `FW_COMPUTE`, `BW_COMPUTE` and `RECOMPUTE` empty. And the axis of a collective comes from its kind, since all-to-all is expert parallelism and gathers and reduce-scatters are FSDP; when TP or CP is active those two are ambiguous and stay unclassified until the rank sets of `communication_matrix.json` are read. Context parallelism takes what only it can send: point to point when there is a single pipeline stage, and all-to-all when there is no expert parallelism.

Then run ND on the same model and cluster with `--real_csv`:

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd -f hyper_v2 -y train.yaml -d 4 \
    --real_csv real.csv -o compare/
```

ND estimates every configuration in the CSV and prints, per configuration, each part's measured value and share next to ND's share of its own score, then the correlation and distance of every part across configurations. Shares make the two comparable despite ND's score units. Correlations need at least two configurations. ND has no idle term, so the measured idle share is what the estimate does not account for.

With `-o`, three files named after the CSV go to that directory: `real.pdf`, the measured parts beside ND's, idle included; `real_no_idle.pdf`, the same without idle and ordered by the measured step less idle, since idle can be half of a short step and differ from run to run; and `real_estimates.csv`, ND's estimate of every configuration with its degrees, the measured step, ND's peak memory in MB, the score and its parts. The measured bars count FSDP waits (`op_wait`) as DP and sequence-parallel waits (`sp_wait`) as MP, as the correlations do.

## Structure

```text
sapp_nd/
|-- README.md
|-- figures/
|-- memory_estimation/
|-- nd/
|   |-- common/
|   |   |-- framework_parsers/
|   |   |-- _cost_model_variables.py
|   |   |-- arch_hooks.py
|   |   |-- config.py
|   |   |-- cost_model_preprocess.py
|   |   |-- generate_partitions.py
|   |   |-- hardware.py
|   |   `-- layer_type.py
|   |-- balancing_adapter.py
|   |-- debug.py
|   |-- dimensions.py
|   |-- global_config.py
|   |-- logger.py
|   |-- parallelize.py
|   |-- run_nd.py
|   `-- trace_classify.py
`-- perf_estimation/
```

## Supported Scope

### Framework Configurations

- HyperParallel yaml configurations (`-f hyper_v2`), with or without a search config.
- MindSpore and MindFormers yaml configurations.
- Megatron json configurations.
- TorchTitan toml configurations are planned but not complete in this PR.

### Models

- Transformer dense models, including Llama-series and Qwen-series up to Qwen 2.5.
- Qwen3-series, including MoE variants, and Qwen3.5, whose stack alternates
  linear and full attention: the model's `layer_types` names each layer's
  kind, and each layer is priced as its kind. A model that states no
  `layer_types`, or names a kind its op profile lacks, is priced as the
  profile's default kind throughout.
- Mixture-of-Experts models, including DeepSeekV3.
- Multimodal models are in progress.

### Hardware

- Ascend A2.
- Ascend A3.
- Other Ascend variants and GPU support are future work.

Offload (`-ao`) prices each copy to the host with two figures per device,
`HostLink` in `nd/common/hardware.py`: the copy bandwidth to pinned host
memory, in GiB/s, and the sustained dense throughput, in TFLOP/s, which turns
a copy's seconds into the estimate's units. The figures there are
placeholders, not measurements: 16 GiB/s is hyper_offload's own default
before it profiles the link, and the throughputs are about half of each
device's dense peak. Measure the link on the target with hyper_offload's
`profile_transfer_bandwidth`, and the throughput from a profiled training
step, or take the vendor's figures; then state them in `hardware.py` or pass
`--host_link_gibps` and `--sustained_tflops`.

### Parallel Dimensions

- `DP`: data parallelism.
- `MP`: model or tensor parallelism.
- `SP`: Megatron sequence parallelism.
- `EP`: expert parallelism.
- `PP`: pipeline parallelism.
- `OP`: optimizer or ZeRO-DP parallelism: how many data-parallel ranks the
  optimizer shards a parameter over, on top of TP.
- `MB`: micro-batch number.
- `MBS`: micro-batch size.
- `VPP`: virtual pipeline parallelism.
- `CP`: context parallelism. Its time is what the algorithm exchanges each
  layer: the keys and values ring and hybrid CP gather, the query, key, value
  and output Ulysses all-to-alls, or a linear-attention layer's recurrent
  state. Its memory is a rank's share of the sequence, plus the other ranks'
  keys and values where the algorithm gathers them.

## Notes

The original toolkit also contains validation scripts, result CSV files, yaml examples, DSL utilities, regression utilities, and PPB code. Those assets are not part of this SAPP-ND PR2 package. Pipeline balancing is maintained separately in the SAPP-PPB module.
