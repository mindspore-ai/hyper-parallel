# Environment profile (fill per site; never commit a filled copy)

> The bring-up checklist §1.4 says how to probe each row read-only. A filled
> copy contains internal hosts and addresses: keep it in site-local notes,
> outside any public repository.

| Field | Value | Probe |
|---|---|---|
| Accelerator / dies per node / HBM per die | `<e.g. 8 x 64GB>` | `npu-smi info` |
| Intra-node die topology (direct-link pairs vs switched) | `<topo>` | `npu-smi info -t topo` |
| Per-die NIC speed / device IPs | `<200Gb/s>` | `hccn_tool -i <die> -speed -g` |
| Host CPU / NUMA / memory | `<arch, cores, GB>` | `lscpu`, `free -g` |
| Shared storage mount + free space | `<path, TB>` | `df -h` |
| CANN toolkit location + source line | `<path>/set_env.sh` | site convention |
| Python env (torch / torch_npu / transformers) | `<versions>` | `pip show` per env |
| Dependency overlay dir (if the repo pins newer deps) | `<path>` | checklist §3 |
| Weight staging host + relay route | `<host, jump>` | checklist §2 |
| Node allocation / etiquette (who else runs here) | `<rules>` | team convention |
| Peak FLOPS per die | measure in S4 (microbenchmark) — do not copy spec sheets | checklist §1.4 |
