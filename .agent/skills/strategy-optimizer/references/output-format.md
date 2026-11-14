# Optimizer outputs

`optimization_result.json` contains `metadata`, `baseline`, `iterations` and `final_result`.
Each iteration records `candidates`, `constraints`, `config_files`, and `validation`.
Validation entries retain candidate ID, success/error, metrics, eligibility, target result and log path.

`final_result.status` is `PLANNED`, `SUCCESS`, `PARTIAL` or `FAILED`.
`optimal_candidate` and `optimal_metrics` are null without an eligible measurement.
A measured winner adds `optimal_strategy`, `production_config` and its validation log when available.
`optimization_report.md` summarizes this final result. The production YAML retains the template's training length.
There is no automatic deployment or generated deployment guide.
