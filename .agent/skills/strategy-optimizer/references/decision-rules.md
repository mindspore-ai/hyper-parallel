# Measured decisions

A candidate is eligible only when execution succeeds and complete finite positive throughput, step time and
peak memory are available. If supplied, `--max-memory-gb` is a hard selection cap.
The fastest eligible candidate is retained across rounds. Meeting
`baseline_throughput * target_improvement` produces `SUCCESS`.

OOM requests full FSDP and full recompute. Communication errors or timeouts reduce the TP/CP search limits.
Candidates already measured are excluded. No new candidates stops the loop; otherwise it runs at most three rounds.
An eligible candidate below target yields `PARTIAL`; no eligible candidate yields `FAILED`.
Planning yields `PLANNED` and has no measured winner or production configuration.
