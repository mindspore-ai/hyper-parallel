# /formal-verify — Training correctness verification

Use the [formal-verification skill](../skills/formal-verification/SKILL.md) with the requested
module/profile. Its [formal-verifier agent](../agents/formal-verifier.md) is a thin entry point
to the same workflow.

Examples: `/formal-verify dtensor`, `/formal-verify fsdp`, `/formal-verify pipeline`.
Inspect the offline plan, check the installed official engine, then run the requested scope.
Report candidate and reproduction status separately. If engine setup is missing, surface the
specific setup requirement and launch plan; do not describe a manual review as an engine run.
