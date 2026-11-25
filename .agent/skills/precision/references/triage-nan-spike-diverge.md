# Numerical Triage: NaN, Loss Spikes, and Non-Convergence

Use the [comparison workflow](align-flow.md) to locate the first divergent
operation before investigating downstream symptoms. Proprietary criteria
without an available implementation remain pending validation.

## NaN / Inf

1. Enable msprobe overflow checking to identify the first API, module, and
   step that produces NaN or Inf.
2. Check division by zero, log of zero, exponential overflow, invalid
   softmax masking, FP16 overflow, and invalid gather indices.
3. Separate device/environment failures from numerical failures. Device
   initialization on a host without an accelerator is not a numerical NaN.
   Check whether a particular input or shape triggers the failure.
4. Re-run the same inputs after a fix and verify that the first failing
   operation and its downstream values recover.

## Loss spikes

1. Record the spike step and the preceding normal step.
2. Compare `grad_norm` and inspect gradient explosion, learning-rate or
   warmup configuration, and anomalous input batches.
3. Repeat with the same seed. A repeatable spike suggests a deterministic
   code/data cause; an inconsistent spike calls for checks of random state,
   communication, and races.
4. Disable one suspected feature, such as a fused kernel or recomputation,
   under the same comparison protocol. Use a controlled experiment to test
   the hypothesis.

## Non-convergence

1. Compare the first-step loss with the baseline. An initial mismatch points
   to forward/backward correctness. A matching first step followed by drift
   directs attention to the optimizer, learning-rate schedule, or data.
2. Verify that trained parameters receive nonzero gradients and belong to
   the optimizer's parameter groups.
3. Check target shifts and masks. Pre-shifted labels followed by another
   shift inside the module are a common source of an incorrect objective.
4. Check reduced-precision scoring and resulting selection/routing changes.
   A dump comparison can expose the first changed operation.

## Output

Use `templates/precision-report.md` to record the first failure, evidence
from dumps or controlled comparisons, command, commit, environment, and
the proposed correction.

## Evidence requirements

- Verify that the intended instrumentation or intervention actually ran.
- Use comparisons that can falsify the hypothesis, rather than correlations
  inferred from logs.
- Apply the acceptance rule's known BF16 tie boundary only when differences
  are confined to selections with a reference-score gap below one ULP.
