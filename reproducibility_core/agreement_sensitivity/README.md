# Agreement and positive-control first-attempt sensitivity

This directory adds frozen-output analyses for the submission artifact.  No
model generation, native rollout, target substitution, sequence repair, or
success-label modification is performed.

## Files

- `agreement_metrics.csv`: original-interface and frozen-projection 2x2
  compositions; raw, positive, negative, chance-corrected, correlation,
  odds-ratio, conditional-probability, and Jaccard measures; and percentile
  95% task-cluster bootstrap intervals.  Pooled rows are fixed-backbone
  descriptive aggregates, not inference to a model population.
- `control_first_attempt_subset.csv`: existing results restricted to the 2,062
  cases whose source-derived positive control satisfied the final condition on
  its first attempt.  It covers original and projected overlap, downstream
  projected-success/full-failure, stage-wise termination, and scaffold-minus-
  full-procedure contrasts.
- `projectability_failure_reasons.csv`: model-specific projectability counts.
  All 1,811 unprojectable rows are exactly the parser-rejected rows; no
  parser-accepted output is unprojectable because of an unsupported action,
  missing field, or ambiguous macro target in the archived data.
- `agreement_sensitivity_audit.json`: analysis configuration, input/output
  checksums, invariants, and headline values.

## Definitions

For a 2x2 table with `a=both success`, `b=left only`, `c=right only`, and
`d=neither`, positive agreement is `2a/(2a+b+c)` and negative agreement is
`2d/(2d+b+c)`.  MCC and phi are identical for these binary tables.  Odds ratios
use the uncorrected cross-product ratio for the reported point estimate.  If a
bootstrap replicate contains a zero cell, the Haldane--Anscombe correction is
applied only to that replicate's odds ratio; the output records the number of
corrected replicates.

Task clusters are sampled with replacement 20,000 times using seed 20260711.
The intervals quantify sensitivity to the empirical composition and weighting
of the 34 retained task clusters.  They are not population-level confidence
intervals over unseen tasks or models.

## Reproduction

From the repository root:

```bash
python analysis/analyze_agreement_and_control_subset.py --repo-root .
python tests/verify_release.py .
```
