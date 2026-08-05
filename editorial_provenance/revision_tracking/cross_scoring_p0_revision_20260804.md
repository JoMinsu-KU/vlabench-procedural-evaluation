# Frozen-output cross-evaluation revision record

Date: 2026-08-04

## Scope

This revision adds a deterministic cross-evaluation of already generated full reference-procedure outputs under the symbolic endpoint. It uses archived output only and performs no model inference and no native rollout.

## Implemented evidence

- Full-procedure output to symbolic projection with a frozen, source-derived mapping.
- No reparsing of parser-rejected output.
- No target replacement, action reordering, insertion of missing model-owned targets, reference-sequence substitution, or invalid-output recovery.
- Exact reproduction of all 6,399 archived type-normalized symbolic endpoint labels before cross-scoring.
- Five passing no-repair projection controls.
- Row-level projection results and model/pooled summaries.
- Explicit feasibility finding that the released symbolic branch has no verified model-choice-preserving symbolic-to-native executor entry point.

## Verified pooled results

| Quantity | Value |
|---|---:|
| Archived model outputs | 6,399 |
| Projectable full outputs | 4,588 (71.70%) |
| Projected symbolic successes | 450 (7.03%) |
| Full-procedure first-attempt successes | 177 (2.77%) |
| Projected symbolic success with full-procedure failure | 273 (4.27%) |
| Original/projected shared successes | 203 |
| Original-only successes | 386 |
| Projected-only successes | 247 |
| Original/projected success-set Jaccard | 24.28% |
| Symbolic endpoint reproduction mismatches | 0 |
| Model inference calls | 0 |
| Native rollouts | 0 |

## Claim boundary

The results support a combined interpretation: original cross-interface disagreement reflects both prompt-conditioned high-level plan elicitation and additional termination under procedure-specific validation or execution. The projection is deterministic and lossy, not a same-plan causal intervention. No reverse symbolic-to-native score is reported.

## Authoritative files

- `analysis/cross_score_frozen_outputs.py`
- `reproducibility_core/cross_scoring/CROSS_SCORING_PROTOCOL.md`
- `reproducibility_core/cross_scoring/projection_spec.json`
- `reproducibility_core/cross_scoring/projection_no_repair_controls.json`
- `reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv`
- `reproducibility_core/cross_scoring/full_to_symbolic_summary.csv`
- `reproducibility_core/cross_scoring/cross_scoring_audit.json`
- `results/reported_statistics/frozen_output_cross_scoring.csv`

