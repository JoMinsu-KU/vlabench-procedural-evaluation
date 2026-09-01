# Reproducibility artifact v1.6

This release supersedes v1.5 as the submission artifact by adding
prevalence-aware agreement statistics, task-cluster composition intervals for
the agreement estimands, a positive-control first-attempt-success subset
sensitivity, and an exact model-level projectability-failure decomposition. It
introduces no model inference, native rollout, semantic repair, or label
modification.

New in v1.6:

- raw, positive, and negative agreement; Cohen's kappa; phi/MCC; odds ratios;
  directional conditional probabilities; and success-set Jaccard for the
  original and frozen-projection comparisons;
- percentile 95% task-cluster bootstrap intervals for those agreement metrics,
  using 20,000 replicates and seed 20260711;
- a sensitivity restricted to the 2,062 cases whose source-derived positive
  control satisfied the final condition on its first attempt, covering overlap,
  downstream termination, stage-wise outcomes, and reference-scaffold
  contrasts;
- model-level confirmation that all 1,811 unprojectable full outputs are
  exactly the parser-rejected rows (210 Qwen, 251 Gemma, and 1,350 InternVL),
  with zero parser-accepted failures due to unsupported actions or
  missing/ambiguous macro targets;
- a standalone reproduction script, machine-readable CSV/JSON outputs, input
  and output checksums, and release-verifier assertions.

The pooled agreement results are fixed-backbone descriptive aggregates and do
not treat the three selected backbones as a random sample from a model
population. The task-cluster intervals quantify sensitivity to the empirical
composition of the 34 retained tasks, not inference to unseen tasks.

Artifact v1.6 does not modify the retained manifest, archived outputs, native
traces, original success labels, projection rules, or earlier sensitivity
results. Earlier versioned tags remain available as release history.

## v1.5 projectable-subset sensitivity

This release supersedes v1.4 as the submission artifact by adding a projectability-conditioned success-set sensitivity analysis. It introduces no model inference, native rollout, or label modification.

New in v1.5:

- original-symbolic versus projected-full success-set composition restricted to the 4,588 projection-possible frozen outputs;
- 203 successes under both symbolic views, 271 original-symbolic-only successes, and 247 projected-full-only successes, yielding a 28.16% projectable-subset Jaccard overlap;
- explicit disclosure that 115 original symbolic successes were paired with unprojectable full-procedure outputs;
- the corresponding whole-macro-family-excluded projectable-subset result: 24.77% Jaccard, with 262 original-only and 227 projected-only successes;
- a standalone reproduction script, machine-readable CSV and JSON summaries, and release-verifier assertions.

Artifact v1.5 does not modify the retained manifest, model outputs, native traces, original success labels, macro-family exclusion rule, or projection rules released in v1.3. The projectable-subset result is a selection-conditioned sensitivity analysis rather than a replacement for the full-pipeline Jaccard estimand. Earlier versioned tags remain available as release history.

## v1.4 macro-family exclusion sensitivity

This release supersedes v1.3 as the submission artifact by adding a deterministic whole-task-family exclusion sensitivity for the source-defined macro projections. It introduces no model inference and no native rollout.

New in v1.4:

- explicit projected-success rates under both the full frozen-output denominator and the projectable-output denominator;
- exact task-level counts for source-defined insert-flower and add-condiment macro mappings;
- whole-family exclusion of all model--case rows from those task families, avoiding output-conditioned selection;
- 6,060 retained rows after excluding 339 macro-family rows, with 4,368 projectable outputs, 388 projected symbolic successes, 223 projected-success/full-procedure-failure outcomes, and a 21.52% original/projected success-set Jaccard;
- a reproduction script, exclusion specification, pooled and model-level outputs, and release-verifier assertions for the sensitivity analysis.

Artifact v1.4 does not modify the retained manifest, model outputs, native traces, original success labels, or projection rules released in v1.3. Earlier immutable tags remain available as release history.

## v1.3 frozen-output cross-evaluation

This release adds a frozen-output cross-evaluation to the v1.2 provenance and robustness evidence. It introduces no model inference and no native rollout.

New in v1.3:

- deterministic projection of all 6,399 archived full-procedure outputs onto the symbolic representation where the source-derived mapping is defined;
- 4,588 projectable outputs (71.70%) and 1,811 explicitly unprojectable outputs, with no semantic repair;
- 450 projected symbolic successes, including 273 that failed the full-procedure first-attempt endpoint;
- 24.28% pooled success-set Jaccard overlap between original symbolic outputs and projected full outputs;
- exact reproduction of all 6,399 archived type-normalized symbolic labels before cross-scoring;
- five passing no-repair projection controls and a row-level audit table;
- an explicit feasibility finding that no verified model-choice-preserving symbolic-to-native expansion path exists in the released symbolic branch.

The cross-evaluation separates evidence for prompt-conditioned high-level plan elicitation from downstream procedure-specific termination only to the extent supported by the deterministic, lossy projection. It is not a causal comparison, and no reverse symbolic-to-native score is claimed.

## v1.2 provenance and composition-sensitivity update

This release preserves every authoritative v1.1 model output and success label while adding source-linked registry provenance, fixed-manifest composition-sensitivity diagnostics, and the claim-boundary records used for the revised manuscript.

New in v1.2:

- a 34-task source-to-registry crosswalk pinned to VLABench commit `cf588fe60c0c7282174fe979f5913170cfe69017`;
- deterministic structural checks plus a documented author-led semantic source review;
- explicit inheritance, effective-method, source-permalink, target-taking-phase, and branch-handling provenance;
- zero D3 substantive mismatches, zero unresolved task or case-to-branch assignments, and zero registry or result-label changes;
- leave-one-task-out and leave-one-category-out composition-sensitivity tables for the primary cross-interface discordance metrics and secondary rate contrasts;
- explicit separation of observed runtime codes, responsibility adjudication, and offline corrected-gate evidence;
- provisional and final claim matrices plus a terminology glossary, separated from the reproducibility core as editorial provenance;
- regenerated framework and scaffold figures using manuscript-facing terminology.

The registry review is not an independent human or inter-rater validation and does not assess whether the fixed contract is unique or enumerates all physically valid procedures. The composition diagnostics characterize the fixed 2,133-case analysis manifest and do not support inference to unseen VLABench tasks.

## v1.1 baseline

This release integrates the completed Oracle runs with the final construct-validity framing, endpoint-sensitivity analysis, task-composition sensitivity, ownership controls, and responsibility adjudication used by the submission manuscript.

Included evidence:

- fixed 2,133-case dual-replay-qualified manifest and 4,000-case selection audit;
- three-backbone Experiment A and full Experiment B outputs;
- GT-derived procedural positive control;
- three-backbone Phase-oracle and Target-oracle full runs;
- pointwise task-cluster bootstrap intervals, six-contrast max-|T| simultaneous intervals, paired recovery, task-level results, and observed/responsibility-adjudicated failure taxonomies;
- recovered Oracle contract validation plus compact native-path and live-parser evidence;
- raw-versus-type-normalized symbolic endpoint sensitivity;
- 10,623 Full-B parser/binder no-repair checks generated from released positive-control fixtures;
- corrected-affordance offline revalidation covering 19,197 unique model-condition rows and reproducing all 472 C5 adjudications without changing success labels;
- the path-sanitized dated Oracle protocol with its original source digest;
- final reference-procedure terminology, figures, and mirrored reported-statistics tables;
- portable as-run evaluation code, prompt archives, environment capture, integrity tests, and checksums.

The artifact excludes abandoned exploratory study branches and historical manuscript drafts.

## Interpretation boundary

The symbolic and reference-procedure interfaces are deliberately different evaluation contracts. Reference-phase and reference-target scaffolds generate separate plans. Cross-condition differences are descriptive or scaffold-conditioned diagnostic associations, not causal decomposition or same-plan repair. Findings are conditional on the fixed 2,133-case retained manifest, deterministic cross-interface transformations, and the source-derived reference-procedure contract.
