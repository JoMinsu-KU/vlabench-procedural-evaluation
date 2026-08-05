# Reproducibility artifact v1.3

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
