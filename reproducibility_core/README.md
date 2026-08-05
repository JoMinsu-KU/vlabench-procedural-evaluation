# Reproducibility core for artifact v1.3

This directory contains evidence that can be recomputed or independently inspected without rerunning model inference or native simulation.

- `registry_audit/` traces the frozen 34-task reference-procedure registry to the pinned upstream VLABench source. The audit is author-led and source-linked; it is not an independent inter-rater validation and does not assess procedure uniqueness.
- `robustness/` contains leave-one-task-out and leave-one-category-out composition-sensitivity results. These are descriptive diagnostics for the fixed analysis manifest, not inference to unseen tasks.
- `result_tables/` separates observed runtime evidence, responsibility adjudication, offline corrected-gate validation, positive controls, and mutation controls.
- `scripts/` contains the standard-library-only audit and sensitivity programs.
- `checksums/` records the immutable v1.1 baseline used to construct this release.
- `cross_scoring/` contains the deterministic full-to-symbolic projection protocol, row-level scores, five no-repair controls, and audit metadata. It performs no model inference or native rollout; the reverse direction is not reported because no verified model-choice-preserving expansion path exists in the released symbolic branch.

No script in this directory modifies a model output, native trace, registry entry, or case-level success label.
