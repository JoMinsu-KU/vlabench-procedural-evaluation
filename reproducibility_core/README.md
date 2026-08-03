# Reproducibility core added in artifact v1.2

This directory contains evidence that can be recomputed or independently inspected without rerunning model inference or native simulation.

- `registry_audit/` traces the frozen 34-task reference-procedure registry to the pinned upstream VLABench source. The audit is author-led and source-linked; it is not an independent inter-rater validation and does not assess procedure uniqueness.
- `robustness/` contains leave-one-task-out and leave-one-category-out composition-sensitivity results. These are descriptive diagnostics for the fixed analysis manifest, not inference to unseen tasks.
- `result_tables/` separates observed runtime evidence, responsibility adjudication, offline corrected-gate validation, positive controls, and mutation controls.
- `scripts/` contains the standard-library-only audit and sensitivity programs.
- `checksums/` records the immutable v1.1 baseline used to construct this release candidate.

No script in this directory modifies a model output, native trace, registry entry, or case-level success label.
