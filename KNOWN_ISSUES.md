# Known issues and reporting boundaries

## Canonical Oracle JSONL

The source InternVL Phase-oracle JSONL contained 4,266 lines with 2,133 exact duplicate lines. The source InternVL Target-oracle JSONL contained 2,682 lines with 549 exact duplicate lines. The public JSONL files apply exact-line deduplication while preserving the first occurrence, yielding 2,133 lines and 2,133 unique case IDs per condition.

The final result CSVs were already complete and unaffected. They are the canonical scoring source. Source and canonical counts are recorded in `results/outcome_audits/oracle_release_build.json` and `oracle_jsonl_deduplication_audit.csv`.

## External validation report

The protocol referenced `oracle_ablation_validation_20260715.json`, but that standalone file was not present in the handoff snapshot. The included `procedural_interface/scripts/validate_oracle_ablations.py` reproduces the static contract checks when the VLABench public-view data are available. The release verifier independently checks row counts, manifest order, summaries, success counts, and claim boundaries without relying on the missing report.

## Observed and adjudicated failures

Five Full-B and 467 Phase-oracle C5 `AttributeError` outcomes occurred after the original affordance gate failed to detect target-object method incompatibility. Raw failure codes remain unchanged in the released result files. A separate responsibility taxonomy classifies these 472 cases as `target_context_failure` with reason `affordance_gate_false_negative`.

This adjudication does not change any primary success outcome.

## Static schema alignment

The as-run static JSON schema required a `target` field on every step, while the runtime parser required integer targets only for target-taking skills and prohibited targets on no-target skills. The public schema is aligned with the runtime contract. This documentation fix does not alter stored experimental outcomes.

## Statistical interpretation

- All six Oracle-versus-Full task-clustered 95% confidence intervals exclude zero.
- All three Target-oracle-versus-Phase-oracle intervals include zero.
- Oracle controls simplify different output-contract components and are not pure causal interventions.
- Absolute Oracle success remains below 30% and must not be described as high procedural accuracy.

## Path sanitization

Machine-specific project and dataset paths are replaced with `${PROJECT_ROOT}` and `${VLABENCH_PUBLIC_ROOT}`. This affects provenance strings only, not model outputs, evaluator fields, or success labels.
