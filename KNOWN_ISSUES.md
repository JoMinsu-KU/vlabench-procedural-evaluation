# Known issues and reporting boundaries

## Canonical Oracle JSONL

The source InternVL Phase-oracle JSONL contained 4,266 lines with 2,133 exact duplicate lines. The source InternVL Target-oracle JSONL contained 2,682 lines with 549 exact duplicate lines. The public JSONL files apply exact-line deduplication while preserving the first occurrence, yielding 2,133 lines and 2,133 unique case IDs per condition.

The final result CSVs were already complete and unaffected. They are the canonical scoring source. Source and canonical counts are recorded in `results/outcome_audits/oracle_release_build.json` and `oracle_jsonl_deduplication_audit.csv`.

## Recovered validation report

The protocol's `oracle_ablation_validation_20260715.json` was absent from the handoff snapshot but remained on the experiment workstation. It was recovered on 2026-07-20 and is now released under `results/validation_evidence/` together with compact native-path and live-parser smoke evidence.

The original report SHA-256 is `6a8a95247c8118851d20bb230ab2143b247f15b6572453d66e21105f6e8c49a4`. The public copy differs only by replacement of machine-specific provenance paths and has SHA-256 `ce1ef08e83b913ef38ef75c380e3b8c4d90eeb623b73eb3eee88b088249039b2`. It records 2,133 cases, 34 tasks, 3,100 target-slot checks, 6,357 mutation checks, no expert-sequence access, no semantic repair, and no errors. The release verifier now asserts these fields directly.

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
