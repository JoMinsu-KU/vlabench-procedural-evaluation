# Known issues and reporting boundaries

## Public Oracle JSONL

The source InternVL Phase-oracle JSONL contained 4,266 lines with 2,133 exact duplicate lines. The source InternVL Target-oracle JSONL contained 2,682 lines with 549 exact duplicate lines. The public JSONL files apply exact-line deduplication while preserving the first occurrence, yielding 2,133 lines and 2,133 unique case IDs per condition.

The final result CSVs were already complete and unaffected. They are the authoritative scoring source. Source and public counts are recorded in `results/outcome_audits/oracle_release_build.json` and `oracle_jsonl_deduplication_audit.csv`.

## Recovered validation report

The protocol's `oracle_ablation_validation_20260715.json` was absent from the handoff snapshot but remained on the experiment workstation. It was recovered on 2026-07-20 and is now released under `results/validation_evidence/` together with compact native-path and live-parser smoke evidence.

The original report SHA-256 is `6a8a95247c8118851d20bb230ab2143b247f15b6572453d66e21105f6e8c49a4`. The public copy differs only by replacement of machine-specific provenance paths and has SHA-256 `ce1ef08e83b913ef38ef75c380e3b8c4d90eeb623b73eb3eee88b088249039b2`. It records 2,133 cases, 34 tasks, 3,100 target-slot checks, 6,357 mutation checks, no expert-sequence access, no semantic repair, and no errors. The release verifier now asserts these fields directly.

## Observed and adjudicated failures

Five Full-B and 467 Phase-oracle C5 `AttributeError` outcomes occurred after the original affordance gate failed to detect target-object method incompatibility. Raw failure codes remain unchanged in the released result files. A separate responsibility taxonomy classifies these 472 cases as `target_context_failure` with reason `affordance_gate_false_negative`.

This adjudication does not change any primary success outcome.

Offline revalidation with the corrected method-availability rules flagged all 472 adjudicated C5 rows, 602 additional rows already rejected as C3 target/context failures, and no successful or other-class row. The corrected gate is therefore documented as a rejection-only refinement: it changes responsibility localization and counterfactual termination order, not raw runtime codes or success labels. Evidence is in `results/outcome_audits/corrected_affordance_revalidation_20260721.{json,csv}`.

The responsibility-adjudicated Full-B table assigns the five raw C5 rows to target/context and zero rows to intrinsic-native responsibility. This does not mean that no raw native exception occurred.

## Static schema alignment

The as-run static JSON schema required a `target` field on every step, while the runtime parser required integer targets only for target-taking skills and prohibited targets on no-target skills. The public schema is aligned with the runtime contract. This documentation fix does not alter stored experimental outcomes.

## Statistical interpretation

- All six pointwise Oracle-versus-Full task-cluster bootstrap intervals exclude zero.
- Five of six max-|T| family-wise simultaneous intervals exclude zero; the InternVL Target-minus-Full-B interval includes zero.
- All three Target-oracle-versus-Phase-oracle intervals include zero.
- The task-cluster bootstrap assesses sensitivity to the empirical composition of the 34 retained tasks; it is not interpreted as inference to unseen tasks.
- Phase-oracle and Target-oracle use different scaffolds and separately generated plans. Their outcome differences are diagnostic associations, not pure causal interventions or same-plan repairs.
- Absolute Oracle success remains below 30% and must not be described as high procedural accuracy.

## Symbolic endpoint representation sensitivity

Digit-only target strings are deterministically cast to integers in the type-normalized symbolic endpoint. Under raw released PM, Full-B success is numerically higher than symbolic full match for all three backbones; normalization reverses the ordering. Neither direction establishes metric superiority. Both representations and their paired differences are released in `results/reported_statistics/symbolic_endpoint_representation_sensitivity.csv`.

## Fixed analysis manifest and reference contract

All findings are conditional on the fixed 2,133-case dual-replay-qualified non-complex analysis manifest. Category retention is uneven, and the manifest does not represent the full VLABench case distribution. The benchmark-derived reference procedure is the fixed execution contract used by this evaluation; it does not enumerate every physically valid procedure.

## Ownership and no-repair controls

The recovered Oracle validation report records 6,357 mutation checks. A separate released Full-B battery applies 10,623 missing-phase, phase-swap, extra-phase, missing-target, and alternative-target preservation checks to the positive-control fixtures. Both batteries report zero semantic repair and zero validation errors. These are offline interface checks, not additional model inference or native-rollout outcomes.

## Path sanitization

Machine-specific project and dataset paths are replaced with `${PROJECT_ROOT}` and `${VLABENCH_PUBLIC_ROOT}`. This affects provenance strings only, not model outputs, evaluator fields, or success labels.
