# Oracle final audit

## Decision

The six Oracle model-condition runs pass the release integrity checks with recorded provenance exceptions. Every final CSV contains the same 2,133 unique cases in canonical manifest order, and its success count agrees with the condition summary and model-level pair summary.

## Main results

| Model | Full B | Phase-oracle | Target-oracle | Phase − Full, clustered 95% CI | Target − Full, clustered 95% CI | Target − Phase, clustered 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 80/2,133 (3.75%) | 427/2,133 (20.02%) | 488/2,133 (22.88%) | +16.27 pp [10.33, 22.54] | +19.13 pp [7.38, 32.34] | +2.86 pp [−12.51, 18.89] |
| Gemma-3-12B-it | 57/2,133 (2.67%) | 344/2,133 (16.13%) | 585/2,133 (27.43%) | +13.46 pp [7.11, 20.95] | +24.75 pp [11.78, 39.02] | +11.30 pp [−5.99, 28.72] |
| InternVL3-8B | 40/2,133 (1.88%) | 314/2,133 (14.72%) | 191/2,133 (8.95%) | +12.85 pp [7.71, 18.45] | +7.08 pp [2.38, 12.97] | −5.77 pp [−13.88, 3.02] |

Intervals use the same 34 `category/task_name` clusters, 20,000 percentile-bootstrap resamples, and seed `20260711` as the A/B analysis. All six Oracle-versus-Full intervals exclude zero. All three Target-versus-Phase intervals include zero, so the ordering of the two Oracle conditions is descriptive only.

## Run integrity

- Each final CSV has 2,133 rows, 2,133 unique case IDs, and exact manifest order.
- The runs used temperature 0, maximum output length 1,024 tokens, and one native attempt.
- No semantic repair was used. The prompts did not expose ground-truth targets, and scoring did not use `operation_sequence`.
- Each condition's prompt archive contains 2,133 files. Prompt names and content digests were identical across the three models within a condition.
- Target-oracle accessed reference targets only after the generated phase sequence passed the frozen contract: 501 Qwen, 602 Gemma, and 210 InternVL cases. It accessed targets for zero invalid sequences.
- One Gemma Target-oracle response ended at the token limit; it was recorded as a C2 JSON-parse failure and not counted as success.

Detailed checks are in `oracle_run_integrity_audit.csv` and `oracle_jsonl_deduplication_audit.csv` in this directory.

## Failure responsibility adjudication

The release preserves observed first-failure codes separately from responsibility adjudication. Five Full-B and 467 Phase-oracle C5 `AttributeError` outcomes were native exceptions caused by target-object method incompatibilities that the pre-execution affordance gate did not detect. The responsibility layer therefore labels these 472 cases `target_context_failure` with reason `affordance_gate_false_negative` without changing success labels.

| Condition/model | Adjudicated cases |
|---|---:|
| Full B / Gemma | 2 |
| Full B / InternVL | 3 |
| Phase-oracle / Qwen | 137 |
| Phase-oracle / Gemma | 173 |
| Phase-oracle / InternVL | 157 |
| Total | 472 |

Canonical sources are `../reported_statistics/oracle_failure_taxonomy_observed.csv`, `../reported_statistics/oracle_failure_taxonomy_adjudicated.csv`, and `oracle_attribute_error_adjudication.csv`.

## Recorded provenance exceptions

1. The source InternVL Phase-oracle JSONL had 4,266 lines, including 2,133 exact duplicates. The source InternVL Target-oracle JSONL had 2,682 lines, including 549 exact duplicates. Public JSONL files retain the first occurrence of each exact line and contain 2,133 cases. Final CSVs were already complete and are the canonical scoring source.
2. The protocol referenced a standalone external file named `oracle_ablation_validation_20260715.json`, but it was absent from the handoff snapshot. The release verifier independently checks the final CSVs, summaries, manifest order, success counts, and claim boundaries. `procedural_interface/scripts/validate_oracle_ablations.py` reproduces the static Oracle-contract checks when the public-view dataset is available.
3. The as-run static procedural schema did not exactly encode the runtime parser's conditional target contract. The public schema is aligned with the as-run runtime behavior; this documentation correction does not alter stored outcomes. See `SCHEMA_RUNTIME_CONTRACT_AUDIT.md`.

## Reporting boundaries

- Both Oracle controls outperform Full B for all three models under the task-clustered intervals.
- The pattern supports the diagnostic conclusion that phase/order generation and target grounding both contribute to low full-contract success.
- Oracle success remains below 30%; it is not evidence of high procedural-planning accuracy.
- The controls simplify different contract components and are not isolated single-factor causal interventions.
- Success sets are not nested, and the data do not support an inferential ranking of Target-oracle versus Phase-oracle.

## Canonical outputs

- `../reported_statistics/oracle_main_results_clustered.csv`
- `../reported_statistics/oracle_paired_recovery.csv`
- `../reported_statistics/oracle_task_level_results.csv`
- `../reported_statistics/oracle_task_recovery_distribution.csv`
- `../reported_statistics/oracle_failure_taxonomy_observed.csv`
- `../reported_statistics/oracle_failure_taxonomy_adjudicated.csv`
- `oracle_run_integrity_audit.csv`
- `oracle_jsonl_deduplication_audit.csv`
- `oracle_attribute_error_adjudication.csv`
- `oracle_preintegration_validation_summary.json`

Regenerate the published Oracle tables from the repository root with:

```bash
python analysis/reproduce_oracle_statistics.py
```
