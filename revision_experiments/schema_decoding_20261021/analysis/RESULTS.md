# JSON-format sensitivity experiment

Fixed 322 cases, 34 task clusters, three BF16 backbones. Both arms are fresh paired outputs.
Native success is first-attempt condition success; rejected plans remain in the denominator.

| Model | JSON object success | JSON Schema success | Difference (pp) | Task-clustered 95% CI (pp) |
|---|---:|---:|---:|---:|
| qwen | 0/322 | 6/322 | 1.86 | [0.00, 4.91] |
| gemma | 4/322 | 3/322 | -0.31 | [-0.96, 0.00] |
| internvl | 12/322 | 13/322 | 0.31 | [0.00, 0.97] |

## Interpretation Boundaries

This experiment isolates response-format constraints within one fixed new serving environment. It does not replace the original 2,133-case A/B study.
Schema validity, procedural gate validity, and native goal success are separate endpoints. A schema-valid plan need not select the correct phases or targets.
Task-clustered confidence intervals are primary. Case-level McNemar results are auxiliary and do not remove task dependence.
No conclusion of improved planning competence follows from formatting improvements alone. Inspect phase/target failures and condition success before revising claims.

The category/task table and full token audit are descriptive diagnostics. Follow-up analyses must retain these fixed results and their provenance.
