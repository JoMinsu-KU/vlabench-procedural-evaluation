# Remote validation evidence reconciliation

The Oracle validation report and its compact implementation checks were recovered from the experiment workstation on 2026-07-20. Their earlier absence was a handoff-packaging omission, not a loss of experimental evidence.

| Workstation evidence | Release decision | Reason |
|---|---|---|
| `oracle_ablation_validation_20260715.json` | Included | Canonical static Oracle-contract and no-repair validation report |
| `oracle_native_path_smoke_20260715/` | Included | Compact two-condition evidence for native-path and condition-check wiring |
| `internvl_oracle_live_inference_smoke_20260715/` | Included | Compact two-schema evidence for live parsing and target-access ordering |
| Phase-oracle prompt-only directory (4,270 files) | Excluded | Superseded by the 2,133-case actual-run archive in `prompts/oracle/`; contains no scored outcomes |
| Target-oracle prompt-only directory (4,270 files) | Excluded | Superseded by the 2,133-case actual-run archive in `prompts/oracle/`; contains no scored outcomes |
| Full-B one-case prompt regression smoke (6 files) | Excluded | Superseded by the complete three-model Full-B runs |
| Root validation logs | Excluded | Operational duplicates of the structured JSON evidence |

The validation report records 2,133 cases, 34 tasks, 3,100 target-slot checks, and 6,357 mutation checks. It also records `operation_sequence_read=false`, `expert_sequence_called=false`, `semantic_repair_applied=false`, and an empty error list. The release verifier checks these fields directly.

All included source files are enumerated in `../validation_evidence/SOURCE_SHA256SUMS.txt`. Public copies replace only machine-specific path strings; model responses, parser decisions, evaluator states, and images are unchanged.
