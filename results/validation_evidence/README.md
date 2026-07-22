# Oracle validation evidence

This directory contains validation evidence recovered from the experiment workstation on 2026-07-20 after it was found to have been omitted from the handoff snapshot.

## Included evidence

- `oracle_ablation_validation_20260715.json`: canonical static contract validation over 2,133 cases and 34 tasks. It records 3,100 target-slot checks, 6,357 no-repair mutation checks, no access to `operation_sequence`, no expert-sequence call, no semantic repair, and no errors.
- `VLABench_phase_target_oracle_ablation_protocol_20260715.md`: path-sanitized copy of the dated protocol defined before the Oracle model runs. The file records the original source SHA-256 and preserves the protocol content apart from workstation user/path tokens.
- `full_b_no_repair_validation_20260721.json`: reproducible offline Full-B parser/binder battery over the 2,133 released positive-control fixtures. It records 10,623 missing, reordered, extra, target-incomplete, and alternative-target preservation checks with zero semantic repair and zero validation errors.
- `oracle_native_path_smoke_20260715/`: one non-model reference fixture for each Oracle condition. Both fixtures reached C0 through the native execution and task-condition path. These files validate interface wiring; they are not model results.
- `internvl_oracle_live_inference_smoke_20260715/`: one inference-only InternVL case for each Oracle output schema. The Phase-oracle response parsed successfully. The intentionally sequence-invalid Target-oracle response was rejected before reference-target loading (`oracle_target_read=false`). These files validate parser and access ordering; they are not full-run outcomes.
- `SOURCE_SHA256SUMS.txt`: SHA-256 digests of the 40 recovered source files before public-path sanitization.

The original validation report digest was `6a8a95247c8118851d20bb230ab2143b247f15b6572453d66e21105f6e8c49a4`. The public copy has digest `ce1ef08e83b913ef38ef75c380e3b8c4d90eeb623b73eb3eee88b088249039b2`; the only change is replacement of machine-specific provenance paths with `${PROJECT_ROOT}` and `${VLABENCH_PUBLIC_ROOT}`.

## Deliberate exclusions

The workstation also contained 4,270 Phase-oracle and 4,270 Target-oracle prompt-only intermediate files. They are not copied because the actual full-run prompt archives are already released under `prompts/oracle/`, and the prompt-only summaries contain no scored outcomes. A one-case Full-B prompt regression smoke and duplicate operational logs were also excluded because the complete Full-B runs and the canonical JSON report supersede them.

See `../outcome_audits/REMOTE_VALIDATION_EVIDENCE_RECONCILIATION.md` for the source-to-release reconciliation.

Corrected-affordance revalidation is stored separately under `../outcome_audits/corrected_affordance_revalidation_20260721.{json,csv}` because it adjudicates frozen model outcomes rather than validating Oracle ownership. Its public script requires the external VLABench public-view `env_config.json` tree, which is not redistributed.
