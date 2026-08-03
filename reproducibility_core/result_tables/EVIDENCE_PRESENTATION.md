# Validation and failure-evidence presentation

## Corrected affordance gate: three non-interchangeable layers

1. **Observed runtime first-termination taxonomy.** The original runs recorded
   472 relevant C5 `AttributeError` rows: five in Full B and 467 in the
   reference-phase scaffold. These raw codes remain unchanged.
2. **Responsibility-adjudicated taxonomy.** Row-level inspection assigned the
   472 failures to target/context responsibility because an incompatible
   model-selected target passed an incomplete pre-native affordance gate.
3. **Offline corrected-gate revalidation.** A deterministic rejection-only
   audit of 19,197 unique frozen model-condition-case outputs flagged all 472
   adjudicated C5 rows and 602 outputs already classified as C3. It flagged no
   successful output and changed no success label. It did not rerun model
   inference or native rollout.

The manuscript may state that zero cases were assigned to intrinsic-native
responsibility **after adjudication**. It must not state that no raw native
exception occurred or that the corrected gate was used in the original runs.

## Positive control and mutation controls

The GT-derived positive control establishes operational coverage of the frozen
reference-procedure path. It does not establish procedure uniqueness. Its
first-attempt result must be called the **observed first-attempt positive-control
success rate** (2,062/2,133, 96.67%), not a theoretical ceiling.

The 10,623 Full-B and 6,357 scaffold mutation checks support the no-repair
boundary. They test rejection/preservation behavior and do not establish the
general physical invalidity of rejected plans.

## Provenance

- Raw/adjudicated rows:
  `results/outcome_audits/oracle_attribute_error_adjudication.csv`
- Offline corrected-gate audit:
  `results/outcome_audits/corrected_affordance_revalidation_20260721.json`
- Positive control: `results/positive_control/`
- Full-B no-repair:
  `results/validation_evidence/full_b_no_repair_validation_20260721.json`
- Scaffold no-repair:
  `results/validation_evidence/oracle_ablation_validation_20260715.json`

