# Source-to-registry provenance audit protocol

## Objective

The audit tests whether the frozen reference-procedure registry accurately and
traceably transcribes the ordered expert-skill sequence exposed by the pinned
VLABench executor source for all 34 task types in the retained manifest. It is
a documented source-linked provenance audit, not an independent human or
inter-rater validation and not a test of procedure uniqueness.

## Frozen inputs

- Artifact baseline: `artifact-v1.1`, commit
  `31f4c8ce50943ac8282b4bab0d567cc5a597d730`
- Registry: `procedural_interface/adapter/task_phase_templates.py`
- Retained manifest: `analysis_data/manifest_2133.json`
- Upstream source: OpenMOSS/VLABench commit
  `cf588fe60c0c7282174fe979f5913170cfe69017`

## Automated structural checks

For every retained task type, the checker records and compares:

1. registered task name and registered class;
2. registry `official_class` and `official_module`;
3. effective defining class and source line of `get_expert_skill_sequence`;
4. ordered `SkillLib` call sequence;
5. inheritance path from the registered class to the defining class;
6. registry phase count and target-taking flags;
7. control-flow nodes inside the effective expert method;
8. case count and the method used to assign retained cases to a branch.

The automatic target-flag check verifies consistency with the released
model-facing contract rule. It is not, by itself, a semantic proof that every
target field is the only possible abstraction of the source arguments.

## Structured semantic review

The generated draft crosswalk is inspected against the pinned source. For each
task, the review records the source/registry mapping rationale, conditional
branch handling, target-taking phase rationale, discrepancy class, resolution,
and impact. The manuscript and artifact must not call it independent, external,
human-expert-confirmed, or inter-rater validation.

## Case-to-branch assignment

Allowed `case_assignment_method` values are:

- `metadata`
- `initial_state`
- `case_configuration`
- `static_branch_resolution`
- `retained_execution_trace`
- `manual_source_interpretation`
- `not_applicable_single_sequence`

If the effective expert method contains a branch that can change skill order,
every retained case must be assigned to a documented branch. Any unresolved
case-to-branch assignment is a stop condition.

## Discrepancy classes

- **D1 — documentation/provenance:** source location, inherited defining class,
  line, wording, or other traceability issue that does not change the procedure.
- **D2 — equivalent abstraction:** naming or representation differs but the
  ordered model-facing skill and target responsibilities are equivalent.
- **D3 — substantive mismatch:** missing/extra phase, changed order,
  incompatible target-taking flag, wrong active branch, or wrong task/class
  mapping that may affect validation, lowering, or reported outcomes.
- **UNRESOLVED:** evidence is insufficient to assign D1–D3.

## Stop conditions

Narrative revision must stop before result claims are finalized if any of the
following occurs:

- at least one D3 mismatch;
- at least one retained case has an unresolved case-to-branch assignment;
- a referenced task, class, method, or source file cannot be traced;
- automated and structured-review results disagree without resolution;
- a computed audit count conflicts with an authoritative CSV/JSON source.

## Pass criteria

- 34/34 retained task types are source-traceable;
- nonexistent source references: 0;
- unresolved phase/order/target mismatches: 0;
- undocumented active branches: 0;
- unresolved retained case-to-branch assignments: 0;
- D3 mismatches: 0.

Passing this audit supports source traceability and transcription fidelity for
the fixed contract. It does not establish that the contract is unique or that
all alternative physically valid procedures have been enumerated.
