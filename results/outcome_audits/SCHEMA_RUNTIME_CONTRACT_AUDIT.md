# Procedural schema–runtime contract audit

## Finding

The as-run static schema required both `skill` and `target` on every step and allowed `target=null`. The runtime parser and prompt used a conditional contract:

- target-taking skills (`pick`, `moveto`, `place`, `press`, and `push`) require an integer target;
- no-target skills (`lift`, `pull`, `pour`, `wait`, `move_offset`, and `open_gripper`) prohibit the target field.

The static schema therefore did not exactly describe the runtime validation contract. This discrepancy does not require recomputing outcomes because evaluation used the runtime parser and evaluator, and the final CSVs record those results.

## Release treatment

`shared/schemas/procedural_plan_schema.json` is aligned with the as-run runtime contract in this public artifact. The experimental outputs and their success labels are unchanged. The original discrepancy and the release correction are retained here as provenance.
