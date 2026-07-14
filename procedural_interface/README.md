# Procedural Execution Interface

This package implements the procedural planning contract and native execution path.

The model selects the skill, numeric target, and sequence order. The interface validates the structured output, binds numeric targets to scene entities, and converts accepted steps to task-specific VLABench operations. Geometric poses, trajectories, and continuous control remain in the native backend.

The interface does not replace targets, reorder steps, insert missing actions, or replay an expert sequence. Invalid structure and incompatible targets are reported before native execution; accepted procedures are evaluated with the benchmark task condition.

Key locations:

- `interface/`: procedural JSON parsing and schema checks
- `adapter/`: target binding, phase templates, and task-specific lowering
- `execution/`: VLABench SkillLib and MuJoCo bridge
- `evaluation/`: outcome taxonomy and condition accounting
- `scripts/run_procedural_vlm_eval_api.py`: model inference and execution runner
- `scripts/run_canonical_procedure_control.py`: reference-procedure viability control
