# Procedural Execution Interface

This folder contains the target-preserving execution interface used for Experiment B.

The interface may instantiate geometry for a VLM-selected step, but it must not choose, reorder, insert, or replace VLM symbolic/procedural steps.

Allowed responsibility of the interface:

```text
VLM-selected skill -> SkillLib-callable operation
VLM-selected target -> scene entity binding
VLM-selected order -> preserved execution order
task template -> geometry and motion parameter instantiation only
```

Forbidden behavior:

```text
expert sequence replay
GT target replacement
operation_sequence-based repair
missing step insertion
order correction
```

Key locations:

- `interface/`: procedural JSON parsing and runtime contract checks
- `adapter/`: numeric target binding, phase templates, and task-specific lowering
- `execution/`: VLABench SkillLib and MuJoCo bridge
- `evaluation/`: outcome taxonomy, Oracle controls, and condition accounting
- `scripts/run_procedural_vlm_eval_api.py`: inference and execution runner
- `scripts/run_canonical_procedure_control.py`: reference-procedure viability control

Inspect the portable command-line interface with:

```bash
python procedural_interface/scripts/run_procedural_vlm_eval_api.py --help
```
