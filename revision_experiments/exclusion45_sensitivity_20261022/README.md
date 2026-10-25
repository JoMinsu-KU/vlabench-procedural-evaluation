# Instruction and goal consistency sensitivity

This module preserves the original 2,133-case audit and separately excludes 45 cases with archived instruction/goal inconsistencies: five density-QA cases and 40 spatial-billiards cases. The filtered population contains 2,088 cases and all 34 task clusters. The same list intersects the 322-case decoding control in 13 cases, leaving 309. Selection does not use model success.

From the repository root:

```bash
python -m pip install -r revision_experiments/requirements-revision-20261023.txt
python revision_experiments/exclusion45_sensitivity_20261022/reproduce.py
```

The command reproduces paired counts, task-cluster intervals, six-contrast joint max-|T| intervals, auxiliary Holm corrections, evaluator component summaries, and decoding sensitivities. It requires no GPU, model server, simulator, credentials, or private paths. Existing expected results are never overwritten. Use --output-dir with a new directory for repeat runs.

The frozen labels were first reproduced against all 6,399 archived outputs in the remote audit. They are supplied as inputs here; this command does not rerun inference or replace the v1.7 cross-scoring implementation. The original matching/projection scripts remain in the base artifact. inputs/source_scene_configs.json retains the 69 relevant archived instruction/configuration pairs. The narrow semantic audit is not an assertion that all other dataset cases are error-free.
