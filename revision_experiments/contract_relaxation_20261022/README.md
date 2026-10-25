# Fixed task-stratified relaxed-contract diagnostic

One archived structural rejection per nonempty task/backbone stratum was selected using SHA256 ranking with seed 20261022. The 86 plans cover 33 tasks; all 102 strata over 34 tasks are recorded. Seven previously inspected scenes were excluded across backbones. Selection used neither reference-target correctness, runtime support, nor observed success.

Only exact reference phase-count/skill-order matching was bypassed. Skills, numeric targets, order, original parser fixture, geometry, and reset policy were retained. Five plans had unsupported phase semantics. The 81 runnable plans each received one no-action, one matched-scene reference, and one unchanged-alternative rollout: 243 calls, no retries.

From the repository root:

```bash
python revision_experiments/contract_relaxation_20261022/reproduce.py
```

This GPU-free command rechecks saved native commands and raw captured traces, preserves model choices, and verifies 5 goal-reaching alternatives, 3 all-stage-success goals, 2 goals with false stage flags, and 74/81 reference goals. It launches neither a model nor a simulator. Use --output-dir with a new directory to repeat it.

The release asset contract_relaxation_inputs_traces_20261023.tar.gz contains original model-visible images, instructions, raw responses, native records, and final camera views. Compact JSON traces and analysis are included in Git for direct auditing. Absolute paths inside archived commands are original provenance values, not portable reanalysis dependencies. Operational launchers, host information, private reviews, and manuscripts are not included.

These are descriptive diagnostic observations, not replacement B scores or a population false-rejection rate. Scenes are matched, not native execution seeds. Native stage flags are backend indicators, not human semantic judgments. Controls that failed are retained. Native re-execution additionally requires the pinned VLABench installation and licensed assets; it is intentionally not triggered by a reproduction or publishing command.
