# Phase-Oracle and Target-Oracle Ablation Protocol

> Public provenance note: this is a path-sanitized copy of the dated pre-run protocol. The original source SHA-256 is `B5596BAD1C4B1C488382ED100BF8C3BF26E7C5FD026ADB3EB2CD16B1C79E364F`. Only workstation user and absolute path tokens were replaced; the protocol content and chronology were otherwise preserved.

## Purpose

The two ablations decompose the procedural contract into the two discrete responsibilities that dominate pre-native validation.

| Condition | Model responsibility | Oracle responsibility | Native endpoint |
|---|---|---|---|
| Full B | phase/skill, order, target | none | unchanged VLABench-native execution |
| Phase oracle | target number for each target-taking reference phase | benchmark-derived phase/skill order | unchanged VLABench-native execution |
| Target oracle | complete phase/skill order | case-specific reference targets | unchanged VLABench-native execution |

These are separate diagnostic conditions. They must not be merged into the main Full-B result.

## Frozen ownership rules

1. Phase oracle exposes the task registry's ordered phase skeleton in the prompt. The model returns only `{"targets": [...]}`.
2. The phase-oracle resolver neither reads nor inserts case-specific reference targets.
3. Target oracle asks the model for only `{"skills": [...]}`.
4. The target-oracle resolver checks the model sequence before reading reference targets. Missing, extra, or reordered skills are rejected and receive no target insertion.
5. Reference targets are read from `env_config.task.conditions` and mapped to numbered components only in the explicitly labeled target-oracle condition.
6. No condition reads `operation_sequence.json` or calls `get_expert_skill_sequence()` at runtime.
7. No model-emitted skill, order, or target is silently repaired.
8. First-attempt task-condition success is the primary native endpoint. Same-plan retry is optional robustness analysis and must use a separate run label.

## Validation before model serving

Run the real-data integrity check on the remote workstation:

```powershell
wsl -d Ubuntu-vlabench --user ${WSL_USER} -- bash -lc "source ${CONDA_PROFILE} && conda activate vlabench310_repaired && cd ${PROJECT_ROOT} && PYTHONPATH=${PROJECT_ROOT} python -u procedural_interface/scripts/validate_oracle_ablations.py"
```

Pass criteria:

- exactly 2,133 cases and 34 retained task types;
- reference targets resolve for every case;
- phase oracle reproduces the frozen skeleton using model-target slots only;
- target oracle reads targets only after a valid model skill sequence;
- missing target, missing skill, and adjacent-swap mutations are not repaired;
- no operation-sequence read or expert-sequence call.

## Prompt-only inspection

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ${PROJECT_ROOT}\procedural_interface\scripts\run_procedural_vlm_eval_api.ps1 `
  -AblationMode phase_oracle `
  -PromptOnly `
  -OutputRoot ${PROJECT_ROOT}/outputs/validation/phase_oracle_prompt_only_2133

powershell -NoProfile -ExecutionPolicy Bypass `
  -File ${PROJECT_ROOT}\procedural_interface\scripts\run_procedural_vlm_eval_api.ps1 `
  -AblationMode target_oracle `
  -PromptOnly `
  -OutputRoot ${PROJECT_ROOT}/outputs/validation/target_oracle_prompt_only_2133
```

## One-case live endpoint test

After the user starts and forwards one vLLM model, run each condition with `-MaxCases 1 -InferenceOnly` first. Inspect the raw response, condition-specific parser record, and oracle provenance before allowing native execution.

## 34-case pilot

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ${PROJECT_ROOT}\procedural_interface\scripts\run_oracle_ablation_pair.ps1 `
  -Model "<served-model-id>" `
  -RunLabel "<model>_oracle_pilot34_20260715" `
  -CaseListFile ${PROJECT_ROOT}/datasets/manifests/pilot_34_one_per_task_20260701.txt `
  -NativeMaxAttempts 1
```

## Full 2,133-case pair

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ${PROJECT_ROOT}\procedural_interface\scripts\run_oracle_ablation_pair.ps1 `
  -Model "<served-model-id>" `
  -RunLabel "<model>_oracle_full2133_20260715" `
  -NativeMaxAttempts 1
```

Run Qwen3-VL-8B-Instruct, Gemma-3-12B-it, and InternVL3-8B sequentially because the GCP instance serves one model at a time. Record the exact served model identifier and revision in each run directory before manuscript analysis.

## Implementation validation completed on 2026-07-15

The following checks were completed on the remote workstation. These are implementation checks and are not manuscript experiment results.

| Check | Result | Evidence |
|---|---:|---|
| Python compile | passed | remote `vlabench310_repaired` conda environment |
| Real manifest coverage | 2,133/2,133 cases, 34 task types | `outputs/validation/oracle_ablation_validation_20260715.json` |
| Reference target slots | 3,100 resolved | same validation report |
| No-repair mutations | 6,357/6,357 rejected as specified | same validation report |
| Phase-oracle prompts | 2,133 generated | `outputs/validation/phase_oracle_prompt_only_2133_20260715` |
| Target-oracle prompts | 2,133 generated | `outputs/validation/target_oracle_prompt_only_2133_20260715` |
| Oracle native-path check | C0 in both modes using a non-model reference fixture | `outputs/validation/oracle_native_path_smoke_20260715` |
| Live InternVL parser check | both schemas parsed; invalid skill sequence did not trigger oracle target loading | `outputs/oracle_ablations/internvl_oracle_live_inference_smoke_20260715` |

The unchanged Full-B prompt function has the same SHA-256 before and after implementation (`c9d3cf1e47472cc568fa45c87c877a31c04d243a774b9e268bafd230bb36ca09`).
