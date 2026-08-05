# VLABench Symbolic and Reference-Procedure Planning Evaluation

This repository provides code, case-level results, audit evidence, and release checks for two deliberately non-equivalent VLM planning interfaces and two reference-scaffold conditions on VLABench. Every reported comparison uses the same 2,133-case dual-replay-qualified non-complex analysis manifest and the same three model families.

- **Symbolic evaluation** follows the released VLABench VLM protocol. Generated high-level skill sequences are scored with Skill Recall (SR), Parameter Recall (PR), Skill-Parameter Recall (SPR), and Precise Matching (PM).
- **Full Reference-Procedure Condition** asks the model to select a benchmark-derived sequence of discrete skills, numeric targets, and order decisions. A target- and order-preserving interface lowers accepted outputs to VLABench SkillLib and MuJoCo execution.
- **Reference-phase scaffold** (artifact field: `phase_oracle`) fixes the benchmark-derived phase/order contract and asks the model to select numeric targets.
- **Reference-target scaffold** (artifact field: `target_oracle`) asks the model to generate the phase/order sequence and inserts reference targets only after that sequence passes the frozen phase contract.

The symbolic and reference-procedure interfaces generate different outputs under different prompts, schemas, and endpoints. Their original results are interface-conditioned classifications rather than the causal effect of adding one component. Artifact v1.5 additionally projects the already generated full-procedure outputs onto the symbolic representation without model inference or semantic repair, reports a whole-task-family exclusion sensitivity for source-defined macro mappings, and quantifies success-set overlap after conditioning on projection-possible outputs. The reference-phase and reference-target conditions each generate a separate plan; their contrasts with the full condition are secondary scaffold-conditioned diagnostics, not same-plan repairs or causal decomposition.

For continuity with released files and scripts, the manuscript display names map to artifact identifiers as follows:

| Manuscript display name | Artifact condition |
|---|---|
| Symbolic Interface | `experiment_a` |
| Full Reference-Procedure Condition | `full_b` |
| Reference-Phase Scaffold | `phase_oracle` |
| Reference-Target Scaffold | `target_oracle` |

## Primary results

| Model | Type-normalized symbolic full match | Full-procedure first-attempt success | Reference-phase scaffold | Reference-target scaffold |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 280/2,133 (13.13%) | 80/2,133 (3.75%) | 427/2,133 (20.02%) | 488/2,133 (22.88%) |
| Gemma-3-12B-it | 221/2,133 (10.36%) | 57/2,133 (2.67%) | 344/2,133 (16.13%) | 585/2,133 (27.43%) |
| InternVL3-8B | 88/2,133 (4.13%) | 40/2,133 (1.88%) | 314/2,133 (14.72%) | 191/2,133 (8.95%) |

All six pointwise scaffold-minus-full-procedure task-cluster bootstrap intervals exclude zero. A six-contrast max-|T| family-wise task-composition sensitivity analysis retains zero-excluding simultaneous intervals for five contrasts; the InternVL reference-target-scaffold-minus-full-procedure interval includes zero. All three direct reference-target-minus-reference-phase intervals include zero. These intervals assess sensitivity to reweighting the 34 retained tasks and are not interpreted as inference to unseen VLABench tasks.

The symbolic-versus-procedure ordering is representation-sensitive: raw released PM places full-procedure success above symbolic full match for all three backbones, whereas deterministic digit-string normalization reverses the ordering. The release therefore treats low cross-interface agreement and stage-wise termination, rather than metric superiority, as the stable findings.

## Frozen-output cross-evaluation

The v1.3 audit, retained unchanged in v1.5, deterministically projects archived full-procedure outputs onto the symbolic skill--target graph. It does not reparse rejected output, insert actions or targets, reorder actions, replace model-selected targets, or invoke a model or simulator. The archived type-normalized symbolic labels are reproduced with zero mismatch before projection is accepted. Projection is considered possible only when the archived output passed the original parser and its emitted fields are sufficient for deterministic symbolic serialization; projectability does not imply symbolic correctness.

| Model | Projectable full outputs | Projected symbolic success | Projected success / full-procedure failure | Original/projected success-set Jaccard |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 1,923/2,133 (90.15%) | 254/2,133 (11.91%); 254/1,923 projectable (13.21%) | 174 | 29.30% |
| Gemma-3-12B-it | 1,882/2,133 (88.23%) | 139/2,133 (6.52%); 139/1,882 projectable (7.39%) | 82 | 25.44% |
| InternVL3-8B | 783/2,133 (36.71%) | 57/2,133 (2.67%); 57/783 projectable (7.28%) | 17 | 6.62% |
| **Pooled** | **4,588/6,399 (71.70%)** | **450/6,399 (7.03%); 450/4,588 projectable (9.81%)** | **273** | **24.28%** |

All 177 full-procedure successes passed symbolic projection, as expected by construction for a contract-conforming procedure. The diagnostically informative group is the additional 273 projected symbolic successes that failed the full-procedure endpoint. Across the full manifest, original and projected symbolic success sets had 24.28% overlap. Restricting the comparison to the 4,588 projection-possible outputs increased the Jaccard overlap to 28.16%, but left 271 original-only and 247 projected-only successes; 115 original symbolic successes were paired with unprojectable full outputs. This conditional analysis therefore shows that unprojectability explains only part of the low agreement. It is a sensitivity analysis conditioned on interface-specific output compliance, not a replacement primary estimand. A reverse symbolic-to-native score is not reported because the released symbolic branch contains no verified model-choice-preserving native expansion path; adding one would require a new adapter and native experiment.

### Macro-family exclusion sensitivity

The source-defined macro projection is used by the insert-flower and add-condiment task families. To avoid output-conditioned exclusion, v1.4 removed every model--case row belonging to those families and recomputed the frozen cross-score on the remaining rows. The exclusion removes 339 of 6,399 rows, leaving 6,060. Projection coverage is 4,368/6,060 (72.08%), projected symbolic success is 388/6,060 (6.40%; 388/4,368 projectable, 8.88%), projected-success/full-procedure-failure remains present in 223 cases, and the overall original/projected success-set Jaccard is 21.52% rather than 24.28%. Within the 4,368 projectable rows, the corresponding Jaccard is 24.77%, with 262 original-only and 227 projected-only successes. Thus, the magnitude changes modestly, but the non-identical success sets and downstream termination pattern do not depend on the macro-mapped task families.

The v1.4 and v1.5 sensitivities add no model inference or native rollout and do not modify the retained manifest, model outputs, native traces, original success labels, or v1.3 projection rules.

## Repository structure

```text
analysis/                   Statistical analysis and figure/table generation
analysis_data/              Compact paired inputs for the analysis
configs/                    Portable model and runtime configuration
datasets/                   Fixed manifests, preparation tools, and selection audit
editorial_provenance/       Claim-boundary and terminology records
procedural_interface/       Parser, validation, target binding, adapters, and execution
prompts/                    Evaluation and diagnostic-control prompts
reproducibility_core/       Source-registry audit and composition-sensitivity evidence
results/                    Model outputs, controls, audits, and reported statistics
shared/                     Shared schemas and utilities
tests/                      Package integrity and anonymity verification
vlabench_official_eval/     Released-style symbolic inference and scoring
```

## Upstream software

The evaluation targets VLABench revision `cf588fe60c0c7282174fe979f5913170cfe69017`. Dataset assets, VLABench source files, and model weights are not redistributed here. Install or obtain them under their respective licenses.

The recorded runs used Python 3.10, MuJoCo/dm_control, and an OpenAI-compatible vLLM endpoint with deterministic decoding (`temperature=0`).

```bash
conda create -n vlabench-proc-eval python=3.10 -y
conda activate vlabench-proc-eval
pip install -r requirements-runtime.txt
pip install -r requirements-analysis.txt
```

Install the pinned VLABench revision separately and expose it on `PYTHONPATH` as described by the upstream project.

## Dataset

The fixed subset contains all non-complex cases that reached the benchmark task condition in both predefined reference replay runs. Rebuild the manifest from the included 4,000-case audit with:

```bash
python datasets/scripts/rebuild_verified_manifest.py \
  --audit-csv datasets/selection_audit/two_replay_results_4000.csv \
  --output-json datasets/manifests/rebuilt_verified_subset_2133_cases.json
```

The model-visible view contains `input.png`, `input_mask.png`, `instruction.txt`, and `env_config.json`. Expert operation sequences are not copied into this view. See `datasets/scripts/prepare_public_view.py` for construction.

## Running the evaluations

The complete prompts are stored in `prompts/`. The symbolic runner is available at `vlabench_official_eval/scripts/run_official_vlm_eval_api.py`; the procedural runner is `procedural_interface/scripts/run_procedural_vlm_eval_api.py`. Both accept an OpenAI-compatible endpoint, a model identifier, the fixed case manifest, and an output directory. Run either module with `--help` for the full command line.

For symbolic evaluation, the repository reports raw and type-normalized SR, PR, SPR, and PM. The released-code total and a recomputed four-metric weighted score are retained as separate audit values. The binary endpoint used in paired analysis is the derived indicator `PM_norm == 100`.

For the full reference-procedure condition and both reference-scaffold conditions, the primary endpoint is first-attempt reference-procedure task success. Result files also report success within up to ten native attempts of the same generated full-condition plan. Native retries do not alter the selected skill, target, or order. Scaffold-condition runs used one native attempt.

The runner exposes `--ablation-mode full|phase_oracle|target_oracle`. The complete case prompts are stored once per scaffold condition under the legacy path `prompts/oracle/` because their contents were identical across backbones within a condition.

## Reproducing the analysis

```bash
python analysis/reproduce_reported_statistics.py
python analysis/cross_score_frozen_outputs.py
python analysis/analyze_macro_family_exclusion.py --repo-root .
python analysis/analyze_projectable_subset_jaccard.py --repo-root .
python analysis/audit_full_b_no_repair.py
python reproducibility_core/scripts/compute_composition_sensitivity.py --repo-root . --output-dir reproducibility_core/robustness
python tests/verify_release.py .
```

The first command regenerates the original cross-interface, scaffold, endpoint-sensitivity, simultaneous-interval, and adjudicated termination tables and figures. The cross-scoring command reproduces the deterministic full-to-symbolic projection, its five no-repair controls, row-level results, and summary table without inference or rollout. The macro-family command reproduces the mapping-use counts and the whole-family exclusion sensitivity. The projectable-subset command recomputes original/projected success-set overlap after conditioning on projection-possible frozen outputs. The next command reproduces 10,623 full-procedure parser/binder ownership and no-repair checks from the released positive-control fixtures. The composition-sensitivity command recomputes leave-one-task-out and leave-one-category-out diagnostics without changing any case-level label. The verifier checks these evidence layers, anonymity constraints, and repository checksums.

Scaffold-specific analysis can also be run independently through the legacy analysis entry points:

```bash
python analysis/reproduce_oracle_statistics.py
python analysis/generate_oracle_manuscript_assets.py
```

The corrected-affordance audit requires the external VLABench public-view case tree because `env_config.json` files are not redistributed:

```bash
python analysis/audit_corrected_affordance_revalidation.py \
  --vlabench-public-root /path/to/vlabench_public_view
```

## Execution traces

Model-specific native traces are distributed as separate release archives because of their size. Verify each archive against the release-level `SHA256SUMS.txt` before use.

## Provenance notes

The final 2,133-row CSV for each condition is the authoritative scoring source. Public scaffold-condition JSONL files preserve the first occurrence of each exact raw line; the duplicate counts found in the InternVL source logs are recorded in `results/outcome_audits/oracle_release_build.json`. The recovered scaffold-validation report verifies 2,133 cases, 34 tasks, 3,100 target slots, and 6,357 no-repair mutations. The separate full-procedure battery verifies 10,623 parser/binder ownership and no-repair behaviors. Corrected-gate offline revalidation flags all 472 adjudicated C5 rows, 602 rows already classified C3, and no successful row. Raw first-failure codes and responsibility-adjudicated classes remain separate. See `KNOWN_ISSUES.md` for the complete boundary.

The source-registry audit traces all 34 retained task types to the pinned VLABench source revision and records effective methods, call order, target-taking phases, inheritance, and branch handling. It is a documented author-led provenance audit, not an independent human or inter-rater validation. No substantive registry mismatch, unresolved case-to-branch assignment, registry modification, or result-label change was found. Leave-one-task-out and leave-one-category-out diagnostics assess sensitivity to the empirical composition of the fixed manifest; they are not inference to unseen tasks.

## License

Original code in this repository is released under the MIT License. VLABench, model weights, and other third-party dependencies remain subject to their own licenses.
