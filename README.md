# VLABench Symbolic and Procedural Planning Evaluation

This repository provides code and case-level results for two matched VLM planning evaluations and two diagnostic oracle controls on VLABench. Every reported model comparison uses the same 2,133 dual-replay-qualified non-complex cases and the same three model families.

- **Symbolic evaluation** follows the released VLABench VLM protocol. Generated high-level skill sequences are scored with Skill Recall (SR), Parameter Recall (PR), Skill-Parameter Recall (SPR), and Precise Matching (PM).
- **Procedural evaluation** asks the model to select an expanded sequence of discrete skills, numeric targets, and order decisions. A target- and order-preserving interface lowers the generated procedure to VLABench SkillLib and MuJoCo execution.
- **Phase-oracle** fixes the benchmark-derived phase/order contract and asks the model to select numeric targets.
- **Target-oracle** asks the model to generate the phase/order sequence and inserts reference targets only after that sequence passes the frozen phase contract.

The symbolic and procedural evaluations generate different outputs. Their results should be interpreted as measurements under two planning contracts rather than as repeated scoring of one plan. The oracle conditions simplify different parts of the procedural output contract and are diagnostic controls, not isolated causal interventions.

## Primary results

| Model | Symbolic full match | Full procedural | Phase-oracle | Target-oracle |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 280/2,133 (13.13%) | 80/2,133 (3.75%) | 427/2,133 (20.02%) | 488/2,133 (22.88%) |
| Gemma-3-12B-it | 221/2,133 (10.36%) | 57/2,133 (2.67%) | 344/2,133 (16.13%) | 585/2,133 (27.43%) |
| InternVL3-8B | 88/2,133 (4.13%) | 40/2,133 (1.88%) | 314/2,133 (14.72%) | 191/2,133 (8.95%) |

All six oracle-versus-full task-clustered 95% confidence intervals exclude zero. All three Target-oracle-versus-Phase-oracle intervals include zero, so the relative ordering of the two controls is descriptive only. Absolute oracle success remains below 30%.

## Repository structure

```text
analysis/                   Statistical analysis and figure/table generation
analysis_data/              Compact paired inputs for the analysis
configs/                    Portable model and runtime configuration
datasets/                   Fixed manifests, preparation tools, and selection audit
procedural_interface/       Parser, validation, target binding, adapters, and execution
prompts/                    Evaluation and diagnostic-control prompts
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

For procedural evaluation and both oracle controls, the primary endpoint is first-attempt task-condition success. Result files also report success within up to ten native attempts of the same generated plan for the full condition. Native retries do not alter the selected skill, target, or order. Oracle runs used one native attempt.

The runner exposes `--ablation-mode full|phase_oracle|target_oracle`. The complete case prompts are stored once per oracle condition under `prompts/oracle/` because their contents were identical across backbones within a condition.

## Reproducing the analysis

```bash
python analysis/reproduce_reported_statistics.py
python tests/verify_release.py .
```

The first command regenerates the A/B and Oracle tables and figures. The second checks the 2,133-case manifest, six A/B files, six Oracle runs, manifest order, success counts, clustered-interval claim boundaries, adjudication counts, anonymity constraints, and repository checksums.

Oracle-specific analysis can also be run independently:

```bash
python analysis/reproduce_oracle_statistics.py
```

## Execution traces

Model-specific native traces are distributed as separate release archives because of their size. Verify each archive against the release-level `SHA256SUMS.txt` before use.

## Provenance notes

The final 2,133-row CSV for each condition is the canonical scoring source. Canonical Oracle JSONL files preserve the first occurrence of each exact raw line; the duplicate counts found in the InternVL source logs are recorded in `results/outcome_audits/oracle_release_build.json`. Observed first-failure stages and responsibility-adjudicated failure classes are stored separately. See `KNOWN_ISSUES.md` for the complete boundary.

## License

Original code in this repository is released under the MIT License. VLABench, model weights, and other third-party dependencies remain subject to their own licenses.
