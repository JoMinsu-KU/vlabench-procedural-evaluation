# VLABench Symbolic and Procedural Planning Evaluation

This repository provides code and case-level results for two matched VLM planning evaluations on VLABench. Both evaluations use the same 2,133 non-complex cases and the same three model families, but they expose planning at different levels of abstraction.

- **Symbolic evaluation** follows the released VLABench VLM protocol. Generated high-level skill sequences are scored with Skill Recall (SR), Parameter Recall (PR), Skill-Parameter Recall (SPR), and Precise Matching (PM).
- **Procedural evaluation** asks the model to select an expanded sequence of discrete skills, numeric targets, and order decisions. A target- and order-preserving interface lowers the generated procedure to VLABench SkillLib and MuJoCo execution.

The two evaluations generate different outputs. Their results should be interpreted as measurements under two planning contracts rather than as repeated scoring of one plan.

## Repository structure

```text
analysis/                   Statistical analysis and figure/table generation
analysis_data/              Compact paired inputs for the analysis
configs/                    Portable model and runtime configuration
datasets/                   Fixed manifests, preparation tools, and selection audit
procedural_interface/       Parser, validation, target binding, adapters, and execution
prompts/                    Complete prompts for both evaluations
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

For procedural evaluation, the primary endpoint is first-attempt task-condition success. Result files also report success within up to ten native attempts of the same generated plan. Native retries do not alter the selected skill, target, or order.

## Reproducing the analysis

```bash
python analysis/reproduce_reported_statistics.py
python tests/verify_release.py .
```

The first command regenerates the reported tables and figures. The second checks the 2,133-case manifest, six paired model result files, anonymity constraints, and repository checksums.

## Execution traces

Model-specific native traces are distributed as separate release archives because of their size. Verify each archive against the release-level `SHA256SUMS.txt` before use.

## License

Original code in this repository is released under the MIT License. VLABench, model weights, and other third-party dependencies remain subject to their own licenses.
