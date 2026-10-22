# Paired schema-decoding control

This control compares JSON object and JSON Schema response formats on the same 322 cases from the original 2,133-case replay-qualified manifest. Each retained task contributes up to ten randomly selected cases, with all cases retained for smaller tasks (selection seed 20261021). The experiment contains 1,932 fresh outputs: 322 cases, two formats, and three models.

The existing Full Reference-Procedure prompt and skill descriptions were reused. Within each new matched request pair, image bytes, instruction, prompt, model revision, temperature 0, inference seed 0, and the 1,024-token limit are identical. Only `response_format` differs. This is a comparison within a common new serving environment, not a before/after rerun of the historical 2,133-case experiment.

## Results

| Model | Parser accepted: object / schema | Phase and target accepted | First-attempt task success |
|---|---:|---:|---:|
| Qwen3-VL-8B | 45 / 322 | 4 / 13 | 0 / 6 |
| Gemma-3-12B | 282 / 322 | 30 / 30 | 4 / 3 |
| InternVL3-8B | 172 / 322 | 29 / 31 | 12 / 13 |

Each arm has 322 cases. InternVL's raw object-mode schema-valid count is 170; two additional responses pass the existing numeric-string normalization. All schema-constrained responses are raw-schema valid. There are no API errors or truncated main-run responses.

The schema enforces required fields, the 11-skill vocabulary, and integer targets for target-taking skills. It contains no task-specific reference sequence, correct target identity, object-index range, or affordance answer. Native evaluation uses one attempt and no semantic repair. Schema validity, reference-phase conformance, target/context compatibility, native completion, and final-condition success are separate outcomes.

Schema-minus-object goal-success differences are +1.86, -0.31, and +0.31 percentage points. The task-clustered 95% intervals touch zero for every model. Auxiliary exact McNemar tests, Holm-adjusted across the three models, are 0.09375, 1, and 1. Thus, output conformance improves, but the experiment does not establish a consistent final-success improvement or prove that all residual errors are pure planning deficiencies.

## Files

- `cloud_bundle/`: fixed case manifest, schema, original cloud runner, per-case prompts and instructions. Input-image hashes are in the manifest; images are not duplicated in Git.
- `gcp_outputs/`: unchanged generated responses, request records, model revisions, serving configuration, and completion manifests.
- `evaluation_outputs/`: unchanged case-level first-attempt evaluations and run configuration. Larger worker traces are a separate release asset.
- `analysis/`: original paired metrics, task tables, token/truncation audit, and failure subtypes.
- `frozen_analysis_rows.jsonl`: a compact join of the generated and evaluated outputs with source checksums. No success label is edited.
- `schema_decoding.py`: image inference, schema audit, and offline native-evaluation commands.
- `reproduce_revision.py`: analysis-only reproduction without models, GPUs, images, or simulator execution.
- `review_followup/`: archived replay-retention and scaffold-transition diagnostics. These are descriptive reanalyses, not new inference runs.
- `EXPORT_PROVENANCE.json`: source checksums and export boundaries.

## Reproduce statistics

From this directory:

```bash
python -m pip install -r requirements-reproduction.txt
python -m unittest test_schema_analysis
python reproduce_revision.py
```

Use Python 3.10--3.12. The last command recalculates all 18 paired metric rows and compares them with the archived counts, 20,000-replicate task-cluster intervals (seed 20261021), and Holm-adjusted auxiliary tests. It writes to a new `reproduced/` directory and refuses to overwrite an existing output. It makes no model API calls and performs no native rollout. `test_schema_decoding.py` additionally tests the POSIX inference runner and installed VLABench adapter; run it in Linux/WSL with the base project's full dependencies and explicit `VLABENCH_PROJECT_ROOT` and `VLABENCH_PUBLIC_ROOT`, not in the lightweight statistics-only environment. The PowerShell wrapper also requires explicit project, dataset, and Python paths. Original test and wrapper files are retained under `execution_source/`; only their publication copies have portable path parameters. Model inference, schema, adapter, evaluator, and stored outcomes are unchanged.

## Full input and native-trace evidence

`schema_decoding_inputs_native_traces_20261022.tar.gz` is distributed separately from Git. It contains the original model-visible inputs and native worker evidence; no model weights, virtual environments, authentication credentials, or private reviewer correspondence are included. VLABench inputs remain subject to upstream licensing. After unpacking the archive into a separate directory, combine its `cloud_bundle/inputs` and `evaluation_outputs` with a working copy of this directory to run the original integrity-checked analysis or to inspect native traces. Do not unpack over the only copy of an experiment.

Fresh inference and native replay require the base repository's adapter/backend and the upstream VLABench public-view case tree, including its original environment configurations. Run `python schema_decoding.py --help` for explicit paths. Do not regenerate or retry an incorrect saved output to change an archived result.

## Runtime and provenance

The two arms used BF16 on H100 80 GB, vLLM 0.21.0+cu129, PyTorch 2.11.0+cu129, Python 3.12.15, and XGrammar with compact JSON (`disable_any_whitespace=true`). Model revisions and full serve commands are preserved per model. A pre-main pilot exposing whitespace repetition was excluded before this common configuration was fixed; the control was not preregistered. The new serving runtime is not claimed to reproduce the historical serving runtime exactly.

Independent human validation is not completed by this artifact. Source audits, mutation tests, and native controls must not be described as independent human expert judgments.

Original code follows the repository's MIT License. Third-party inputs and dependencies retain their original licenses.
