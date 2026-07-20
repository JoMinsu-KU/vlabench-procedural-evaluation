# VLABench Official-Style Symbolic Evaluation

This module runs Experiment A: VLABench VLM symbolic planning evaluation on the same 2,133 examples used by the procedural execution branch.

The implementation follows the official VLABench VLM evaluation protocol:

- official skill-library prompt: `prompts/eval_vlm_en.txt`
- official input modality: scene image, numbered-mask image, and natural-language instruction
- official output schema: JSON skill sequence with `name` and `params`
- official symbolic scorer: `VLABench.evaluation.utils.get_final_score`

Only the model invocation backend is changed. Instead of loading a local Hugging Face model inside the official script, `inference/openai_compatible_official_vlm.py` calls a vLLM OpenAI-compatible API endpoint.

## Folder Layout

| Folder | Purpose |
|---|---|
| `dataset/` | Loads the shared 2,133 manifest, public model-visible inputs, and scoring-only GT operation sequences. |
| `inference/` | Builds official VLABench multimodal messages and calls the OpenAI-compatible VLM backend. |
| `prompts/` | Vendored official VLABench VLM prompt. |
| `scoring/` | Thin wrapper around the official VLABench symbolic scorer. |
| `scripts/` | User-facing runners for smoke tests, pilots, and full runs. |

## Main command

```bash
python vlabench_official_eval/scripts/run_official_vlm_eval_api.py \
  --public-root /path/to/public_view \
  --gt-root /path/to/vlabench_dataset \
  --max-cases 1 \
  --output-root outputs/official_sequence_eval/smoke
```

Run the script with `--help` for endpoint, model, resume, and scoring-only options. Do not enable `--response-format-json` unless a backend requires it. The official VLABench prompt asks for a JSON list, while JSON-object response mode can force some models to return an object-shaped refusal or explanation.

## Outputs

| File | Meaning |
|---|---|
| `case_outputs/*.json` | Per-example raw response, parsed skill sequence, GT sequence for scoring audit, and official score. |
| `official_vlm_symbolic_rows.jsonl` | Streaming row log for long runs. |
| `official_vlm_symbolic_scores.csv` | Compact spreadsheet-friendly score table. |
| `final_score_official_format.json` | Task/example nested score format compatible with VLABench-style reports. |
| `summary.json` | Aggregate symbolic success and official metric means. |

Primary binary success:

```text
G0 symbolic success = exact_match_score == 100
```

Auxiliary official metrics:

```text
skill_match_score
entity_match_score
skill_with_entity_match_score
exact_match_score
total_score
```
