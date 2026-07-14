# Released-Style VLABench Symbolic Evaluation

This package implements the symbolic evaluation used for the first planning contract. It follows the released VLABench VLM input and graph-matching procedure while using an OpenAI-compatible endpoint for model inference.

The model receives the scene image, numbered-mask image, instruction, and released skill-library prompt. It returns a JSON skill sequence with `name` and `params`. The scoring wrapper reports Skill Recall, Parameter Recall, Skill-Parameter Recall, and Precise Matching.

Directory guide:

- `dataset/`: fixed manifest, model-visible inputs, and scoring references
- `inference/`: multimodal request construction and API client
- `prompts/`: vendored released prompt
- `scoring/`: VLABench graph-matching wrapper and deterministic type normalization
- `scripts/`: command-line inference and scoring runners

Run the Python module with `--help` to inspect all endpoint, model, manifest, and output options:

```bash
python -m vlabench_official_eval.scripts.run_official_vlm_eval_api --help
```

Per-case files preserve the raw response, parsed sequence, scoring reference, and all reported metrics. The raw released-evaluator metrics and type-normalized derived metrics are stored separately.
