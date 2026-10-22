# Publication-only changes

The generated responses, case-level evaluations, prompts, schema, model runner, inference settings, and original statistics are copied byte-for-byte. The export inventory records their original hashes.

The publication copy of `test_schema_decoding.py` reads project and public-input paths from environment variables instead of workstation-specific constants. The PowerShell wrapper requires explicit project, dataset, and interpreter paths and accepts the WSL distribution/user as arguments. Their unmodified execution-time sources remain in `execution_source/`. These path changes do not modify experimental outputs or evaluator behavior.

`reproduce_revision.py` and `reproduce_followup.py` are additional analysis-only entry points. They do not call a model or simulator. `frozen_analysis_rows.jsonl` is a compact join of unchanged generation and evaluation records, linked by hashes. Original raw records remain available for inspection.

The nested `.gitattributes` disables text newline conversion so that Git checkout does not invalidate frozen-record hashes.
