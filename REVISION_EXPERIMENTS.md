# Additional evaluation control

The original 2,133-case symbolic, full-procedure, scaffold, and frozen-output results remain unchanged. Their artifact-v1.7 release and checksummed files are retained as historical evidence.

The [paired schema-decoding control](revision_experiments/schema_decoding_20261021/README.md) uses a fixed, task-stratified 322-case subset and 1,932 fresh outputs. It compares JSON object and JSON Schema response formats in a common new BF16 serving environment, reusing the original Full Reference-Procedure prompt and input images. It is not pooled with the historical outputs or treated as a before/after rerun of the original inference environment.

The control separates schema compliance, reference-phase conformance, target/context compatibility, native completion, and final-condition success. Scripts, raw model responses, case-level evaluations, and GPU-free statistics reproduction are under the linked directory. Larger input/native-trace evidence is a separate release asset; no manuscript or Overleaf project is included.

## Contract and instruction-goal diagnostics

- [Fixed relaxed-contract diagnostic](revision_experiments/contract_relaxation_20261022/README.md)
- [45-case exclusion sensitivity](revision_experiments/exclusion45_sensitivity_20261022/README.md)

These additive modules do not change the original 2,133-case labels, projection rules, backend, schema-control outputs, or previous release tags. Both analysis commands are GPU-free.
