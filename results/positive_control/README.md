# Canonical Procedure Control

This directory contains a method-level viability control for the procedural interface. It is not included in model-performance totals.

For each of the 2,133 fixed cases, a canonical procedure and target identities were constructed offline from reference metadata and frozen as numeric steps. Runtime validation and execution consumed the frozen fixture. The adapter and lowerer did not use expert operation sequences or replace the frozen targets and order.

| Endpoint | Result |
|---|---:|
| Validation and binding | 2,133/2,133 |
| Native completion | 2,133/2,133 |
| First-attempt task-condition success | 2,062/2,133 (96.67%) |
| Success within 10 same-plan attempts | 2,133/2,133 (100.00%) |

The selected-attempt distribution was: attempt 1, 2,062 cases; attempt 2, 46; attempt 3, 17; attempt 4, 2; attempt 5, 3; attempt 6, 2; and attempt 8, 1.

`gt_procedural_positive_control_report.csv` contains one portable row per case. `summary.json` defines the endpoints, `verification_report.json` records integrity checks, and `provenance/fixtures.json` contains the frozen procedures. Full native traces are retained outside the Git repository.
