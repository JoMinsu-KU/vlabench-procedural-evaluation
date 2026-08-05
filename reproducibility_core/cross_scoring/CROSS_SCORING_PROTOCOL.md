# Frozen-Output Cross-Scoring Protocol

## Scope

This protocol cross-scores the archived outputs without any additional model
generation.  Its primary implemented direction is:

```text
archived Full-Procedure output
    -> deterministic lossy symbolic projection
    -> type-normalized symbolic exact-match endpoint
```

The protocol was written before inspecting the cross-score results.  All three
models contribute 2,133 archived Full-Procedure outputs and 2,133 independently
elicited symbolic outputs.

## Projection boundary

The projector may remove executor-level intermediate phases and serialize the
remaining information in the symbolic schema.  Task-aware macro projection is
used only where the source executor vocabulary and the symbolic vocabulary have
different abstraction levels.  In particular, insert-flower `moveto(target)`
phases are represented as the symbolic `insert(target)` macro.  This mapping is
lossy and source-derived; it is not evidence that the model literally emitted
the word `insert`.

The transformation must not:

- replace a model-selected target;
- reorder emitted actions;
- insert a missing model-owned target;
- substitute a reference target or reference action sequence;
- repair an invalid parsed output.

Malformed or parser-rejected Full-Procedure outputs are not reparsed by the
cross-scorer.  They remain unsuccessful under the projected endpoint.  The
analysis reports projection coverage separately from projected success.

## Endpoint reproduction

The local binary exact-match implementation is accepted only if it exactly
reproduces every archived type-normalized symbolic success label across all
6,399 original symbolic outputs.  Any mismatch aborts the analysis.

## Reverse direction

The released symbolic branch provides a scorer, not a verified executor entry
point for arbitrary model-generated symbolic sequences.  The procedural runner
requires an already expanded procedure.  Therefore Symbolic-to-Native execution
is not reported unless an existing path can be shown to preserve the model's
skill, target, and order without reading or injecting reference decisions.
Building a new macro-to-procedure adapter would be a new execution experiment,
not frozen-output cross-scoring under the present protocol.

## Interpretation

The original comparison uses independently elicited outputs.  The projection
audit partially separates high-level plan content from downstream
procedure-contract termination, but it remains a deterministic, lossy
transformation.  It is not a causal intervention and does not establish that
the symbolic or procedural endpoint is universally superior.
