# Frozen-Output Cross-Scoring Protocol

## Scope

This protocol cross-scores the archived outputs without any additional model
generation.  Its primary implemented direction is:

```text
archived Full-Procedure output
    -> deterministic lossy symbolic projection
    -> symbolic endpoint tiers (v1.7):
         released type-normalized PM=100 (the v1.3 endpoint)
         strict normalized sequence equality
         strict equality after removing executor-only phase names
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

The local PM=100 implementation (`vlabench_pm100_match`, called
`symbolic_exact_match` in v1.3--v1.6) is accepted only if it exactly
reproduces every archived type-normalized symbolic success label across all
6,399 original symbolic outputs.  Any mismatch aborts the analysis.

## Symbolic endpoint tiers (v1.7)

The released Precise Matching rule is a reference-coverage test: it walks the
reference subtask patterns in order, skips predicted actions outside the
evaluator vocabulary, and never inspects predicted subtasks beyond the
reference count.  A prediction with a surplus leading, trailing, or
unrecognised action can therefore score 100, and a prediction with an
executor phase interposed inside a subtask pattern (`pick -> moveto -> place`)
scores below 100 even though its symbolic content is intact.  v1.7 keeps the
PM=100 endpoint as released and adds:

- `strict`: `strict_sequence_equal` -- the normalized prediction must equal
  the normalized reference in length, skill names, order, and every target
  key and value;
- `strict_executor_filtered`: `strict_sequence_equal` after removing the
  executor-only phase names `moveto`, `wait`, `open_gripper`, and
  `move_offset` from the projected side.  The list comes from the executor
  phase templates; the filter deletes only those names and never adds,
  reorders, relabels, or retargets an action.  Original symbolic outputs
  contain no executor phase names, so this tier reuses the strict original
  label.

Every tier is applied to the same archived outputs and the same frozen
projection.  Adding intermediate-phase deletions to the projection itself in
order to raise strict agreement is not permitted; any such change would be a
new projection protocol and would have to be reported separately.

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
