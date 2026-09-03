#!/usr/bin/env python3
"""Behavioural tests for the three symbolic endpoint tiers.

Run from the repository root:

    python -m unittest tests.test_symbolic_endpoints

The cases pin down the difference between the released VLABench Precise Matching
rule (ground-truth subtask coverage, reported as ``exact_match_score``) and strict
whole-sequence equality, and show that the executor-phase filter removes only
projection residue.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import cross_score_frozen_outputs as cs  # noqa: E402


def step(name: str, **params):
    return {"name": name, "params": params}


GT_PICK_PULL = [step("pick", target_entity_name=2), step("pull")]
GT_PICK_PLACE = [step("pick", target_entity_name=3), step("place", target_container_name=1)]


class EndpointTierCases(unittest.TestCase):
    def assert_tiers(self, prediction, ground_truth, pm100, strict, filtered):
        self.assertEqual(cs.vlabench_pm100_match(prediction, ground_truth), pm100, "pm100")
        self.assertEqual(cs.strict_sequence_equal(prediction, ground_truth), strict, "strict")
        self.assertEqual(
            cs.strict_sequence_equal_executor_filtered(prediction, ground_truth), filtered, "filtered"
        )

    def test_identical_sequence_passes_every_tier(self):
        self.assert_tiers(GT_PICK_PULL, GT_PICK_PULL, True, True, True)

    def test_digit_string_target_is_normalised_in_every_tier(self):
        prediction = [step("pick", target_entity_name=" 02 "), step("pull")]
        self.assert_tiers(prediction, GT_PICK_PULL, True, True, True)

    def test_surplus_leading_action_passes_pm100_but_fails_strict(self):
        # Archived example: Qwen3-VL-8B, M&T/select_drink/example3.
        prediction = [step("open_door"), step("pick", target_entity_name="2"), step("pull")]
        self.assert_tiers(prediction, GT_PICK_PULL, True, False, False)
        self.assertEqual(
            cs.strict_failure_reason(prediction, GT_PICK_PULL), "surplus_evaluator_vocabulary_action"
        )

    def test_surplus_trailing_action_passes_pm100_but_fails_strict(self):
        prediction = GT_PICK_PULL + [step("place", target_container_name=1)]
        self.assert_tiers(prediction, GT_PICK_PULL, True, False, False)

    def test_missing_action_fails_every_tier(self):
        self.assert_tiers([step("pick", target_entity_name=2)], GT_PICK_PULL, False, False, False)
        self.assertEqual(
            cs.strict_failure_reason([step("pick", target_entity_name=2)], GT_PICK_PULL), "missing_action"
        )

    def test_target_mismatch_fails_every_tier(self):
        prediction = [step("pick", target_entity_name=5), step("pull")]
        self.assert_tiers(prediction, GT_PICK_PULL, False, False, False)
        self.assertEqual(cs.strict_failure_reason(prediction, GT_PICK_PULL), "parameter_mismatch")

    def test_reordered_actions_fail_every_tier(self):
        prediction = [step("place", target_container_name=1), step("pick", target_entity_name=3)]
        self.assert_tiers(prediction, GT_PICK_PLACE, False, False, False)

    def test_trailing_executor_phase_passes_pm100_and_filtered_but_fails_strict(self):
        # A ``moveto`` after the matched subtask is skipped by the released matcher,
        # fails strict equality, and is the only thing the executor-phase filter removes.
        prediction = [step("pick", target_entity_name=2), step("pull"), step("moveto", target_entity_name=0)]
        self.assert_tiers(prediction, GT_PICK_PULL, True, False, True)
        self.assertEqual(
            cs.strict_failure_reason(prediction, GT_PICK_PULL), "executor_only_phase_residue:moveto"
        )

    def test_interposed_executor_phase_fails_pm100_but_passes_filtered(self):
        # The released matcher requires the subtask pattern to be contiguous, so an
        # executor phase *inside* ``pick -> pull`` breaks the match even though the
        # symbolic content is intact.  This is why the filtered tier is not nested in
        # the PM=100 tier (123 pooled projected rows in artifact v1.7).
        prediction = [step("pick", target_entity_name=2), step("moveto", target_entity_name=0), step("pull")]
        self.assert_tiers(prediction, GT_PICK_PULL, False, False, True)
        self.assertEqual(
            cs.strict_failure_reason(prediction, GT_PICK_PULL), "executor_only_phase_residue:moveto"
        )

    def test_filter_never_repairs_a_wrong_plan(self):
        wrong_target = [step("pick", target_entity_name=9), step("moveto"), step("pull")]
        self.assert_tiers(wrong_target, GT_PICK_PULL, False, False, False)
        swapped = [step("moveto"), step("pull"), step("pick", target_entity_name=2)]
        self.assert_tiers(swapped, GT_PICK_PULL, False, False, False)

    def test_unnormalizable_prediction_fails_every_tier(self):
        self.assert_tiers(None, GT_PICK_PULL, False, False, False)
        self.assert_tiers([{"name": 7}], GT_PICK_PULL, False, False, False)

    def test_legacy_name_still_points_at_pm100(self):
        self.assertIs(cs.symbolic_exact_match, cs.vlabench_pm100_match)


class ArchivedOutputCases(unittest.TestCase):
    """Checks against the archived outputs when they are present."""

    def setUp(self):
        self.path = cs.MODELS["Qwen3-VL-8B"]["symbolic"]
        if not self.path.exists():
            self.skipTest("archived Qwen symbolic outputs not present")

    def test_archived_select_drink_example3_is_pm100_pass_strict_fail(self):
        rows = {row["case_id"]: row for row in cs.read_jsonl(self.path)}
        row = rows["M&T/select_drink/example3"]
        self.assertEqual([s["name"] for s in row["skill_sequence"]], ["open_door", "pick", "pull"])
        self.assertTrue(row["type_normalized_symbolic_success"])
        self.assertTrue(cs.vlabench_pm100_match(row["skill_sequence"], row["gt_skill_sequence"]))
        self.assertFalse(cs.strict_sequence_equal(row["skill_sequence"], row["gt_skill_sequence"]))

    def test_archived_pm100_labels_are_reproduced_exactly(self):
        rows = cs.read_jsonl(self.path)
        self.assertEqual(cs.validate_archived_symbolic_endpoint(rows), 0)


if __name__ == "__main__":
    unittest.main()
