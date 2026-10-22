"""Check the retention and non-monotonic scaffold facts used in the revision."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "review_followup/validation_v1/physics_1200_replay_trace_audit.csv"
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    assert len(rows) == 1200 and len(cases) == 600
    retention = Counter()
    for pair in cases.values():
        assert len(pair) == 2 and {r["repeat"] for r in pair} == {"1", "2"}
        successful = sum(r["condition_success"] == "True" for r in pair)
        if all(r["execution_status"] == "exception" for r in pair):
            assert all("reflectance" in r["error"] for r in pair)
            retention["both_initialization_exception"] += 1
        else:
            assert all(r["execution_status"] == "completed" for r in pair)
            retention[{0: "neither_success", 1: "one_success", 2: "both_success"}[successful]] += 1
    assert dict(retention) == {"both_success": 42, "one_success": 162,
                               "neither_success": 296, "both_initialization_exception": 100}
    source = HERE / "review_followup/verified_v2/scaffold_nonmonotonicity.csv"
    with source.open(encoding="utf-8-sig") as stream:
        scaffolds = {(r["model"], r["condition"]): r for r in csv.DictReader(stream)}
    qwen = json.loads(scaffolds["qwen", "target_oracle"]["lost_failure_subtypes"])
    gemma = json.loads(scaffolds["gemma", "phase_oracle"]["lost_failure_subtypes"])
    assert qwen == {"extra_phase": 30}
    assert gemma == {"incompatible_target_affordance": 34, "target_out_of_range": 2, "C1": 3}
    print(json.dumps({"physics_cases": dict(retention), "qwen_target_scaffold_lost": qwen,
                      "gemma_phase_scaffold_lost": gemma}, indent=2))


if __name__ == "__main__":
    main()
