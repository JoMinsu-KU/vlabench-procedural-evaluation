"""Public-view VLABench dataset loader for VLABench procedural evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from procedural_interface.core.io import read_json


@dataclass(frozen=True)
class SampleRef:
    """One public-view VLABench sample."""

    category: str
    task_name: str
    example_id: str
    root: Path

    @property
    def case_id(self) -> str:
        """Return stable slash-form case id."""

        return f"{self.category}/{self.task_name}/{self.example_id}"

    @property
    def env_config_path(self) -> Path:
        """Return the internal scene config path."""

        return self.root / "env_config.json"

    def load_env_config(self) -> dict:
        """Load the sample env_config strictly."""

        return read_json(self.env_config_path)


def iter_samples(dataset_root: str | Path) -> Iterator[SampleRef]:
    """Yield all samples from a the procedural evaluation project public-view dataset root."""

    root = Path(dataset_root)
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            for example_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                if (example_dir / "env_config.json").exists():
                    yield SampleRef(category_dir.name, task_dir.name, example_dir.name, example_dir)


def task_inventory(dataset_root: str | Path) -> list[dict]:
    """Summarize public-view tasks and example counts."""

    counts: dict[tuple[str, str], int] = {}
    for sample in iter_samples(dataset_root):
        counts[(sample.category, sample.task_name)] = counts.get((sample.category, sample.task_name), 0) + 1
    return [
        {"category": category, "task_name": task_name, "example_count": count}
        for (category, task_name), count in sorted(counts.items())
    ]
