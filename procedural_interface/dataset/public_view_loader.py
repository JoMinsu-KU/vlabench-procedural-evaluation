"""Public-view VLABench dataset loader for VLABench procedural evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from procedural_interface.core.io import read_json


@dataclass(frozen=True)
class SampleRef:
    """One public-view VLABench sample."""

    # High-level VLABench category directory.
    category: str
    # Public task directory name.
    task_name: str
    # Example directory name such as example0.
    example_id: str
    # Absolute path to the example directory.
    root: Path

    @property
    def case_id(self) -> str:
        """Return stable slash-form case id."""

        # The case id is used in manifests and trace directories.
        return f"{self.category}/{self.task_name}/{self.example_id}"

    @property
    def env_config_path(self) -> Path:
        """Return the internal scene config path."""

        # Public view intentionally keeps env_config for internal binding/execution only.
        return self.root / "env_config.json"

    def load_env_config(self) -> dict:
        """Load the sample env_config strictly."""

        # The caller decides which fields are allowed for its phase.
        return read_json(self.env_config_path)


def iter_samples(dataset_root: str | Path) -> Iterator[SampleRef]:
    """Yield all samples from a legacy public-view dataset root."""

    # Normalize the root path once for deterministic child paths.
    root = Path(dataset_root)
    # Iterate category directories in lexical order for reproducible manifests.
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        # Iterate task directories in lexical order for reproducible inventory.
        for task_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            # Iterate examples in numeric-like lexical order; VLABench names are exampleN.
            for example_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                # Skip malformed rows early; Gate0 reports completeness separately.
                if (example_dir / "env_config.json").exists():
                    # Yield a lightweight reference instead of loading all JSON eagerly.
                    yield SampleRef(category_dir.name, task_dir.name, example_dir.name, example_dir)


def task_inventory(dataset_root: str | Path) -> list[dict]:
    """Summarize public-view tasks and example counts."""

    # Use a dictionary keyed by category/task to aggregate examples.
    counts: dict[tuple[str, str], int] = {}
    # Count samples without reading env_config bodies.
    for sample in iter_samples(dataset_root):
        # Increment the stable category/task bucket.
        counts[(sample.category, sample.task_name)] = counts.get((sample.category, sample.task_name), 0) + 1
    # Convert buckets to stable audit rows.
    return [
        {"category": category, "task_name": task_name, "example_count": count}
        for (category, task_name), count in sorted(counts.items())
    ]
