"""Portable project location for the unchanged diagnostic dispatcher."""
import os
from pathlib import Path
PROJECT=Path(os.environ.get("VLABENCH_PROJECT_ROOT",str(Path(__file__).resolve().parents[3])))
def project_steps(executable):
    return [{"skill":step["skill"], **({"target":step["target"]["number"]}
        if isinstance(step.get("target"),dict) else {})} for step in executable]
