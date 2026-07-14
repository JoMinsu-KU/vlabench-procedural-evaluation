"""Native VLABench execution launcher for procedural evaluation runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from procedural_interface.core.io import read_json, write_json


def run_native_execution(command: dict, output_dir: str | Path, timeout_sec: int = 300) -> dict:
    """Run one native VLABench execution command and return a structured result."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command_path = write_json(out_dir / "native_command.json", command)
    result_path = out_dir / "native_result.json"
    worker_path = Path(__file__).with_name("vlabench_native_worker.py")
    python_executable = sys.executable
    argv = [python_executable, str(worker_path), "--command", str(command_path), "--result", str(result_path)]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(out_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_result = {
            "ok": False,
            "execution_completed": False,
            "native_timeout": True,
            "timeout_sec": timeout_sec,
            "worker_argv": argv,
            "stdout": stdout,
            "stderr": stderr,
            "gt_used_during_execution": False,
            "operation_sequence_read": False,
            "expert_sequence_called": False,
        }
        write_json(result_path, timeout_result)
        return timeout_result
    if completed.returncode != 0:
        return {
            "ok": False,
            "infrastructure_error": True,
            "worker_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    if not result_path.exists():
        return {
            "ok": False,
            "infrastructure_error": True,
            "worker_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "detail": "native worker did not create native_result.json",
        }
    result = read_json(result_path)
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    return result

