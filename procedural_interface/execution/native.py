"""Native VLABench execution launcher for Gate3 and later procedural-evaluation runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from procedural_interface.core.io import read_json, write_json


def run_native_execution(command: dict, output_dir: str | Path, timeout_sec: int = 300) -> dict:
    """Run one native VLABench execution command and return a structured result."""

    # Normalize the caller-provided output directory once.
    out_dir = Path(output_dir)
    # Create the execution directory before writing command/result artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact native command used for audit and reproducibility.
    command_path = write_json(out_dir / "native_command.json", command)
    # Define the result path that the worker must create.
    result_path = out_dir / "native_result.json"
    # Locate the worker next to this launcher so imports stay package-local.
    worker_path = Path(__file__).with_name("vlabench_native_worker.py")
    # Use the current Python interpreter because the conda launcher launches inside WSL conda.
    python_executable = sys.executable
    # Build an argv list instead of a shell string to avoid quoting path bugs.
    argv = [python_executable, str(worker_path), "--command", str(command_path), "--result", str(result_path)]
    # Launch the worker as a child process so timeout and stdout/stderr are captured.
    try:
        # Keep the simulator in a child process so a hung native step cannot hang the controller forever.
        completed = subprocess.run(
            argv,
            cwd=str(out_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    # Convert native execution timeout into a structured evaluation finding instead of aborting the whole gate.
    except subprocess.TimeoutExpired as exc:
        # Preserve any partial worker output because MuJoCo warnings can explain timeout causes.
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        # Preserve any partial worker stderr for low-level diagnostics.
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        # Write a native_result artifact so every fixture still has a trace directory.
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
        # Persist the timeout result at the same path a normal worker would have used.
        write_json(result_path, timeout_result)
        # Return the structured timeout to the evaluation controller.
        return timeout_result
    # Convert a non-zero worker process exit into an infrastructure-style result.
    if completed.returncode != 0:
        # Return stdout/stderr because simulator import failures often appear there.
        return {
            "ok": False,
            "infrastructure_error": True,
            "worker_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    # Treat a missing result file as infrastructure failure, not model failure.
    if not result_path.exists():
        # Preserve worker output to make the missing file debuggable.
        return {
            "ok": False,
            "infrastructure_error": True,
            "worker_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "detail": "native worker did not create native_result.json",
        }
    # Read the worker-produced structured result.
    result = read_json(result_path)
    # Attach stdout for low-level simulator diagnostics.
    result["stdout"] = completed.stdout
    # Attach stderr for low-level simulator diagnostics.
    result["stderr"] = completed.stderr
    # Return the complete result to the evaluation controller.
    return result

