"""Setup-controller tests without installation, model downloads, or GPU use."""

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jsonschema
import gcp_bootstrap as gb


def fixture_bundle(root):
    import schema_decoding as sd

    bundle = Path(root) / "bundle"
    bundle.mkdir()
    shutil.copyfile(Path(sd.__file__), bundle / "schema_decoding.py")
    shutil.copyfile(sd.HERE / "procedural_plan_schema.json", bundle / "procedural_plan_schema.json")
    entries = []
    for index in (1, 2):
        case_id = f"Test/task/example{index}"
        directory = bundle / "inputs" / sd.safe_case_id(case_id)
        directory.mkdir(parents=True)
        for name in ("input.png", "input_mask.png", "instruction.txt", "prompt.txt"):
            (directory / name).write_text("TEST-ONLY", encoding="utf-8")
        entries.append({"case_id": case_id, "input_dir": str(directory.relative_to(bundle)),
                        "sha256": {name: sd.digest(directory / name) for name in ("input.png", "input_mask.png", "instruction.txt", "prompt.txt")}})
    sd.write_json(bundle / "bundle_manifest.json", {"cases": entries, "case_count": 2,
        "schema_sha256": sd.digest(bundle / "procedural_plan_schema.json"), "runner_sha256": sd.digest(bundle / "schema_decoding.py")})
    return bundle


def options(root, bundle, action="ready"):
    return gb.parser().parse_args(["--work-root", str(Path(root) / "work"), "--bundle", str(bundle), "--action", action])


class FakeBootstrap(gb.Bootstrap):
    def __init__(self, args, fail_at=None):
        super().__init__(args)
        self.events = []
        self.fail_at = fail_at

    def event(self, value):
        self.events.append(value)
        if self.fail_at == value:
            raise RuntimeError("TEST-ONLY injected setup failure")

    def setup_system(self):
        self.event("system")

    def setup_driver(self):
        self.event("driver")

    def setup_environment(self):
        self.event("environment")
        (self.root / "environment.freeze.txt").write_text("TEST-ONLY", encoding="utf-8")

    def call_worker(self, stage, model=None, output=None):
        self.event(stage + ("_" + model if model else ""))
        result = {"test_only": True, "passed": True}
        if output:
            gb.write_json(output, result)
        return result

    def stable_metadata(self, model):
        self.event("metadata_" + model)

    @contextmanager
    def server(self, model):
        self.event("start_" + model)
        try:
            yield
        finally:
            self.event("stop_" + model)

    def inference(self, model, smoke):
        self.event(("smoke_infer_" if smoke else "full_infer_") + model)

    def capability_probe(self, model):
        self.event("probe_" + model)


class ConfigurationTests(unittest.TestCase):
    def test_all_models_have_pinned_hf_revisions(self):
        self.assertEqual(gb.selected_models(["all"]), ["qwen", "gemma", "internvl"])
        for model in gb.MODELS.values():
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")

    def test_invalid_or_duplicate_models_rejected(self):
        for values in (["qwen", "qwen"], ["other"], [], ["all", "qwen"]):
            with self.assertRaises(ValueError):
                gb.selected_models(values)

    def test_serve_command_is_bf16_loopback_and_pinned(self):
        for model in gb.MODELS:
            argv = gb.serving_argv(Path("/isolated/venv"), model, 8000)
            self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
            self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")
            for flag in ("--revision", "--tokenizer-revision", "--code-revision"):
                self.assertEqual(argv[argv.index(flag) + 1], gb.MODELS[model]["revision"])
            self.assertEqual(argv[argv.index("--max-model-len") + 1], "16384")
            self.assertEqual(argv[argv.index("--generation-config") + 1], "vllm")
            self.assertTrue(json.loads(argv[argv.index("--structured-outputs-config") + 1])["disable_any_whitespace"])
            self.assertNotIn("--device", argv)
            self.assertNotIn("--quantization", argv)

    def test_runtime_env_is_isolated_without_changing_parent_env(self):
        with mock.patch.dict(gb.os.environ, {"PYTHONPATH": "/old", "TRANSFORMERS_CACHE": "/old", "HF_TOKEN": "TEST-SECRET"}):
            env = gb.runtime_env(Path("/dedicated"), Path("/dedicated/venv"))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("TRANSFORMERS_CACHE", env)
            self.assertEqual(env["HF_HOME"], "/dedicated/cache/huggingface")
            self.assertEqual(env["HF_TOKEN"], "TEST-SECRET")
            self.assertEqual(gb.os.environ["PYTHONPATH"], "/old")

    def test_plan_has_no_install_network_or_file_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.Popen", side_effect=AssertionError("No install")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("No network")), redirect_stdout(io.StringIO()) as output:
            root = Path(tmp) / "not-created"
            self.assertEqual(gb.main(["--plan", "--work-root", str(root)]), 0)
            self.assertFalse(root.exists())
            self.assertFalse(json.loads(output.getvalue())["changes_made"])

    def test_pinned_requirements_match_controller(self):
        content = (gb.HERE / "gcp_requirements.txt").read_text()
        self.assertIn("vllm @ " + gb.VLLM_WHEEL, content)
        self.assertIn("+cu129", gb.VLLM_VERSION)
        self.assertIn("#sha256=", gb.VLLM_WHEEL)
        self.assertIn("transformers==4.57.6", content)
        self.assertIn("huggingface-hub==0.36.2", content)

    def test_resolved_lock_contains_every_top_level_pin(self):
        pinned = {line for line in (gb.HERE / "gcp_requirements.txt").read_text().splitlines() if line and not line.startswith("#")}
        resolved = set((gb.HERE / "gcp_dependencies.lock.txt").read_text().splitlines())
        self.assertTrue(pinned <= resolved)
        self.assertIn("torch==2.11.0+cu129", resolved)


class SafetyTests(unittest.TestCase):
    def test_port_probe_enables_time_wait_reuse(self):
        with mock.patch.object(gb.socket, "socket") as factory:
            probe = factory.return_value.__enter__.return_value
            gb.ensure_free_port(8000)
            probe.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def test_wsl_cannot_install(self):
        with mock.patch.object(gb.platform, "system", return_value="Linux"), \
             mock.patch.object(gb.platform, "machine", return_value="x86_64"), \
             mock.patch.object(gb.platform, "release", return_value="6.6-microsoft-standard-WSL2"), \
             mock.patch("urllib.request.build_opener", side_effect=AssertionError("No metadata request on WSL")):
            with self.assertRaisesRegex(RuntimeError, "WSL"):
                gb.verify_gcp_host()

    def test_failed_host_guard_does_not_initialize_work_directory(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gb, "verify_gcp_host", side_effect=RuntimeError("not GCP")):
            root = Path(tmp) / "work"
            with self.assertRaises(RuntimeError):
                gb.main(["--work-root", str(root)])
            self.assertFalse(root.exists())

    def test_busy_port_is_not_killed(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                gb.ensure_free_port(listener.getsockname()[1])
            self.assertNotEqual(listener.fileno(), -1)

    def test_existing_gpu_processes_are_not_killed(self):
        completed = SimpleNamespace(stdout="1234\n5678\n")
        with mock.patch("subprocess.run", return_value=completed), mock.patch("os.killpg", side_effect=AssertionError("No kill")):
            with self.assertRaisesRegex(RuntimeError, "1234"):
                gb.ensure_idle_gpu({})

    def test_model_switch_waits_for_gpu_release_without_killing_other_jobs(self):
        with mock.patch.object(gb, "ensure_idle_gpu", side_effect=[RuntimeError("busy"), None]) as check, \
             mock.patch.object(gb.time, "sleep"), mock.patch("os.killpg", side_effect=AssertionError("No kill")):
            gb.wait_idle_gpu({}, timeout=60)
        self.assertEqual(check.call_count, 2)

    def test_owned_process_group_is_cleaned(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            gb.stop_owned_server(process, timeout=5)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_crashed_parent_still_signals_only_its_own_group(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = 0
        with mock.patch("os.killpg") as kill:
            gb.stop_owned_server(process)
        kill.assert_called_once_with(12345, gb.signal.SIGTERM)

    def test_work_directory_with_foreign_data_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(tmp)
            args = options(tmp, bundle)
            args.work_root.mkdir()
            sentinel = args.work_root / "old_data.txt"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "unowned"):
                gb.Bootstrap(args).initialize()
            self.assertEqual(sentinel.read_text(), "keep")

    def test_changed_requirements_or_bundle_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(tmp)
            boot = gb.Bootstrap(options(tmp, bundle))
            boot.initialize()
            marker = boot.root / "bootstrap_project.json"
            data = gb.read_json(marker)
            data["requirements_sha256"] = "changed"
            gb.write_json(marker, data)
            with self.assertRaisesRegex(RuntimeError, "different setup"):
                boot.initialize()

    def test_resume_disk_requirement_does_not_require_another_120_gib(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertGreaterEqual(gb.required_free_gib(root, ["qwen"], 120), 120)
            (root / "environment.freeze.txt").write_text("locked")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.safetensors").write_bytes(b"0" * 1024)
            gb.write_json(root / "models/qwen.json", {"snapshot_path": str(snapshot)})
            with mock.patch.dict(gb.MODELS, {"qwen": {"weight_bytes": 1024}}):
                self.assertEqual(gb.required_free_gib(root, ["qwen"], 120), 10)

    def test_export_excludes_tokens_models_and_environments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("cache/huggingface/token", "venv-vllm/bin/python", "outputs/qwen/result.json", "serving/qwen.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("TEST-ONLY")
            report = gb.export_artifacts(root, "test")
            with tarfile.open(report["archive"]) as tar:
                names = tar.getnames()
            self.assertIn("outputs/qwen/result.json", names)
            self.assertNotIn("cache/huggingface/token", names)
            self.assertFalse(any("venv-vllm" in name for name in names))


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.schema = gb.read_json(gb.HERE / "procedural_plan_schema.json")
        self.rows = [(None, {"condition": condition, "transport_ok": True,
                     "origin_output": '{"steps":[{"skill":"pick","target":999}]}'} )
                     for condition in ("json_object", "json_schema") for _ in range(2)]

    def test_smoke_checks_format_not_correct_target_or_phase(self):
        result = gb.verify_smoke_rows(self.rows, self.schema, jsonschema.validate)
        self.assertTrue(result["passed"])
        self.assertFalse(result["semantic_plan_correctness_required"])

    def test_missing_case_refuses_readiness(self):
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            gb.verify_smoke_rows(self.rows[:-1], self.schema, jsonschema.validate)

    def test_transport_failure_refuses_readiness(self):
        self.rows[0][1]["transport_ok"] = False
        with self.assertRaisesRegex(RuntimeError, "transport"):
            gb.verify_smoke_rows(self.rows, self.schema, jsonschema.validate)

    def test_invalid_json_schema_is_not_repaired(self):
        self.rows[-1][1]["origin_output"] = '{"steps":[{"skill":"pick","target":"2"}]}'
        with self.assertRaises(jsonschema.ValidationError):
            gb.verify_smoke_rows(self.rows, self.schema, jsonschema.validate)

    def test_invalid_baseline_output_is_retained_as_model_behavior(self):
        self.rows[0][1]["origin_output"] = "bad model JSON output"
        self.assertTrue(gb.verify_smoke_rows(self.rows, self.schema, jsonschema.validate)["passed"])

    def test_truncation_is_not_repaired_or_mistaken_for_an_infrastructure_failure(self):
        self.rows[-1][1].update(origin_output='{"steps":[', finish_reason="length")
        result = gb.verify_smoke_rows(self.rows, self.schema, jsonschema.validate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["schema_truncated"], 1)
        self.assertFalse(result["raw_schema_enforcement_verified"])


class ControllerTests(unittest.TestCase):
    @contextmanager
    def simulated_host(self):
        with mock.patch.object(gb, "verify_gcp_host", return_value={"TEST_ONLY": True}), \
             mock.patch("shutil.disk_usage", return_value=SimpleNamespace(free=500 * 1024**3)), \
             redirect_stdout(io.StringIO()):
            yield

    def test_ready_does_not_start_full_inference(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulated_host():
            boot = FakeBootstrap(options(tmp, fixture_bundle(tmp)))
            self.assertEqual(boot.run(), 0)
            self.assertFalse(any(event.startswith("full_infer") for event in boot.events))
            self.assertEqual(gb.read_json(boot.root / "status.json")["status"], "ready")
            self.assertEqual(boot.events.count("stop_qwen"), 1)

    def test_full_run_starts_only_after_all_three_models_pass(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulated_host():
            boot = FakeBootstrap(options(tmp, fixture_bundle(tmp), "run"))
            boot.run()
            first_full = boot.events.index("full_infer_qwen")
            self.assertLess(boot.events.index("smoke_internvl"), first_full)
            self.assertEqual(sum(event.startswith("full_infer_") for event in boot.events), 3)
            self.assertEqual(gb.read_json(boot.root / "status.json")["status"], "inference_completed")

    def test_failed_smoke_blocks_every_full_run_and_cleans_server(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulated_host():
            boot = FakeBootstrap(options(tmp, fixture_bundle(tmp), "run"), fail_at="smoke_infer_gemma")
            with self.assertRaises(RuntimeError):
                boot.run()
            self.assertFalse((boot.root / "READY.json").exists())
            self.assertFalse(any(event.startswith("full_infer_") for event in boot.events))
            self.assertIn("stop_gemma", boot.events)
            self.assertEqual(gb.read_json(boot.root / "status.json")["status"], "failed")

    def test_access_failure_prevents_model_weight_downloads(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulated_host():
            boot = FakeBootstrap(options(tmp, fixture_bundle(tmp)), fail_at="authorize")
            with self.assertRaises(RuntimeError):
                boot.run()
            self.assertFalse(any(event.startswith("download_") for event in boot.events))

    def test_reboot_requirement_blocks_environment_and_models(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulated_host():
            boot = FakeBootstrap(options(tmp, fixture_bundle(tmp)))
            with mock.patch.object(boot, "setup_driver", side_effect=gb.RebootRequired("TEST reboot")):
                with self.assertRaises(gb.RebootRequired):
                    boot.run()
            self.assertNotIn("environment", boot.events)
            self.assertEqual(gb.read_json(boot.root / "status.json")["status"], "reboot_required")


if __name__ == "__main__":
    unittest.main(verbosity=2)
