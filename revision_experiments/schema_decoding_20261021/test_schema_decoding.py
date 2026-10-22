"""Remote regression checks; mock API results are never experimental results."""

import argparse
import base64
import builtins
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest import mock

import jsonschema
import schema_decoding as sd


PROJECT = Path(os.environ.get("VLABENCH_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
PUBLIC = Path(os.environ.get("VLABENCH_PUBLIC_ROOT", str(PROJECT / "datasets/public_view")))
CASE = "CommenSence/select_chemistry_tube_common_sense/example4"


def make_bundle(root):
    source = Path(tempfile.mkdtemp(dir=root))
    inputs = source / "inputs" / sd.safe_case_id(CASE)
    inputs.mkdir(parents=True)
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nm1cAAAAASUVORK5CYII=")
    (inputs / "input.png").write_bytes(png)
    (inputs / "input_mask.png").write_bytes(png)
    (inputs / "instruction.txt").write_text("Select the chemistry tube", encoding="utf-8")
    (inputs / "prompt.txt").write_text("TEST TRANSPORT ONLY. Output a JSON object with steps.", encoding="utf-8")
    schema = sd.read_json(sd.HERE / "procedural_plan_schema.json")
    sd.write_json(source / "procedural_plan_schema.json", schema)
    (source / "schema_decoding.py").write_bytes(Path(sd.__file__).read_bytes())
    entry = {"case_id": CASE, "input_dir": "inputs/" + sd.safe_case_id(CASE),
             "sha256": {name: sd.digest(inputs / name) for name in (*sd.INPUT_NAMES, "prompt.txt")}}
    sd.write_json(source / "bundle_manifest.json", {"case_count": 1, "cases": [entry],
                  "schema_sha256": sd.digest(source / "procedural_plan_schema.json"),
                  "runner_sha256": sd.digest(source / "schema_decoding.py")})
    return source, entry


@contextmanager
def mock_server(mode="valid"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "TEST/transport-only"}]}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if (mode == "unsupported" or (mode == "unsupported_once" and len(requests) == 2)) and payload["response_format"]["type"] == "json_schema":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"TEST unsupported schema"}')
                return
            self.send_response(200)
            self.end_headers()
            content = {"skill": "pick", "target": 1} if payload["response_format"]["type"] == "json_schema" else {"skill": "pick"}
            self.wfile.write(json.dumps({"model": "TEST/transport-only", "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps({"steps": [content]})}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 12}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = sd.read_json(sd.HERE / "procedural_plan_schema.json")
        jsonschema.Draft202012Validator.check_schema(cls.schema)

    def test_all_eleven_skills_follow_target_policy(self):
        target = ["pick", "moveto", "place", "press", "push"]
        no_target = ["lift", "pull", "pour", "wait", "move_offset", "open_gripper"]
        for skill in target:
            with self.subTest(skill=skill):
                jsonschema.validate({"steps": [{"skill": skill, "target": 7}]}, self.schema)
                for wrong in ({"skill": skill}, {"skill": skill, "target": "7"}, {"skill": skill, "target": None}, {"skill": skill, "target": True}):
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate({"steps": [wrong]}, self.schema)
        for skill in no_target:
            jsonschema.validate({"steps": [{"skill": skill}]}, self.schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({"steps": [{"skill": skill, "target": None}]}, self.schema)

    def test_schema_does_not_include_task_answers(self):
        for steps in ([{"skill": "lift"}], [{"skill": "place", "target": -999}, {"skill": "pick", "target": 999999}],
                      [{"skill": "wait"}] * 17):
            jsonschema.validate({"steps": steps}, self.schema)
        encoded = json.dumps(self.schema)
        for forbidden in ("prefixItems", "maxItems", "task_name", "phase", "role", "const"):
            self.assertNotIn(forbidden, encoded)

    def test_unknown_fields_and_empty_plan_rejected(self):
        for value in ({"steps": []}, {"steps": [{"skill": "teleport"}]},
                      {"steps": [{"skill": "lift", "role": "actor"}]}, {"steps": [{"skill": "lift"}], "answer": 1}):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(value, self.schema)

    def test_only_response_format_differs(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "same"}]}]
        a = sd.build_payload(messages, "model", "json_object", self.schema)
        b = sd.build_payload(messages, "model", "json_schema", self.schema)
        self.assertNotEqual(a.pop("response_format"), b.pop("response_format"))
        self.assertEqual(a, b)


class BundleTests(unittest.TestCase):
    def test_selection_is_reproducible_and_model_independent(self):
        ids = [f"Category/task{task}/example{i}" for task in range(3) for i in range(1, 14)] + ["Category/small/example1"]
        selected = sd.select_cases(ids, 10, 20261021)
        self.assertEqual(selected, sd.select_cases(ids, 10, 20261021))
        self.assertEqual(len(selected), 31)
        self.assertIn("Category/small/example1", selected)
        self.assertEqual(selected, [item for item in ids if item in set(selected)])

    def test_bom_duplicate_and_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.txt"
            path.write_text(CASE + "\n", encoding="utf-8-sig")
            self.assertEqual(sd.load_case_ids(path), [CASE])
            path.write_text(CASE + "\n" + CASE, encoding="utf-8")
            with self.assertRaises(ValueError):
                sd.load_case_ids(path)
        for case_id in ("../task/example1", "Category/task/../../evil", "Category/task/examplex"):
            with self.assertRaises(ValueError):
                sd.safe_case_id(case_id)

    def test_bundle_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, entry = make_bundle(tmp)
            sd.load_bundle(bundle)
            (bundle / entry["input_dir"] / "instruction.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                sd.load_bundle(bundle)

    def test_overwrite_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sd.initialize_run(root, {"a": 1}, False)
            with self.assertRaises(FileExistsError):
                sd.initialize_run(root, {"a": 1}, False)
            with self.assertRaises(ValueError):
                sd.initialize_run(root, {"a": 2}, True)
            sd.initialize_run(root, {"a": 1}, True)
        with self.assertRaises(ValueError):
            sd.require_separate("/input/output", "/input")

    def test_concurrent_writer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with sd.run_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with sd.run_lock(tmp):
                        pass


class TransportTests(unittest.TestCase):
    def run_inference(self, bundle, root, url, resume=False, retry=False):
        config = root.parent / "test_serving.json"
        sd.write_json(config, {"model_revision": "test-only", "vllm_version": "mock-not-vllm", "dtype": "bfloat16",
                               "server_command": "TEST MOCK", "processor_revision": "test-only", "chat_template": "mock", "image_preprocessing": "test-only"})
        argv = ["infer", "--bundle", str(bundle), "--output-root", str(root), "--model", "TEST/transport-only",
                "--serving-config", str(config), "--base-url", url]
        return sd.main(argv + (["--resume"] if resume else []) + (["--retry-api-errors"] if retry else []))

    def test_http_pair_raw_evidence_resume_and_frozen_replay_validation(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server() as (url, requests):
            bundle, entry = make_bundle(tmp)
            root = Path(tmp) / "generation"
            with mock.patch.object(sd, "import_legacy", side_effect=AssertionError("Cloud inference must not import VLABench")):
                self.assertEqual(self.run_inference(bundle, root, url), 0)
            self.assertEqual(len(requests), 2)
            for request in requests:
                self.assertTrue(request["messages"][0]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,"))
            a, b = [dict(item) for item in requests]
            a.pop("response_format")
            b.pop("response_format")
            self.assertEqual(a, b)
            config, rows = sd.load_frozen_generation(root, bundle)
            self.assertEqual(len(rows), 2)
            legacy = sd.import_legacy(PROJECT)
            self.assertFalse(sd.evaluate_content(legacy, rows[0][1]["origin_output"])["model_parse_ok"])
            self.assertTrue(sd.evaluate_content(legacy, rows[1][1]["origin_output"])["model_parse_ok"])
            self.run_inference(bundle, root, url, resume=True)
            self.assertEqual(len(requests), 2)
            with self.assertRaises(FileExistsError):
                self.run_inference(bundle, root, url)
            body = root / rows[0][1]["attempt_history"][-1] / "response_body.bin"
            body.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                sd.load_frozen_generation(root, bundle)

    def test_unsupported_schema_is_recorded_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server("unsupported") as (url, requests):
            bundle, _ = make_bundle(tmp)
            root = Path(tmp) / "generation"
            with self.assertRaises(RuntimeError):
                self.run_inference(bundle, root, url)
            self.assertEqual(len(requests), 2)
            path = root / "json_schema" / "case_outputs" / sd.safe_case_id(CASE) / "row.json"
            row = sd.read_json(path)
            self.assertEqual(row["http_status"], 400)
            self.assertFalse(row["transport_ok"])
            self.assertIn("unsupported schema", (root / row["attempt_history"][-1] / "response_body.bin").read_text())
            self.assertFalse((root / "generation_complete.json").exists())

    def test_explicit_transport_retry_retains_all_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server("unsupported_once") as (url, requests):
            bundle, _ = make_bundle(tmp)
            root = Path(tmp) / "generation"
            with self.assertRaises(RuntimeError):
                self.run_inference(bundle, root, url)
            with self.assertRaises(RuntimeError):
                self.run_inference(bundle, root, url, resume=True)
            self.assertEqual(len(requests), 2)
            self.run_inference(bundle, root, url, resume=True, retry=True)
            self.assertEqual(len(requests), 3)
            _, rows = sd.load_frozen_generation(root, bundle)
            schema_row = next(row for _, row in rows if row["condition"] == "json_schema")
            self.assertEqual(len(schema_row["attempt_history"]), 2)
            first_body = root / schema_row["attempt_history"][0] / "response_body.bin"
            self.assertIn("unsupported schema", first_body.read_text())
            self.assertTrue(schema_row["transport_ok"])


class LegacyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = sd.import_legacy(PROJECT)

    def test_numeric_cast_does_not_fill_missing_steps_or_targets(self):
        value = '{"steps":[{"skill":"pick","target":"2"},{"skill":"lift"}]}'
        result = sd.evaluate_content(self.legacy, value)
        self.assertFalse(result["strict_parse_ok"])
        self.assertTrue(result["model_parse_ok"])
        self.assertEqual(result["parsed_plan"]["steps"], [{"skill": "pick", "target": 2}, {"skill": "lift"}])
        missing = sd.evaluate_content(self.legacy, '{"steps":[{"skill":"pick"}]}')
        self.assertEqual(missing["parsed_plan"]["status"]["subtype"], "missing_target")
        self.assertFalse(missing["semantic_repair_applied"])

    def test_offline_parse_failure_cannot_call_api_or_native(self):
        source = {"case_id": CASE, "condition": "json_object", "transport_ok": True,
                  "origin_output": '{"steps":[{"skill":"pick"}]}'}
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", side_effect=AssertionError("No API")), \
             mock.patch.object(self.legacy, "execute_model_fixture", side_effect=AssertionError("No native on parser failure")):
            row = sd.evaluate_one(self.legacy, source, {}, Path(tmp), PUBLIC, False, 240)
        self.assertEqual(row["actual_failure_subtype"], "missing_target")
        self.assertFalse(row["api_called"])

    def test_replay_is_single_attempt_and_preserves_plan(self):
        case = self.legacy.split_public_case(PUBLIC, CASE)
        source = {"case_id": CASE, "condition": "json_schema", "transport_ok": True,
                  "origin_output": '{"steps":[{"skill":"pick","target":2},{"skill":"lift"}]}'}
        native = {"native_attempt_count": 1, "actual_gate1_ok": True, "actual_native_ok": True,
                  "actual_condition_success": True, "native_execution_attempted": True, "procedural_success": True}
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", side_effect=AssertionError("No API")), \
             mock.patch.object(self.legacy, "execute_model_fixture", return_value=native) as execute:
            row = sd.evaluate_one(self.legacy, source, case, Path(tmp), PUBLIC, False, 240)
        self.assertEqual(execute.call_args.kwargs["native_max_attempts"], 1)
        self.assertEqual(execute.call_args.kwargs["fixture"]["steps"], json.loads(source["origin_output"])["steps"])
        self.assertTrue(row["procedural_success"])

    def test_exported_real_images_and_messages_match_existing_runner(self):
        bundle = Path(__file__).parent / "cloud_bundle"
        manifest = sd.load_bundle(bundle)
        entry = manifest["cases"][0]
        case = self.legacy.split_public_case(PUBLIC, entry["case_id"])
        expected = self.legacy.build_messages(input_image_path=case["input_image_path"], input_mask_path=case["input_mask_path"],
                   instruction=case["instruction"], task_name=case["task_name"], ablation_mode=self.legacy.FULL_MODE)
        self.assertEqual(sd.messages_for_entry(bundle, entry), expected)
        self.assertEqual(manifest["case_count"], 322)
        self.assertEqual(manifest["task_count"], 34)
        self.assertEqual(len(list((bundle / "inputs").rglob("*.*"))), 322 * 4)
        for forbidden in ("env_config.json", "operation_sequence.json", "expert_sequence.json"):
            self.assertEqual(list(bundle.rglob(forbidden)), [])

    def test_schema_audit_is_separate_from_numeric_normalization(self):
        schema = sd.read_json(sd.HERE / "procedural_plan_schema.json")
        result = sd.schema_audit(self.legacy, '{"steps":[{"skill":"pick","target":"2"}]}', schema)
        self.assertFalse(result["raw_json_schema_valid"])
        self.assertTrue(result["type_normalized_json_schema_valid"])
        result = sd.schema_audit(self.legacy, '{"steps":[{"skill":"pick"}]}', schema)
        self.assertFalse(result["type_normalized_json_schema_valid"])

    def test_incomplete_generation_refused_before_api_or_native(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", side_effect=AssertionError("No API")), \
             mock.patch.object(self.legacy, "execute_model_fixture", side_effect=AssertionError("No native")):
            bundle, _ = make_bundle(tmp)
            with self.assertRaises(FileNotFoundError):
                sd.load_frozen_generation(Path(tmp) / "missing_generation", bundle)

    def test_wrong_local_image_refused_before_native(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server() as (url, requests):
            bundle, _ = make_bundle(tmp)
            root = Path(tmp) / "generation"
            TransportTests().run_inference(bundle, root, url)
            argv = ["evaluate", "--bundle", str(bundle), "--generation-root", str(root), "--project-root", str(PROJECT),
                    "--public-root", str(PUBLIC), "--output-root", str(Path(tmp) / "evaluation")]
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("No API")), \
                 mock.patch.object(self.legacy, "execute_model_fixture", side_effect=AssertionError("No native")):
                with self.assertRaisesRegex(ValueError, "differs"):
                    sd.main(argv)

    def test_validation_only_uses_real_binder_but_never_native(self):
        case = self.legacy.split_public_case(PUBLIC, CASE)
        source = {"case_id": CASE, "condition": "json_object", "transport_ok": True,
                  "origin_output": '{"steps":[{"skill":"pick","target":2}]}'}
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", side_effect=AssertionError("No API")), \
             mock.patch.object(self.legacy, "execute_model_fixture", side_effect=AssertionError("No native")):
            row = sd.evaluate_one(self.legacy, source, case, Path(tmp), PUBLIC, True, 240)
            self.assertTrue(list(Path(tmp).glob("*/validation_traces/*.json")))
        self.assertFalse(row["native_execution_attempted"])

    def test_multiple_native_attempts_are_rejected(self):
        case = self.legacy.split_public_case(PUBLIC, CASE)
        source = {"case_id": CASE, "condition": "json_schema", "transport_ok": True,
                  "origin_output": '{"steps":[{"skill":"pick","target":2},{"skill":"lift"}]}'}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.legacy, "execute_model_fixture", return_value={"native_attempt_count": 2}):
            with self.assertRaisesRegex(RuntimeError, "Single-attempt"):
                sd.evaluate_one(self.legacy, source, case, Path(tmp), PUBLIC, False, 240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
