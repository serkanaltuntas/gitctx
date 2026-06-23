import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import tempfile
from pathlib import Path
from threading import Thread
from typing import Optional
import unittest

from gitctx.ollama_generate import (
    generate_labels,
    generate_smoke_labels,
    validate_generated_labels,
    validate_smoke_generated_labels,
)


class OllamaGenerateTests(unittest.TestCase):
    def test_generates_and_validates_label_with_fake_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama()
            try:
                report = generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    ollama_options={"num_ctx": 4096},
                    resume=False,
                )
                summary = validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(report["generated_records"], 1)
            self.assertEqual(summary["generated_label_records"], 1)
            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["header"], "fix(parser): handle empty values")
            self.assertEqual(label["teacher_runtime_model_id"], "deepseek-r1:latest")
            self.assertEqual(label["human_review_status"], "not_reviewed")
            self.assertEqual(server.seen_payload["format"], "json")  # type: ignore[attr-defined]
            self.assertFalse(server.seen_payload["think"])  # type: ignore[attr-defined]
            self.assertEqual(server.seen_payload["options"]["num_ctx"], 4096)  # type: ignore[attr-defined]
            self.assertEqual(server.seen_payload["options"]["num_predict"], 256)  # type: ignore[attr-defined]

    def test_generates_and_validates_named_label_with_fake_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root, artifact_name="pilot")
            server, thread = self._start_fake_ollama()
            try:
                report = generate_labels(
                    root,
                    artifact_name="pilot",
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                summary = validate_generated_labels(root, artifact_name="pilot")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(report["artifact_name"], "pilot")
            self.assertEqual(summary["artifact_name"], "pilot")
            label_path = root / "artifacts/teacher/generated-labels.pilot.jsonl"
            self.assertTrue(label_path.exists())
            label = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(label["header"], "fix(parser): handle empty values")

    def test_resume_skips_existing_generated_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            generated_path = root / "artifacts/teacher/generated-labels.smoke.jsonl"
            generated_path.write_text(
                json.dumps({"id": "generated-example-repo-111111111111"}) + "\n",
                encoding="utf-8",
            )

            report = generate_smoke_labels(
                root,
                ollama_url="http://127.0.0.1:9",
                resume=True,
            )

            self.assertEqual(report["generated_records"], 0)
            self.assertEqual(report["skipped_existing_records"], 1)
            self.assertEqual(report["failed_records"], 0)

    def test_progress_callback_reports_generation_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            events = []
            server, thread = self._start_fake_ollama()
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    progress_every=1,
                    resume=False,
                    progress_callback=events.append,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(events[0]["processed"], 0)
            self.assertEqual(events[0]["total"], 1)
            self.assertEqual(events[-1]["processed"], 1)
            self.assertEqual(events[-1]["generated"], 1)
            self.assertEqual(events[-1]["failed"], 0)
            self.assertEqual(events[-1]["current_record_id"], "generated-example-repo-111111111111")

    def test_scope_only_parser_error_does_not_lower_verifier_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(
                header="fix(parser_module): handle empty values",
                scope="parser_module",
            )
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("scope 'parser_module' is not visible", label["parser_result"]["errors"][0])
            self.assertEqual(label["verifier_score"], 1.0)

    def test_invalid_evidence_paths_become_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(
                evidence_paths=["tests/test_parser.py#L12", "tests/missing.py"],
            )
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["evidence_paths"], ["tests/test_parser.py"])
            self.assertIn(
                "dropped evidence_paths not present in changed_paths: tests/missing.py",
                label["warnings"],
            )

    def test_string_footers_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(footers="Refs: #123")
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["footers"], ["Refs: #123"])

    def test_plain_conventional_commit_output_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(
                raw_content="fix(parser): handle empty values"
            )
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["header"], "fix(parser): handle empty values")
            self.assertIn("normalized to JSON candidate", label["warnings"][0])

    def test_plain_commit_inside_prose_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(
                raw_content=(
                    "Here is the commit message:\n\n"
                    "```text\n"
                    "fix(parser): handle empty values\n"
                    "```\n"
                )
            )
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["header"], "fix(parser): handle empty values")

    def test_json_string_commit_output_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_teacher_inputs(root)
            server, thread = self._start_fake_ollama(
                raw_content=json.dumps("fix(parser): handle empty values")
            )
            try:
                generate_smoke_labels(
                    root,
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    resume=False,
                )
                validate_smoke_generated_labels(root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            label = json.loads(
                (root / "artifacts/teacher/generated-labels.smoke.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(label["header"], "fix(parser): handle empty values")

    def _write_teacher_inputs(self, root: Path, *, artifact_name: str = "smoke") -> None:
        (root / "artifacts/teacher").mkdir(parents=True)
        record = {
            "id": "teacher-input-example-repo-111111111111",
            "source_diff_id": "example-repo-111111111111",
            "review_decision_id": "review-example-repo-111111111111",
            "source_repo_url": "https://github.com/example/repo",
            "source_license": "MIT",
            "source_commit": "1111111111111111111111111111111111111111",
            "parent_commit": "0000000000000000000000000000000000000000",
            "data_split": "DEV",
            "changed_paths": ["src/parser.py", "tests/test_parser.py"],
            "diff_stat": " src/parser.py | 2 +-",
            "historical_subject": "fix parser",
            "teacher_model_id": "ollama/deepseek-r1:latest",
            "teacher_runtime": "ollama",
            "teacher_runtime_model_id": "deepseek-r1:latest",
            "teacher_revision": "6995872bfe4c",
            "teacher_license": "MIT",
            "teacher_size": "5.2 GB",
            "teacher_context_length": "128K",
            "prompt_version": "commit-message-teacher-v0.1",
            "prompt_path": "prompts/commit-message-teacher-v0.1.md",
            "decoding_config": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 256},
            "system_message": "Return JSON only.",
            "user_message": "Generate one Conventional Commit message.",
            "diff": "diff --git a/src/parser.py b/src/parser.py\n",
            "diff_sha256": "0" * 64,
            "input_status": "ready_for_generation",
        }
        (root / f"artifacts/teacher/teacher-inputs.{artifact_name}.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )

    def _start_fake_ollama(
        self,
        *,
        header: str = "fix(parser): handle empty values",
        scope: str = "parser",
        evidence_paths: Optional[list[str]] = None,
        footers: object = None,
        raw_content: Optional[str] = None,
    ) -> tuple[HTTPServer, Thread]:
        evidence_paths = evidence_paths or ["src/parser.py", "tests/test_parser.py"]
        if footers is None:
            footers = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback.
                length = int(self.headers["Content-Length"])
                self.server.seen_payload = json.loads(self.rfile.read(length))  # type: ignore[attr-defined]
                content = raw_content
                if content is None:
                    content = json.dumps(
                        {
                            "header": header,
                            "body": [],
                            "footers": footers,
                            "type": "fix",
                            "scope": scope,
                            "confidence": 0.9,
                            "warnings": [],
                            "evidence_paths": evidence_paths,
                        }
                    )
                payload = {
                    "message": {
                        "content": content
                    }
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread


if __name__ == "__main__":
    unittest.main()
