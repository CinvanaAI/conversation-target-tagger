import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from conversation_target_tagger.cli import run
from conversation_target_tagger.normalize import normalize_export, public_row
from conversation_target_tagger.ollama import validate_base_url
from conversation_target_tagger.tagging import (
    build_user_prompt,
    parse_and_validate,
    plan_batches,
    tag_rows,
)


def newer_message(identifier: str, timestamp: int, text: str) -> dict:
    return {
        "id": identifier,
        "author": {"role": "user"},
        "create_time": timestamp,
        "content": {"content_type": "text", "parts": [text]},
    }


class NormalizeTests(unittest.TestCase):
    def test_newer_and_mapping_shapes_are_both_preserved(self) -> None:
        export = [
            {"id": "new", "create_time": 5, "messages": [newer_message("n1", 20, "new")]},
            {
                "id": "old",
                "create_time": 4,
                "mapping": {
                    "o1": {
                        "parent": None,
                        "children": [],
                        "message": newer_message("ignored", 10, "old"),
                    }
                },
            },
        ]
        rows = normalize_export(export)
        self.assertEqual([row["conversation_id"] for row in rows], ["old", "new"])
        self.assertEqual([row["gidx"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["node_id"], "o1")

    def test_ties_are_stable_by_conversation_and_sequence(self) -> None:
        rows = normalize_export(
            [
                {"id": "a", "create_time": 1, "messages": [newer_message("a1", 10, "one"), newer_message("a2", 10, "two")]},
                {"id": "b", "create_time": 1, "messages": [newer_message("b1", 10, "three")]},
            ]
        )
        self.assertEqual([row["node_id"] for row in rows], ["a1", "a2", "b1"])

    def test_raw_data_is_opt_in_and_public_content_is_opt_in(self) -> None:
        export = [{"id": "a", "messages": [newer_message("a1", 1, "private")]}]
        row = normalize_export(export, retain_raw=True)[0]
        self.assertIn("raw_message", row)
        stripped = public_row(row)
        self.assertNotIn("raw_message", stripped)
        self.assertNotIn("content_text", stripped)
        self.assertEqual(public_row(row, include_content=True)["content_text"], "private")

    def test_invalid_root_fails(self) -> None:
        with self.assertRaises(ValueError):
            normalize_export({"not": "a list"})


class TaggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = normalize_export(
            [{"id": "a", "messages": [newer_message(f"n{i}", i, f"text {i}") for i in range(1, 6)]}]
        )

    def test_batches_have_context_without_retargeting_it(self) -> None:
        batches = plan_batches(self.rows, window=2, context=1)
        self.assertEqual([batch.target_gidxs for batch in batches], [(1, 2), (3, 4), (5,)])
        self.assertEqual([row["gidx"] for row in batches[1].rows], [2, 3, 4])

    def test_prompt_marks_context_and_targets(self) -> None:
        prompt = build_user_prompt(plan_batches(self.rows, window=2, context=1)[1])
        self.assertIn("[CTX] [2]", prompt)
        self.assertIn("[MSG] [3]", prompt)
        self.assertIn("TARGET_GIDXS: [3, 4]", prompt)

    def test_strict_json_and_alignment_are_required(self) -> None:
        with self.assertRaises(ValueError):
            parse_and_validate("```json\n[]\n```", (1,))
        with self.assertRaises(ValueError):
            parse_and_validate('[{"gidx": 2, "targets": []}]', (1,))

    def test_target_contract_rejects_unknown_and_duplicate_labels(self) -> None:
        with self.assertRaises(ValueError):
            parse_and_validate('[{"gidx": 1, "targets": ["mystery:x"]}]', (1,))
        with self.assertRaises(ValueError):
            parse_and_validate('[{"gidx": 1, "targets": ["self", "self"]}]', (1,))

    def test_injected_tagger_assigns_every_row(self) -> None:
        def fake(_system: str, prompt: str) -> str:
            ids = json.loads(prompt.split("TARGET_GIDXS: ", 1)[1].splitlines()[0])
            return json.dumps([{"gidx": value, "targets": ["thing:sensor"]} for value in ids])

        tagged = tag_rows(self.rows, fake, window=2, context=1)
        self.assertEqual([row["targets"] for row in tagged], [["thing:sensor"]] * 5)

    def test_batch_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            plan_batches(self.rows, window=0)
        with self.assertRaises(ValueError):
            plan_batches(self.rows, context=101)


class BoundaryTests(unittest.TestCase):
    def test_ollama_defaults_to_loopback(self) -> None:
        self.assertEqual(validate_base_url("http://localhost:11434/"), "http://localhost:11434")
        with self.assertRaises(ValueError):
            validate_base_url("https://models.example.test")
        self.assertEqual(
            validate_base_url("https://models.example.test", allow_remote=True),
            "https://models.example.test",
        )

    def test_ollama_url_must_be_an_origin(self) -> None:
        with self.assertRaises(ValueError):
            validate_base_url("http://localhost:11434/private/path")
        with self.assertRaises(ValueError):
            validate_base_url("http://localhost:11434?token=hidden")

    def test_cli_plan_performs_no_model_call_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.json"
            source.write_text(json.dumps([{"id": "a", "messages": [newer_message("a1", 1, "hello")]}]), encoding="utf-8")
            output = root / "output"
            self.assertEqual(run([str(source), str(output), "--plan"]), 0)
            self.assertFalse(output.exists())

    def test_module_entrypoint_prints_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.json"
            source.write_text(json.dumps([{"id": "a", "messages": [newer_message("a1", 1, "hello")]}]), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "conversation_target_tagger.cli",
                    str(source),
                    str(root / "output"),
                    "--plan",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["message_count"], 1)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
