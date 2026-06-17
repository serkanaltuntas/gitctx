import unittest
from contextlib import redirect_stdout
from io import StringIO

from gitctx.conventional import (
    CommitContext,
    parse_commit_message,
    run_fixture_cases,
    score_commit_message,
)
from gitctx.eval import main as eval_main


class ConventionalCommitParserTests(unittest.TestCase):
    def test_parses_scoped_commit(self) -> None:
        parsed = parse_commit_message("fix(parser): reject malformed headers")

        self.assertEqual(parsed.type, "fix")
        self.assertEqual(parsed.scope, "parser")
        self.assertEqual(parsed.subject, "reject malformed headers")
        self.assertFalse(parsed.breaking)

    def test_parses_breaking_footer(self) -> None:
        parsed = parse_commit_message(
            "feat(api): require explicit project id\n\nBREAKING CHANGE: callers must pass project_id"
        )

        self.assertTrue(parsed.breaking)
        self.assertEqual(parsed.footers, ("BREAKING CHANGE: callers must pass project_id",))

    def test_preserves_body(self) -> None:
        parsed = parse_commit_message(
            "refactor(core): split parser stages\n\n"
            "The parser now separates header and footer handling so future scorers can\n"
            "inspect each part independently."
        )

        self.assertEqual(len(parsed.body), 3)
        self.assertEqual(parsed.footers, ())

    def test_rejects_invalid_header(self) -> None:
        with self.assertRaises(ValueError):
            parse_commit_message("fix parser missing colon")


class ConventionalCommitScorerTests(unittest.TestCase):
    def test_scores_valid_grounded_message(self) -> None:
        score = score_commit_message(
            "fix(parser): reject malformed headers",
            CommitContext(
                changed_paths=("src/gitctx/parser.py",),
                expected_type="fix",
                expected_scope="parser",
            ),
        )

        self.assertTrue(score.format_validity)
        self.assertTrue(score.type_accuracy)
        self.assertTrue(score.scope_quality)
        self.assertTrue(score.specificity)
        self.assertTrue(score.brevity)
        self.assertEqual(score.errors, ())

    def test_flags_vague_subject(self) -> None:
        score = score_commit_message("fix: update")

        self.assertFalse(score.specificity)
        self.assertIn("subject is too vague", score.errors)

    def test_flags_forbidden_claim(self) -> None:
        score = score_commit_message(
            "feat(auth): add OAuth login",
            CommitContext(forbidden_claims=("OAuth",)),
        )

        self.assertFalse(score.factuality)

    def test_scores_breaking_marker(self) -> None:
        score = score_commit_message(
            "feat(api)!: change request schema",
            CommitContext(expect_breaking_change=True),
        )

        self.assertTrue(score.breaking_change_detection)

    def test_scores_expected_body_and_footer(self) -> None:
        score = score_commit_message(
            "feat(api): require project id\n\n"
            "Calls must provide project_id before request validation can pass.\n\n"
            "Refs: #123",
            CommitContext(expect_body=True, expect_footer=True),
        )

        self.assertTrue(score.body_presence)
        self.assertTrue(score.footer_presence)

    def test_scores_missing_expected_body(self) -> None:
        score = score_commit_message(
            "refactor(core): split parser stages",
            CommitContext(expect_body=True),
        )

        self.assertFalse(score.body_presence)
        self.assertIn("expected commit body", score.errors)


class FixtureRunnerTests(unittest.TestCase):
    def test_dev_fixture_cases_pass(self) -> None:
        passed, failures = run_fixture_cases("fixtures/dev/commit_message_cases.jsonl")

        self.assertGreaterEqual(passed, 1)
        self.assertEqual(failures, [])

    def test_eval_main_returns_success(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = eval_main(["fixtures/dev/commit_message_cases.jsonl"])

        self.assertEqual(exit_code, 0)
        self.assertIn("10 passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
