import unittest

from gitctx.conventional import CommitContext, parse_commit_message, score_commit_message


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


if __name__ == "__main__":
    unittest.main()
