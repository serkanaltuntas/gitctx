import json
import tempfile
import unittest
from pathlib import Path

from gitctx.split_plan import load_split_plan, select_split_for_commit, validate_split_plan


class SplitPlanTests(unittest.TestCase):
    def test_validates_and_selects_split_by_repo_time_window(self) -> None:
        plan = _plan(
            [
                _window(
                    "dev",
                    "https://github.com/example/repo",
                    "DEV",
                    "2025-01-01T00:00:00Z",
                    "2025-07-01T00:00:00Z",
                ),
                _window(
                    "report",
                    "https://github.com/example/repo",
                    "REPORT",
                    "2025-07-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ),
            ]
        )

        self.assertEqual(validate_split_plan(plan), ())
        self.assertEqual(
            select_split_for_commit(
                plan,
                "https://github.com/example/repo.git",
                "2025-08-01T00:00:00+00:00",
            ),
            "REPORT",
        )
        self.assertIsNone(
            select_split_for_commit(
                plan,
                "https://github.com/example/repo",
                "2026-02-01T00:00:00+00:00",
            )
        )

    def test_rejects_overlapping_repo_windows(self) -> None:
        plan = _plan(
            [
                _window(
                    "dev",
                    "https://github.com/example/repo",
                    "DEV",
                    "2025-01-01T00:00:00Z",
                    "2025-08-01T00:00:00Z",
                ),
                _window(
                    "report",
                    "https://github.com/example/repo",
                    "REPORT",
                    "2025-07-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ),
            ]
        )

        self.assertIn("overlapping split windows", "\n".join(validate_split_plan(plan)))

    def test_load_split_plan_raises_on_invalid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split-plan.json"
            path.write_text(json.dumps(_plan([])), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_split_plan(path)


def _plan(windows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "id": "test-plan",
        "version": "v0",
        "created_at": "2026-06-19",
        "windows": windows,
    }


def _window(
    window_id: str,
    repo_url: str,
    split: str,
    start: str,
    end: str,
) -> dict[str, str]:
    return {
        "id": window_id,
        "repo_url": repo_url,
        "split": split,
        "start": start,
        "end": end,
        "reason": "test window",
    }


if __name__ == "__main__":
    unittest.main()
