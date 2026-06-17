"""Small fixture runner for gitctx deterministic evals."""

from __future__ import annotations

import argparse
from pathlib import Path

from gitctx.conventional import load_fixture_cases, run_fixture_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run gitctx JSONL fixture cases.")
    parser.add_argument("fixture", type=Path, help="Path to a JSONL fixture file.")
    args = parser.parse_args(argv)

    total = len(load_fixture_cases(args.fixture))
    passed, failures = run_fixture_cases(args.fixture)
    failed = len(failures)

    print(f"{total} cases")
    print(f"{passed} passed")
    print(f"{failed} failed")
    for failure in failures:
        print(f"FAIL {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
