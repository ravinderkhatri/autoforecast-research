"""CI hand-off: verify a completed bundle is recoverable from its named Git tag."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlops import ProvenanceError, verify_retained_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify one completed bundle's retained source ref."
    )
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("--retained-ref", required=True)
    args = parser.parse_args()
    try:
        result = verify_retained_run(args.bundle, retained_ref=args.retained_ref)
    except ProvenanceError as error:
        parser.error(str(error))
    print(f"verified retained run {result.run_id}: {result.retained_ref} -> {result.source_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
