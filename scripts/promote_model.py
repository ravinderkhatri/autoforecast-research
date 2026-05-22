"""Record a local, reviewable AutoForecast champion/candidate alias."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlops.promotion import PromotionError, promote_completed_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a verified local model promotion.")
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--retained-ref", required=True)
    parser.add_argument("--record-root", type=pathlib.Path, default=ROOT / "promotions")
    args = parser.parse_args()
    try:
        promotion = promote_completed_run(
            args.bundle, retained_ref=args.retained_ref, model_id=args.model_id,
            alias=args.alias, record_root=args.record_root, repository=ROOT,
        )
    except PromotionError as error:
        parser.error(str(error))
    print(f"Promotion record: {promotion.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
