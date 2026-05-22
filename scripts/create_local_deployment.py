"""Create a local batch-deployment hand-off from an approved promotion record."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlops.deployment import DeploymentError, create_local_batch_deployment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable local batch-forecast deployment hand-off."
    )
    parser.add_argument("--promotion", type=pathlib.Path, required=True)
    parser.add_argument("--deployment-root", type=pathlib.Path, default=ROOT / "deployments")
    args = parser.parse_args()
    try:
        deployment = create_local_batch_deployment(
            args.promotion,
            deployment_root=args.deployment_root,
            repository=ROOT,
        )
    except DeploymentError as error:
        parser.error(str(error))
    print(f"Local batch deployment: {deployment.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
