"""Write a TensorBoard Text-card link to a separately served observability hub."""

from __future__ import annotations

import argparse
import pathlib
from urllib.parse import urlparse

__all__ = ["ObservabilityCardError", "write_tensorboard_navigation"]


class ObservabilityCardError(ValueError):
    """A TensorBoard card cannot safely point to the requested hub URL."""


def write_tensorboard_navigation(log_dir: str | pathlib.Path, hub_url: str) -> pathlib.Path:
    """Write one Markdown link into a dedicated TensorBoard Text run.

    The linked hub owns report navigation. TensorBoard owns its scalar and text plugins; this
    adapter deliberately does not copy report content into event logs.
    """
    parsed = urlparse(hub_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ObservabilityCardError("hub_url must be a reachable HTTP(S) URL")
    root = pathlib.Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(root)) as writer:
        writer.add_text("Navigation/Observability Hub", f"[Open Observability Hub]({hub_url})", 0)
    return root


def main() -> int:
    """Write or refresh the navigation card in the TensorBoard log tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--hub-url", required=True)
    args = parser.parse_args()
    root = write_tensorboard_navigation(args.log_dir, args.hub_url)
    print(f"TensorBoard navigation card: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
