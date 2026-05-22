"""Check the tracked publication inventory and an optional clean-build wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tomllib
import zipfile

from autoforecast._exact_json import loads_exact_json

ROOT_FILES = {
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "uv.lock",
    "pyrightconfig.json",
    "pyrightconfig.governed.json",
}
PRIVATE_PATH = re.compile(r"(^|/)(conversations?|transcripts?|sessions?|chats?)(/|[._-])", re.I)
ATTRIBUTION = re.compile(
    r"(?im)^\s*(?:co-authored-by|generated-by|assistant-signature|ai-signature)\s*:"
    r"|^\s*(?:human|assistant)\s*:\s+"
    r"|<\|(?:im_start|im_end|assistant|user)\|>"
    r'|"role"\s*:\s*"(?:assistant|user|system)"'
)


def git(root: pathlib.Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.PIPE)


def tracked_inventory(root: pathlib.Path) -> tuple[set[str], dict]:
    """Only maintained code, configuration, CI and the public README are distributable."""
    if pathlib.Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("public inventory requires the repository root")
    config = tomllib.loads((root / "pyproject.toml").read_text())
    roots = {
        pattern.removesuffix("*")
        for pattern in config["tool"]["setuptools"]["packages"]["find"]["include"]
    }
    names = {name.decode() for name in git(root, "ls-files", "-z").split(b"\0") if name}
    if not names:
        raise ValueError("public release has no tracked files")
    for name in sorted(names):
        path = pathlib.PurePosixPath(name)
        allowed = (
            name in ROOT_FILES
            or (path.parts[0] in roots and path.suffix == ".py")
            or (path.parts[0] == "configs" and path.suffix in {".json", ".yaml", ".yml"})
            or (path.parent.as_posix() == ".github/workflows" and path.suffix in {".yaml", ".yml"})
            or name == "observability/sources.json"
        )
        if (
            not allowed
            or PRIVATE_PATH.search(name)
            or name.startswith("migration/")
            or (root / name).is_symlink()
        ):
            raise ValueError(f"file is outside the public inventory: {name}")
        if ATTRIBUTION.search((root / name).read_text()):
            raise ValueError(f"conversation or attribution marker in public file: {name}")
    messages = git(root, "log", "--all", "--format=%B").decode()
    if ATTRIBUTION.search(messages):
        raise ValueError("conversation or attribution marker in commit history")
    return names, config


def verify_wheel(root: pathlib.Path, wheel: pathlib.Path, names: set[str], config: dict) -> int:
    """Require exact tracked runtime modules and bytes, including the wheel's RECORD."""
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("wheel verification requires a clean tracked checkout")
    settings = config["tool"]["setuptools"]
    roots = {pattern.removesuffix("*") for pattern in settings["packages"]["find"]["include"]}
    expected = {
        name
        for name in names
        if name.split("/", 1)[0] in roots
        and name.endswith(".py")
        and not pathlib.PurePosixPath(name).stem.startswith("test_")
        and not pathlib.PurePosixPath(name).stem.endswith("_test")
    }
    for package, patterns in settings.get("package-data", {}).items():
        prefix = package.replace(".", "/") + "/"
        expected.update(
            name
            for name in names
            if name.startswith(prefix)
            and any(fnmatch.fnmatchcase(name[len(prefix) :], pattern) for pattern in patterns)
        )
    project = config["project"]
    info = f"{project['name'].replace('-', '_')}-{project['version']}.dist-info"
    metadata = {
        f"{info}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "entry_points.txt",
            "top_level.txt",
            "RECORD",
        )
    }
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        if len(set(members)) != len(members):
            raise ValueError("wheel has duplicate members")
        wanted = expected | metadata | {"autoforecast/_build.json"}
        if set(members) != wanted:
            raise ValueError(
                f"wheel inventory differs: extra={sorted(set(members) - wanted)}, "
                f"missing={sorted(wanted - set(members))}"
            )
        for name in expected:
            if archive.read(name) != (root / name).read_bytes():
                raise ValueError(f"wheel source differs from tracked source: {name}")
        code = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in expected
            if name.endswith(".py")
        }
        build = loads_exact_json(archive.read("autoforecast/_build.json"))
        expected_build = {
            "schema": "autoforecast.build.v1",
            "source_commit": git(root, "rev-parse", "--verify", "HEAD").decode().strip(),
            "code_sha256": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
        }
        if build != expected_build:
            raise ValueError("wheel build identity differs from the checked commit and code")
        rows = list(csv.reader(io.StringIO(archive.read(f"{info}/RECORD").decode())))
        if any(len(row) != 3 for row in rows) or len(rows) != len(members):
            raise ValueError("wheel RECORD has an invalid inventory")
        if {row[0] for row in rows} != wanted:
            raise ValueError("wheel RECORD has an invalid inventory")
        for name, recorded_hash, size in rows:
            payload = archive.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            expected_record = (
                ("", "")
                if name == f"{info}/RECORD"
                else (
                    f"sha256={digest}",
                    str(len(payload)),
                )
            )
            if (recorded_hash, size) != expected_record:
                raise ValueError(f"wheel RECORD differs from member bytes: {name}")
        return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--wheel", type=pathlib.Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        names, config = tracked_inventory(root)
        members = verify_wheel(root, args.wheel, names, config) if args.wheel else None
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"public release inventory failed: {exc}\n")
    print(f"public release inventory: {len(names)} tracked files; wheel members: {members}")


if __name__ == "__main__":
    main()
