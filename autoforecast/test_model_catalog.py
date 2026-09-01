"""Tests for the non-claim-bearing, declarative AutoForecast candidate catalog."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast import model_catalog as catalog

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _implementation_profile(root: pathlib.Path, relative: str, payload: bytes) -> pathlib.Path:
    profile = root / "models" / "revision" / "model.yaml"
    profile.parent.mkdir(parents=True)
    digest = hashlib.sha256(payload).hexdigest()
    profile.write_text(
        "schema_version: 1\n"
        "implementation:\n"
        f"  config: {relative}\n"
        f"  sha256: {digest}\n"
        "model:\n"
        "  id: revision_test\n"
        "  adapter: sft\n"
        "  family: custom\n"
        "  status: candidate\n"
        "  tags: [autoforecast]\n"
        "  runtime: custom_torch\n",
        encoding="utf-8",
    )
    return profile.parent.parent


def test_checked_in_catalog_resolves_all_registered_candidates() -> None:
    candidates = catalog.discover_candidates(ROOT / "configs/models")
    policy = catalog.load_selection_policy(
        ROOT / "configs/autoforecast/selection/development_top_k.yaml"
    )
    selected = catalog.resolve_candidates(candidates, policy)
    assert len(candidates) == 16
    assert len(selected) == 16
    assert {candidate.id for candidate in candidates} == {
        "seasonal_naive",
        "croston",
        "auto_ets",
        "auto_theta",
        "shallow_global",
        "tft",
        "tft_q",
        "tft_hnb",
        "autogluon_zoo",
        "timesfm_standalone",
        "timesfm_hnb",
        "timesfm_v8",
        "mft",
        "sft",
        "hmfnet",
        "pact",
    }
    assert {candidate.adapter for candidate in candidates} == {
        "seasonal_naive",
        "croston",
        "auto_ets",
        "auto_theta",
        "shallow_global",
        "tft",
        "tft_q",
        "tft_hnb",
        "autogluon_zoo",
        "timesfm_standalone",
        "timesfm_hnb",
        "timesfm_v8",
        "mft",
        "sft",
        "hmfnet",
        "pact",
    }


@pytest.mark.parametrize("alias", ["final_symlink", "intermediate_symlink", "hardlink"])
def test_implementation_config_rejects_filesystem_aliases(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.json"
    payload = b'{}\n'
    outside.write_bytes(payload)
    if alias == "final_symlink":
        target = project / "recipe.json"
        target.symlink_to(outside)
        relative = "recipe.json"
    elif alias == "intermediate_symlink":
        linked = project / "linked"
        linked.symlink_to(tmp_path, target_is_directory=True)
        relative = "linked/outside.json"
    else:
        target = project / "recipe.json"
        os.link(outside, target)
        relative = "recipe.json"
    catalog_root = _implementation_profile(project, relative, payload)
    monkeypatch.setattr(catalog, "_PROJECT_ROOT", project)
    with pytest.raises(catalog.CatalogError, match="missing or aliased"):
        catalog.discover_candidates(catalog_root)


def test_implementation_config_retains_the_single_read_bytes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    payload = b'{}\n'
    (project / "recipe.json").write_bytes(payload)
    catalog_root = _implementation_profile(project, "recipe.json", payload)
    monkeypatch.setattr(catalog, "_PROJECT_ROOT", project)
    candidate = catalog.discover_candidates(catalog_root)[0]
    assert candidate.implementation_config_bytes == payload
    assert candidate.source_bytes == candidate.source_path.read_bytes()


def test_catalog_refuses_an_unknown_adapter() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        path = root / "bad"
        path.mkdir()
        (path / "model.yaml").write_text(
            "schema_version: 1\nmodel:\n"
            "  id: bad\n  adapter: not_registered\n  family: baseline\n"
            "  runtime: builtin\n  status: candidate\n  tags: [autoforecast]\n"
        )
        try:
            catalog.discover_candidates(root)
        except catalog.CatalogError as exc:
            assert "unknown" in str(exc)
            return
        raise AssertionError("catalog accepted an adapter that cannot be registered")


def test_catalog_refuses_duplicate_yaml_keys() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        path = root / "ambiguous"
        path.mkdir()
        (path / "model.yaml").write_text(
            "schema_version: 1\nschema_version: 1\nmodel: {}\n",
            encoding="utf-8",
        )
        try:
            catalog.discover_candidates(root)
        except catalog.CatalogError as exc:
            assert "duplicate YAML mapping key" in str(exc)
            return
        raise AssertionError("catalog accepted an ambiguous YAML mapping")


def test_catalog_requires_exact_integer_schema_versions_and_string_direction() -> None:
    for version in ("true", "1.0"):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "candidate"
            path.mkdir()
            (path / "model.yaml").write_text(
                f"schema_version: {version}\nmodel:\n"
                "  id: exact\n  adapter: seasonal_naive\n  family: baseline\n"
                "  runtime: builtin\n  status: candidate\n  tags: [autoforecast]\n"
            )
            try:
                catalog.discover_candidates(root)
            except catalog.CatalogError as error:
                assert "integer 1" in str(error)
            else:
                raise AssertionError(f"catalog accepted schema_version={version}")

    policy = pathlib.Path(tempfile.mkdtemp()) / "policy.yaml"
    policy.write_text(
        """schema_version: 1
selection:
  id: exact
  candidate_tags: [autoforecast]
  statuses: [candidate]
  metric: wrmsse
  direction: [minimize]
  top_k: 1
"""
    )
    try:
        catalog.load_selection_policy(policy)
    except catalog.CatalogError as error:
        assert "direction" in str(error)
    else:
        raise AssertionError("catalog accepted a non-string direction")


def test_catalog_refuses_duplicate_model_id() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        for name in ("first", "second"):
            path = root / name
            path.mkdir()
            (path / "model.yaml").write_text(
                "schema_version: 1\nmodel:\n"
                "  id: duplicate\n  adapter: seasonal_naive\n  family: baseline\n"
                "  runtime: builtin\n  status: candidate\n  tags: [autoforecast]\n"
            )
        try:
            catalog.discover_candidates(root)
        except catalog.CatalogError as exc:
            assert "duplicate model ids" in str(exc)
            return
        raise AssertionError("catalog accepted two candidates with one identity")


def test_catalog_refuses_executable_or_unrecognised_yaml_fields() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        path = root / "unsafe"
        path.mkdir()
        (path / "model.yaml").write_text(
            "schema_version: 1\nmodel:\n"
            "  id: unsafe\n  adapter: seasonal_naive\n  family: baseline\n"
            "  runtime: builtin\n  status: candidate\n  tags: [autoforecast]\n"
            "  python_target: os.system\n"
        )
        try:
            catalog.discover_candidates(root)
        except catalog.CatalogError as exc:
            assert "exactly" in str(exc)
            return
        raise AssertionError("catalog accepted an executable YAML extension point")


def test_policy_refuses_insufficient_pre_ranking_candidates() -> None:
    candidates = catalog.discover_candidates(ROOT / "configs/models")
    policy = catalog.SelectionPolicy(
        "too_many",
        frozenset({"autoforecast", "development"}),
        frozenset({"candidate"}),
        "wrmsse",
        "minimize",
        len(candidates) + 1,
        pathlib.Path("policy.yaml"),
        "0" * 64,
    )
    try:
        catalog.resolve_candidates(candidates, policy)
    except catalog.CatalogError as exc:
        assert f"fewer than top_k={len(candidates) + 1}" in str(exc)
        return
    raise AssertionError("policy promised more selected candidates than exist")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)


def test_implementation_config_uses_the_active_experiment_project(tmp_path, monkeypatch):
    from autoforecast.project import experiment_project

    checkout = tmp_path / "checkout"
    project = tmp_path / "project"
    checkout.mkdir()
    project.mkdir()
    first = b'{"owner":"checkout"}\n'
    second = b'{"owner":"project"}\n'
    (checkout / "recipe.json").write_bytes(first)
    (project / "recipe.json").write_bytes(second)
    checkout_catalog = _implementation_profile(checkout, "recipe.json", first)
    project_catalog = _implementation_profile(project, "recipe.json", second)
    monkeypatch.setattr(catalog, "_PROJECT_ROOT", checkout)
    with experiment_project(project):
        candidate = catalog.discover_candidates(project_catalog)[0]
        assert candidate.implementation_config_bytes == second
        assert candidate.implementation_config_path == project / "recipe.json"
        with pytest.raises(catalog.CatalogError, match="has changed"):
            catalog.discover_candidates(checkout_catalog)
    assert catalog.discover_candidates(checkout_catalog)[0].implementation_config_bytes == first
