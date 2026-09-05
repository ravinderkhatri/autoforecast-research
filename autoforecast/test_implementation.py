"""Saved-source identity is portable, explicit and independent of study data."""

import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sys

import pytest

from autoforecast.implementation import (
    resolve_implementation,
    source_scope,
    verify_retained,
    worker_implementation,
)


def scope_at(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text("def predict(): return -2.0\n")
    return source_scope({"source_roots": {"model": str(root)}}, root.parent)


def interactive():
    # A generated callback has no saved module; its files remain saved-source evidence.
    return eval(compile("lambda request: request", "<notebook>", "eval"))


def capture(scope):
    return resolve_implementation(None, scope, interactive(), "synthetic")


def test_portable_identity_includes_edits_additions_deletions_and_scope(tmp_path):
    scope = scope_at(tmp_path / "a")
    first = capture(scope)
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    other = source_scope({"source_roots": {"model": str(tmp_path / "b")}})
    assert capture(other)["source_sha256"] == first["source_sha256"]
    helper = tmp_path / "a/helper.py"
    helper.write_text("VALUE = 1\n")
    second = capture(scope)
    assert second["source_sha256"] != first["source_sha256"]
    helper.write_text("VALUE = 2\n")
    assert capture(scope)["source_sha256"] != second["source_sha256"]
    with pytest.raises(ValueError, match="changed"):
        first.verify()
    helper.unlink()
    assert capture(scope)["source_sha256"] == first["source_sha256"]
    changed_scope = {**scope, "exclude": ["unused.py"]}
    assert capture(changed_scope)["source_sha256"] != first["source_sha256"]


def test_excluded_files_never_enter_saved_source(tmp_path):
    scope = scope_at(tmp_path / "source")
    first = capture(scope)
    for directory in ("data", "outputs", ".git", ".venv", "__pycache__"):
        folder = tmp_path / "source" / directory
        folder.mkdir()
        (folder / "file.py").write_text("excluded")
    for name in (".env", "credentials.json", "secrets.yaml"):
        (tmp_path / "source" / name).write_text("excluded")
    assert capture(scope)["source_sha256"] == first["source_sha256"]
    destination = tmp_path / "retained"
    destination.mkdir()
    first.retain(destination)
    assert len(list((destination / "metadata/implementation/source").rglob("*.py"))) == 1
    evidence = json.loads((destination / "metadata/implementation/evidence.json").read_bytes())
    assert evidence["callback"]["evidence"] == "saved_source"
    assert evidence["callback"]["execution_attested"] is False
    assert "installed_packages" in evidence["environment"]
    shutil.rmtree(tmp_path / "source")
    verify_retained(destination, first, first.binding)
    source = destination / "metadata/implementation/source/model/model.py"
    source.write_text("tampered")
    with pytest.raises(ValueError, match="source differs"):
        verify_retained(destination, first, first.binding)


@pytest.mark.parametrize("fault", ["missing", "empty", "overlap", "file_link", "dir_link"])
def test_invalid_source_trees_fail_before_capture(tmp_path, fault):
    scope = scope_at(tmp_path / "source")
    if fault == "missing":
        shutil.rmtree(tmp_path / "source")
    elif fault == "empty":
        (tmp_path / "source/model.py").unlink()
    elif fault == "overlap":
        scope_at(tmp_path / "source/sub")
        scope["source_roots"]["nested"] = str(tmp_path / "source/sub")
    elif fault == "file_link":
        (tmp_path / "source/model.py").unlink()
        (tmp_path / "outside.py").write_text("outside")
        (tmp_path / "source/model.py").symlink_to(tmp_path / "outside.py")
    else:
        (tmp_path / "source/sub").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        capture(scope)


@pytest.mark.parametrize("scope", [{}, {"source_roots": {}}, {"source_roots": {"../bad": "a"}},
                                   {"source_roots": {"m": "a"}, "include": []},
                                   {"source_roots": {"m": "a"}, "unknown": True},
                                   None,
                                   [],
                                   {'source_roots': []},
                                   {'source_roots': {'m': 3}},
                                   {'source_roots': {'m': ''}},
                                   {'source_roots': {'m': '   '}},
                                   {'source_roots': {'m': 'file:///tmp/source'}},
                                   {'source_roots': {'m': 'a'}, 'include': None},
                                   {'source_roots': {'m': 'a'}, 'include': '*.py'},
                                   {'source_roots': {'m': 'a'}, 'include': ('*.py',)},
                                   {'source_roots': {'m': 'a'}, 'include': ['']},
                                   {'source_roots': {'m': 'a'}, 'include': [None]},
                                   {'source_roots': {'m': 'a'}, 'include': ['/a.py']},
                                   {'source_roots': {'m': 'a'}, 'include': ['a/../b.py']},
                                   {'source_roots': {'m': 'a'}, 'exclude': None},
                                   {'source_roots': {'m': 'a'}, 'exclude': '*.py'},
                                   {'source_roots': {'m': 'a'}, 'exclude': ('*.py',)},
                                   {'source_roots': {'m': 'a'}, 'exclude': ['']},
                                   {'source_roots': {'m': 'a'}, 'exclude': [None]},
                                   {'source_roots': {'m': 'a'}, 'exclude': ['/a.py']},
                                   {'source_roots': {'m': 'a'}, 'exclude': ['a/../b.py']},
])
def test_invalid_scope_is_actionable(scope):
    with pytest.raises(ValueError):
        source_scope(scope)


def test_explicit_identity_skips_scanning_and_missing_scope_is_actionable():
    declared = {"name": "model", "version": "1", "source_sha256": "a" * 64, "dependencies": {}}
    assert resolve_implementation(declared, {"bad": "scope"}, interactive(), "model") is declared
    with pytest.raises(ValueError, match="source_roots"):
        resolve_implementation(None, None, interactive(), "model")


def test_importable_callback_must_be_in_scope_and_worker_uses_frozen_baseline(tmp_path, monkeypatch):
    scope = scope_at(tmp_path / "source")
    with pytest.raises(ValueError, match="callback source"):
        resolve_implementation(None, scope, capture, "model")
    first = capture(scope)
    saved = tmp_path / "retained"
    saved.mkdir()
    first.retain(saved)
    record = {"path": str(saved), "identity": dict(first), "binding": first.binding, "scope": scope}
    shutil.copytree(tmp_path / "source", tmp_path / "worker")
    monkeypatch.setenv("AUTOFORECAST_SOURCE_ROOTS", json.dumps({"model": str(tmp_path / "worker")}))
    second = worker_implementation(record, interactive())
    assert dict(second) == dict(first)
    with pytest.raises(dataclasses.FrozenInstanceError):
        second.manifest_bytes = b"changed"
    (tmp_path / "worker/model.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        worker_implementation(record)
    monkeypatch.setenv("AUTOFORECAST_SOURCE_ROOTS", json.dumps({"wrong": str(tmp_path)}))
    with pytest.raises(ValueError, match="labels"):
        worker_implementation(record)


def test_selected_unreadable_source_is_not_silently_omitted(tmp_path, monkeypatch):
    import autoforecast.implementation as module

    scope = scope_at(tmp_path / "source")
    def unreadable(*args):
        raise PermissionError("source cannot be read")
    monkeypatch.setattr(module, "read_regular_bytes", unreadable)
    with pytest.raises(PermissionError):
        capture(scope)


def test_capture_refuses_edit_between_initial_read_and_verification(tmp_path, monkeypatch):
    import autoforecast.implementation as module

    scope = scope_at(tmp_path / "source")
    original = module.installed_dependencies
    def edit_source():
        (tmp_path / "source/model.py").write_text("changed during capture")
        return original()
    monkeypatch.setattr(module, "installed_dependencies", edit_source)
    with pytest.raises(ValueError, match="changed"):
        capture(scope)


def test_explicit_notebook_export_is_retained_as_saved_bytes(tmp_path):
    scope = scope_at(tmp_path / "source")
    (tmp_path / "source/executed.ipynb").write_text('{"cells": []}')
    scope["include"].append("*.ipynb")
    result = capture(scope)
    assert "model/executed.ipynb" in json.loads(result.manifest_bytes)["files"]
    assert not json.loads(result.evidence_bytes)["callback"]["execution_attested"]


def test_bound_partial_and_callable_objects_check_their_saved_source(tmp_path, monkeypatch):
    import functools
    import importlib.util
    import sys

    scope = scope_at(tmp_path / "source")
    path = tmp_path / "source/model.py"
    path.write_text("def predict(request): return request\n"
                    "class Model:\n    def __call__(self, request): return request\n")
    spec = importlib.util.spec_from_file_location("scoped_callable", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    for callback in (module.Model, functools.partial(module.Model), staticmethod(module.Model),
                     functools.partial(staticmethod(module.Model)),
                     functools.lru_cache(maxsize=0)(module.Model),
                     functools.partial(functools.lru_cache(maxsize=0)(module.Model))):
        with pytest.raises(ValueError, match="function wrapper for class callbacks"):
            resolve_implementation(None, scope, callback, "model")
        declared = {"name": "model", "version": "1", "source_sha256": "a" * 64, "dependencies": {}}
        assert resolve_implementation(declared, scope, callback, "model") is declared
    for callback in (functools.partial(module.predict), module.Model(), module.Model().__call__,
                     staticmethod(module.predict),
                     functools.partial(staticmethod(module.predict)),
                     functools.lru_cache(maxsize=0)(module.predict),
                     functools.partial(functools.lru_cache(maxsize=0)(module.predict))):
        result = resolve_implementation(None, scope, callback, "model")
        assert json.loads(result.evidence_bytes)["callback"]["source"] == "model/model.py"
        outside = scope_at(tmp_path / "outside")
        with pytest.raises(ValueError, match="callback source"):
            resolve_implementation(None, outside, callback, "model")


def two_callbacks(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    filename = root / 'model.py'
    filename.write_text('def first(request): return -1.0\ndef second(request): return -2.0\n')
    spec = importlib.util.spec_from_file_location('synthetic_two_callbacks', filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, source_scope({'source_roots': {'model': str(root)}})


def test_distinct_saved_callbacks_have_distinct_implementation_identities(tmp_path, monkeypatch):
    module, scope = two_callbacks(tmp_path, monkeypatch)
    first = resolve_implementation(None, scope, module.first, 'same-profile')
    second = resolve_implementation(None, scope, module.second, 'same-profile')
    assert dict(first) != dict(second)


def test_existing_capture_rejects_a_different_saved_callback(tmp_path, monkeypatch):
    module, scope = two_callbacks(tmp_path, monkeypatch)
    first = resolve_implementation(None, scope, module.first, 'same-profile')
    with pytest.raises(ValueError, match='callback|implementation'):
        resolve_implementation(first, None, module.second, 'same-profile')


@pytest.mark.parametrize("field", ["callback", "environment", "library", "root_locations"])
def test_retained_evidence_cannot_be_replaced_without_changing_binding(tmp_path, monkeypatch, field):
    from autoforecast.implementation import verify_retained

    module, scope = two_callbacks(tmp_path, monkeypatch)
    first = resolve_implementation(None, scope, module.first, 'same-profile')
    retained = tmp_path / 'retained'
    retained.mkdir()
    first.retain(retained)
    evidence = retained / 'metadata/implementation/evidence.json'
    forged = json.loads(evidence.read_bytes())
    forged[field] = {"changed": True}
    evidence.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='evidence|binding|implementation'):
        verify_retained(retained, first, first.binding)


@pytest.mark.parametrize('fault', ['identity', 'binding', 'schema', 'empty', 'evidence_missing'])
def test_reload_rejects_each_independent_saved_source_fault(tmp_path, fault):
    first = capture(scope_at(tmp_path / 'source'))
    root = tmp_path / 'retained'
    root.mkdir()
    first.retain(root)
    identity, binding = dict(first), first.binding
    if fault == 'identity':
        identity['source_sha256'] = '0' * 64
    elif fault == 'binding':
        binding['manifest_sha256'] = '0' * 64
    elif fault in {'schema', 'empty'}:
        path = root / 'metadata/implementation/manifest.json'
        manifest = json.loads(path.read_bytes())
        manifest['schema' if fault == 'schema' else 'files'] = 'unsupported' if fault == 'schema' else {}
        payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        path.write_bytes(payload)
        identity['source_sha256'] = binding['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
    else:
        (root / 'metadata/implementation/evidence.json').unlink()
    with pytest.raises((ValueError, OSError)):
        verify_retained(root, identity, binding)


def test_worker_rechecks_importable_callback_with_relocated_scope(tmp_path, monkeypatch):

    scope = scope_at(tmp_path / 'source')
    first = capture(scope)
    root = tmp_path / 'retained'
    root.mkdir()
    first.retain(root)
    record = {'path': str(root), 'identity': dict(first), 'binding': first.binding, 'scope': scope}
    with pytest.raises(ValueError, match='callback source'):
        worker_implementation(record, capture)


@pytest.mark.parametrize('form', ['decorated', 'repeated_name'])
def test_distinct_saved_function_code_locations_have_distinct_selectors(tmp_path, monkeypatch, form):
    import importlib.util
    import sys

    root = tmp_path / 'saved'
    root.mkdir()
    path = root / 'model.py'
    if form == 'decorated':
        code = '''import functools

def shared(request): return 0.0
@functools.wraps(shared)
def first(request): return -1.0
@functools.wraps(shared)
def second(request): return -2.0
'''
    else:
        code = '''def shared(request): return -1.0
first = shared
def shared(request): return -2.0
second = shared
'''
    path.write_text(code)
    spec = importlib.util.spec_from_file_location('synthetic_code_location', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.first(None) == -1.0 and module.second(None) == -2.0
    scope = source_scope({'source_roots': {'model': str(root)}})
    first = resolve_implementation(None, scope, module.first, 'same-profile')
    second = resolve_implementation(None, scope, module.second, 'same-profile')
    first_record = json.loads(first.evidence_bytes)['callback']
    second_record = json.loads(second.evidence_bytes)['callback']
    assert first_record != second_record


@pytest.mark.parametrize('binding', [None, {'mode': 'declared'}])
def test_declared_reload_requires_no_source_artifacts(tmp_path, binding):
    verify_retained(tmp_path / 'absent', {}, binding)


def test_checkout_library_revision_tracks_actual_python_bytes_and_sorted_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import mlops.provenance as provenance

    import autoforecast.project as project

    monkeypatch.setattr(project, 'LIBRARY_ROOT', tmp_path)
    monkeypatch.setattr(provenance, 'capture_source_identity', lambda root, require_clean: SimpleNamespace(as_record=lambda: {'commit': 'a' * 40}))
    package = tmp_path / 'autoforecast'
    package.mkdir()
    model = package / 'model.py'
    model.write_bytes(b'VALUE = 1\n')
    first = project.library_revision()
    expected = {'autoforecast/model.py': hashlib.sha256(model.read_bytes()).hexdigest()}
    assert first['schema'] == 'autoforecast.checkout-library.v1'
    assert first['code_sha256'] == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    model.write_bytes(b'VALUE = 2\n')
    assert project.library_revision()['code_sha256'] != first['code_sha256']
    model.write_bytes(b'VALUE = 1\n')
    (package / 'ignored.txt').write_text('not Python source')
    assert project.library_revision() == first


def test_installed_library_revision_authenticates_wheel_before_reporting(tmp_path, monkeypatch):
    import autoforecast.project as project
    from autoforecast.test_build_identity import installed_fixture

    package, _, _ = installed_fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(project, 'LIBRARY_ROOT', tmp_path)
    result = project.library_revision()
    assert result['schema'] == 'autoforecast.installed-library.v1'
    assert result == project.installed_library_identity().as_record()
    (package / 'project.py').write_text('changed')
    with pytest.raises(ValueError, match='file changed'):
        project.library_revision()


@pytest.mark.parametrize("partial", ["empty", "source", "manifest", "evidence"])
def test_retaining_source_completes_interrupted_writes_without_replacing_bytes(tmp_path, partial):
    from mlops._safe_files import publish_bytes_set

    first = capture(scope_at(tmp_path / "source"))
    root = tmp_path / "retained"
    root.mkdir()
    prefix = "metadata/implementation/"
    payloads = {
        "source": {prefix + "source/" + p: b for p, b in first.source_bytes},
        "manifest": {prefix + "manifest.json": first.manifest_bytes},
        "evidence": {prefix + "evidence.json": first.evidence_bytes},
    }
    if partial != "empty":
        publish_bytes_set(root, payloads[partial])
    first.retain(root)
    first.retain(root)
    verify_retained(root, first, first.binding)
    (root / (prefix + "evidence.json")).write_bytes(b"changed")
    with pytest.raises((ValueError, OSError), match="differs"):
        first.retain(root)
