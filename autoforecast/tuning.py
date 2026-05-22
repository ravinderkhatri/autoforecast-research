"""Framework-neutral hyperparameter contracts embedded in a model YAML.

The YAML here describes only the parameters a study may sample.  It is deliberately not an
execution plan: the evaluation manifest, data/profile facts, optimiser budget, trial history and
chosen result have different lifecycles and belong to a governed run artifact.  Optuna and Ray
can both compile these three primitive domain types without the catalog importing either optional
package.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Protocol, cast

from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast.tuning_support import tuning_support

__all__ = [
    "PARAMETER_EXECUTION_COMPATIBILITY",
    "SearchDomain",
    "ParameterConstraint",
    "TuningError",
    "TuningParameter",
    "TuningSpec",
    "load_tuning_spec",
    "loads_tuning_spec",
    "materialize_manual_parameters",
    "materialize_runtime_parameters",
    "materialize_trial_parameters",
    "ray_param_space",
    "suggest_optuna_parameters",
]


PARAMETER_EXECUTION_COMPATIBILITY = "autoforecast.typed-parameter-execution.v1"
_MAX_DISCRETE_DOMAIN_SIZE = 100_000
_NAME = re.compile(r"^(model|training)\.[a-z][a-z0-9_]{0,63}$")
_ROOT_KEYS = frozenset({"schema_version", "model", "hyperparameters"})
_ROOT_KEYS_WITH_CONSTRAINTS = _ROOT_KEYS | {"constraints"}
_OPTIONAL_IMPLEMENTATION = frozenset({"implementation"})
_PARAMETER_KEYS_V1 = frozenset({"default", "search"})
_PARAMETER_KEYS_V2 = frozenset({"default", "valid", "search"})


class TuningError(ValueError):
    """A tuning YAML is malformed, unsafe, or internally inconsistent."""


class _OptunaTrial(Protocol):
    """The stable subset of Optuna's trial API this project needs."""

    def suggest_categorical(self, name: str, choices: tuple[object, ...]) -> object: ...

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float: ...


@dataclass(frozen=True)
class SearchDomain:
    """One portable categorical, integer, or floating-point parameter domain."""

    kind: str
    values: tuple[object, ...] | None = None
    low: int | float | None = None
    high: int | float | None = None
    step: int | None = None
    log: bool = False


@dataclass(frozen=True)
class TuningParameter:
    """A setting with separate runtime-valid and narrower HPO sampling domains."""

    name: str
    default: object
    valid: SearchDomain
    search: SearchDomain


@dataclass(frozen=True)
class ParameterConstraint:
    """One cross-parameter invariant evaluated after scalar-domain validation."""

    kind: str
    dividend: str
    divisor: str


@dataclass(frozen=True)
class TuningSpec:
    """A hashable search-space definition embedded in one candidate model file."""

    model_id: str
    adapter: str
    parameters: tuple[TuningParameter, ...]
    constraints: tuple[ParameterConstraint, ...]
    source_path: pathlib.Path
    source_sha256: str

    def defaults(self) -> dict[str, dict[str, object]]:
        """Return defaults grouped for the model-spec and training-control bridge.

        For the current custom model, a suite builds ``ForecastModelSpec(**model)`` and passes
        ``training["learning_rate"]`` to the adapter as ``lr``.  The bridge remains adapter-owned:
        YAML may name supported settings but must never bypass ``ResolvedBuild`` validation.
        """
        result: dict[str, dict[str, object]] = {"model": {}, "training": {}}
        for parameter in self.parameters:
            namespace, name = parameter.name.split(".", maxsplit=1)
            result[namespace][name] = parameter.default
        return result


def _contains(domain: SearchDomain, value: object) -> bool:
    if domain.kind == "categorical":
        return domain.values is not None and any(
            _same_value(value, choice) for choice in domain.values
        )
    if domain.kind == "int":
        return (
            type(value) is int
            and domain.low is not None
            and domain.step is not None
            and domain.low <= value
            and (domain.high is None or value <= domain.high)
            and (value - domain.low) % domain.step == 0
        )
    if domain.kind == "float":
        if type(value) not in (int, float) or domain.low is None or domain.high is None:
            return False
        numeric_value = cast(int | float, value)
        return math.isfinite(float(numeric_value)) and domain.low <= numeric_value <= domain.high
    raise AssertionError(f"unsupported validated domain kind {domain.kind!r}")


def materialize_trial_parameters(
    spec: TuningSpec, values: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Validate one complete sampled configuration and group it for an adapter-owned bridge.

    A study may not silently use a model default for a missing suggested value, nor may it smuggle
    an undeclared setting into an adapter.  This function is deliberately independent of Optuna
    and Ray so an objective has one validation path under either backend.
    """
    expected = {parameter.name for parameter in spec.parameters}
    if set(values) != expected:
        raise TuningError(
            f"trial parameters must be exactly {sorted(expected)}, got {sorted(values)}"
        )
    result = materialize_runtime_parameters(spec, values)
    for parameter in spec.parameters:
        value = values[parameter.name]
        if not _contains(parameter.search, value):
            raise TuningError(
                f"trial parameter {parameter.name!r}={value!r} is outside its declared range"
            )
    return result


def materialize_runtime_parameters(
    spec: TuningSpec, values: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Validate one complete executable map without imposing the HPO search range.

    Runtime validity and search policy have different lifecycles. A deliberate manual value or a
    previously selected value may be safe outside today's deliberately narrow search range, while
    an HPO trial must still remain inside that range.
    """
    if type(values) is not dict:
        raise TuningError("runtime parameters must be a mapping")
    expected = {parameter.name for parameter in spec.parameters}
    if set(values) != expected:
        raise TuningError(
            f"runtime parameters must be exactly {sorted(expected)}, got {sorted(values)}"
        )
    result: dict[str, dict[str, object]] = {"model": {}, "training": {}}
    for parameter in spec.parameters:
        value = values[parameter.name]
        if not _contains(parameter.valid, value):
            raise TuningError(
                f"runtime parameter {parameter.name!r}={value!r} is outside its valid domain"
            )
        namespace, name = parameter.name.split(".", maxsplit=1)
        result[namespace][name] = value
    _validate_constraints(spec, values)
    return result


def _validate_constraints(spec: TuningSpec, values: dict[str, object]) -> None:
    """Apply profile-owned relationships without embedding model names in the loader."""
    for constraint in spec.constraints:
        dividend = values[constraint.dividend]
        divisor = values[constraint.divisor]
        if constraint.kind == "divisible":
            if (
                type(dividend) is not int
                or type(divisor) is not int
                or divisor == 0
                or dividend % divisor
            ):
                raise TuningError(
                    f"runtime parameter {constraint.dividend!r}={dividend!r} must be divisible "
                    f"by non-zero {constraint.divisor!r}={divisor!r}"
                )
        elif constraint.kind == "less_equal":
            if (
                type(dividend) not in {int, float}
                or type(divisor) not in {int, float}
                or cast(int | float, dividend) > cast(int | float, divisor)
            ):
                raise TuningError(
                    f"runtime parameter {constraint.dividend!r}={dividend!r} must be less than "
                    f"or equal to {constraint.divisor!r}={divisor!r}"
                )
        else:  # pragma: no cover - the strict loader rejects unknown constraint kinds
            raise AssertionError(f"unsupported validated constraint {constraint.kind!r}")


def _integer_domain_values(domain: SearchDomain, label: str) -> tuple[int, ...]:
    """Enumerate one finite integer search domain used by a cross-parameter constraint."""
    if domain.kind == "categorical" and domain.values is not None:
        if not all(type(value) is int for value in domain.values):
            raise TuningError(f"{label} must contain only integers")
        return cast(tuple[int, ...], domain.values)
    if domain.kind == "int":
        assert isinstance(domain.low, int)
        assert isinstance(domain.high, int)
        assert isinstance(domain.step, int)
        count = ((domain.high - domain.low) // domain.step) + 1
        if count > _MAX_DISCRETE_DOMAIN_SIZE:
            raise TuningError(
                f"{label} contains more than {_MAX_DISCRETE_DOMAIN_SIZE} discrete values"
            )
        return tuple(range(domain.low, domain.high + 1, domain.step))
    raise TuningError(f"{label} must be a finite integer domain")


def _validate_search_constraints(spec: TuningSpec) -> None:
    """Ensure an independent sampler cannot emit a structurally invalid combination."""
    parameters = {parameter.name: parameter for parameter in spec.parameters}
    for constraint in spec.constraints:
        dividends = _integer_domain_values(
            parameters[constraint.dividend].search,
            f"search domain for {constraint.dividend!r}",
        )
        divisors = _integer_domain_values(
            parameters[constraint.divisor].search,
            f"search domain for {constraint.divisor!r}",
        )
        if constraint.kind == "less_equal":
            if max(dividends) > min(divisors):
                raise TuningError(
                    "independent HPO search domains violate less_equal constraint "
                    f"{constraint.dividend!r} <= {constraint.divisor!r}"
                )
            continue
        if constraint.kind != "divisible":  # pragma: no cover - strict loader owns kinds
            raise AssertionError(f"unsupported validated constraint {constraint.kind!r}")
        if 0 in divisors:
            raise TuningError(f"search domain for divisor {constraint.divisor!r} must be non-zero")
        common_dividend = math.gcd(*dividends)
        if any(common_dividend % divisor for divisor in divisors):
            raise TuningError(
                "independent HPO search domains violate divisible constraint "
                f"{constraint.dividend!r} / {constraint.divisor!r}"
            )


def materialize_manual_parameters(
    spec: TuningSpec, overrides: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Merge a partial manual experiment over profile defaults and validate the full result."""
    if type(overrides) is not dict:
        raise TuningError("manual parameter overrides must be a mapping")
    expected = {parameter.name for parameter in spec.parameters}
    unknown = sorted(set(overrides) - expected)
    if unknown:
        raise TuningError(f"manual parameter overrides are undeclared: {unknown}")
    values = {parameter.name: parameter.default for parameter in spec.parameters}
    values.update(overrides)
    return materialize_runtime_parameters(spec, values)


def suggest_optuna_parameters(spec: TuningSpec, trial: _OptunaTrial) -> dict[str, object]:
    """Suggest one complete trial from ``spec`` through Optuna's public trial API.

    The caller installs ``.[hpo-optuna]`` and owns study construction, storage, objective metric,
    seed, and trial budget.  Keeping those experiment facts out of model YAML prevents a model
    edit from silently changing a registered evaluation protocol.
    """
    values: dict[str, object] = {}
    for parameter in spec.parameters:
        domain = parameter.search
        if domain.kind == "categorical":
            assert domain.values is not None
            values[parameter.name] = trial.suggest_categorical(parameter.name, domain.values)
        elif domain.kind == "int":
            assert isinstance(domain.low, int) and isinstance(domain.high, int)
            assert isinstance(domain.step, int)
            values[parameter.name] = trial.suggest_int(
                parameter.name, domain.low, domain.high, step=domain.step
            )
        elif domain.kind == "float":
            assert domain.low is not None and domain.high is not None
            values[parameter.name] = trial.suggest_float(
                parameter.name, float(domain.low), float(domain.high), log=domain.log
            )
        else:  # pragma: no cover - load_tuning_spec refuses unknown kinds
            raise AssertionError(f"unsupported validated domain kind {domain.kind!r}")
    materialize_trial_parameters(spec, values)
    return values


def ray_param_space(spec: TuningSpec) -> dict[str, object]:
    """Compile ``spec`` to Ray Tune's public ``param_space`` representation.

    The values remain small, serializable scalars.  Dataset objects, Spark sessions, and metric
    state must be constructed inside a Ray trainable rather than placed in ``param_space``.
    """
    try:
        from ray import tune
    except ImportError as exc:  # pragma: no cover - installation-dependent boundary
        raise TuningError("Ray Tune requires: pip install '.[hpo-ray]'") from exc
    values: dict[str, object] = {}
    for parameter in spec.parameters:
        domain = parameter.search
        if domain.kind == "categorical":
            assert domain.values is not None
            values[parameter.name] = tune.choice(list(domain.values))
        elif domain.kind == "int":
            assert isinstance(domain.low, int) and isinstance(domain.high, int)
            assert isinstance(domain.step, int)
            # qrandint quantises around zero rather than around ``low``.  For an offset
            # domain such as 31, 63, 95, 127 it would emit only invalid values.
            if domain.step == 1:
                values[parameter.name] = tune.randint(domain.low, domain.high + 1)
            else:
                count = ((domain.high - domain.low) // domain.step) + 1
                if count > _MAX_DISCRETE_DOMAIN_SIZE:
                    raise TuningError(
                        f"{parameter.name} contains more than "
                        f"{_MAX_DISCRETE_DOMAIN_SIZE} discrete search values"
                    )
                values[parameter.name] = tune.choice(
                    list(range(domain.low, domain.high + 1, domain.step))
                )
        elif domain.kind == "float":
            assert domain.low is not None and domain.high is not None
            if domain.log:
                values[parameter.name] = tune.loguniform(float(domain.low), float(domain.high))
            else:
                values[parameter.name] = tune.uniform(float(domain.low), float(domain.high))
        else:  # pragma: no cover - load_tuning_spec refuses unknown kinds
            raise AssertionError(f"unsupported validated domain kind {domain.kind!r}")
    return values


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TuningError(f"{label} must be a mapping")
    return value


def _exact_keys(value: dict[str, object], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise TuningError(f"{label} must contain exactly {sorted(keys)}, got {sorted(value)}")


def _finite_number(value: object, label: str) -> int | float:
    if type(value) not in (int, float):
        raise TuningError(f"{label} must be a finite built-in number")
    number = cast(int | float, value)
    try:
        finite = math.isfinite(float(number))
    except OverflowError:
        finite = False
    if not finite:
        raise TuningError(f"{label} must be a finite built-in number")
    return number


def _same_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _domain(
    value: object,
    default: object,
    label: str,
    *,
    allow_unbounded_int: bool = False,
) -> SearchDomain:
    body = _mapping(value, label)
    kind = body.get("kind")
    if kind == "categorical":
        _exact_keys(body, frozenset({"kind", "values"}), label)
        choices = body["values"]
        if type(choices) is not list or not choices:
            raise TuningError(f"{label}.values must be a non-empty list")
        if len(choices) > _MAX_DISCRETE_DOMAIN_SIZE:
            raise TuningError(
                f"{label}.values may contain at most {_MAX_DISCRETE_DOMAIN_SIZE} choices"
            )
        if any(type(choice) not in (str, int, float) for choice in choices):
            raise TuningError(f"{label}.values may contain only built-in strings or numbers")
        if any(type(choice) is float and not math.isfinite(choice) for choice in choices):
            raise TuningError(f"{label}.values numeric choices must be finite")
        if len({(type(choice), choice) for choice in choices}) != len(choices):
            raise TuningError(f"{label}.values must not contain duplicates")
        if not any(_same_value(default, choice) for choice in choices):
            raise TuningError(f"{label} must include the parameter default")
        return SearchDomain(kind="categorical", values=tuple(choices))
    if kind == "int":
        finite_keys = frozenset({"kind", "low", "high", "step"})
        unbounded_keys = frozenset({"kind", "low", "step"})
        if set(body) != finite_keys and not (
            allow_unbounded_int and set(body) == unbounded_keys
        ):
            expected = (
                "['kind', 'low', 'step'] or ['high', 'kind', 'low', 'step']"
                if allow_unbounded_int
                else "['high', 'kind', 'low', 'step']"
            )
            raise TuningError(f"{label} must contain exactly {expected}, got {sorted(body)}")
        low, step = body["low"], body["step"]
        has_high = "high" in body
        high = body["high"] if has_high else None
        bounds = (low, high, step) if has_high else (low, step)
        if any(type(item) is not int for item in bounds):
            raise TuningError(f"{label} integer bounds and step must be built-in integers")
        low, step = cast(int, low), cast(int, step)
        high = cast(int | None, high)
        if step < 1 or (
            high is not None and (low > high or (high - low) % step)
        ):
            raise TuningError(f"{label} needs low <= high and a step that reaches high")
        if (
            type(default) is not int
            or default < low
            or (high is not None and default > high)
            or (default - low) % step
        ):
            raise TuningError(f"{label} must contain the integer parameter default")
        return SearchDomain(kind="int", low=low, high=high, step=step)
    if kind == "float":
        _exact_keys(body, frozenset({"kind", "low", "high", "log"}), label)
        low = _finite_number(body["low"], f"{label}.low")
        high = _finite_number(body["high"], f"{label}.high")
        log = body["log"]
        if type(log) is not bool or low >= high or (log and low <= 0):
            raise TuningError(f"{label} has invalid float bounds or log flag")
        numeric_default = _finite_number(default, f"{label} default")
        if not low <= numeric_default <= high:
            raise TuningError(f"{label} must contain the parameter default")
        return SearchDomain(kind="float", low=low, high=high, log=log)
    raise TuningError(f"{label}.kind must be categorical, int, or float")


def _domain_is_subset(search: SearchDomain, valid: SearchDomain) -> bool:
    """Return whether every value emitted by *search* is admitted by *valid*."""
    if search.kind == "categorical":
        assert search.values is not None
        return all(_contains(valid, value) for value in search.values)
    if search.kind == "int" and valid.kind == "int":
        assert isinstance(search.low, int) and isinstance(search.high, int)
        assert isinstance(search.step, int)
        assert isinstance(valid.low, int)
        assert isinstance(valid.step, int)
        return (
            _contains(valid, search.low)
            and _contains(valid, search.high)
            and search.step % valid.step == 0
        )
    if search.kind == "int" and valid.kind == "float":
        assert search.low is not None and search.high is not None
        return _contains(valid, search.low) and _contains(valid, search.high)
    if search.kind == "int" and valid.kind == "categorical":
        assert isinstance(search.low, int) and isinstance(search.high, int)
        assert isinstance(search.step, int)
        return all(
            _contains(valid, value) for value in range(search.low, search.high + 1, search.step)
        )
    if search.kind == "float" and valid.kind == "float":
        assert search.low is not None and search.high is not None
        return _contains(valid, search.low) and _contains(valid, search.high)
    return False


def load_tuning_spec(path: str | pathlib.Path) -> TuningSpec:
    """Read the finite HPO space from one ``configs/models/**/model.yaml``.

    The caller owns framework compilation and must record this file hash beside every study.  A
    malformed source-controlled space is refused before a trial reaches any data or hardware.
    """
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise TuningError(f"model config must be a regular file, not {source}")
    raw = source.read_bytes()
    return loads_tuning_spec(raw, source_path=source)


def loads_tuning_spec(
    raw: bytes, *, source_path: str | pathlib.Path = "<retained-model-profile>"
) -> TuningSpec:
    """Parse already-authenticated model-profile bytes without reopening a mutable path."""
    source = pathlib.Path(source_path)
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as exc:
        raise TuningError(f"invalid YAML in {source}: {exc}") from exc
    root = _mapping(value, str(source))
    root_without_implementation = frozenset(root) - _OPTIONAL_IMPLEMENTATION
    if root_without_implementation not in {
        _ROOT_KEYS,
        _ROOT_KEYS_WITH_CONSTRAINTS,
        _ROOT_KEYS - {"hyperparameters"},
    }:
        raise TuningError(
            f"{source} must contain model/schema_version with optional hyperparameters"
        )
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise TuningError(f"{source}: schema_version must be integer 1 or 2")
    if schema_version == 1 and "constraints" in root:
        raise TuningError(f"{source}: constraints require schema_version 2")
    model = _mapping(root.get("model"), f"{source}: model")
    model_id = model.get("id")
    if type(model_id) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", model_id):
        raise TuningError(f"{source}: model.id is invalid")
    adapter = model.get("adapter")
    if type(adapter) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", adapter):
        raise TuningError(f"{source}: model.adapter is invalid")
    if not isinstance(adapter, str):
        raise TuningError(f"{source}: model.adapter is invalid")
    try:
        support = tuning_support(adapter)
    except ValueError as error:
        raise TuningError(f"{source}: model.adapter is not HPO-registered: {adapter!r}") from error
    raw_parameters = root.get("hyperparameters")
    if raw_parameters is None and not support.supported:
        raise TuningError(f"{model_id} is intentionally not tuneable: {support.reason}")
    if raw_parameters is None:
        raise TuningError(f"{model_id} has no declared tuneable execution settings")
    if not support.supported:
        raise TuningError(f"{model_id} is intentionally not tuneable: {support.reason}")
    raw_parameters = _mapping(raw_parameters, f"{source}: hyperparameters")
    if not raw_parameters:
        raise TuningError(f"{source}: hyperparameters must not be empty")
    parameters: list[TuningParameter] = []
    for name, raw_parameter in sorted(raw_parameters.items()):
        if type(name) is not str or not _NAME.fullmatch(name):
            raise TuningError(f"{source}: invalid parameter name {name!r}")
        parameter = _mapping(raw_parameter, f"{source}: hyperparameters.{name}")
        expected_keys = _PARAMETER_KEYS_V1 if schema_version == 1 else _PARAMETER_KEYS_V2
        _exact_keys(parameter, expected_keys, f"{source}: hyperparameters.{name}")
        default = parameter["default"]
        search = _domain(parameter["search"], default, f"{source}: hyperparameters.{name}.search")
        if (
            search.kind == "int"
            and isinstance(search.low, int)
            and isinstance(search.high, int)
            and isinstance(search.step, int)
            and ((search.high - search.low) // search.step) + 1 > _MAX_DISCRETE_DOMAIN_SIZE
        ):
            raise TuningError(
                f"{source}: hyperparameters.{name}.search may contain at most "
                f"{_MAX_DISCRETE_DOMAIN_SIZE} discrete values"
            )
        valid = (
            search
            if schema_version == 1
            else _domain(
                parameter["valid"],
                default,
                f"{source}: hyperparameters.{name}.valid",
                allow_unbounded_int=True,
            )
        )
        if not _domain_is_subset(search, valid):
            raise TuningError(f"{source}: hyperparameters.{name}.search must be a subset of valid")
        parameters.append(
            TuningParameter(
                name=name,
                default=default,
                valid=valid,
                search=search,
            )
        )
    names = {parameter.name for parameter in parameters}
    constraints: list[ParameterConstraint] = []
    raw_constraints = root.get("constraints", [])
    if type(raw_constraints) is not list:
        raise TuningError(f"{source}: constraints must be a list")
    for index, raw_constraint in enumerate(raw_constraints):
        label = f"{source}: constraints[{index}]"
        constraint = _mapping(raw_constraint, label)
        _exact_keys(
            constraint,
            frozenset({"kind", "dividend", "divisor"}),
            label,
        )
        kind = constraint["kind"]
        dividend, divisor = constraint["dividend"], constraint["divisor"]
        if type(kind) is not str or kind not in {"divisible", "less_equal"}:
            raise TuningError(f"{label}.kind must be divisible or less_equal")
        if (
            type(dividend) is not str
            or type(divisor) is not str
            or dividend == divisor
            or dividend not in names
            or divisor not in names
        ):
            raise TuningError(f"{label} must reference two distinct declared parameters")
        constraints.append(ParameterConstraint(kind, dividend, divisor))
    if len(set(constraints)) != len(constraints):
        raise TuningError(f"{source}: constraints must not contain duplicates")
    spec = TuningSpec(
        model_id,
        adapter,
        tuple(parameters),
        tuple(constraints),
        source,
        hashlib.sha256(raw).hexdigest(),
    )
    _validate_search_constraints(spec)
    materialize_runtime_parameters(
        spec, {parameter.name: parameter.default for parameter in spec.parameters}
    )
    return spec
