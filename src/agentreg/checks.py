"""Deterministic checks: cheap, exact and explainable.

Each check is a pure function (expectations, trace) -> CheckResult | None.
Returning None means "not configured for this case", so reports only show
checks that were actually asked for. Register custom checks with @check.
"""

from __future__ import annotations

import re
from typing import Callable

from agentreg.schema import CheckResult, Expectations, Trace

CheckFn = Callable[[Expectations, Trace], CheckResult | None]
REGISTRY: dict[str, CheckFn] = {}


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    def register(fn: CheckFn) -> CheckFn:
        REGISTRY[name] = fn
        return fn

    return register


def run_checks(expect: Expectations, trace: Trace) -> list[CheckResult]:
    results = []
    for name, fn in REGISTRY.items():
        result = fn(expect, trace)
        if result is not None:
            result.name = name
            results.append(result)
    return results


def _ok(detail: str = "") -> CheckResult:
    return CheckResult(name="", passed=True, detail=detail)


def _fail(detail: str) -> CheckResult:
    return CheckResult(name="", passed=False, detail=detail)


def _output_text(trace: Trace) -> str:
    return trace.output if isinstance(trace.output, str) else str(trace.output or "")


# --------------------------------------------------------------------------- built-ins


@check("completed")
def completed(expect: Expectations, trace: Trace) -> CheckResult:
    return _fail(f"agent raised {trace.error}") if trace.error else _ok()


@check("required_tools")
def required_tools(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.required_tools:
        return None
    called = set(trace.tool_sequence)
    missing = [t for t in expect.required_tools if t not in called]
    return _fail(f"not called: {', '.join(missing)}") if missing else _ok()


@check("forbidden_tools")
def forbidden_tools(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.forbidden_tools:
        return None
    hit = sorted(set(expect.forbidden_tools) & set(trace.tool_sequence))
    return _fail(f"called: {', '.join(hit)}") if hit else _ok()


@check("tool_order")
def tool_order(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.tool_order:
        return None
    it = iter(trace.tool_sequence)
    in_order = all(name in it for name in expect.tool_order)  # subsequence test
    if in_order:
        return _ok()
    return _fail(f"expected order {' → '.join(expect.tool_order)}, got {' → '.join(trace.tool_sequence) or '(no tools)'}")


@check("tool_args")
def tool_args(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.tool_args:
        return None
    problems = []
    for tool, patterns in expect.tool_args.items():
        calls = [t for t in trace.tool_calls if t.name == tool]
        matched = any(
            all(re.search(rx, str(c.args.get(arg, ""))) for arg, rx in patterns.items()) for c in calls
        )
        if not matched:
            problems.append(f"{tool} never called with {patterns}")
    return _fail("; ".join(problems)) if problems else _ok()


@check("no_tool_errors")
def no_tool_errors(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.no_tool_errors:
        return None
    errors = [f"{t.name}: {t.error}" for t in trace.tool_calls if t.error]
    return _fail("; ".join(errors)) if errors else _ok()


def _budget(actual: float | None, limit: float | None, unit: str) -> CheckResult | None:
    if limit is None:
        return None
    if actual is None:
        return _ok("not measured")
    detail = f"{actual:g}{unit} / limit {limit:g}{unit}"
    return _ok(detail) if actual <= limit else _fail(detail)


@check("max_tokens")
def max_tokens(expect: Expectations, trace: Trace) -> CheckResult | None:
    return _budget(trace.total_tokens, expect.max_tokens, " tok")


@check("max_cost_usd")
def max_cost(expect: Expectations, trace: Trace) -> CheckResult | None:
    return _budget(trace.cost_usd, expect.max_cost_usd, " USD")


@check("max_latency_s")
def max_latency(expect: Expectations, trace: Trace) -> CheckResult | None:
    return _budget(round(trace.latency_s, 2), expect.max_latency_s, "s")


@check("max_steps")
def max_steps(expect: Expectations, trace: Trace) -> CheckResult | None:
    return _budget(len(trace.steps), expect.max_steps, " steps")


@check("output_contains")
def output_contains(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.output_contains:
        return None
    text = _output_text(trace).lower()
    missing = [s for s in expect.output_contains if s.lower() not in text]
    return _fail(f"missing: {missing}") if missing else _ok()


@check("output_regex")
def output_regex(expect: Expectations, trace: Trace) -> CheckResult | None:
    if not expect.output_regex:
        return None
    text = _output_text(trace)
    missing = [rx for rx in expect.output_regex if not re.search(rx, text, re.MULTILINE)]
    return _fail(f"no match for: {missing}") if missing else _ok()
