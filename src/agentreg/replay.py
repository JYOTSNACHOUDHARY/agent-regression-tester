"""Replay test cases against the current agent and classify each outcome."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any, Callable, Literal

from agentreg.checks import run_checks
from agentreg.judge import Judge, format_evidence
from agentreg.recorder import record_run
from agentreg.schema import Attempt, CaseResult, Report, Status, TestCase, ToolStub, Verdict
from agentreg.stubs import stubbing

ToolMode = Literal["live", "stub"]


def run_attempt(
    case: TestCase,
    target_name: str,
    run: Callable[[Any], Any],
    tool_mode: ToolMode,
    judge: Judge | None,
) -> Attempt:
    ctx = stubbing(case.tool_stubs) if tool_mode == "stub" else nullcontext(None)
    with ctx as store:
        trace = record_run(target_name, run, case.input, case.input)

    checks = run_checks(case.expect, trace)
    verdict = None
    if judge is not None and case.judge.enabled and trace.error is None:
        verdict = judge.compare(
            case.judge.criteria,
            case.input,
            case.reference.output,
            trace.output,
            reference_evidence=format_evidence(case.reference.tool_results),
            candidate_evidence=format_evidence(trace.tool_calls),
        )

    return Attempt(
        trace=trace,
        checks=checks,
        judge=verdict,
        stub_misses=list(store.misses) if store is not None else [],
    )


def classify(attempts: list[Attempt]) -> Status:
    """Majority vote across repeats, so one unlucky sample is 'flaky', not 'regressed'."""
    n = len(attempts)
    failures = sum(a.failed for a in attempts)
    if failures == 0:
        better = sum(a.judge is not None and a.judge.verdict == Verdict.better for a in attempts)
        return Status.improved if better * 2 > n else Status.passed
    if failures * 2 > n:
        if all(a.trace.error for a in attempts):
            return Status.error
        return Status.regressed
    return Status.flaky


def run_suite(
    cases: list[TestCase],
    target_name: str,
    run: Callable[[Any], Any],
    tool_mode: ToolMode = "live",
    repeats: int = 1,
    judge: Judge | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Report:
    report = Report(target=target_name, tool_mode=tool_mode, repeats=repeats)
    for case in cases:
        attempts = [run_attempt(case, target_name, run, tool_mode, judge) for _ in range(repeats)]
        result = CaseResult(
            case_id=case.id, status=classify(attempts), reference=case.reference, attempts=attempts
        )
        report.results.append(result)
        if on_result:
            on_result(result)
    if judge is not None:
        report.judge_cost_usd = judge.total_cost_usd
    return report


def accept(case: TestCase, result: CaseResult) -> TestCase:
    """Snapshot-style update: make the latest run the new reference for this case."""
    trace = result.attempts[0].trace
    case = case.model_copy(deep=True)
    case.reference = trace.summary()
    seen = {s.name + json.dumps(s.args, sort_keys=True, default=str) for s in case.tool_stubs}
    for call in trace.tool_calls:
        key = call.name + json.dumps(call.args, sort_keys=True, default=str)
        if key not in seen and call.error is None:
            case.tool_stubs.append(ToolStub(name=call.name, args=call.args, output=call.output, artifact=call.artifact))
            seen.add(key)
    return case
