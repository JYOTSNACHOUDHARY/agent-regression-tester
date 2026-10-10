"""Render a Report as Markdown (for PR comments) and JSON (for machines)."""

from __future__ import annotations

from collections import Counter

from agentreg.schema import CaseResult, Report, RunSummary, Status, Trace

MARKER = "<!-- agentreg-report -->"
MAX_OUTPUT_CHARS = 1500

ICON = {
    Status.regressed: "❌",
    Status.error: "💥",
    Status.flaky: "⚠️",
    Status.improved: "⬆️",
    Status.passed: "✅",
}
ORDER = [Status.regressed, Status.error, Status.flaky, Status.improved, Status.passed]


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _delta(before: float | None, after: float | None, fmt: str = "{:,.0f}", unit: str = "") -> str:
    if before is None or after is None:
        return "n/a"
    text = f"{fmt.format(before)}{unit} → {fmt.format(after)}{unit}"
    if before:
        pct = (after - before) / before * 100
        if abs(pct) >= 1:
            text += f" ({pct:+.0f}%)"
    return text


def tool_diff(reference: list[str], now: list[str]) -> str:
    """Compare tool calls as multisets: what disappeared, what appeared."""
    ref, cur = Counter(reference), Counter(now)
    removed = ref - cur
    added = cur - ref
    if not removed and not added:
        return f"unchanged ({len(now)} calls)"
    parts = [f"−{name}" + (f" ×{n}" if n > 1 else "") for name, n in sorted(removed.items())]
    parts += [f"+{name}" + (f" ×{n}" if n > 1 else "") for name, n in sorted(added.items())]
    return " ".join(parts)


def _judge_cell(result: CaseResult) -> str:
    verdicts = [a.judge.verdict.value for a in result.attempts if a.judge]
    if not verdicts:
        return "–"
    counts = Counter(verdicts)
    cell = ", ".join(f"{v}" + (f" ×{n}" if n > 1 else "") for v, n in counts.most_common())
    judge = result.representative.judge
    if judge and judge.criteria and judge.reference_passed is not None:
        total = len(judge.criteria)
        cell += f" ({sum(c.passed for c in judge.criteria)}/{total} vs ref {judge.reference_passed}/{total})"
    return cell


def _headline(report: Report) -> str:
    parts = []
    for status in ORDER:
        n = report.count(status)
        if n or status in (Status.regressed, Status.passed):
            label = status.value
            parts.append(f"{ICON[status]} {n} {label}")
    verdict = "**Regressions found**" if report.has_regressions else "**No regressions**"
    return f"{verdict} · " + " · ".join(parts)


def _details(result: CaseResult) -> str:
    attempt = result.representative
    trace: Trace = attempt.trace
    ref: RunSummary = result.reference
    lines = [f"<details><summary>{ICON[result.status]} <b>{result.case_id}</b> — {result.status.value}</summary>", ""]

    if len(result.attempts) > 1:
        failed = sum(a.failed for a in result.attempts)
        lines.append(f"Failed **{failed} of {len(result.attempts)}** repeats.\n")
    if trace.error:
        lines.append(f"**Agent error:** `{trace.error}`\n")

    failed_checks = [c for c in attempt.checks if not c.passed]
    if failed_checks:
        lines.append("**Failed checks**")
        lines += [f"- `{c.name}`: {c.detail}" for c in failed_checks]
        lines.append("")

    lines.append("**Tool calls**")
    lines.append(f"- reference: {', '.join(ref.tool_sequence) or '(none)'}")
    lines.append(f"- now: {', '.join(trace.tool_sequence) or '(none)'}")
    if attempt.stub_misses:
        lines.append(f"- new calls with no recorded stub (ran live): {', '.join(f'`{m}`' for m in attempt.stub_misses)}")
    lines.append("")

    if attempt.judge and attempt.judge.criteria:
        lines.append(f"**Judge: {attempt.judge.verdict.value}**")
        for c in attempt.judge.criteria:
            lines.append(f"- {'✅' if c.passed else '❌'} {c.criterion} — {c.reason}")
        lines.append("")

    output = trace.output if isinstance(trace.output, str) else str(trace.output)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n… (truncated)"
    lines += ["**Output now**", "", "```text", output, "```", "", "</details>", ""]
    return "\n".join(lines)


def to_markdown(report: Report) -> str:
    results = sorted(report.results, key=lambda r: ORDER.index(r.status))
    lines = [
        MARKER,
        "## 🧪 Agent regression report",
        "",
        _headline(report),
        "",
        f"`{report.target}` · {len(report.results)} cases · tools: **{report.tool_mode}** · "
        f"repeats: {report.repeats} · agent cost {_money(report.agent_cost_usd)} · "
        f"judge cost {_money(report.judge_cost_usd)}",
        "",
        "| Case | Status | Tool calls vs reference | Tokens | Latency | Judge |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.representative.trace
        lines.append(
            f"| {r.case_id} | {ICON[r.status]} {r.status.value} "
            f"| {tool_diff(r.reference.tool_sequence, t.tool_sequence)} "
            f"| {_delta(r.reference.total_tokens, t.total_tokens)} "
            f"| {_delta(r.reference.latency_s, t.latency_s, '{:.1f}', 's')} "
            f"| {_judge_cell(r)} |"
        )
    lines.append("")
    for r in results:
        if r.status != Status.passed:
            lines.append(_details(r))
    lines.append("<sub>Generated by agentreg · deterministic checks + checklist LLM judge</sub>")
    return "\n".join(lines)


def to_json(report: Report) -> str:
    return report.model_dump_json(indent=2)
