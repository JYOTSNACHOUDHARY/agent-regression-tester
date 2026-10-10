"""agentreg command line: record, capture, run, comment."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agentreg import report as report_mod
from agentreg.adapters import load_target
from agentreg.capture import load_cases, save_case, trace_to_case
from agentreg.config import DEFAULT_CONFIG, load_config
from agentreg.recorder import record_run
from agentreg.schema import CaseResult, Status, Trace

app = typer.Typer(add_completion=False, help="Regression tests for LLM agents.", no_args_is_help=True)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to agentreg.yaml.")]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "case"


def _trace_table(trace: Trace) -> Table:
    table = Table(title=f"Trace {trace.id}", show_lines=False)
    for col in ("#", "step", "detail", "tokens", "latency"):
        table.add_column(col)
    for i, step in enumerate(trace.steps, 1):
        if step.kind == "llm":
            tools = f" → wants {', '.join(step.requested_tools)}" if step.requested_tools else ""
            table.add_row(str(i), "llm", f"{step.model}{tools}", f"{step.input_tokens}+{step.output_tokens}", f"{step.latency_s:.2f}s")
        else:
            table.add_row(str(i), "tool", f"{step.name}({step.args})" + (" [stub]" if step.stubbed else ""), "", f"{step.latency_s:.2f}s")
    return table


@app.command()
def record(
    input: Annotated[str, typer.Argument(help="The user input to send to the agent.")],
    name: Annotated[Optional[str], typer.Option(help="Trace / case name (default: from input).")] = None,
    capture: Annotated[bool, typer.Option(help="Also turn the trace into a test case.")] = True,
    config: ConfigOpt = Path(DEFAULT_CONFIG),
) -> None:
    """Run the agent once on INPUT and save the trace (and a test case)."""
    cfg = load_config(config)
    run = load_target(cfg)
    name = name or _slug(input)

    with console.status(f"Running {cfg.target} …"):
        trace = record_run(cfg.target, run, input, input)
    path = cfg.path(cfg.traces_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2))

    console.print(_trace_table(trace))
    console.print(f"tokens {trace.total_tokens:,} · cost {report_mod._money(trace.cost_usd)} · {trace.latency_s:.1f}s")
    if trace.error:
        console.print(f"[red]Agent error:[/red] {trace.error}")
    console.print(f"Saved trace → {path}")
    if capture and not trace.error:
        case_path = cfg.path(cfg.cases_dir) / f"{name}.yaml"
        save_case(trace_to_case(trace, name, cfg), case_path)
        console.print(f"Saved case  → {case_path}")


@app.command()
def capture(
    trace_path: Annotated[Path, typer.Argument(help="A trace JSON written by `record`.")],
    case_id: Annotated[Optional[str], typer.Option("--id", help="Case id (default: trace file name).")] = None,
    description: Annotated[str, typer.Option(help="What this case protects.")] = "",
    config: ConfigOpt = Path(DEFAULT_CONFIG),
) -> None:
    """Turn a recorded trace into a test case."""
    cfg = load_config(config)
    trace = Trace.model_validate_json(trace_path.read_text())
    case_id = case_id or trace_path.stem
    path = cfg.path(cfg.cases_dir) / f"{case_id}.yaml"
    save_case(trace_to_case(trace, case_id, cfg, description), path)
    console.print(f"Saved case → {path}")


def _print_result(result: CaseResult) -> None:
    attempt = result.representative
    icon = report_mod.ICON[result.status]
    console.print(f"{icon} {result.case_id}: {result.status.value}")
    for c in attempt.checks:
        if not c.passed:
            console.print(f"    [red]✗ {c.name}[/red] {c.detail}")
    if attempt.judge and attempt.judge.verdict.value in ("worse", "better"):
        console.print(f"    judge: {attempt.judge.verdict.value} — {attempt.judge.reasoning}")


@app.command()
def run(
    tools: Annotated[str, typer.Option(help="'live' runs real tools; 'stub' replays recorded tool outputs.")] = "live",
    repeats: Annotated[int, typer.Option(min=1, help="Runs per case; majority vote decides.")] = 1,
    judge: Annotated[bool, typer.Option(help="Use the LLM judge.")] = True,
    case: Annotated[Optional[list[str]], typer.Option(help="Only run these case ids.")] = None,
    accept: Annotated[bool, typer.Option(help="Make this run the new reference (like snapshot update).")] = False,
    fail_on_flaky: Annotated[bool, typer.Option(help="Exit non-zero on flaky cases too.")] = False,
    config: ConfigOpt = Path(DEFAULT_CONFIG),
) -> None:
    """Replay every test case and report regressions. Exits 1 if any are found."""
    from agentreg.judge import Judge
    from agentreg.replay import accept as accept_case
    from agentreg.replay import run_suite

    if tools not in ("live", "stub"):
        raise typer.BadParameter("--tools must be 'live' or 'stub'")
    cfg = load_config(config)
    cases = load_cases(cfg.path(cfg.cases_dir), case)
    if not cases:
        console.print("[yellow]No test cases found. Record some with `agentreg record`.[/yellow]")
        raise typer.Exit(2)

    target = load_target(cfg)
    reports_dir = cfg.path(cfg.reports_dir)
    the_judge = None
    if judge and cfg.judge.enabled:
        the_judge = Judge(cfg.judge.model, cache_path=reports_dir / ".judge-cache.json")

    console.print(f"Replaying {len(cases)} case(s) against [bold]{cfg.target}[/bold] (tools: {tools}, repeats: {repeats})")
    rep = run_suite(cases, cfg.target, target, tools, repeats, the_judge, on_result=_print_result)  # type: ignore[arg-type]
    if the_judge:
        the_judge.save_cache()

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(report_mod.to_markdown(rep))
    (reports_dir / "report.json").write_text(report_mod.to_json(rep))
    console.print(f"\n{report_mod._headline(rep)}")
    console.print(f"Report → {reports_dir / 'report.md'}")

    if accept:
        by_id = {c.id: c for c in cases}
        for result in rep.results:
            save_case(accept_case(by_id[result.case_id], result), cfg.path(cfg.cases_dir) / f"{result.case_id}.yaml")
        console.print("[green]References updated from this run.[/green]")
        return

    if rep.has_regressions or (fail_on_flaky and rep.count(Status.flaky)):
        raise typer.Exit(1)


@app.command()
def comment(
    report_path: Annotated[Path, typer.Argument(help="Markdown report to post.")] = Path("reports/report.md"),
    repo: Annotated[Optional[str], typer.Option(help="owner/name (default: from GitHub Actions).")] = None,
    pr: Annotated[Optional[int], typer.Option(help="PR number (default: from GitHub Actions).")] = None,
) -> None:
    """Post or update the report as a single sticky PR comment."""
    from agentreg.github import pr_from_env, upsert_comment

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print("[red]GITHUB_TOKEN is not set.[/red]")
        raise typer.Exit(2)
    if repo is None or pr is None:
        repo, pr = pr_from_env()
    url = upsert_comment(report_path.read_text(), repo, pr, token)
    console.print(f"Comment → {url}")


def main() -> None:
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
