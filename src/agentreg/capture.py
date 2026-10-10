"""Turn a recorded Trace into a human-editable YAML TestCase."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from agentreg.config import Config
from agentreg.schema import Expectations, JudgeSpec, TestCase, ToolStub, Trace


def trace_to_case(trace: Trace, case_id: str, cfg: Config, description: str = "") -> TestCase:
    tol = cfg.budget_tolerance
    tools = trace.tool_calls
    expect = Expectations(
        required_tools=sorted({t.name for t in tools}),
        max_tokens=math.ceil(trace.total_tokens * tol) if trace.total_tokens else None,
        max_cost_usd=round(trace.cost_usd * tol, 6) if trace.cost_usd else None,
        # At least 1s of headroom, so very fast runs don't fail on timer noise
        max_latency_s=round(max(trace.latency_s * cfg.latency_tolerance, trace.latency_s + 1.0), 2),
        max_steps=math.ceil(len(trace.steps) * tol),
        no_tool_errors=not any(t.error for t in tools),
    )
    stubs: dict[str, ToolStub] = {}
    for t in tools:
        if t.error is None:
            key = t.name + json.dumps(t.args, sort_keys=True, default=str)
            stubs.setdefault(key, ToolStub(name=t.name, args=t.args, output=t.output, artifact=t.artifact))

    return TestCase(
        id=case_id,
        description=description or f"Captured from trace {trace.id}",
        input=trace.input,
        reference=trace.summary(),
        expect=expect,
        judge=JudgeSpec(enabled=cfg.judge.enabled),
        tool_stubs=list(stubs.values()),
    )


# ---------------------------------------------------------------------- YAML io


class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> Any:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)


def save_case(case: TestCase, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = case.model_dump(mode="json")
    path.write_text(yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100))


def load_case(path: Path) -> TestCase:
    return TestCase.model_validate(yaml.safe_load(path.read_text()))


def load_cases(directory: Path, only: list[str] | None = None) -> list[TestCase]:
    cases = [load_case(p) for p in sorted(directory.glob("*.y*ml"))]
    if only:
        cases = [c for c in cases if c.id in only]
    return cases
