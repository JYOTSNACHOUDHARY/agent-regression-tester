"""Data model shared by every stage: traces, test cases and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> Any:
    """Strip trailing whitespace per line: keeps YAML readable and diffs meaningful."""
    if not isinstance(value, str):
        return value
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


# --------------------------------------------------------------------------- traces


class LLMCall(BaseModel):
    kind: Literal["llm"] = "llm"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    requested_tools: list[str] = Field(default_factory=list)
    error: str | None = None


class ToolCall(BaseModel):
    kind: Literal["tool"] = "tool"
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    artifact: Any = None
    latency_s: float = 0.0
    error: str | None = None
    stubbed: bool = False


Step = Annotated[LLMCall | ToolCall, Field(discriminator="kind")]


class Trace(BaseModel):
    """Everything observable about one agent run."""

    id: str
    target: str
    input: Any
    output: Any = None
    steps: list[Step] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def llm_calls(self) -> list[LLMCall]:
        return [s for s in self.steps if isinstance(s, LLMCall)]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s for s in self.steps if isinstance(s, ToolCall)]

    @property
    def tool_sequence(self) -> list[str]:
        return [t.name for t in self.tool_calls]

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.llm_calls]
        if not costs or any(c is None for c in costs):
            return None
        return round(sum(costs), 6)

    def summary(self) -> "RunSummary":
        return RunSummary(
            output=normalize_text(self.output),
            tool_sequence=self.tool_sequence,
            total_tokens=self.total_tokens,
            cost_usd=self.cost_usd,
            latency_s=round(self.latency_s, 3),
            steps=len(self.steps),
            tool_results=[
                ToolStub(name=t.name, args=t.args, output=t.output) for t in self.tool_calls if not t.error
            ],
        )


class ToolStub(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    artifact: Any = None


class RunSummary(BaseModel):
    """The compact part of a trace that a test case keeps as its reference."""

    output: Any = None
    tool_sequence: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    steps: int = 0
    tool_results: list[ToolStub] = Field(
        default_factory=list, description="What the tools returned: the evidence the output was based on."
    )


# --------------------------------------------------------------------------- test cases


class Expectations(BaseModel):
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_order: list[str] = Field(
        default_factory=list, description="Must appear as a subsequence of the tool calls."
    )
    tool_args: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="tool -> {arg: regex} that some call must match."
    )
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_latency_s: float | None = None
    max_steps: int | None = None
    output_contains: list[str] = Field(default_factory=list)
    output_regex: list[str] = Field(default_factory=list)
    no_tool_errors: bool = True


class JudgeSpec(BaseModel):
    enabled: bool = True
    criteria: list[str] = Field(
        default_factory=lambda: [
            "Directly answers the user's request.",
            "Every factual claim is supported by the evidence from the tools (nothing invented).",
            "Uses the specific results returned by the tools rather than generic statements.",
            "Is complete: nothing the user asked for is missing.",
        ],
        description="Yes/no checklist the judge grades every response against.",
    )


class TestCase(BaseModel):
    __test__ = False  # stop pytest collecting this class

    id: str
    description: str = ""
    input: Any
    reference: RunSummary
    expect: Expectations = Field(default_factory=Expectations)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    tool_stubs: list[ToolStub] = Field(default_factory=list)


# --------------------------------------------------------------------------- results


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Verdict(str, Enum):
    better = "better"
    same = "same"
    worse = "worse"
    skipped = "skipped"


class CriterionResult(BaseModel):
    criterion: str
    passed: bool
    reason: str = ""


class JudgeResult(BaseModel):
    verdict: Verdict
    reasoning: str = ""
    criteria: list[CriterionResult] = Field(default_factory=list)
    reference_passed: int | None = None
    cost_usd: float | None = None


class Status(str, Enum):
    passed = "passed"
    regressed = "regressed"
    improved = "improved"
    flaky = "flaky"
    error = "error"


class Attempt(BaseModel):
    """One replay of a case (there are several when --repeats > 1)."""

    trace: Trace
    checks: list[CheckResult]
    judge: JudgeResult | None = None
    stub_misses: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        if self.trace.error:
            return True
        if any(not c.passed for c in self.checks):
            return True
        return self.judge is not None and self.judge.verdict == Verdict.worse


class CaseResult(BaseModel):
    case_id: str
    status: Status
    reference: RunSummary
    attempts: list[Attempt]

    @property
    def representative(self) -> Attempt:
        """The attempt shown in the report: the first failing one if any failed."""
        return next((a for a in self.attempts if a.failed), self.attempts[0])


class Report(BaseModel):
    target: str
    tool_mode: str
    repeats: int
    started_at: datetime = Field(default_factory=_now)
    results: list[CaseResult] = Field(default_factory=list)
    judge_cost_usd: float | None = None

    def count(self, status: Status) -> int:
        return sum(r.status == status for r in self.results)

    @property
    def has_regressions(self) -> bool:
        return any(r.status in (Status.regressed, Status.error) for r in self.results)

    @property
    def agent_cost_usd(self) -> float | None:
        costs = [a.trace.cost_usd for r in self.results for a in r.attempts]
        if not costs or any(c is None for c in costs):
            return None
        return round(sum(costs), 6)
