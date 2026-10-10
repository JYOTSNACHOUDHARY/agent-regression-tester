"""Checklist LLM judge: grade a response against explicit yes/no criteria.

Why a checklist, not pairwise comparison: the first version of agentreg compared
reference and candidate side by side (in both orders, to cancel position bias).
On this project's agent that judge preferred whichever response came first in
almost every comparison - and stayed blind to a prompt change that removed all
figures from the answers. Grading one response at a time against binary
criteria removes position bias entirely, is more stable than 1-10 scores, and
makes the report actionable: it names the exact criterion that started failing.

* Evidence-grounded - each response is graded together with the tool results of
  its own run, because agent runs legitimately differ in content.
* Relative - the reference is graded with the same checklist (cached), and the
  verdict compares failures: new failures mean "worse", fewer mean "better".
* Cached by content hash; identical outputs skip the LLM entirely.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentreg.recorder import recording
from agentreg.schema import CriterionResult, JudgeResult, Verdict, normalize_text

SYSTEM = """You are a strict, consistent grader of AI agent responses.
You grade ONE response against a checklist of yes/no criteria.
The response comes with the evidence (tool results) its own agent run produced; use it to check
faithfulness and completeness. Grade each criterion independently and literally:
pass it only if the response clearly satisfies it. Point to the specific text that decides it."""

USER = """## User input
{input}

## Evidence from this run's tools
{evidence}

## Response to grade
{response}

## Criteria
{criteria}

Grade every criterion, in order."""


class _Grade(BaseModel):
    criterion: int = Field(description="1-based criterion number.")
    reason: str = Field(description="One sentence pointing to the deciding text.")
    passed: bool


class _Grades(BaseModel):
    grades: list[_Grade]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, default=str)


def format_evidence(tool_results) -> str:
    """Render tool calls as compact evidence lines for the judge."""
    lines = [f"- {t.name}({json.dumps(t.args, default=str)}) -> {t.output}" for t in tool_results]
    return "\n".join(lines) or "(no tool calls)"


class Judge:
    def __init__(self, model: str, cache_path: Path | None = None, llm: Any = None) -> None:
        self.model = model
        self._llm = llm
        self._cache_path = cache_path
        self._cache: dict[str, list[dict]] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text())
        self.total_cost_usd: float | None = 0.0

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from langchain.chat_models import init_chat_model

            self._llm = init_chat_model(self.model, temperature=0)
        return self._llm

    # ------------------------------------------------------------------ grading

    def _ask(self, criteria: list[str], case_input: str, response: str, evidence: str) -> list[_Grade]:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1))
        structured = self.llm.with_structured_output(_Grades)
        prompt = USER.format(input=case_input, evidence=evidence, response=response, criteria=numbered)
        return structured.invoke([("system", SYSTEM), ("user", prompt)]).grades

    def grade(self, criteria: list[str], case_input: Any, response: Any, evidence: str) -> list[CriterionResult]:
        inp, resp = _text(case_input), normalize_text(_text(response))
        key = hashlib.sha256(json.dumps([self.model, criteria, inp, resp, evidence]).encode()).hexdigest()
        if key not in self._cache:
            with recording() as rec:  # reuse the recorder to meter the judge's own spend
                grades = self._ask(criteria, inp, resp, evidence)
            self._add_cost(_sum_cost(rec.steps))
            by_number = {g.criterion: g for g in grades}
            results = []
            for i, criterion in enumerate(criteria, 1):
                g = by_number.get(i)
                results.append(
                    CriterionResult(
                        criterion=criterion,
                        passed=bool(g and g.passed),
                        reason=g.reason if g else "Judge did not grade this criterion.",
                    )
                )
            self._cache[key] = [r.model_dump() for r in results]
        return [CriterionResult.model_validate(r) for r in self._cache[key]]

    def compare(
        self,
        criteria: list[str],
        case_input: Any,
        reference: Any,
        candidate: Any,
        reference_evidence: str = "(not recorded)",
        candidate_evidence: str = "(not recorded)",
    ) -> JudgeResult:
        if normalize_text(_text(reference)) == normalize_text(_text(candidate)):
            return JudgeResult(verdict=Verdict.skipped, reasoning="Output identical to reference.")

        before = self.total_cost_usd
        ref = self.grade(criteria, case_input, reference, reference_evidence)
        cand = self.grade(criteria, case_input, candidate, candidate_evidence)

        newly_failed = [c for r, c in zip(ref, cand) if r.passed and not c.passed]
        newly_passed = [c for r, c in zip(ref, cand) if not r.passed and c.passed]
        if len(newly_failed) > len(newly_passed):
            verdict = Verdict.worse
            reasoning = "Now failing: " + "; ".join(f"{c.criterion} ({c.reason})" for c in newly_failed)
        elif len(newly_passed) > len(newly_failed):
            verdict = Verdict.better
            reasoning = "Now passing: " + "; ".join(f"{c.criterion} ({c.reason})" for c in newly_passed)
        else:
            verdict = Verdict.same
            reasoning = f"{sum(c.passed for c in cand)}/{len(cand)} criteria pass, same as reference."

        cost = None
        if before is not None and self.total_cost_usd is not None:
            cost = round(self.total_cost_usd - before, 6)
        return JudgeResult(
            verdict=verdict,
            reasoning=reasoning,
            criteria=cand,
            reference_passed=sum(r.passed for r in ref),
            cost_usd=cost,
        )

    # ------------------------------------------------------------------ bookkeeping

    def _add_cost(self, cost: float | None) -> None:
        if cost is None or self.total_cost_usd is None:
            self.total_cost_usd = None
        else:
            self.total_cost_usd = round(self.total_cost_usd + cost, 6)

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=1))


def _sum_cost(steps) -> float | None:
    costs = [getattr(s, "cost_usd", None) for s in steps if getattr(s, "kind", "") == "llm"]
    if not costs or any(c is None for c in costs):
        return None
    return round(sum(costs), 6)
