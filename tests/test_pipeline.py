import pytest

from agentreg.adapters import extract
from agentreg.capture import load_case, save_case, trace_to_case
from agentreg.checks import run_checks
from agentreg.config import Config
from agentreg.judge import Judge, _Grade
from agentreg.recorder import record_run
from agentreg.replay import accept, classify, run_attempt, run_suite
from agentreg.schema import Attempt, CheckResult, Expectations, JudgeResult, Status, ToolStub, Trace, Verdict

from conftest import ai, build_agent

CFG = Config(target="toy:agent")


def runner(agent):
    return lambda q: agent.invoke({"messages": [("user", q)]})["messages"][-1].content


def captured_case(agent, case_id="paris"):
    trace = record_run("toy", runner(agent), "weather in Paris?", "weather in Paris?")
    return trace_to_case(trace, case_id, CFG)


# ------------------------------------------------------------------- capture


def test_capture_derives_expectations_and_stubs(weather_agent, tmp_path):
    case = captured_case(weather_agent)
    assert case.expect.required_tools == ["get_weather"]
    assert case.expect.max_tokens == 423  # ceil(282 * 1.5)
    assert case.expect.max_steps == 5  # ceil(3 * 1.5)
    assert case.reference.output == "It is sunny and 24C in Paris."
    assert [s.name for s in case.tool_stubs] == ["get_weather"]

    save_case(case, tmp_path / "paris.yaml")
    assert load_case(tmp_path / "paris.yaml") == case


# ------------------------------------------------------------------- replay


def test_unchanged_agent_passes(weather_agent):
    case = captured_case(weather_agent)
    report = run_suite([case], "toy", runner(weather_agent))
    assert report.results[0].status == Status.passed
    assert not report.has_regressions


def test_agent_that_skips_tool_regresses(weather_agent):
    case = captured_case(weather_agent)
    lazy = build_agent([ai("Probably sunny in Paris.")])  # answers without calling the tool
    report = run_suite([case], "toy", runner(lazy))
    result = report.results[0]
    assert result.status == Status.regressed
    failed = {c.name: c.detail for c in result.representative.checks if not c.passed}
    assert failed == {"required_tools": "not called: get_weather"}


def test_stub_mode_serves_recorded_outputs(weather_agent):
    case = captured_case(weather_agent)
    case.tool_stubs = [ToolStub(name="get_weather", args={"city": "Paris"}, output="STUBBED")]
    attempt = run_attempt(case, "toy", runner(weather_agent), "stub", judge=None)
    call = attempt.trace.tool_calls[0]
    assert call.output == "STUBBED" and call.stubbed
    assert attempt.stub_misses == []


def test_stub_miss_falls_back_to_live_tool_and_is_reported(weather_agent):
    case = captured_case(weather_agent)
    london = build_agent(
        [ai(tool_calls=[("get_weather", {"city": "London"})]), ai("Sunny in London.")]
    )
    attempt = run_attempt(case, "toy", runner(london), "stub", judge=None)
    call = attempt.trace.tool_calls[0]
    assert call.output == "Sunny, 24C in London" and not call.stubbed
    assert attempt.stub_misses == ['get_weather({"city": "London", "unit": "C"})']


def test_stub_matches_when_only_defaults_differ(weather_agent):
    case = captured_case(weather_agent)
    case.tool_stubs = [ToolStub(name="get_weather", args={"city": "Paris", "unit": "C"}, output="S")]
    attempt = run_attempt(case, "toy", runner(weather_agent), "stub", judge=None)
    assert attempt.trace.tool_calls[0].output == "S"


def test_accept_updates_reference_and_adds_new_stubs(weather_agent):
    case = captured_case(weather_agent)
    london = build_agent(
        [ai(tool_calls=[("get_weather", {"city": "London"})]), ai("Sunny in London.")]
    )
    result = run_suite([case], "toy", runner(london)).results[0]
    updated = accept(case, result)
    assert updated.reference.output == "Sunny in London."
    assert {s.args["city"] for s in updated.tool_stubs} == {"Paris", "London"}


# ------------------------------------------------------------------- classification


def _attempt(failed: bool, verdict: Verdict | None = None) -> Attempt:
    return Attempt(
        trace=Trace(id="t", target="toy", input="x"),
        checks=[CheckResult(name="c", passed=not failed)],
        judge=JudgeResult(verdict=verdict) if verdict else None,
    )


@pytest.mark.parametrize(
    "fails, expected",
    [([False, False, False], Status.passed), ([True, False, False], Status.flaky), ([True, True, False], Status.regressed)],
)
def test_majority_vote(fails, expected):
    assert classify([_attempt(f) for f in fails]) == expected


def test_judge_better_majority_is_improvement():
    assert classify([_attempt(False, Verdict.better)] * 2 + [_attempt(False, Verdict.same)]) == Status.improved


# ------------------------------------------------------------------- checks


def test_checks_order_args_output_and_budgets(weather_agent):
    trace = record_run("toy", runner(weather_agent), "q", "q")
    expect = Expectations(
        tool_order=["get_time", "get_weather"],
        tool_args={"get_weather": {"city": "^Par"}},
        output_regex=[r"\d+C"],
        max_tokens=100,
    )
    results = {c.name: c for c in run_checks(expect, trace)}
    assert not results["tool_order"].passed
    assert results["tool_args"].passed
    assert results["output_regex"].passed
    assert not results["max_tokens"].passed and "282 tok / limit 100 tok" in results["max_tokens"].detail
    assert "required_tools" not in results  # unconfigured checks are omitted


# ------------------------------------------------------------------- judge


class ScriptedJudge(Judge):
    """Grades from a lookup: response text -> list of pass/fail per criterion."""

    def __init__(self, grades: dict[str, list[bool]]):
        super().__init__(model="fake")
        self.grades = grades
        self.calls = 0

    def _ask(self, criteria, case_input, response, evidence):
        self.calls += 1
        return [_Grade(criterion=i, reason="r", passed=p) for i, p in enumerate(self.grades[response], 1)]


CRITERIA = ["cites figures", "decides every idea"]


def test_judge_flags_newly_failing_criteria():
    judge = ScriptedJudge({"good": [True, True], "vague": [False, True]})
    result = judge.compare(CRITERIA, "q", "good", "vague")
    assert result.verdict == Verdict.worse
    assert result.reasoning.startswith("Now failing: cites figures")
    assert result.reference_passed == 2


def test_judge_better_and_same():
    judge = ScriptedJudge({"meh": [False, True], "great": [True, True], "also meh": [True, False]})
    assert judge.compare(CRITERIA, "q", "meh", "great").verdict == Verdict.better
    assert judge.compare(CRITERIA, "q", "meh", "also meh").verdict == Verdict.same  # one gained, one lost


def test_judge_skips_identical_outputs_and_caches_grades():
    judge = ScriptedJudge({"a": [True, True], "b": [True, True]})
    assert judge.compare(CRITERIA, "q", "same", "same  ").verdict == Verdict.skipped
    judge.compare(CRITERIA, "q", "a", "b")
    judge.compare(CRITERIA, "q", "a", "b")
    assert judge.calls == 2  # reference and candidate graded once each


# ------------------------------------------------------------------- adapters


def test_extract_paths():
    data = {"messages": [{"content": "a"}, {"content": "b"}]}
    assert extract(data, "messages[-1].content") == "b"
    assert extract(data, None) is data
    with pytest.raises(ValueError):
        extract(data, "messages[-1]..content")
