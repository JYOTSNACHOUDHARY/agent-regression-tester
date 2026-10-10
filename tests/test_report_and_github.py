import json

import httpx

from agentreg.github import upsert_comment
from agentreg.report import MARKER, to_markdown, tool_diff
from agentreg.replay import run_suite
from agentreg.capture import trace_to_case
from agentreg.config import Config
from agentreg.recorder import record_run

from conftest import ai, build_agent


def runner(agent):
    return lambda q: agent.invoke({"messages": [("user", q)]})["messages"][-1].content


def test_tool_diff():
    assert tool_diff(["a", "b"], ["b", "a"]) == "unchanged (2 calls)"
    assert tool_diff(["a", "b", "b"], ["a", "c"]) == "−b ×2 +c"


def test_markdown_report_explains_the_regression(weather_agent):
    trace = record_run("toy", runner(weather_agent), "weather?", "weather?")
    case = trace_to_case(trace, "paris", Config(target="toy:agent"))
    lazy = build_agent([ai("Probably sunny.")])
    md = to_markdown(run_suite([case], "toy", runner(lazy)))

    assert md.startswith(MARKER)
    assert "**Regressions found**" in md
    assert "| paris | ❌ regressed | −get_weather |" in md
    assert "`required_tools`: not called: get_weather" in md
    assert "Probably sunny." in md


def _github(comments: list[dict], log: list):
    def handler(request: httpx.Request) -> httpx.Response:
        log.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json=comments)
        body = json.loads(request.content)["body"]
        return httpx.Response(200, json={"html_url": "https://github.com/x", "body": body})

    return httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(handler))


def test_creates_comment_when_none_exists():
    log = []
    upsert_comment("report", "me/repo", 7, "t", client=_github([{"id": 1, "body": "lgtm"}], log))
    assert log == [("GET", "/repos/me/repo/issues/7/comments"), ("POST", "/repos/me/repo/issues/7/comments")]


def test_edits_existing_sticky_comment():
    log = []
    existing = [{"id": 1, "body": "lgtm"}, {"id": 42, "body": f"{MARKER}\nold report"}]
    upsert_comment("new report", "me/repo", 7, "t", client=_github(existing, log))
    assert log[-1] == ("PATCH", "/repos/me/repo/issues/comments/42")
