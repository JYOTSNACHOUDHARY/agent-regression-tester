from agentreg.recorder import record_run
from agentreg.schema import LLMCall, ToolCall


def _run(agent):
    return lambda payload: agent.invoke(payload)["messages"][-1].content


def test_records_llm_and_tool_steps_in_order(weather_agent):
    trace = record_run("toy", _run(weather_agent), {"messages": [("user", "weather?")]}, "weather?")

    assert trace.error is None
    assert trace.output == "It is sunny and 24C in Paris."
    assert [type(s) for s in trace.steps] == [LLMCall, ToolCall, LLMCall]
    assert trace.tool_sequence == ["get_weather"]

    tool = trace.tool_calls[0]
    assert tool.args == {"city": "Paris"}
    assert tool.output == "Sunny, 24C in Paris"

    first = trace.llm_calls[0]
    assert first.model == "gpt-4o-mini"
    assert first.requested_tools == ["get_weather"]
    assert trace.total_tokens == 100 + 20 + 150 + 12
    # gpt-4o-mini: $0.15 / $0.60 per 1M tokens
    assert trace.cost_usd == round((250 * 0.15 + 32 * 0.60) / 1e6, 6)


def test_agent_exception_is_captured_not_raised():
    def boom(_):
        raise RuntimeError("tool backend down")

    trace = record_run("toy", boom, {}, "x")
    assert trace.error == "RuntimeError: tool backend down"
    assert trace.output is None


def test_recording_does_not_leak_outside_block(weather_agent):
    record_run("toy", _run(weather_agent), {"messages": [("user", "hi")]}, "hi")
    from agentreg.recorder import _active

    assert _active.get() is None
