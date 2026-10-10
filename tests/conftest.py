"""Shared fixtures: a tiny LangGraph agent driven by a scripted fake chat model."""

from __future__ import annotations

from typing import Annotated, Any, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class ScriptedChatModel(FakeMessagesListChatModel):
    """Returns the scripted messages in order; tool binding is a no-op."""

    model_name: str = "gpt-4o-mini"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # type: ignore[override]
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        result.llm_output = {"model_name": self.model_name}
        return result


def ai(content: str = "", tool_calls: list[dict] | None = None, tokens=(100, 20)) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"}
            for i, (n, a) in enumerate(tool_calls or [])
        ],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )


@tool
def get_weather(city: str, unit: str = "C") -> str:
    """Get the current weather for a city."""
    return f"Sunny, 24{unit} in {city}"


@tool
def get_time(city: str) -> str:
    """Get the local time in a city."""
    return f"14:05 in {city}"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(responses: list[AIMessage]):
    llm = ScriptedChatModel(responses=responses).bind_tools([get_weather, get_time])

    def agent(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([get_weather, get_time]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile()


@pytest.fixture
def weather_agent():
    """Calls get_weather(Paris), then answers."""
    return build_agent(
        [
            ai(tool_calls=[("get_weather", {"city": "Paris"})]),
            ai("It is sunny and 24C in Paris.", tokens=(150, 12)),
        ]
    )
