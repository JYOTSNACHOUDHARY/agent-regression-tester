
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from typing import List
from tools.competitor_finder import competitor_finder
from tools.legal_risk_checker import legal_risk_checker
from tools.market_size_estimator import market_size_estimator
from tools.risk_score_calculator import risk_score_calculator
from tools.trend_finder import trend_finder
from prompts.startup_validator import prompt as validator_prompt
from prompts.decision_maker import prompt as decision_prompt
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o", model_provider="openai")

validator_prompt_template = ChatPromptTemplate.from_template(validator_prompt)
decision_prompt_template = ChatPromptTemplate.from_template(decision_prompt)


class State(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    domain: str
    startup_ideas_list: List[str]


class StartupIdeas(BaseModel):
    startup_ideas: List[str] = Field(
        description="List of Generated startup ideas")


def startup_ideas(state: State):
    print("---human input----")
    domain_input = state["user_input"]
    print(f"received user input: {domain_input}")
    # domain_ideas = ["a", "b", "c"]
    structured_llm = llm.with_structured_output(StartupIdeas)
    ai_msg = structured_llm.invoke(f"Generate 3 startup ideas related to domain: {domain_input}. Just List the ideas. Don't provide a description or additional information on ideas. Just the ideas which are self-descriptive.")
    print("startup idea generator: ", end="\n")
    print(ai_msg.startup_ideas)
    return {"domain": domain_input, "startup_ideas_list": ai_msg.startup_ideas, "messages": ai_msg.startup_ideas}


def validate_ideas(state: State):
    print("Here's the user domain input to process: ", state["domain"])
    print(f"Here are the stored startup ideas: ", state["startup_ideas_list"])
    domain = state["domain"]
    startup_ideas = state["startup_ideas_list"]

    llm_with_tools = llm.bind_tools(tools)
    validator_prompt = validator_prompt_template.invoke({"idea": f"{startup_ideas}", "domain": f"{domain}"})
    ai_msg = llm_with_tools.invoke(validator_prompt)
    print(f"decided tools to be used are: ", ai_msg.tool_calls)
    return {"messages": ai_msg}


def decision_maker(state: State):
    # Collect every tool result from the research turn, not just the last message
    research = []
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        research.append(msg.content)
    market_research = "\n".join(reversed(research))
    domain = state["domain"]
    startup_ideas = state["startup_ideas_list"]
    decision_maker_prompt = decision_prompt_template.invoke({"idea": f"{startup_ideas}", "domain": f"{domain}", "market_research": f"{market_research}"})
    ai_msg = llm.invoke(decision_maker_prompt)
    print("Decision maker response:: ", end="\n")
    print(ai_msg.content)
    return {"messages": ai_msg}


builder = StateGraph(State)
builder.add_node("startup_ideas", startup_ideas)
tools = [competitor_finder, legal_risk_checker, market_size_estimator, risk_score_calculator, trend_finder]
tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)
builder.add_node("validate_ideas", validate_ideas)
builder.add_node("decision_maker", decision_maker)
builder.add_edge(START, "startup_ideas")
builder.add_edge("startup_ideas", "validate_ideas")
builder.add_conditional_edges(
    "validate_ideas",
    tools_condition,
    ["tools", END]
)
builder.add_edge("tools", "decision_maker")
builder.add_edge("decision_maker", END)

graph = builder.compile()


# To create image for compiled graph
# from PIL import Image as PILImage
# from io import BytesIO
# img_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
# img = PILImage.open(BytesIO(img_bytes))
# img.show()


# Run to interact through console
# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"user_input": user_input}):
#         for value in event.values():
#             response = value["messages"]
#             if hasattr(response, "content"):
#                 print("Assistant:", value["messages"].content)
#             else:
#                 print("Assistant:", value["messages"])
#             if isinstance(value["messages"], List):
#                 print("Assistant:", len(value["messages"]))
#
#
# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break
