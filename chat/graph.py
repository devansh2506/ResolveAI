"""Builds the chat graph. server.py runs it once per message."""
from functools import partial
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from chat.llm import ChatState, build_prompt, call_model, save_turn, tools_node, update_summary_node, use_tool_answer


def after_tools(state: ChatState) -> Literal["use_tool_answer", "call_model"]:
    """A finished web_search answer goes straight to the user; any other tool result (including a web_search
    that found too little, so the model writes a new query) goes back to the model."""
    last: ToolMessage = state["prompt"][-1]
    return "use_tool_answer" if last.name == "web_search" and last.artifact else "call_model"


# Graph: build_prompt -> call_model -> (tools -> call_model, repeat) -> save_turn -> update_summary
# A web_search result skips the second model call: tools -> use_tool_answer -> save_turn
_builder = StateGraph(ChatState)
_builder.add_node("build_prompt", build_prompt)
_builder.add_node("call_model", call_model)
_builder.add_node("tools", tools_node)
_builder.add_node("use_tool_answer", use_tool_answer)
_builder.add_node("save_turn", save_turn)
_builder.add_node("update_summary", update_summary_node)
_builder.add_edge(START, "build_prompt")
_builder.add_edge("build_prompt", "call_model")
# tools_condition returns "tools" if the last prompt message asks for tools, otherwise END
_builder.add_conditional_edges(
    "call_model", partial(tools_condition, messages_key="prompt"), {"tools": "tools", END: "save_turn"}
)
_builder.add_conditional_edges("tools", after_tools)
_builder.add_edge("use_tool_answer", "save_turn")
_builder.add_edge("save_turn", "update_summary")
_builder.add_edge("update_summary", END)
graph = _builder.compile()