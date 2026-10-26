"""LLM calls through the LiteLLM router (models from PROJECT_CONFIG.md)."""
import logging
from typing import Annotated, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLMRouter
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from chat.tools import TOOLS
from core.chat_store import (
    count_messages_after_summary,
    get_chat_context,
    get_messages_to_summarise,
    get_summary,
    save_message,
    save_summary,
    set_chat_title,
)
from core.llm_router import router
from core.prompts import CHAT_INSTRUCTIONS, Message, build_summary_messages

RECENT_MESSAGES = 10  # raw messages sent with every prompt (5 user + 5 AI, see get_chat_context)
SUMMARY_TRIGGER = RECENT_MESSAGES + 2  # fold the next exchange once this many messages are unsummarised
USE_WEB_TOOL_NOTE = "\n\nuse web tool to answer"  # added to the model's copy of the user message when "Use web tool" is ticked

log = logging.getLogger("summary")

# LangChain chat model on top of the router (keeps its load balancing and fallbacks)
model = ChatLiteLLMRouter(router=router, model_name="chat").bind_tools(TOOLS)
tools_node = ToolNode(TOOLS, messages_key="prompt", handle_tool_errors=True)  # runs the requested tools
summary_model = ChatLiteLLMRouter(router=router, model_name="small")  # Gemini, for summaries

class ChatState(TypedDict):
    """State shared by all graph nodes for one turn (one user message)."""

    chat_id: UUID
    # full prompt for the model; add_messages appends what nodes return, so each node returns only new messages
    prompt: Annotated[list[BaseMessage], add_messages]
    current_convo: list[Message]  # [user's new message, then the AI reply once call_model has run]
    use_web_tool: bool  # "Use web tool" ticked: the model's copy of the user message gets USE_WEB_TOOL_NOTE


def build_prompt(state: ChatState) -> dict:
    """Load the chat's history and summary from the DB and build the full prompt."""
    chat_id = state["chat_id"]
    try:
        recent = get_chat_context(chat_id)
        summary, _ = get_summary(chat_id)
    except Exception as exc:
        raise RuntimeError(f"Could not load chat history from the DB: {exc}") from exc
    system_prompt = CHAT_INSTRUCTIONS
    if summary:
        system_prompt += f"\n\n<conversation_summary>\n{summary}\n</conversation_summary>"
    user_query = state["current_convo"][0]
    history = [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content) for m in recent
    ]
    question = user_query.content + (USE_WEB_TOOL_NOTE if state["use_web_tool"] else "")  # the DB keeps the original
    prompt = [SystemMessage(system_prompt)] + history + [HumanMessage(question)]
    return {"prompt": prompt}


async def call_model(state: ChatState) -> dict:
    """Call the model with the tools attached.

    A tool request is added to the prompt (tools_node answers it); a text answer goes to current_convo.
    """
    # WARNING: real external API call (uses free-tier quota)
    ai_message = await model.ainvoke(state["prompt"])
    if ai_message.tool_calls:
        return {"prompt": [ai_message]}
    if not ai_message.content:
        raise RuntimeError("The model returned an empty reply")
    return {"current_convo": state["current_convo"] + [Message("assistant", ai_message.content)]}


def use_tool_answer(state: ChatState) -> dict:
    """The web_search tool already wrote the final answer: use it as the AI reply, without another model call."""
    return {"current_convo": state["current_convo"] + [Message("assistant", state["prompt"][-1].content)]}


def save_turn(state: ChatState) -> None:
    """Save the user message and the AI reply to the DB, in that order, and title the chat if it has no title."""
    # WARNING: writes two rows to public.messages and may set public.chats.title
    for message in state["current_convo"]:
        save_message(state["chat_id"], message.role, message.content)
    set_chat_title(state["chat_id"], state["current_convo"][0].content)


async def update_summary(chat_id: UUID) -> None:
    """Fold every message that has left the recent window into the chat summary.

    Does nothing until SUMMARY_TRIGGER or more messages are unsummarised. Saves the new
    summary and moves summary_upto_message_id to the last message covered.
    Raises if the summary call or a DB call fails; the pointer then stays where it was.
    """
    summary, pointer = get_summary(chat_id)
    pending = count_messages_after_summary(chat_id, pointer)  # pointer None = all messages
    if pending < SUMMARY_TRIGGER:
        log.info("    Summary not needed yet: %d unsummarised messages (it runs at %d)", pending, SUMMARY_TRIGGER)
        return
    batch = get_messages_to_summarise(chat_id, pointer)
    if batch is None:
        log.info("    Summary: nothing outside the recent window to fold in")
        return
    new_messages, last_message_id = batch
    log.info("    Summary: folding %d messages into it with Gemini", len(new_messages))
    total = count_messages_after_summary(chat_id, None)
    messages_summarised = total - pending + len(new_messages)  # messages covered once this is saved
    prompt = build_summary_messages(summary, new_messages, messages_summarised)
    # WARNING: real external API call (uses free-tier quota)
    new_summary = (await summary_model.ainvoke(prompt)).content
    if not new_summary or not new_summary.strip():
        raise RuntimeError("The model returned an empty summary")
    save_summary(chat_id, new_summary.strip(), last_message_id)
    log.info("    Summary saved (%d characters)", len(new_summary.strip()))


async def update_summary_node(state: ChatState) -> None:
    """Graph node: update the chat summary using only the chat_id."""
    await update_summary(state["chat_id"])