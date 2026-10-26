"""Tools the chat model can call (LangChain @tool)."""
import os
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from core.utils import run_query
from rag.web_rag import web_answer

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")  # 384 dims, matches vector(384)
# Loaded once at import (utils has already loaded .env); normalized so cosine similarity works
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})


class SaveMemoryInput(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "One short fact about the user, in third person, understandable on its own. "
            'Example: "User is a Python developer building a ChatGPT clone."'
        ),
    )


@tool(args_schema=SaveMemoryInput)
def save_memory(content: str) -> str:
    """Save a long-term memory about the user, so future chats can use it.

    Call this ONLY when the user shares something lasting that will help in future chats:
    - who they are: name, job, location, languages, skill level
    - preferences: how they like answers, tools and tech they use or avoid
    - ongoing projects and goals
    - anything they ask you to remember

    Do NOT call it for:
    - one-time questions or tasks
    - general facts about the world
    - things only the assistant said
    - passwords, API keys, card numbers, health details or other sensitive data
    - guesses: save only what the user clearly said

    If nothing is worth saving, do not call this tool.
    """
    # WARNING: writes one new row to public.memories on every call
    vector = embeddings.embed_documents([content])[0]
    run_query(
        "insert into public.memories (content, embedding) values (%s, %s::vector)",
        (content, str(vector)),
    )
    return f"Saved memory: {content}"


MEMORY_SEARCH_LIMIT = 5


class RetrieveMemoryInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=300,
        description='A short search phrase for the information needed, e.g. "user\'s job" or "languages the user codes in".',
    )


@tool(args_schema=RetrieveMemoryInput)
def retrieve_memory(query: str) -> str:
    """Search the user's saved long-term memories for facts relevant to the current question.

    Call this ONLY when you need personal information about the user to answer well, for example:
    - the user refers to something about themselves or earlier chats ("my project", "as I told you")
    - the answer depends on their preferences, background or setup
    - the user asks what you remember about them

    Do NOT call it for:
    - general knowledge questions
    - questions that do not depend on anything personal
    - information already given in this conversation

    Returns the most relevant saved memories, or a message saying none were found.
    """
    rows = run_query(
        "select content from public.memories where embedding is not null "
        "order by embedding <=> %s::vector limit %s",  # <=> is cosine distance
        (str(embeddings.embed_query(query)), MEMORY_SEARCH_LIMIT),
    )
    return "\n".join(f"- {content}" for (content,) in rows) or "No saved memories found."


class WebSearchInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=300,
        description='A short, specific web search query, e.g. "latest Python release" or "weather in Pune today".',
    )
    context: str = Field(
        min_length=1,
        max_length=350,
        description=(
            "About 50 words of background on the user's question: what they want to know and why, "
            "and any detail from the conversation that helps pick the most useful search results."
        ),
    )
    # Filled in by ToolNode with the graph state; the model never sees or sets it
    state: Annotated[dict, InjectedState]


# content_and_artifact: the text goes to the model, the bool (answered or not) is read by the graph in graph.py
@tool(args_schema=WebSearchInput, response_format="content_and_artifact")
def web_search(query: str, context: str, state: Annotated[dict, InjectedState]) -> tuple[str, bool]:
    """Search the web for current or factual information you cannot answer reliably from memory.

    Call this when:
    - the question is about recent events, news, prices, weather, versions or anything that may have changed
    - the user asks you to search, look something up or find sources
    - you are not sure of a fact and an up-to-date answer matters

    Do NOT call it for:
    - general knowledge that does not change, explanations, writing or coding help
    - questions about the user (use retrieve_memory for that)
    - information already given in this conversation

    Searches the web and checks the pages. Returns one of two things:
    - the finished answer with its sources: it is shown to the user as it is, so add nothing after it
    - a note on what the search was missing: call this tool again with a better query and context
    The note says which search this was. The 3rd search always returns a final answer.
    """
    # Searches already made this turn = web_search results already in the prompt
    earlier_searches = sum(isinstance(m, ToolMessage) and m.name == "web_search" for m in state["prompt"])
    # WARNING: one real Tavily call (about 1 credit) plus Gemini and Groq calls on every call
    return web_answer(query, context, search_number=earlier_searches + 1)


TOOLS = [save_memory, retrieve_memory, web_search]  # add new tools here; llm.py binds and runs everything in this list