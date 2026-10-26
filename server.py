"""Web server: serves the chat UI and runs the chat graph. Start it with: python server.py

Every step of a message (graph nodes, tool calls, web search rounds, DB saves) is logged to this terminal.
"""
import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for _noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "huggingface_hub", "filelock", "opentelemetry"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("server")
log.info("Loading models and building the graph (takes a few seconds)...")

# Imported after the logging setup, so the start-up messages of these modules use the same format
from chat.graph import graph  # noqa: E402
from chat.llm import ChatState  # noqa: E402
from chat.tools import TOOLS  # noqa: E402
from core.chat_store import (  # noqa: E402
    chat_exists,
    count_messages_after_summary,
    create_chat,
    get_all_messages,
    get_summary,
    list_chats,
)
from core.prompts import Message  # noqa: E402
from core.utils import run_query  # noqa: E402

HOST = os.environ.get("APP_HOST", "127.0.0.1")
PORT = int(os.environ.get("APP_PORT", "8000"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
RECURSION_LIMIT = 12  # caps runaway tool loops (each tool round uses 2 steps)
TOOL_STEPS = {"save_memory": "Saving a memory", "retrieve_memory": "Checking saved memories"}


class ChatOut(BaseModel):
    chat_id: UUID
    title: str


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SummaryOut(BaseModel):
    summary: str | None  # None until a summary has been written
    messages_summarised: int  # how many of the oldest messages it covers


class SendIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=8000)
    use_web: bool = False  # "Use web tool" ticked in the UI


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Checking the database connection...")
    try:
        await asyncio.to_thread(run_query, "select 1")
    except Exception:
        log.exception("Cannot reach the database, check DATABASE_URL in .env")
        raise
    log.info("Database reachable")
    log.info("Tools the model can use: %s", ", ".join(tool.name for tool in TOOLS))
    log.info("Ready: open http://%s:%s in your browser", HOST, PORT)
    yield
    log.info("Server stopped")


app = FastAPI(title="Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _short(text: str, limit: int = 160) -> str:
    """One-line preview of a text, for the logs."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _line(event: dict) -> str:
    """One streamed event: a JSON object on its own line."""
    return json.dumps(event) + "\n"


async def run_turn(chat_id: UUID, content: str, use_web: bool) -> AsyncIterator[str]:
    """Run the graph for one message and yield events: steps while it works, then the reply, then done."""
    state: ChatState = {
        "chat_id": chat_id,
        "prompt": [],
        "current_convo": [Message("user", content)],
        "use_web_tool": use_web,
    }
    finished: set[str] = set()  # nodes that completed, used to label an error
    replied = False  # True once the final AI reply has been sent to the browser
    started = last = time.perf_counter()
    step = 0
    log.info("-" * 72)
    log.info("New message in chat %s (web tool ticked: %s)", chat_id, "yes" if use_web else "no")
    log.info("You: %s", _short(content))
    try:
        async for update in graph.astream(state, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
            for node, out in update.items():
                finished.add(node)
                step += 1
                now = time.perf_counter()
                took, last = now - last, now  # time since the previous step finished
                head = f"Step {step} finished ({took:.1f}s):"
                if node == "build_prompt":
                    prompt = out["prompt"]
                    has_summary = "<conversation_summary>" in prompt[0].content
                    log.info("%s read the conversation", head)
                    log.info("        %d messages in the prompt, summary of the older chat: %s", len(prompt), "yes" if has_summary else "no")
                    yield _line({"type": "step", "text": "Reading the conversation"})
                elif node == "call_model" and "prompt" in out:  # the model asked for tools
                    log.info("%s the model asked to use a tool", head)
                    for call in out["prompt"][0].tool_calls:
                        log.info("        %s  %s", call["name"], _short(json.dumps(call["args"]), 300))
                        text = (
                            f"Searching the web for “{call['args'].get('query', '')}”"
                            if call["name"] == "web_search"
                            else TOOL_STEPS.get(call["name"], f"Using {call['name']}")
                        )
                        yield _line({"type": "step", "text": text})
                elif node == "call_model":  # the model answered in text
                    reply = out["current_convo"][-1].content
                    replied = True
                    log.info("%s the model wrote the answer (%d characters)", head, len(reply))
                    yield _line({"type": "reply", "content": reply})
                elif node == "tools":
                    log.info("%s ran the tool", head)
                    for msg in out["prompt"]:
                        if msg.status == "error":
                            result = f"FAILED: {_short(msg.content, 200)}"
                        elif msg.name != "web_search":
                            result = _short(msg.content, 200)
                        elif msg.artifact:
                            result = f"final answer ready ({len(msg.content)} characters)"
                        else:
                            result = f"not enough, the model will search again. {_short(msg.content.splitlines()[0], 200)}"
                        log.info("        %s: %s", msg.name, result)
                        if msg.status == "error":
                            text = f"{msg.name.replace('_', ' ')} failed"
                        elif msg.name == "web_search":
                            text = "Found enough to answer" if msg.artifact else msg.content.splitlines()[0]
                        else:
                            text = f"{msg.name.replace('_', ' ')} done"
                        yield _line({"type": "step", "text": text})
                elif node == "use_tool_answer":
                    reply = out["current_convo"][-1].content
                    replied = True
                    log.info("%s used the web search answer as the reply (%d characters)", head, len(reply))
                    yield _line({"type": "reply", "content": reply})
                elif node == "save_turn":
                    log.info("%s saved your message and the reply to the database", head)
                elif node == "update_summary":
                    log.info("%s checked the chat summary", head)
    except asyncio.CancelledError:
        log.warning("The browser disconnected, so the remaining steps were cancelled")
        raise
    except Exception as exc:
        if not replied:
            label = "Something went wrong before a reply was ready, nothing was saved. You can send the message again."
        elif "save_turn" not in finished:
            label = "The reply was shown, but this exchange could not be saved to the chat."
        else:
            label = "The reply was saved, but the chat summary could not be updated. It will be retried next time."
        log.exception("FAILED: %s", label)
        yield _line({"type": "error", "message": f"{label} ({type(exc).__name__}: {exc})"})
        return
    log.info("Done in %.1fs", time.perf_counter() - started)
    yield _line({"type": "done"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/chats", response_model=list[ChatOut])
def api_list_chats():
    """All chats that have messages, most recent first."""
    try:
        return [ChatOut(chat_id=chat_id, title=title) for chat_id, title in list_chats()]
    except Exception as exc:
        log.exception("Could not list the chats")
        raise HTTPException(status_code=500, detail="Could not load the chat list") from exc


@app.post("/api/chats", response_model=ChatOut, status_code=201)
def api_create_chat():
    """Create an empty chat."""
    try:
        # WARNING: writes one new row to public.chats
        chat_id = create_chat()
    except Exception as exc:
        log.exception("Could not create a chat")
        raise HTTPException(status_code=500, detail="Could not create a new chat") from exc
    log.info("Created chat %s", chat_id)
    return ChatOut(chat_id=chat_id, title="New chat")


@app.get("/api/chats/{chat_id}/messages", response_model=list[MessageOut])
def api_get_messages(chat_id: UUID):
    """Every message of one chat, oldest first."""
    try:
        exists = chat_exists(chat_id)
        messages = get_all_messages(chat_id) if exists else []
    except Exception as exc:
        log.exception("Could not load the messages of chat %s", chat_id)
        raise HTTPException(status_code=500, detail="Could not load this chat") from exc
    if not exists:
        raise HTTPException(status_code=404, detail="Chat not found")
    log.info("Opened chat %s (%d messages)", chat_id, len(messages))
    return [MessageOut(role=m.role, content=m.content) for m in messages]


@app.get("/api/chats/{chat_id}/summary", response_model=SummaryOut)
def api_get_summary(chat_id: UUID):
    """The saved summary of the older part of one chat, and how many messages it covers."""
    try:
        exists = chat_exists(chat_id)
        summary, pointer = get_summary(chat_id) if exists else (None, None)
        covered = count_messages_after_summary(chat_id, None) - count_messages_after_summary(chat_id, pointer) if exists else 0
    except Exception as exc:
        log.exception("Could not load the summary of chat %s", chat_id)
        raise HTTPException(status_code=500, detail="Could not load the summary") from exc
    if not exists:
        raise HTTPException(status_code=404, detail="Chat not found")
    log.info("Showed the summary of chat %s (%s)", chat_id, f"covers {covered} messages" if summary else "none written yet")
    return SummaryOut(summary=summary, messages_summarised=covered)


@app.post("/api/chats/{chat_id}/messages")
async def api_send_message(chat_id: UUID, body: SendIn) -> StreamingResponse:
    """Run one message through the graph; the response streams one JSON event per line."""
    try:
        exists = await asyncio.to_thread(chat_exists, chat_id)
    except Exception as exc:
        log.exception("Could not check chat %s", chat_id)
        raise HTTPException(status_code=500, detail="Could not reach the database") from exc
    if not exists:
        raise HTTPException(status_code=404, detail="Chat not found")
    return StreamingResponse(
        run_turn(chat_id, body.content, body.use_web),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # Pass the app object (not "server:app"), so the models are not loaded a second time
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)  # no line per HTTP request
