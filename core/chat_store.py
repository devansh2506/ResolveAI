"""Database helpers for chats, messages and summaries."""
from uuid import UUID

from core.prompts import Message
from core.utils import run_query

RECENT_MESSAGES = 10  # latest messages kept raw in the prompt (5 user + 5 AI, see get_chat_context)

# "After message X": X's sent_at, or -infinity when X is None, so None means "every message"
_SENT_AFTER = "sent_at > coalesce((select sent_at from public.messages where message_id = %s), '-infinity')"


def create_chat() -> UUID:
    """Create a new, empty chat row and return its chat_id."""
    return run_query("insert into public.chats default values returning chat_id")[0][0]


def set_chat_title(chat_id: UUID, first_message: str) -> None:
    """Title a chat with its first message (cut to 50 characters plus "...") unless it already has a title."""
    text = " ".join(first_message.split())  # one line, no extra whitespace
    title = text if len(text) <= 50 else text[:50] + "..."
    run_query(
        "insert into public.chats (chat_id, title) values (%s, %s) "
        "on conflict (chat_id) do update set title = coalesce(chats.title, excluded.title)",
        (chat_id, title),
    )


def save_message(chat_id: UUID, role: str, content: str) -> UUID:
    """Save one message (role 'user' or 'assistant') and return its message_id."""
    return run_query(
        "insert into public.messages (chat_id, role, content) values (%s, %s, %s) returning message_id",
        (chat_id, role, content),
    )[0][0]


def count_messages_after_summary(chat_id: UUID, summary_upto_message_id: UUID | None) -> int:
    """Count the chat's messages sent after the given message (all of them if it is None)."""
    return run_query(
        f"select count(*) from public.messages where chat_id = %s and {_SENT_AFTER}",
        (chat_id, summary_upto_message_id),
    )[0][0]


def get_summary(chat_id: UUID) -> tuple[str | None, UUID | None]:
    """Return (summary, summary_upto_message_id), or (None, None) if there is no summary yet."""
    rows = run_query(
        "select summary, summary_upto_message_id from public.chats where chat_id = %s", (chat_id,)
    )
    return rows[0] if rows else (None, None)


def save_summary(chat_id: UUID, summary: str, summary_upto_message_id: UUID) -> None:
    """Save the summary and the last message it covers; creates the chats row if missing."""
    run_query(
        "insert into public.chats (chat_id, summary, summary_upto_message_id) values (%s, %s, %s) "
        "on conflict (chat_id) do update set "
        "summary = excluded.summary, summary_upto_message_id = excluded.summary_upto_message_id",
        (chat_id, summary, summary_upto_message_id),
    )


def get_messages_to_summarise(
    chat_id: UUID, after_message_id: UUID | None
) -> tuple[list[Message], UUID] | None:
    """Return (messages to summarise, id of the last one) for everything after a message.

    That is every message after after_message_id (from the start if None), oldest first,
    except the latest RECENT_MESSAGES, which stay in the prompt as raw history.
    The id becomes the new summary pointer. Returns None if nothing is left to summarise.
    """
    rows = run_query(
        f"select message_id, role, content from public.messages where chat_id = %s and {_SENT_AFTER} "
        "order by sent_at",
        (chat_id, after_message_id),
    )[:-RECENT_MESSAGES]
    if not rows:
        return None
    return [Message(role, content) for _, role, content in rows], rows[-1][0]


def get_chat_context(chat_id: UUID) -> list[Message]:
    """Return the latest 5 user + 5 AI messages, oldest first (fewer if the chat is short)."""
    rows = run_query(
        "select role, content from ("
        " (select role, content, sent_at from public.messages"
        "  where chat_id = %s and role = 'user' order by sent_at desc limit 5)"
        " union all"
        " (select role, content, sent_at from public.messages"
        "  where chat_id = %s and role = 'assistant' order by sent_at desc limit 5)"
        ") latest order by sent_at",
        (chat_id, chat_id),
    )
    return [Message(role, content) for role, content in rows]


def chat_exists(chat_id: UUID) -> bool:
    """Return True if a chats row with this chat_id exists."""
    return bool(run_query("select 1 from public.chats where chat_id = %s", (chat_id,)))


def list_chats() -> list[tuple[UUID, str]]:
    """Return (chat_id, title) of every chat that has messages, most recently active first."""
    return run_query(
        "select c.chat_id, coalesce(c.title, 'New chat') from public.chats c "
        "join (select chat_id, max(sent_at) as last_sent from public.messages group by chat_id) m "
        "on m.chat_id = c.chat_id order by m.last_sent desc"
    )


def get_all_messages(chat_id: UUID) -> list[Message]:
    """Return every message of the chat, oldest first (for showing the chat in the UI)."""
    rows = run_query("select role, content from public.messages where chat_id = %s order by sent_at", (chat_id,))
    return [Message(role, content) for role, content in rows]
