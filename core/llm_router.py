"""The LiteLLM router (models from PROJECT_CONFIG.md), shared by llm.py and the web search pipeline."""
import logging
import os

import litellm
from litellm import Router

from core import utils  # noqa: F401  (importing it loads .env, so the API keys are available)

# Langfuse tracing is turned on only if both keys are present in .env
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    # SECRETS CONSUMED HERE: LiteLLM reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.
    # The docs disagree on the host variable name, so both are set to the same value.
    _langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    os.environ["LANGFUSE_HOST"] = _langfuse_host
    os.environ["LANGFUSE_OTEL_HOST"] = _langfuse_host
    litellm.callbacks = ["langfuse_otel"]

# The web search answer runs on a second Groq account, so it has its own tokens-per-minute budget
# SECRET CONSUMED HERE: GROQ_API_KEY2 (a key from a different Groq account than GROQ_API_KEY, never printed)
_web_groq_key = os.environ.get("GROQ_API_KEY2")
if not _web_groq_key:
    logging.getLogger("llm_router").warning(
        "GROQ_API_KEY2 is not set in .env, so web answers share the chat model's Groq limit"
    )
    _web_groq_key = os.environ.get("GROQ_API_KEY")

# API KEYS CONSUMED HERE: LiteLLM reads GROQ_API_KEY (chat), OPENROUTER_API_KEY and GEMINI_API_KEY from the
# environment automatically; the web-answer entry below is given its own Groq key.
router = Router(
    model_list=[
        {"model_name": "chat",
         "litellm_params": {"model": "groq/openai/gpt-oss-120b", "rpm": 30}},
        {"model_name": "web-answer",
         "litellm_params": {"model": "groq/openai/gpt-oss-120b", "api_key": _web_groq_key, "rpm": 30}},
        {"model_name": "chat-backup",
         "litellm_params": {"model": "openrouter/nvidia/nemotron-3.5-super-vl:free", "rpm": 20}},
        {"model_name": "small",
         "litellm_params": {"model": "gemini/gemini-3.5-flash-lite"}},
    ],
    fallbacks=[{"chat": ["chat-backup"]}, {"web-answer": ["chat-backup"]}],
    num_retries=2,
    timeout=30,
)
