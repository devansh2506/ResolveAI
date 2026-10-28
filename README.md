<div align="center">

# ResolveAI

**A chat assistant that remembers you, keeps long chats under control, and checks the web before it answers.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-chat%20flow-1C3C3C)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-F55036)
![Gemini](https://img.shields.io/badge/Gemini-3.5%20Flash--Lite-4285F4?logo=googlegemini&logoColor=white)

[Features](#features) ·
[Architecture](#architecture) ·
[How it works](#how-it-works) ·
[Tech stack](#tech-stack) ·
[Getting started](#getting-started) ·
[Project structure](#project-structure)

</div>

<!--
  Add a 10 to 20 second demo here (a chat, a memory recall, a web answer with sources):
  ![ResolveAI demo](docs/demo.gif)
-->

ResolveAI is a ChatGPT-style assistant with long-term memory and a Perplexity-style web search. It is a learning
project built to explore context engineering, RAG, memory, routing and LLMOps with free-tier models, and it runs
on a single machine for a single user.

---

## Features

- **Long-term memory.** The model saves lasting facts about you and looks them up later by meaning (pgvector).
- **Rolling chat summaries.** Older messages are folded into a short summary, so long chats stay cheap. You can open the saved summary from the UI.
- **Web search you can switch on or leave to the model.** A *Use web tool* checkbox forces a search. When it is off, the model decides if it needs one.
- **Checked answers with citations.** Search results are chunked, ranked and checked by Gemini before Groq writes the answer with inline `[n]` citations and a numbered source list.
- **Self-correcting search.** If the evidence is not enough, the reason is fed back and the model writes a better query. At most 3 searches per message.
- **Live steps.** The UI shows what is happening ("Searching the web for ...", "Search 1 of 3 was not enough ...") while a reply is being prepared.
- **Readable chat UI.** Chat history sidebar, Markdown with code highlighting and copy buttons, smooth loading states, light sober design.
- **Every step in the terminal.** Each graph step, tool call, search round and database save is logged in plain language.
- **Optional tracing.** Langfuse traces for every model call when its keys are set.

---

## Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "fontSize": "13px", "primaryColor": "#F1F0EC", "primaryTextColor": "#FFFFFF", "secondaryTextColor": "#FFFFFF", "labelTextColor": "#FFFFFF", "primaryBorderColor": "#BDBAB0", "background": "transparent", "lineColor": "#D0D0CC", "textColor": "#FFFFFF", "tertiaryTextColor": "#FFFFFF", "clusterBkg": "transparent", "clusterBorder": "#BDBAB0", "titleColor": "#FFFFFF", "edgeLabelBackground": "#1F2328"}, "flowchart": {"curve": "basis", "nodeSpacing": 24, "rankSpacing": 56, "padding": 10}}}%%
flowchart LR
    subgraph client["Browser"]
        UI("Chat UI"):::grey
    end

    subgraph server["FastAPI server"]
        API("API"):::blue
        GRAPH("LangGraph<br/>chat flow + tools"):::blue
    end

    subgraph models["LiteLLM router"]
        GROQ("Groq gpt-oss-120b<br/>chat + final answers"):::green
        GEM("Gemini Flash-Lite<br/>summaries + evidence check"):::green
        BACKUP("OpenRouter<br/>backup for Groq"):::green
    end

    subgraph services["Data and services"]
        PG[("Postgres + pgvector<br/>chats, messages, memories")]:::yellow
        TAV("Tavily<br/>web search"):::yellow
        LF("Langfuse<br/>traces"):::yellow
    end

    UI <--> API
    API --> GRAPH
    GRAPH <--> GROQ
    GRAPH --> GEM
    GRAPH --> PG
    GRAPH --> TAV
    GRAPH -.-> LF
    GROQ -.->|"if it fails"| BACKUP

    classDef blue fill:#D6E8FF,stroke:#6FA3E6,color:#16345F,stroke-width:2px,font-weight:bold
    classDef green fill:#D3F0DA,stroke:#5DB57A,color:#14442A,stroke-width:2px,font-weight:bold
    classDef yellow fill:#FFF1A8,stroke:#E0BE3A,color:#5A4600,stroke-width:2px,font-weight:bold
    classDef grey fill:#F1F0EC,stroke:#BDBAB0,color:#3B3934,stroke-width:2px,font-weight:bold
    style client fill:none,stroke:#BDBAB0,stroke-width:2px
    style server fill:none,stroke:#BDBAB0,stroke-width:2px
    style models fill:none,stroke:#BDBAB0,stroke-width:2px
    style services fill:none,stroke:#BDBAB0,stroke-width:2px
    linkStyle default stroke:#D0D0CC,stroke-width:2px,fill:none
```

<sub>Blue: our code · Green: language models · Yellow: data and external services · Dashed: fallback or optional</sub>

### The chat graph

One run of this graph handles one user message. Nothing is kept in memory between messages: the database is the
only memory.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "fontSize": "13px", "primaryColor": "#F1F0EC", "primaryTextColor": "#FFFFFF", "secondaryTextColor": "#FFFFFF", "labelTextColor": "#FFFFFF", "primaryBorderColor": "#BDBAB0", "background": "transparent", "lineColor": "#D0D0CC", "textColor": "#FFFFFF", "tertiaryTextColor": "#FFFFFF", "edgeLabelBackground": "#1F2328"}, "flowchart": {"curve": "basis", "nodeSpacing": 22, "rankSpacing": 34, "padding": 8}}}%%
flowchart LR
    BP("build_prompt<br/>summary + 10 messages"):::blue --> CM("call_model<br/>Groq + tools"):::green
    CM -->|"text answer"| ST("save_turn<br/>write to database"):::blue
    CM -->|"tool request"| TL("tools<br/>run the tool"):::rose
    TL -->|"web answer ready"| UA("use_tool_answer<br/>answer = reply"):::blue
    TL -.->|"anything else"| CM
    UA --> ST
    ST --> US("update_summary<br/>fold old messages"):::blue

    classDef blue fill:#D6E8FF,stroke:#6FA3E6,color:#16345F,stroke-width:2px,font-weight:bold
    classDef green fill:#D3F0DA,stroke:#5DB57A,color:#14442A,stroke-width:2px,font-weight:bold
    classDef rose fill:#FFDDE3,stroke:#E98BA0,color:#6B1F33,stroke-width:2px,font-weight:bold
    classDef grey fill:#F1F0EC,stroke:#BDBAB0,color:#3B3934,stroke-width:2px,font-weight:bold
    linkStyle default stroke:#D0D0CC,stroke-width:2px,fill:none
```

<sub>Blue: steps in our code · Green: model call · Rose: tool run · Dashed: the loop back to the model · The run starts at build_prompt and ends after update_summary</sub>

---

## How it works

### Context engineering

Every prompt is built in the same order, with XML-style tags marking what each part is:

| Order | Part | Where it comes from |
|---|---|---|
| 1 | Instructions (`CHAT_INSTRUCTIONS`) | `core/prompts.py` |
| 2 | `<conversation_summary>` (only if one exists) | `chats.summary` |
| 3 | The latest 10 messages (5 user + 5 assistant), oldest first | `messages` table |
| 4 | Your new message, plus the note `use web tool to answer` if the checkbox is ticked | the request |

Memories are **not** pasted into every prompt. The model fetches them on demand with the `retrieve_memory` tool,
so the prompt stays small. The note from the checkbox is added only to the copy the model sees: the saved and
displayed message is exactly what you typed.

### Tools and tool calling

The tools in `chat/tools.py` are LangChain `@tool` functions with Pydantic input schemas. Their names, docstrings
and field descriptions are sent to the model with every request, so the docstrings are what tell the model *when*
to use each tool. The same `TOOLS` list is used to bind the tools to the model and to run them.

When the model answers with a tool request, it is appended to the prompt, `ToolNode` runs the tool, and the result
is appended as a `ToolMessage`. A finished `web_search` answer carries a flag (`artifact = True`), which the graph
reads to skip the second model call and show the answer directly.

### Memory

- `save_memory(content)` stores one short, third-person fact (up to 500 characters) with its 384-dimension embedding.
- `retrieve_memory(query)` embeds the query and returns the 5 closest memories by cosine distance (`<=>`).
- The model decides when to call either tool, guided by the tool descriptions.

Not built yet: updating or deleting memories, and duplicate checks.

### Rolling summary

- The latest 10 messages always stay raw in the prompt.
- Once 12 or more messages are unsummarised, everything that has left that window is folded by Gemini into the
  chat summary, together with the previous summary.
- The summary length grows with the chat: `min(500, 60 + 4 x messages summarised)` words.
- `chats.summary_upto_message_id` points at the last message the summary covers. If a summary call fails, the
  pointer does not move and the next message retries it.

### Web search pipeline

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "fontSize": "13px", "primaryColor": "#F1F0EC", "primaryTextColor": "#FFFFFF", "secondaryTextColor": "#FFFFFF", "labelTextColor": "#FFFFFF", "primaryBorderColor": "#BDBAB0", "background": "transparent", "lineColor": "#D0D0CC", "textColor": "#FFFFFF", "tertiaryTextColor": "#FFFFFF", "clusterBkg": "transparent", "clusterBorder": "#BDBAB0", "titleColor": "#FFFFFF", "edgeLabelBackground": "#1F2328"}, "flowchart": {"curve": "basis", "nodeSpacing": 22, "rankSpacing": 34, "padding": 8}}}%%
flowchart LR
    Q("web_search<br/>query + context"):::blue --> S("Tavily<br/>5 pages"):::blue
    S --> C("Chunk + embed<br/>top 20 chunks"):::blue
    C --> G{"Gemini check<br/>enough?"}:::yellow
    G -->|"yes"| A("Groq answer<br/>with [n] citations"):::green
    G -->|"no"| R("New query<br/>and context"):::green
    R -.->|"search again"| Q
    A --> O("Answer + sources<br/>shown to the user"):::rose

    classDef blue fill:#D6E8FF,stroke:#6FA3E6,color:#16345F,stroke-width:2px,font-weight:bold
    classDef green fill:#D3F0DA,stroke:#5DB57A,color:#14442A,stroke-width:2px,font-weight:bold
    classDef yellow fill:#FFF1A8,stroke:#E0BE3A,color:#5A4600,stroke-width:2px,font-weight:bold
    classDef rose fill:#FFDDE3,stroke:#E98BA0,color:#6B1F33,stroke-width:2px,font-weight:bold
    linkStyle default stroke:#D0D0CC,stroke-width:2px,fill:none
```

<sub>Blue: steps in our code · Green: model calls · Yellow: Gemini's decision · Rose: result · Dashed: the retry loop · The 3rd search skips the check and always answers</sub>

1. **Search.** Tavily returns up to 5 pages, with their full text as Markdown.
2. **Chunk and clean** each page (see below), **embed** the chunks with the local `bge-small-en-v1.5` model, and keep the **20 closest** to the query by cosine similarity.
3. **Check.** Gemini reads the chunks, the question and the background, and returns a reason and a yes/no verdict. The instructions are strict: every part of the question must be directly supported, otherwise the answer is no.
4. **Retry.** On "not enough", the reason goes back to the main model as the tool result. It writes a new query and context and calls the tool again. The 3rd search is final: its chunks are used without the check.
5. **Answer.** Groq writes the answer using only the chunks, with `[n]` after every fact, exact numbers and dates, and an explicit "I couldn't find information about X" for gaps. Any citation in the model's own `【3†L1-L3】` style is converted to `[3]`, or removed if no such source exists. A numbered **Sources** list is added.

### Chunking

`rag/md_chunker.py` turns a page into chunks that are small enough to embed and still make sense alone:

1. **Headings.** `MarkdownHeaderTextSplitter` splits the page into one piece per heading section.
2. **Length.** Sections over the budget (400 tokens including the header) are split with `RecursiveCharacterTextSplitter`, with a 40-token overlap. Tokens are counted with the bge tokenizer.
3. **Merge.** Pieces under 50 tokens are merged into a neighbour, or dropped if they contain no numbers.
4. **Header.** Each chunk starts with `[Page: <title> | Section: <heading path>]`, so it carries its context when embedded.
5. **Clean.** Images are removed, `[text](url)` becomes `text`, and navigation blocks, cookie and newsletter boilerplate and tiny chunks are dropped.

## Tech stack

<div align="center">

<img src="https://cdn.simpleicons.org/python" height="44" alt="Python" title="Python">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/fastapi" height="44" alt="FastAPI" title="FastAPI">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/langchain/2F7D6D" height="44" alt="LangChain and LangGraph" title="LangChain and LangGraph">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://www.google.com/s2/favicons?domain=litellm.ai&sz=64" height="44" alt="LiteLLM" title="LiteLLM">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/html5" height="44" alt="HTML5" title="HTML, CSS and JavaScript">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/javascript" height="44" alt="JavaScript" title="HTML, CSS and JavaScript">

<br>

<img src="https://www.google.com/s2/favicons?domain=groq.com&sz=64" height="44" alt="Groq" title="Groq">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/openrouter" height="44" alt="OpenRouter" title="OpenRouter">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/googlegemini" height="44" alt="Google Gemini" title="Google Gemini">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/huggingface" height="44" alt="Hugging Face" title="Hugging Face embeddings">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://www.google.com/s2/favicons?domain=tavily.com&sz=64" height="44" alt="Tavily" title="Tavily">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/postgresql" height="44" alt="PostgreSQL" title="PostgreSQL with pgvector">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://cdn.simpleicons.org/supabase" height="44" alt="Supabase" title="Supabase">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://www.google.com/s2/favicons?domain=langfuse.com&sz=64" height="44" alt="Langfuse" title="Langfuse">

</div>

---

## Getting started

### Requirements

- **Python 3.10 or newer**
- A **PostgreSQL** database with the **pgvector** extension (a free Supabase project works)
- API keys for **Groq**, **Gemini** and **Tavily** (OpenRouter and Langfuse are optional)
- Internet access: the page loads fonts and Markdown libraries from CDNs, and the embedding model (about 130 MB) is downloaded on the first run

Node.js is not needed.

> **Privacy.** Free model tiers may use your prompts for training. Do not put private data in the chat.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the database tables

Run this once in your database's SQL editor. It is the minimum schema the code uses; skip it if you already created these tables.

```sql
create extension if not exists vector;

create table if not exists public.chats (
  chat_id uuid primary key default gen_random_uuid(),
  title text,
  summary text,
  summary_upto_message_id uuid
);

create table if not exists public.messages (
  message_id uuid primary key default gen_random_uuid(),
  chat_id uuid not null references public.chats (chat_id),
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  sent_at timestamptz not null default now()
);
create index if not exists messages_chat_sent_idx on public.messages (chat_id, sent_at);

create table if not exists public.memories (
  memory_id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(384),
  created_at timestamptz not null default now()
);
```

The `384` must match the embedding model. If you switch to a different model, change it and re-embed your memories.

### 4. Add your environment variables

Create a file named `.env` in the project root. Never commit it.

```bash
# LLM providers
GROQ_API_KEY=<YOUR_KEY_HERE>
GROQ_API_KEY2=<YOUR_KEY_HERE>             # optional: a key from a second Groq account, used for web answers
GEMINI_API_KEY=<YOUR_KEY_HERE>
GEMINI_API_KEY2=<YOUR_KEY_HERE>           # optional: a second Gemini account, used only by the evals bot
OPENROUTER_API_KEY=<YOUR_KEY_HERE>        # only used if Groq fails

# Web search
TAVILY_API_KEY=<YOUR_KEY_HERE>

# Database (Supabase: Project Settings > Database > Connection string)
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME

# Optional: tracing
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: defaults shown
EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBED_DIM=384
APP_HOST=127.0.0.1
APP_PORT=8000
```

### 5. Start the app

```bash
python server.py
```

The terminal loads the models, checks the database and then prints the address. Open **http://127.0.0.1:8000**.
All logs for the app appear in that terminal.

> There is no Docker setup yet, and there are no migrations to run: the SQL above is the whole schema.

---

## Using the app

- **New chat** starts a draft. The chat is created in the database when you send its first message, and it appears in the sidebar with the start of that message as its title.
- **Use web tool** (next to Send): when ticked, the model is told to use the web tool for that message. When unticked, it decides for itself.
- **Show summary** (top right of a chat): opens the saved summary of the older part of that chat.
- Click a **steps** line above an answer to see what happened while it was being prepared.

---

## Project structure

```text
ResolveAI/
├── server.py              Entry point: FastAPI app, API endpoints, streamed replies, terminal logs
├── static/                The UI (index.html, style.css, app.js), served at /static
├── chat/                  The chat flow
│   ├── graph.py             Builds the LangGraph graph and its routing
│   ├── llm.py               Graph state and nodes: prompt, model call, save, summary
│   └── tools.py             Tools the model can call: save_memory, retrieve_memory, web_search
├── core/                  Shared foundations
│   ├── utils.py             Loads .env, database connection and query helper
│   ├── chat_store.py        Database helpers for chats, messages and summaries
│   ├── llm_router.py        The LiteLLM router (models and fallback) and Langfuse setup
│   └── prompts.py           Chat and summary prompts, and the Message type
├── rag/                   The web search pipeline
│   ├── web_rag.py           One search round: Tavily, chunks, check, answer
│   ├── chunk_judge.py       The Gemini evidence check
│   ├── md_chunker.py        Markdown chunking and cleaning
│   └── embedder.py          Embeddings and cosine top-k
├── requirements.txt
├── PROJECT_CONFIG.md      Original plan: stack, models, limits, feature checklist
└── README.md
```

---