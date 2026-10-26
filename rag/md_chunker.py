"""
md_chunker.py — Heading-based chunking for markdown (e.g. Tavily raw_content).

Steps
  1. HEADINGS  LangChain MarkdownHeaderTextSplitter → one section per heading,
               with the heading path in metadata ({"h1": ..., "h2": ...})
  2. LENGTH    sections longer than the budget → RecursiveCharacterTextSplitter
               (markdown-aware separators, counted in tokens, with overlap)
  3. MERGE     pieces under min_tokens → merged into a neighbour
               (same section first, then same top section), or dropped if they
               have no numbers in them
  4. HEADER    "[Page: <title> | Section: <path>]" + content → Chunk.text (embed this)
  5. CLEAN     every chunk body: images removed, [text](url) → text;
               navigation, boilerplate (cookies, newsletter ...) and tiny chunks dropped

Install:
  pip install langchain-text-splitters
  pip install transformers          # optional: exact token counts with bge's tokenizer

Usage:
  chunker = MarkdownChunker()       # create once, reuse
  chunks = chunker.chunk(markdown_text, url=url, page_title=title)   # list[Chunk], already cleaned
  for c in chunks:
      c.chunk_id, c.text, c.url

Known limits (from MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter):
  - a 4-space-indented "# comment" can be taken as a heading
  - blank lines between paragraphs are not kept
  - very long tables / code blocks can be cut in the middle
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    max_tokens: int = 400          # whole chunk INCLUDING header (bge-small hard limit is 512)
    min_tokens: int = 50           # smaller pieces get merged (or dropped if no numbers)
    overlap_tokens: int = 40       # overlap between pieces of one long section
    header_path_levels: int = 3    # how many heading levels to show in the header
    tokenizer_name: str = "BAAI/bge-small-en-v1.5"
    nav_link_line_ratio: float = 0.5   # navigation chunk: at least this share of lines is only a link ...
    nav_max_prose_words: int = 40      # ... and its other lines hold fewer words than this
    min_chars: int = 40                # chunks shorter than this are dropped after cleaning


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str                  # "<url>#<n>"
    text: str                      # header + content  ← embed this
    url: str
    embedding: Optional[list[float]] = None   # filled in by embedder.embed_chunks


# ---------------------------------------------------------------------------
# Internal piece (before the header is added)
# ---------------------------------------------------------------------------

@dataclass
class _Piece:
    path: list[str]                # ["LangGraph Guide", "Memory"]
    content: str


_HEADING_KEYS = ["h1", "h2", "h3", "h4", "h5", "h6"]
_DIGIT = re.compile(r"\d")

# Cleaning (step 5)
_URL = r"\((?:[^()]|\([^()]*\))*\)"                      # (https://...), allows one level of () inside
_IMAGE = re.compile(rf"!\[[^\]]*\]{_URL}")
_LINK = re.compile(rf"\[([^\]]*)\]{_URL}")
_LINK_ONLY_LINE = re.compile(rf"^\s*(?:[*+-]\s+)?\[[^\]]*\]{_URL}\s*$")
_EMPTY_ITEM = re.compile(r"^\s*(?:[*+-]|\d+\.)?\s*$")   # a bullet or line left empty by the cleaning
_BOILERPLATE = re.compile(r"cookie|newsletter|subscribe|privacy statement|skip to|all fields required|accept all", re.I)

# Tried in order: the biggest boundary that makes pieces fit wins.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _remove_overlap(a: str, b: str, min_chars: int = 20) -> str:
    """If b starts with the end of a (overlap from splitting), drop that part from b."""
    for k in range(min(len(a), len(b)), min_chars - 1, -1):
        if a.endswith(b[:k]):
            return b[k:].lstrip()
    return b


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def approx_token_count(text: str) -> int:
    """Rough fallback: ~4 characters per token for English."""
    return max(1, len(text) // 4)


def make_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Exact counts with the embedding model's tokenizer; falls back to approx."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception as e:  # not installed or no network
        warnings.warn(f"Using approximate token counts ({e.__class__.__name__}).")
        return approx_token_count


# ---------------------------------------------------------------------------
# The chunker
# ---------------------------------------------------------------------------

class MarkdownChunker:
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ):
        self.cfg = config or ChunkerConfig()
        self.count = count_tokens or make_token_counter(self.cfg.tokenizer_name)
        self.heading_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#" * (i + 1), key) for i, key in enumerate(_HEADING_KEYS)],
            strip_headers=True,    # heading text goes to metadata, not into the content
        )

    # ---- public -----------------------------------------------------------

    def chunk(self, markdown: str, url: str = "", page_title: str = "") -> list[Chunk]:
        """Split markdown into chunks and return them cleaned (step 5 included)."""
        pieces = self._split_by_headings(markdown)                  # step 1
        pieces = self._split_long(pieces, page_title)               # step 2
        pieces = self._merge_small(pieces, page_title)              # step 3
        chunks = [                                                  # step 4
            Chunk(chunk_id=f"{url}#{i}", text=self._with_header(p, page_title), url=url)
            for i, p in enumerate(pieces)
        ]
        return self._clean_chunks(chunks)                           # step 5

    # ---- step 1: headings -------------------------------------------------

    def _split_by_headings(self, markdown: str) -> list[_Piece]:
        markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
        pieces = []
        for doc in self.heading_splitter.split_text(markdown):
            content = doc.page_content.replace("  \n", "\n").strip()   # splitter joins lines with "  \n"
            if not content:
                continue
            path = [doc.metadata[k] for k in _HEADING_KEYS if k in doc.metadata]
            pieces.append(_Piece(path=path, content=content))
        return pieces

    # ---- step 2: long sections -------------------------------------------

    def _split_long(self, pieces: list[_Piece], page_title: str) -> list[_Piece]:
        out = []
        for piece in pieces:
            budget = self._content_budget(piece.path, page_title)
            if self.count(piece.content) <= budget:
                out.append(piece)
                continue
            splitter = RecursiveCharacterTextSplitter(
                separators=_SEPARATORS,              # paragraphs → lines → sentences → words
                keep_separator="end",                # "." stays at the end of its sentence
                chunk_size=budget,
                chunk_overlap=min(self.cfg.overlap_tokens, budget // 4),
                length_function=self.count,
            )
            for part in splitter.split_text(piece.content):
                if part.strip():
                    out.append(_Piece(path=piece.path, content=part.strip()))
        return out

    # ---- step 3: merge small pieces --------------------------------------

    def _merge_small(self, pieces: list[_Piece], page_title: str) -> list[_Piece]:
        """
        A piece under min_tokens is merged into, in this order:
          1. the previous piece, if it has the SAME path (e.g. the small tail of a long section)
          2. the next piece, if it has the same path
          3. the next piece, if it is in the same top section (same h1),
             or if this piece is the page intro (text before the first heading)
          4. the previous piece, if it is in the same top section
        if the result still fits. Otherwise it is kept only if it contains a number
        (prices, dates, versions); else dropped.
        """
        items = list(pieces)
        out: list[_Piece] = []
        i = 0
        while i < len(items):
            p = items[i]
            if self.count(p.content) >= self.cfg.min_tokens:
                out.append(p)
                i += 1
                continue

            prev = out[-1] if out else None
            nxt = items[i + 1] if i + 1 < len(items) else None
            options = [
                ("prev", prev, prev is not None and prev.path == p.path),
                ("next", nxt, nxt is not None and nxt.path == p.path),
                ("next", nxt, nxt is not None and (nxt.path[:1] == p.path[:1] or not p.path)),
                ("prev", prev, prev is not None and prev.path[:1] == p.path[:1]),
            ]

            merged_ok = False
            for side, other, allowed in options:
                if not allowed:
                    continue
                merged = self._merge(other, p) if side == "prev" else self._merge(p, other)
                if self._fits(merged, page_title):
                    if side == "prev":
                        out[-1] = merged
                    else:
                        items[i + 1] = merged          # may merge again if still small
                    merged_ok = True
                    break

            if not merged_ok and _DIGIT.search(p.content):
                out.append(p)                          # keep small facts with numbers
            i += 1
        return out

    @staticmethod
    def _merge(a: _Piece, b: _Piece) -> _Piece:
        """Merged path = shared parent. Lower headings are kept as bold labels."""
        path = []
        for x, y in zip(a.path, b.path):
            if x != y:
                break
            path.append(x)

        def labelled(p: _Piece, content: str) -> str:
            extra = p.path[len(path):]
            return f"**{' > '.join(extra)}**\n{content}" if extra else content

        # pieces of the same long section share overlap text → don't repeat it
        b_content = _remove_overlap(a.content, b.content) if a.path == b.path else b.content
        if not b_content:
            return _Piece(path=a.path, content=a.content)
        return _Piece(path=path, content=f"{labelled(a, a.content)}\n\n{labelled(b, b_content)}")

    # ---- header + sizes ---------------------------------------------------

    def _header(self, path: list[str], page_title: str) -> str:
        section = " > ".join(path[-self.cfg.header_path_levels:]) if path else "Main"
        return f"[Page: {page_title or 'Untitled'} | Section: {section}]"

    def _with_header(self, piece: _Piece, page_title: str) -> str:
        return f"{self._header(piece.path, page_title)}\n{piece.content}"

    def _content_budget(self, path: list[str], page_title: str) -> int:
        return self.cfg.max_tokens - self.count(self._header(path, page_title) + "\n")

    def _fits(self, piece: _Piece, page_title: str) -> bool:
        return self.count(self._with_header(piece, page_title)) <= self.cfg.max_tokens

    # ---- step 5: clean ----------------------------------------------------

    def _clean_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Clean every chunk's body (header kept); drop navigation, boilerplate and tiny chunks.

        chunk_id values stay as they were, so a gap in the numbers means a chunk was dropped.
        """
        cleaned = []
        for chunk in chunks:
            header, _, body = chunk.text.partition("\n")
            text = self._clean_text(body)
            if self._is_navigation(body) or self._is_boilerplate(text) or len(text) < self.cfg.min_chars:
                continue
            cleaned.append(replace(chunk, text=f"{header}\n{text}"))
        return cleaned

    @staticmethod
    def _clean_text(body: str) -> str:
        """Remove images, turn [text](url) into text, and drop the lines this leaves empty."""
        body = _LINK.sub(r"\1", _IMAGE.sub("", body))
        lines = (re.sub(r"[ \t]+", " ", line).strip() for line in body.splitlines())
        return "\n".join(line for line in lines if not _EMPTY_ITEM.match(line))

    def _is_navigation(self, body: str) -> bool:
        lines = [line for line in body.splitlines() if line.strip()]
        link_lines = [line for line in lines if _LINK_ONLY_LINE.match(line)]
        prose_words = sum(len(line.split()) for line in lines if line not in link_lines)
        return (
            bool(lines)
            and len(link_lines) / len(lines) >= self.cfg.nav_link_line_ratio
            and prose_words < self.cfg.nav_max_prose_words
        )

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        hits = len(_BOILERPLATE.findall(text))
        return hits >= 2 or (hits == 1 and len(text) < 300)