"""Parse judge message bodies from WildBench batch results (lenient JSON and fallbacks)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def unwrap_markdown_json_fence(content: str) -> str:
    """Strip optional ``` / ```json fences so json.loads sees raw JSON."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if not lines:
        return text
    lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


_SMART_QUOTE_TRANS = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)


def normalize_typographic_quotes(s: str) -> str:
    return s.translate(_SMART_QUOTE_TRANS)


def repair_trailing_commas_json(s: str) -> str:
    """Remove trailing commas before } or ] (best-effort for model JSON glitches)."""
    return re.sub(r",(\s*[}\]])", r"\1", s)


def unwrap_one_json_string_layer(s: str) -> str:
    """If s is a JSON-encoded string whose value looks like JSON, return the inner string."""
    t = s.strip()
    if len(t) < 2 or t[0] != '"':
        return s
    try:
        inner = json.loads(t)
    except json.JSONDecodeError:
        return s
    if not isinstance(inner, str):
        return s
    return inner.strip()


def coerce_judge_message_content(content: Any) -> str:
    """OpenAI message.content is usually str; batch responses may use text-part lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.replace("\ufeff", "").strip()
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "text" in p:
                    parts.append(str(p["text"]))
        return "\n".join(parts).replace("\ufeff", "").strip()
    return str(content).replace("\ufeff", "").strip()


def first_balanced_json_object(text: str) -> str | None:
    """First top-level {...} respecting JSON double-quoted strings and escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    i = start
    in_str = False
    escape = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def try_json_load_dict(s: str) -> dict[str, Any] | None:
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def try_ast_literal_dict(s: str) -> dict[str, Any] | None:
    t = s.strip()
    if not t.startswith("{"):
        return None
    try:
        out = ast.literal_eval(t)
        return out if isinstance(out, dict) else None
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return None


_SCORE_RES: list[re.Pattern[str]] = [
    re.compile(r'"(?:score|Score)"\s*:\s*(-?(?:[0-9]+\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?)'),
    re.compile(r"'(?:score|Score)'\s*:\s*(-?(?:[0-9]+\.[0-9]+|[0-9]+))"),
    re.compile(r"\bscore\s*[:=]\s*(-?(?:[0-9]+\.[0-9]+|[0-9]+))\b", re.I),
]


def regex_extract_score_dict(text: str) -> dict[str, Any] | None:
    """Last-resort: pull a numeric score from messy judge text."""
    for rx in _SCORE_RES:
        m = rx.search(text)
        if not m:
            continue
        raw = m.group(1)
        try:
            v = float(raw)
        except ValueError:
            continue
        return {
            "score": v,
            "_parse_recovery": "regex_score",
            "_parse_recovery_snippet": text[:800],
        }
    return None


def expand_parse_blobs(surface: str) -> list[str]:
    """Surface strings to try (fences, smart quotes, double-encoded JSON strings)."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        t = s.strip()
        if not t or t in seen:
            return
        seen.add(t)
        out.append(t)

    add(surface)
    add(unwrap_markdown_json_fence(surface))
    add(normalize_typographic_quotes(surface))
    add(normalize_typographic_quotes(unwrap_markdown_json_fence(surface)))

    peeled = unwrap_markdown_json_fence(surface)
    for _ in range(5):
        nxt = unwrap_one_json_string_layer(peeled)
        if nxt == peeled:
            break
        peeled = nxt
        add(peeled)
        add(unwrap_markdown_json_fence(peeled))
        add(normalize_typographic_quotes(peeled))
    return out


def parse_judge_json_object(content: Any) -> dict[str, Any] | None:
    """Parse judge message content into a dict; tolerate fences, prose, and mild JSON breakage."""
    text = coerce_judge_message_content(content)
    if not text:
        return None

    blobs = expand_parse_blobs(text)

    for blob in blobs:
        candidates: list[str] = [blob]
        bal = first_balanced_json_object(blob)
        if bal:
            candidates.append(bal)
        idx = blob.find("{")
        if idx != -1:
            tail = blob[idx:].strip()
            if tail not in candidates:
                candidates.append(tail)

        for sl in candidates:
            if not sl or not sl.strip().startswith("{"):
                continue
            base = sl.strip()
            variants = [
                base,
                normalize_typographic_quotes(base),
                repair_trailing_commas_json(base),
                repair_trailing_commas_json(normalize_typographic_quotes(base)),
            ]
            for v in variants:
                if not v:
                    continue
                d = try_json_load_dict(v)
                if d is not None:
                    return d
                d = try_ast_literal_dict(v)
                if d is not None:
                    out = dict(d)
                    out.setdefault("_parse_recovery", "ast_literal_eval")
                    return out

    for blob in blobs:
        d = regex_extract_score_dict(blob)
        if d is not None:
            return d
    return None


def score_value_from_parsed(parsed: dict[str, Any]) -> Any:
    if "score" in parsed:
        return parsed.get("score")
    return parsed.get("Score")
