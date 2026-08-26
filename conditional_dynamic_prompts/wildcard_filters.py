from __future__ import annotations

import re

ANIMA_PREFIX = "@anima{"


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _find_closing_brace(text: str, body_start: int) -> int | None:
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def remove_anima_wildcard_content(value: str) -> str:
    """Remove balanced @anima{...} sections loaded through __wildcard__."""
    chunks: list[str] = []
    cursor = 0
    search_from = 0

    while True:
        start = value.find(ANIMA_PREFIX, search_from)
        if start < 0:
            break
        if _is_escaped(value, start):
            search_from = start + len(ANIMA_PREFIX)
            continue

        body_start = start + len(ANIMA_PREFIX)
        end = _find_closing_brace(value, body_start)
        if end is None:
            break

        chunks.append(value[cursor:start])
        chunks.append(" ")
        cursor = end + 1
        search_from = cursor

    chunks.append(value[cursor:])
    filtered = "".join(chunks)

    # Removing a complete comma-separated item should not leave empty tags.
    filtered = re.sub(r"[ \t]+", " ", filtered)
    filtered = re.sub(r",\s*(?:,\s*)+", ", ", filtered)
    filtered = re.sub(r"^\s*,\s*", "", filtered)
    filtered = re.sub(r"\s*,\s*$", "", filtered)
    return filtered.strip()
