from __future__ import annotations

import re
from typing import Iterable

from .data import Chunk, Document
from .text import normalize_text


HEADING_RE = re.compile(
    r"(?im)^(?P<heading>\s*(?:chương\s+[IVXLCDM0-9]+|mục\s+[IVXLCDM0-9]+|"
    r"điều\s+\d+[a-zđ]?\.?|phần\s+(?:thứ\s+)?[A-ZÀ-ỸIVXLCDM0-9]+)"
    r"(?:\s*[-.:–—]?\s*[^\n]{0,180})?)\s*$"
)


def _legal_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    result: list[tuple[str, str]] = []
    if text[: matches[0].start()].strip():
        result.append(("Mở đầu", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((normalize_text(match.group("heading")), normalize_text(text[match.start() : end])))
    return result


def _window_text(text: str, max_chars: int, overlap_chars: int, min_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 1:
        paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    if len(paragraphs) == 1:
        step = max(1, max_chars - overlap_chars)
        return [text[start : start + max_chars].strip() for start in range(0, len(text), step) if len(text[start:].strip()) >= min_chars]

    windows: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                windows.append("\n\n".join(current))
                current, current_size = [], 0
            windows.extend(_window_text(paragraph, max_chars, overlap_chars, min_chars))
            continue
        added = len(paragraph) + (2 if current else 0)
        if current and current_size + added > max_chars:
            completed = "\n\n".join(current)
            windows.append(completed)
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail and tail_size + len(previous) + 2 > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + (2 if len(tail) > 1 else 0)
            current, current_size = tail, len("\n\n".join(tail))
        current.append(paragraph)
        current_size += len(paragraph) + (2 if len(current) > 1 else 0)
    if current:
        windows.append("\n\n".join(current))
    return [window for window in windows if len(window) >= min_chars]


def chunk_documents(documents: Iterable[Document], config: dict) -> list[Chunk]:
    cfg = config["corpus"]
    result: list[Chunk] = []
    for document in documents:
        section_number = 0
        for heading, section in _legal_sections(document.text):
            windows = _window_text(
                section,
                int(cfg["max_chunk_chars"]),
                int(cfg["overlap_chars"]),
                int(cfg["min_chunk_chars"]),
            )
            for window_number, window in enumerate(windows):
                prefix_parts = [part for part in (document.title, heading) if part]
                prefix = " — ".join(prefix_parts)
                text = f"{prefix}\n{window}" if prefix and prefix.lower() not in window[:300].lower() else window
                result.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::s{section_number:04d}::w{window_number:03d}",
                        doc_id=document.doc_id,
                        title=document.title,
                        heading=heading,
                        text=normalize_text(text),
                        metadata={
                            **document.metadata,
                            "_section_index": section_number,
                            "_window_index": window_number,
                            "_legal_heading": heading,
                        },
                    )
                )
            section_number += 1
    return result
