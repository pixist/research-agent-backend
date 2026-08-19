"""Split source documents into overlapping chunks for embedding."""
from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Break ``text`` into ~``size`` character windows overlapping by ``overlap``.

    Windows prefer to end on whitespace so we don't cut words in half.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Nudge the boundary back to the last whitespace inside the window.
        if end < n:
            window = text.rfind(" ", start, end)
            if window > start:
                end = window
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
