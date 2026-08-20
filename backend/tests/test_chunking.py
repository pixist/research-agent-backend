from app.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", size=100, overlap=10) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=120, overlap=30)
    assert len(chunks) > 1
    assert chunks[0].split()[-1] in chunks[1]


def test_overlap_not_smaller_than_size_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text here", size=10, overlap=10)
