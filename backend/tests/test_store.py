from app.store import Chunk, VectorStore


def test_empty_store_returns_no_matches():
    store = VectorStore()
    assert store.search([0.1, 0.2, 0.3], k=3) == []


def test_search_ranks_by_cosine_similarity():
    store = VectorStore()
    store.add([Chunk("apple", "a"), Chunk("orange", "b")], [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([0.9, 0.1], k=2)
    assert results[0].chunk.text == "apple"
    assert results[0].score > results[1].score


def test_add_rejects_mismatched_lengths():
    import pytest

    store = VectorStore()
    with pytest.raises(ValueError):
        store.add([Chunk("x", "s")], [[1.0], [2.0]])
