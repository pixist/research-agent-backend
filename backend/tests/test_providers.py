from app.config import Settings


def test_openrouter_chat_keeps_embeddings_offline_without_embed_key():
    # chat on openrouter, no embed key -> embeddings must NOT go to openrouter
    # (it has no embeddings endpoint), so they fall back to the offline hash.
    s = Settings(
        openai_api_key="sk-or-test",
        openai_base_url="https://openrouter.ai/api/v1",
    )
    assert s.use_fake_provider is False
    assert s.use_fake_embeddings is True


def test_explicit_embed_provider_is_used():
    s = Settings(
        openai_api_key="sk-or-test",
        openai_base_url="https://openrouter.ai/api/v1",
        embed_api_key="sk-oai",
        embed_base_url="https://api.openai.com/v1",
    )
    assert s.use_fake_embeddings is False
    assert s.effective_embed_base_url == "https://api.openai.com/v1"
