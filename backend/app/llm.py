"""Chat client that streams tokens, with an offline fallback."""
from __future__ import annotations

from collections.abc import AsyncIterator

from .config import Settings


class ChatClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        if not settings.use_fake_provider:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key, base_url=settings.openai_base_url
            )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if self._client is None:
            async for token in _fake_stream(messages):
                yield token
            return
        stream = await self._client.chat.completions.create(
            model=self._settings.chat_model, messages=messages, stream=True
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta


async def _fake_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Echo a plausible markdown answer built from the supplied context.

    Streams word by word so the frontend's incremental rendering is exercised.
    """
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    question = user.split("Question:")[-1].strip() if "Question:" in user else user
    answer = (
        f"## Research summary\n\n"
        f"Here is what I found on **{question[:120]}**.\n\n"
        f"- The offline demo model is answering because no API key is configured.\n"
        f"- Retrieved context and web results were passed in as grounding.\n\n"
        f"Set `OPENAI_API_KEY` to get a real model's answer.\n"
    )
    for word in answer.split(" "):
        yield word + " "
