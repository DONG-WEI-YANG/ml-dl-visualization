import anthropic
from .base import LLMProvider, LLMMessage, LLMResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(
            content=resp.content[0].text,
            model=resp.model,
            usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        )

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            async for text in stream.text_stream:
                yield text
