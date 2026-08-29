import anthropic
from .base import LLMProvider, LLMMessage, LLMProviderError, LLMResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except Exception as exc:
            raise LLMProviderError("anthropic", "request_failed") from exc
        text_blocks = [getattr(block, "text", "") for block in resp.content]
        content = "".join(text_blocks).strip()
        if not content:
            raise LLMProviderError("anthropic", "empty_response")
        usage = None
        if resp.usage is not None:
            usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        return LLMResponse(
            content=content,
            model=resp.model,
            usage=usage,
        )

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        emitted = False
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        emitted = True
                        yield text
        except Exception as exc:
            raise LLMProviderError("anthropic", "request_failed") from exc
        if not emitted:
            raise LLMProviderError("anthropic", "empty_response")
