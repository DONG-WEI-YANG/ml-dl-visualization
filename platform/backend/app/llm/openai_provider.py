from openai import AsyncOpenAI
from .base import LLMProvider, LLMMessage, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        resp = await self.client.chat.completions.create(model=self.model, messages=msgs)
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=resp.model,
            usage={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
        )

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        stream = await self.client.chat.completions.create(
            model=self.model, messages=msgs, stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
