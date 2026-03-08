import json
import httpx
from .base import LLMProvider, LLMMessage, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False},
                timeout=120,
            )
            data = resp.json()
        return LLMResponse(content=data["message"]["content"], model=self.model)

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": True},
                timeout=120,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and data["message"].get("content"):
                            yield data["message"]["content"]
