import json
import httpx
from .base import LLMProvider, LLMMessage, LLMProviderError, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": msgs, "stream": False},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            content = data.get("message", {}).get("content", "")
        except Exception as exc:
            raise LLMProviderError("ollama", "request_failed") from exc
        if not content or not content.strip():
            raise LLMProviderError("ollama", "empty_response")
        return LLMResponse(content=content, model=self.model)

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        emitted = False
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": msgs, "stream": True},
                    timeout=120,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            emitted = True
                            yield content
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError("ollama", "request_failed") from exc
        if not emitted:
            raise LLMProviderError("ollama", "empty_response")
