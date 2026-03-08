from abc import ABC, abstractmethod
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        ...

    @abstractmethod
    async def stream(self, messages: list[LLMMessage], system: str = ""):
        ...
