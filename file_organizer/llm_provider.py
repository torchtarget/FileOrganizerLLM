"""LLM provider utilities for FileOrganizerLLM."""
from __future__ import annotations

from typing import Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

try:
    from langchain_fireworks import ChatFireworks
except Exception:  # pragma: no cover - optional dependency may not be installed
    ChatFireworks = None  # type: ignore
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(
    model_name: str,
    provider: str = "ollama",
    *,
    openai_api_key: Optional[str] = None,
    fireworks_api_key: Optional[str] = None,
) -> BaseChatModel:
    """Return a LangChain chat model instance.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    provider : str, optional
        Provider name, one of ``"ollama"``, ``"openai"`` or ``"fireworks"``.
    openai_api_key : str, optional
        API key for OpenAI models. If not provided, ``OPENAI_API_KEY`` env var is used.
    fireworks_api_key : str, optional
        API key for Fireworks models. If not provided, ``FIREWORKS_API_KEY`` env
        var is used.

    Returns
    -------
    BaseChatModel
        Initialized LangChain chat model.
    """
    provider = provider.lower()
    if provider == "ollama":
        return ChatOllama(model=model_name)
    if provider == "openai":
        return ChatOpenAI(model=model_name, api_key=openai_api_key)
    if provider == "fireworks":
        if ChatFireworks is None:
            raise ImportError(
                "langchain-fireworks is not installed. Install with 'pip install langchain-fireworks'."
            )
        kwargs = {}
        if fireworks_api_key is not None:
            kwargs["api_key"] = fireworks_api_key
        return ChatFireworks(
            model=model_name,
            **kwargs,
        )
    raise ValueError(f"Unknown provider: {provider}")
