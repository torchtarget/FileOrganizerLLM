"""LLM provider utilities for FileOrganizerLLM."""
from __future__ import annotations

from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(model_name: str, provider: str = "ollama", *, openai_api_key: Optional[str] = None) -> BaseChatModel:
    """Return a LangChain chat model instance.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    provider : str, optional
        Provider name, either ``"ollama"`` or ``"openai"``.
    openai_api_key : str, optional
        API key for OpenAI models. If not provided, ``OPENAI_API_KEY`` env var is used.

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
    raise ValueError(f"Unknown provider: {provider}")
