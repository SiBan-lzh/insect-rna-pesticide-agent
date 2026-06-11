"""
llm_config.py — LLM factory with provider registry (openai/anthropic/google).

Usage:
    from llm_config import get_llm, get_default_llm, Settings

    llm = get_llm()
    llm = get_llm(Settings(model_name="gpt-4o"))
    llm = get_default_llm()
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load .env — find_dotenv walks up from this file to locate .env
# so the script works regardless of which directory it is run from
load_dotenv(find_dotenv(".env", usecwd=True), override=True)
for _k, _v in list(os.environ.items()):
    if "$" in _v:
        os.environ[_k] = os.path.expandvars(_v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: str = ""
    model_name: str = ""
    api_key: str = ""
    base_url: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        missing = [k for k in ("model_provider", "model_name", "api_key") if not getattr(self, k)]
        if missing:
            raise ValueError(
                f"Missing LLM config: {', '.join(missing)}. "
                f"Set them in .env or pass a Settings(...) instance."
            )
        if self.model_provider not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported provider '{self.model_provider}'. "
                f"Available: {', '.join(MODEL_REGISTRY)}"
            )
        return self


def build_openai(s: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key,
        base_url=s.base_url or "https://api.openai.com/v1",
    )


def build_anthropic(s: Settings) -> ChatAnthropic:
    kwargs: dict = {"model": s.model_name, "api_key": s.api_key}
    if s.base_url:
        kwargs["base_url"] = s.base_url
    return ChatAnthropic(**kwargs)


def build_google(s: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=s.model_name,
        google_api_key=s.api_key,
    )


MODEL_REGISTRY: dict[str, Callable[[Settings], object]] = {
    "openai": build_openai,
    "anthropic": build_anthropic,
    "google": build_google,
}


def get_llm(settings: Optional[Settings] = None):
    s = settings or Settings()
    return MODEL_REGISTRY[s.model_provider](s)


@lru_cache(maxsize=1)
def get_default_llm():
    # Cached on first call. Restart the process to pick up .env changes.
    return get_llm()
