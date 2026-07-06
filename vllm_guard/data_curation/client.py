import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: Optional[str]
    api_key_env: str


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", None, "OPENAI_API_KEY"),
    "gemini": ProviderConfig("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ProviderConfig("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "vllm": ProviderConfig("vllm", None, "OPENAI_API_KEY"),
}


def resolve_api_key(explicit_api_key: str | None, env_var: str) -> str:
    api_key = explicit_api_key or os.getenv(env_var, "")
    if not api_key:
        raise ValueError(f"No API key provided. Set --api-key or export {env_var}")
    return api_key


def build_openai_compatible_client(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> AsyncOpenAI:
    if provider not in PROVIDER_CONFIGS:
        supported = ", ".join(sorted(PROVIDER_CONFIGS))
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

    config = PROVIDER_CONFIGS[provider]
    resolved_api_key = resolve_api_key(api_key, config.api_key_env)
    resolved_base_url = base_url or config.base_url

    kwargs = {
        "api_key": resolved_api_key,
        "timeout": timeout,
    }
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return AsyncOpenAI(**kwargs)
