"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./langtutor.db"

    # Legacy STT provider default
    stt_provider: str = "whisper_local"
    # Explicit STT providers by use case
    live_stt_provider: str = "whisper_local"
    final_stt_provider: str = "whisper_api"

    # LLM provider: ollama | openai | anthropic
    llm_provider: str = "openai"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = 300.0

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-mini"

    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-20250514"

    # App defaults
    default_language: str = "en"
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_origin_regex: str = r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$"

    # STT confidence threshold for marking uncertain
    stt_confidence_threshold: float = 0.6

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
