"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./langtutor.db"

    # STT provider: dummy | whisper_api | whisper_local
    stt_provider: str = "whisper_local"

    # LLM provider: ollama | openai | anthropic
    llm_provider: str = "ollama"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = 300.0

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-20250514"

    # App defaults
    default_language: str = "en"

    # STT confidence threshold for marking uncertain
    stt_confidence_threshold: float = 0.6

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
