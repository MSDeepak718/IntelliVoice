"""
IntelliVoice — Application Settings

Pydantic-based settings loaded from environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- General ----------
    app_name: str = "IntelliVoice"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ---------- Server ----------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ---------- CORS ----------
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # ---------- MongoDB ----------
    mongodb_uri: str = "mongodb://intellivoice:intellivoice@localhost:27017"
    mongodb_db_name: str = "intellivoice"


    # ---------- HuggingFace ----------
    hf_token: str = ""
    hf_home: str = "./models"

    # ---------- GPU ----------
    cuda_visible_devices: str = "0"
    gpu_memory_fraction: float = 0.92

    # ---------- Model Paths (auto-downloaded if empty) ----------
    silero_vad_model_path: str = ""
    whisper_model_path: str = ""
    emotion_model_path: str = ""
    ecapa_tdnn_model_path: str = ""
    qwen3_14b_model_path: str = ""
    piper_model_path: str = ""

    # ---------- Audio ----------
    sample_rate: int = 16000
    chunk_size_ms: int = 30
    vad_threshold: float = 0.35
    vad_min_speech_ms: int = 80
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 120
    agc_target_db: float = -20.0
    max_audio_length_s: int = 30

    # ---------- LLM ----------
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


    @property
    def project_root(self) -> Path:
        """Return project root directory."""
        return Path(__file__).resolve().parent.parent

    @property
    def models_dir(self) -> Path:
        """Return models directory."""
        p = Path(self.hf_home)
        if not p.is_absolute():
            p = self.project_root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
