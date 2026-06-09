"""
Tests for the pipeline integration and the model registry.
"""

import pytest
import torch

from config.model_registry import ModelRegistry, LoadingOrder, ModelPrecision
from config.settings import Settings, get_settings
from backend.core.gpu_manager import GPUManager
from backend.core.session_manager import SessionManager, SessionState


class TestSettings:
    """Tests for application settings."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.app_name == "IntelliVoice"
        assert settings.sample_rate == 16000
        assert settings.port == 8000

    def test_models_dir_creation(self, tmp_path):
        settings = Settings(hf_home=str(tmp_path / "models"))
        models_dir = settings.models_dir
        assert models_dir.exists()

    def test_is_production(self):
        assert not Settings(app_env="development").is_production
        assert Settings(app_env="production").is_production


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_all_models_listed(self):
        models = ModelRegistry.get_all_models()
        # 2 (preprocessing) + 1 (asr) + 2 (understanding) + 1 (reasoning) + 1 (tts) = 7
        assert len(models) == 7

    def test_peak_vram_under_16gb(self):
        # Peak VRAM is just the sum of all estimated VRAM
        peak = sum(m.estimated_vram_mb for m in ModelRegistry.get_all_models())
        assert peak < 16 * 1024, f"Peak {peak / 1024:.2f} GB exceeds 16 GB"

    def test_fast_llm_config(self):
        llm = ModelRegistry.FAST_LLM
        assert llm.hf_model_id == "Qwen/Qwen2.5-3B-Instruct"
        assert llm.precision == ModelPrecision.INT4
        assert llm.quantization_config is not None
        assert llm.quantization_config["load_in_4bit"] is True
        assert llm.quantization_config["bnb_4bit_use_double_quant"] is True


class TestGPUManager:
    """Tests for GPUManager."""

    def test_device_detection(self):
        gpu = GPUManager()
        assert gpu.device is not None

    def test_memory_stats(self):
        gpu = GPUManager()
        stats = gpu.get_memory_stats()
        for key in ("allocated_gb", "reserved_gb", "total_gb", "free_gb"):
            assert key in stats

    def test_register_and_unregister(self):
        gpu = GPUManager()
        dummy = torch.nn.Linear(2, 2)
        gpu.register_model(
            "test_dummy", dummy, order=LoadingOrder.PREPROCESSING, vram_mb=10
        )
        assert gpu.is_loaded("test_dummy")
        gpu.unregister_model("test_dummy")
        assert not gpu.is_loaded("test_dummy")


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        manager = SessionManager()
        session = SessionState(session_id="test-1", websocket=None)
        manager._sessions["test-1"] = session
        assert manager.active_count == 1

    @pytest.mark.asyncio
    async def test_session_audio_buffer(self):
        session = SessionState(session_id="test-1", websocket=None)
        session.append_audio(b"\x00" * 32000)  # 1 second
        assert session.audio_duration_s == pytest.approx(1.0, abs=0.01)
        session.reset_audio_buffer()
        assert session.audio_duration_s == 0.0

    @pytest.mark.asyncio
    async def test_conversation_history(self):
        session = SessionState(session_id="test-1", websocket=None)
        session.add_to_history("user", "Hello")
        session.add_to_history("assistant", "Hi!")
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
