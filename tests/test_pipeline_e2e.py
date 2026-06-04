"""
Tests for the pipeline integration.
"""

import pytest
import torch

from config.model_registry import ModelRegistry, LoadingPhase, ModelPrecision
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
        dev = Settings(app_env="development")
        assert not dev.is_production
        prod = Settings(app_env="production")
        assert prod.is_production


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_all_models_listed(self):
        models = ModelRegistry.get_all_models()
        assert len(models) == 9

    def test_phase_grouping(self):
        always = ModelRegistry.get_models_by_phase(LoadingPhase.ALWAYS)
        assert len(always) == 2  # VAD + DeepFilterNet

        understanding = ModelRegistry.get_models_by_phase(LoadingPhase.UNDERSTANDING)
        assert len(understanding) == 4  # XLS-R, Qwen-Audio, Emotion2Vec, WavLM

        reasoning = ModelRegistry.get_models_by_phase(LoadingPhase.REASONING)
        assert len(reasoning) == 1  # Qwen3

        generation = ModelRegistry.get_models_by_phase(LoadingPhase.GENERATION)
        assert len(generation) == 2  # CosyVoice + HiFi-GAN

    def test_vram_budget(self):
        always_vram = ModelRegistry.get_phase_vram_mb(LoadingPhase.ALWAYS)
        assert always_vram < 200  # Should be small

    def test_model_configs(self):
        xlsr = ModelRegistry.XLSR_1B
        assert xlsr.hf_model_id == "facebook/wav2vec2-xls-r-1b"
        assert xlsr.precision == ModelPrecision.FP16
        assert xlsr.phase == LoadingPhase.UNDERSTANDING

        qwen3 = ModelRegistry.QWEN3_MOE
        assert qwen3.quantization_config is not None
        assert qwen3.quantization_config["load_in_4bit"] is True


class TestGPUManager:
    """Tests for GPUManager."""

    def test_device_detection(self):
        gpu = GPUManager()
        assert gpu.device is not None

    def test_memory_stats(self):
        gpu = GPUManager()
        stats = gpu.get_memory_stats()
        assert "allocated" in stats
        assert "free" in stats
        assert "total" in stats

    def test_torch_dtype_conversion(self):
        gpu = GPUManager()
        assert gpu.get_torch_dtype("float32") == torch.float32
        assert gpu.get_torch_dtype("float16") == torch.float16
        assert gpu.get_torch_dtype("bfloat16") == torch.bfloat16


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        manager = SessionManager()
        # Create with mock websocket
        session = SessionState(session_id="test-1", websocket=None)
        manager._sessions["test-1"] = session
        assert manager.active_count == 1

    @pytest.mark.asyncio
    async def test_session_audio_buffer(self):
        session = SessionState(session_id="test-1", websocket=None)

        # Append audio
        session.append_audio(b"\x00" * 32000)  # 1 second
        assert session.audio_duration_s == pytest.approx(1.0, abs=0.01)

        # Reset
        session.reset_audio_buffer()
        assert session.audio_duration_s == 0.0

    @pytest.mark.asyncio
    async def test_conversation_history(self):
        session = SessionState(session_id="test-1", websocket=None)
        session.add_to_history("user", "Hello")
        session.add_to_history("assistant", "Hi!")

        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
