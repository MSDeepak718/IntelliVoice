"""
Tests for the pipeline integration and the model registry.
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

    def test_qdrant_settings_present(self):
        settings = Settings()
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "qdrant_collection")
        assert settings.qdrant_vector_size == 1024

    def test_rag_settings_present(self):
        settings = Settings()
        assert settings.rag_top_k >= 1
        assert 0.0 <= settings.rag_min_score <= 1.0

    def test_is_production(self):
        assert not Settings(app_env="development").is_production
        assert Settings(app_env="production").is_production


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_all_models_listed(self):
        models = ModelRegistry.get_all_models()
        # 2 (always) + 3 (understanding) + 1 (rag) + 1 (reasoning) + 2 (gen) + 1 (clone)
        assert len(models) == 10

    def test_phase_grouping(self):
        always = ModelRegistry.get_models_by_phase(LoadingPhase.ALWAYS)
        assert len(always) == 2  # VAD + DeepFilterNet

        understanding = ModelRegistry.get_models_by_phase(LoadingPhase.UNDERSTANDING)
        assert len(understanding) == 3  # XLS-R, Qwen2-Audio, Emotion2Vec

        rag = ModelRegistry.get_models_by_phase(LoadingPhase.RAG)
        assert len(rag) == 1  # BGE-M3

        reasoning = ModelRegistry.get_models_by_phase(LoadingPhase.REASONING)
        assert len(reasoning) == 1  # Qwen3-14B

        generation = ModelRegistry.get_models_by_phase(LoadingPhase.GENERATION)
        assert len(generation) == 2  # CosyVoice + HiFi-GAN

        clone = ModelRegistry.get_models_by_phase(LoadingPhase.VOICE_CLONE)
        assert len(clone) == 1  # OpenVoice V2

    def test_vram_budget_under_16gb(self):
        """Every phase should fit comfortably in 16GB."""
        for phase in LoadingPhase:
            vram_mb = ModelRegistry.get_phase_vram_mb(phase)
            assert vram_mb < 16 * 1024, (
                f"Phase {phase.name} needs {vram_mb / 1024:.2f} GB > 16 GB"
            )

    def test_peak_vram_under_16gb(self):
        peak = ModelRegistry.get_peak_vram_mb()
        assert peak < 16 * 1024, f"Peak {peak / 1024:.2f} GB exceeds 16 GB"

    def test_xlsr_config(self):
        xlsr = ModelRegistry.XLSR_1B
        assert xlsr.hf_model_id == "facebook/wav2vec2-xls-r-1b"
        assert xlsr.precision == ModelPrecision.FP16
        assert xlsr.phase == LoadingPhase.UNDERSTANDING

    def test_qwen2_audio_config(self):
        qa = ModelRegistry.QWEN2_AUDIO
        assert qa.hf_model_id == "Qwen/Qwen2-Audio-7B-Instruct"
        assert qa.precision == ModelPrecision.INT4
        assert qa.quantization_config["load_in_4bit"] is True
        assert qa.quantization_config["bnb_4bit_quant_type"] == "nf4"

    def test_qwen3_14b_config(self):
        q3 = ModelRegistry.QWEN3_14B
        assert q3.hf_model_id == "Qwen/Qwen3-14B"
        assert q3.precision == ModelPrecision.INT4
        assert q3.quantization_config is not None
        assert q3.quantization_config["load_in_4bit"] is True
        assert q3.quantization_config["bnb_4bit_use_double_quant"] is True

    def test_bge_m3_config(self):
        bge = ModelRegistry.BGE_M3
        assert bge.hf_model_id == "BAAI/bge-m3"
        assert bge.phase == LoadingPhase.RAG

    def test_openvoice_v2_config(self):
        ov = ModelRegistry.OPENVoice_V2
        assert ov.hf_model_id == "myshell-ai/OpenVoiceV2"
        assert ov.phase == LoadingPhase.VOICE_CLONE

    def test_get_model_by_name(self):
        assert ModelRegistry.get_model_by_name("bge_m3") is not None
        assert ModelRegistry.get_model_by_name("nonexistent") is None


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

    def test_torch_dtype_conversion(self):
        gpu = GPUManager()
        assert gpu.get_torch_dtype("float32") == torch.float32
        assert gpu.get_torch_dtype("float16") == torch.float16
        assert gpu.get_torch_dtype("bfloat16") == torch.bfloat16

    def test_register_and_unregister(self):
        gpu = GPUManager()
        dummy = torch.nn.Linear(2, 2)
        gpu.register_model(
            "test_dummy", dummy, phase=LoadingPhase.UNDERSTANDING, vram_mb=10
        )
        assert gpu.is_loaded("test_dummy")
        assert "test_dummy" in gpu.loaded_names()
        gpu.unregister_model("test_dummy")
        assert not gpu.is_loaded("test_dummy")

    def test_models_in_phase(self):
        gpu = GPUManager()
        dummy1 = torch.nn.Linear(2, 2)
        dummy2 = torch.nn.Linear(2, 2)
        gpu.register_model("a", dummy1, phase=LoadingPhase.UNDERSTANDING, vram_mb=10)
        gpu.register_model("b", dummy2, phase=LoadingPhase.REASONING, vram_mb=10)
        assert "a" in gpu.models_in_phase(LoadingPhase.UNDERSTANDING)
        assert "b" in gpu.models_in_phase(LoadingPhase.REASONING)
        assert "a" not in gpu.models_in_phase(LoadingPhase.REASONING)


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
