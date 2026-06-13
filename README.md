# IntelliVoice

## Multilingual Real-Time Speech-to-Speech AI Assistant

A low-latency multilingual speech-to-speech AI assistant optimized for the **RTX 4080 16GB** VRAM budget. Processes audio end-to-end with streaming LLM output and overlapped TTS synthesis for minimal perceived latency.

---

## Architecture

```
Audio Input → VAD → DeepFilterNet → ASR + Emotion → LLM (streaming) → TTS (overlapped) → Audio Output
```

### Pipeline Layers

| Order | Component | Model | VRAM |
|-------|-----------|-------|------|
| 1 | VAD | Silero VAD v5 | 50 MB |
| 1 | Noise Suppression | DeepFilterNet (CPU) | 0 MB |
| 2 | ASR | Whisper large-v3-turbo (FP16) | 1.5 GB |
| 3 | Emotion | wav2vec2-base-superb-er (FP16) | 0.35 GB |
| 4 | Reasoning | Qwen2.5-7B-Instruct (INT4 NF4) | 5.0 GB |
| 5 | TTS | OmniVoice (FP16) | 4.0 GB |

### VRAM Management (16GB target)

| Total | Notes |
|-------|-------|
| ~10.9 GB | Leaves ~5.1 GB headroom for KV cache and batch inference. |

> **Concurrency & Execution:** All models are loaded at startup with no lazy loading. ASR and Emotion run concurrently via `asyncio.gather`. LLM text output streams sentence-by-sentence to TTS with overlapped execution — TTS for sentence N runs in parallel with LLM generating sentence N+1. Memory is session-scoped (no external database required).

---

## Quick Start

### 1. Clone and Setup

```bash
cd IntelliVoice
cp .env.example .env
# Edit .env with your HuggingFace token if needed
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Show the VRAM budget
python scripts/download_models.py --budget

# List every model
python scripts/download_models.py --list

# Download all HF models
python scripts/download_models.py

# Or download specific ones
python scripts/download_models.py --model whisper
python scripts/download_models.py --model emotion
python scripts/download_models.py --model fast_llm
python scripts/download_models.py --model omnivoice
```

### 4. Run the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run Tests

```bash
pytest tests/ -v
python scripts/test_pipeline.py --layer preprocessing  # Preprocessing only
python scripts/test_pipeline.py --layer full            # End-to-end
```

---

## API Endpoints

### WebSocket (Real-time Audio)

```
ws://localhost:8000/ws/audio
```

Protocol:
- **Client → Server**: binary PCM chunks (int16, 16kHz, mono)
- **Server → Client**: JSON messages (transcription, response text, base64 audio chunks)

### REST (Text Chat)

```
POST /chat
{
    "message": "Hello, how are you?",
    "session_id": "optional-session-id"
}
```

### Health

```
GET /health          # basic
GET /health/gpu      # GPU + model status
GET /health/config   # config + VRAM budget
```

---

## Project Structure

```
IntelliVoice/
├── config/                  # settings, model registry, logging
├── backend/
│   ├── api/routes/          # FastAPI endpoints (WebSocket, REST, health)
│   ├── core/                # pipeline orchestrator, GPU manager, sessions
│   ├── layers/
│   │   ├── preprocessing/   # VAD, DeepFilterNet, audio utilities
│   │   ├── asr/             # Whisper ASR
│   │   ├── speaker/         # Emotion analyzer
│   │   ├── reasoning/       # Qwen2.5-7B-Instruct LLM
│   │   ├── memory/          # session-scoped conversation memory
│   │   └── speech_generation/  # OmniVoice TTS
│   └── services/            # model loader, audio streaming
├── frontend/                # React UI (Vite)
├── scripts/                 # download, benchmark, test
├── tests/                   # unit + integration tests
└── assets/                  # reference audio (Thiru.wav)
```

---

## Streaming TTS Strategy

The pipeline uses **overlapped sentence-level TTS** to balance audio quality with latency:

1. LLM tokens stream in and accumulate in a buffer
2. When a **complete sentence** is detected (`.` `!` `?` or `\n`), TTS starts immediately
3. While TTS synthesizes sentence N, the LLM continues generating sentence N+1
4. Audio is yielded in-order — no words are dropped

This avoids the problems of clause/word-level splitting (which causes OmniVoice to drop words on tiny fragments) while overlapping LLM and TTS execution for maximum throughput.

---

## Supported Languages

- English
- Hindi (हिन्दी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Code-mixed (Hinglish, Tanglish, etc.)
- 600+ via Whisper ASR + Qwen2.5 reasoning + OmniVoice TTS

---

## License

MIT
