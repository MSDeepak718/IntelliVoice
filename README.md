# IntelliVoice

## Multilingual Real-Time Speech-to-Speech AI Assistant

A low-latency multilingual speech-to-speech AI assistant optimized for the **RTX 4080 16GB** VRAM budget. Processes audio end-to-end with streaming LLM output and incremental TTS synthesis for minimal perceived latency.

---

## Architecture

```
Audio Input → VAD → ASR → LLM (streaming) → TTS (incremental) → Audio Output
```

### Pipeline Layers

| Order | Component | Model | VRAM |
|-------|-----------|-------|------|
| 1 | VAD | Silero VAD v5 | 50 MB |
| 2 | ASR | Whisper large-v3-turbo (FP16) | 1.5 GB |
| 3 | Reasoning | Qwen2.5-7B-Instruct (INT4 NF4) | 5.0 GB |
| 4 | TTS | OmniVoice (FP16) | 4.0 GB |

### VRAM Management (16GB target)

| Total | Notes |
|-------|-------|
| ~10.5 GB | Leaves ~5.5 GB headroom for KV cache and batch inference. |

> **Concurrency & Execution:** All models are loaded at startup with no lazy loading. LLM text output streams incrementally to TTS on clause/sentence boundaries for minimum first-audio latency. Memory is session-scoped (no external database required).

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
│   │   ├── preprocessing/   # VAD, audio utilities
│   │   ├── asr/             # Whisper ASR
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

## Latency Optimizations

- **No noise suppression in critical path** — VAD + Whisper handle noisy audio robustly
- **No emotion/speaker analysis** — removed GPU contention from critical path
- **Clause-boundary TTS chunking** — TTS fires on commas/semicolons, not just sentence ends
- **Word-count fallback flush** — ensures first audio within ~15 tokens regardless of punctuation
- **0.6s silence threshold** — reduced from 1.2s for faster end-of-speech detection
- **150 max tokens** — voice-optimized concise responses (1-3 sentences)
- **Session-scoped memory** — no external database latency

---

## Supported Languages

- English
- Hindi (हिन्दी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Code-mixed (Hinglish, Tanglish, etc.)
- 100+ via Whisper ASR + Qwen2.5 reasoning + OmniVoice TTS

---

## License

MIT
