# IntelliVoice

## Multilingual Real-Time Speech-to-Speech AI Assistant

A production-grade multilingual speech-to-speech AI assistant that processes audio end-to-end — preserving emotion, tone, accent, and speaker characteristics — while supporting Indian languages and code-mixed conversations. Tuned for the **RTX 4080 16GB** VRAM budget.

---

## Architecture

```
Audio Input -> Preprocessing -> ASR + Emotion/Speaker -> Prompt Assembly -> LLM Reasoning -> TTS Synthesis -> Audio Output
```

### Pipeline Layers

| Order | Component | Model | VRAM |
|-------|-----------|-------|------|
| 1 | VAD | Silero VAD v5 | 50 MB |
| 1 | Noise Suppression | resemble-enhance | 0.3 GB |
| 2 | ASR | Whisper large-v3-turbo (FP16) | 1.5 GB |
| 3 | Emotion | wav2vec2-base-superb-er (FP16) | 0.3 GB |
| 3 | Speaker Emb. | ECAPA-TDNN (FP16) | 0.1 GB |
| 4 | Memory | Conversation + MongoDB | CPU |
| 5 | Reasoning | Qwen2.5-7B-Instruct (INT4 NF4) | 5.0 GB |
| 6 | TTS | XTTS-v2 (FP16) | 1.5 GB |

### VRAM Management (16GB target)

| Sub-Total | Notes |
|------|-------|
| ~12 GB | Total resident VRAM. Leaves ~4 GB headroom for KV cache and batch inference. |

> **Concurrency & Execution:** All models are loaded at startup. No lazy loading is implemented to eliminate latency. Whisper and Emotion execute concurrently via asyncio. LLM text output is streamed directly to the XTTS-v2 module.

---

## Quick Start

### 1. Clone and Setup

```bash
cd IntelliVoice
cp .env.example .env
# Edit .env with your HuggingFace token if needed
```

### 2. Start MongoDB and Qdrant

```bash
docker compose up -d mongodb qdrant
```

MongoDB at `localhost:27017`, Qdrant at `localhost:6333`. The in-memory Qdrant fallback (no docker needed) is enabled by default in `.env.example` — set `QDRANT_IN_MEMORY=false` to use the server.

### 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Download Models

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
python scripts/download_models.py --model ecapa_tdnn
python scripts/download_models.py --model fast_llm
python scripts/download_models.py --model xtts
```

### 5. Run the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Tests

```bash
pytest tests/ -v
python scripts/test_pipeline.py --layer rag     # RAG only
python scripts/test_pipeline.py --layer full    # end-to-end
```

---

## API Endpoints

### WebSocket (Real-time Audio)

```
ws://localhost:8000/ws/audio
```

Protocol:
- **Client -> Server**: binary PCM chunks (int16, 16kHz, mono)
- **Server -> Client**: JSON messages (transcription, response text, base64 audio)

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
│   ├── api/routes/          # FastAPI endpoints
│   ├── core/                # pipeline orchestrator, GPU manager, sessions
│   ├── layers/
│   │   ├── preprocessing/   # VAD, denoising
│   │   ├── speaker/         # ECAPA-TDNN + SenseVoice
│   │   ├── reasoning/       # Fast LLM (Qwen2.5-3B)
│   │   ├── memory/          # conversation, long-term (MongoDB)
│   │   └── speech_generation/  # CosyVoice 2
│   └── services/            # model loader, audio streaming
├── scripts/                 # download, benchmark, test
├── tests/                   # unit + integration tests
├── docker/                  # Dockerfile
└── docker-compose.yml       # MongoDB
```

---

## Supported Languages

- English
- Hindi (हिन्दी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Code-mixed (Hinglish, Tanglish, etc.)
- 100+ via Qwen2-Audio ASR, Fast LLM reasoning, CosyVoice 2 TTS

---

## License

MIT
