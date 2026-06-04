# IntelliVoice 🎙️

## Multilingual Real-Time Speech-to-Speech AI Assistant

A production-grade multilingual speech-to-speech AI assistant that processes audio end-to-end — preserving emotion, tone, accent, and speaker characteristics — while supporting Indian languages and code-mixed conversations.

---

## 🏗️ Architecture

```
Audio Input → Preprocessing → Acoustic Encoder → Semantic Understanding → Reasoning → Memory → Planning → Speech Generation → Audio Output
```

### Pipeline Layers

| Layer | Component | Model | Purpose |
|-------|-----------|-------|---------|
| 1 | VAD | Silero VAD | Speech detection |
| 1 | Noise Suppression | DeepFilterNet | Audio cleaning |
| 2 | Acoustic Encoder | XLS-R 1B | Speech embeddings |
| 3 | Semantic Understanding | Qwen-Audio | Audio comprehension |
| 4 | Prosody & Emotion | Emotion2Vec | Emotion detection |
| 5 | Speaker Understanding | WavLM Large | Speaker identity |
| 6 | Reasoning | Qwen3 30B-A3B MoE | Response generation |
| 7 | Memory | LangGraph + MongoDB | Conversation memory |
| 9 | Response Planning | Custom | Tone/emotion planning |
| 10 | Speech Generation | CosyVoice 2 | Text-to-speech |
| 12 | Audio Synthesis | HiFi-GAN | Waveform generation |

### VRAM Management

Uses **three-phase model loading** to fit within RTX 4080 (16GB):
- **Phase 1**: Understanding models (~8GB)
- **Phase 2**: Reasoning model (~10GB)
- **Phase 3**: Generation models (~2.5GB)

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd IntelliVoice
cp .env.example .env
# Edit .env with your HuggingFace token
```

### 2. Start MongoDB (Docker)

```bash
docker compose up -d mongodb
```

### 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Download Models

```bash
# Download all models
python scripts/download_models.py

# Or download specific models
python scripts/download_models.py --model silero_vad
python scripts/download_models.py --model xlsr_1b

# Check status
python scripts/download_models.py --status
```

### 5. Run the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Tests

```bash
pytest tests/ -v
python scripts/test_pipeline.py
```

---

## 📡 API Endpoints

### WebSocket (Real-time Audio)

```
ws://localhost:8000/ws/audio
```

**Protocol:**
- **Client → Server**: Binary PCM audio chunks (int16, 16kHz, mono)
- **Server → Client**: JSON messages (transcription, response, audio)

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
GET /health          # Basic health
GET /health/gpu      # GPU + model status
GET /health/config   # Configuration info
```

---

## 🛠️ Development

### Project Structure

```
IntelliVoice/
├── config/              # Settings, model registry, logging
├── backend/
│   ├── api/routes/      # FastAPI endpoints
│   ├── core/            # Pipeline, GPU manager, session manager
│   ├── layers/          # All 12 pipeline layers
│   └── services/        # Model loader, audio streaming
├── scripts/             # Download, benchmark, test scripts
├── tests/               # Test suite
└── docker/              # Docker configuration
```

### Benchmarking

```bash
python scripts/benchmark.py
python scripts/benchmark.py --layer vad --iterations 50
```

---

## 🐳 Docker

### Full Stack

```bash
docker compose up -d
```

### Backend Only

```bash
docker compose up -d backend mongodb
```

---

## 🌐 Supported Languages

- English
- Hindi (हिंदी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Code-mixed (Hinglish, Tanglish, etc.)

---

## 📋 License

MIT
