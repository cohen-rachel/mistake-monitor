# Language Tutor — Personal Language Learning MVP

Hello and welcome! I made this repo because I am both a part-time Language Teacher and a Language learner myself! This is the exact app that I have been looking for, while also encorporating practices I use in my own lessons with my students to help them understand mistakes they are making often. 

This app is developed by me personally, with the help of AI. 

This is a full-stack web and mobile app that:
  - listens to you speak and transcribes your audio, 
  - detects your mistakes in the language you are learning,
  - categorizes them, and 
  - tracks your improvement over time.

This is an MVP which is in active development :) 

## Architecture

```
frontend/    → React (Vite) web app
mobile/      → React Native app with Expo SDK 54
backend/     → Python FastAPI async API
SQLite       → Local database (swap to Postgres via env var)
```

**Pluggable providers:**
- **STT (Speech-to-Text):** Dummy (default, no deps), OpenAI Whisper API, faster-whisper local
- **LLM (Analysis):** Ollama local (default), OpenAI, Anthropic

## Quick Start (Local Development)

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Ollama** (for LLM analysis — default provider)
- **Expo Go** or an iOS/Android simulator for the mobile app

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Copy env config (edit as needed)
cp .env.example .env

# Generate initial migration and start
alembic revision --autogenerate -m "initial"
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies /api to localhost:8000)
npm run dev
```

Open **http://localhost:3000** in your browser.

### 3. Mobile App

```bash
cd mobile

# Install dependencies
npm install

# Start Expo
npx expo start
```

The mobile app talks directly to the backend API. Set `EXPO_PUBLIC_API_BASE_URL`
before starting Expo if `localhost` is not correct for your simulator or device.

Examples:

```bash
EXPO_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api npx expo start
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.10:8000/api npx expo start
```

Notes:

- Expo Go on a physical phone cannot use `localhost` for your Mac-hosted backend.
- Start the backend with `--host 0.0.0.0` if you want a phone on the same network to reach it:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Ollama (for LLM analysis)

```bash
# Install Ollama: https://ollama.ai
# Pull a model:
ollama pull llama3

# Ollama runs on port 11434 by default — no config needed
```

## Docker

```bash
# Start both services
docker compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# API docs: http://localhost:8000/docs
```

Note: Docker currently covers the backend and web frontend. The Expo mobile app runs separately from `mobile/`.

For Ollama access from Docker, the compose file uses `host.docker.internal:11434`. Ensure Ollama is running on the host machine.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STT_PROVIDER` | `dummy` | STT engine: `dummy`, `whisper_api`, `whisper_local` |
| `LLM_PROVIDER` | `ollama` | LLM engine: `ollama`, `openai`, `anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OPENAI_API_KEY` | — | Required for `whisper_api` STT or `openai` LLM |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` LLM |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model |
| `DATABASE_URL` | `sqlite+aiosqlite:///./langtutor.db` | DB connection string |
| `DEFAULT_LANGUAGE` | `en` | Target language code |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated explicit CORS origins |
| `CORS_ALLOW_ORIGIN_REGEX` | local/LAN dev regex | Regex for localhost and common LAN Expo/web development hosts |

## Swapping Providers

### Use OpenAI Whisper API for STT

```bash
STT_PROVIDER=whisper_api
OPENAI_API_KEY=sk-your-key-here
```

### Use local faster-whisper for STT

```bash
pip install faster-whisper
STT_PROVIDER=whisper_local
```

### Use OpenAI for LLM analysis

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Use Anthropic for LLM analysis

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### Use Postgres instead of SQLite

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/langtutor
pip install asyncpg
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions` | Create session (upload audio or transcript) |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/sessions/{id}` | Get session detail with transcript & mistakes |
| `POST` | `/api/transcribe` | Transcribe audio file (standalone) |
| `WS` | `/api/transcribe/stream` | Real-time audio transcription via WebSocket |
| `POST` | `/api/analyze` | Run LLM analysis on a session |
| `GET` | `/api/insights` | Get aggregated mistake insights & trends |
| `GET` | `/api/topics` | Topic suggestions based on estimated level |
| `GET` | `/api/topics/history` | Historical attempts for a specific topic |
| `GET` | `/api/user/language_profiles` | List available language profiles |
| `GET` | `/api/user/language_profiles/current` | Get current language profile |
| `PUT` | `/api/user/language_profiles/set_current` | Switch current language profile |
| `GET` | `/api/health` | Health check |

Interactive API docs available at **http://localhost:8000/docs** (Swagger UI).

## Pages

| Route | Description |
|---|---|
| `/` | Home — Record audio (real-time transcription) or upload a file |
| `/history` | Browse past sessions with transcripts and analysis |
| `/insights` | Top mistakes, error trend chart, and recent corrections list |
| `/rewrite` | Rewrite training based on previously detected mistakes |

## Mobile App

- Built with **Expo SDK 54**
- Uses the same FastAPI backend as the web app
- Supports recording/upload flows, history, insights, rewrite practice, and language-profile switching
- Expo/mobile-specific backend support includes:
  - broader local/LAN CORS support
  - audio uploads that preserve file type metadata such as `m4a`, `mp3`, `wav`, and `webm`
  - websocket transcription support compatible with web binary chunks and Expo-style JSON/base64 chunk messages

## Language-Aware Analysis

- The analyzer now uses language-specific system prompts for:
  - French (`fr`)
  - Spanish (`es`)
  - Japanese (`ja`)
- Other languages use a generic multilingual grammar prompt.
- Sessions are linked to per-user language profiles in DB tables:
  - `user_language_profiles`
  - `session_language_profiles`
- You can filter insights by language:
  - `GET /api/insights?language=fr`

## Data Model

- **User** — minimal user record
- **Session** — one recording/upload session with metadata
- **Transcript** — raw text + optional word-level timestamps/confidence
- **MistakeType** — canonical error categories (verb-tense, article, preposition, etc.)
- **Mistake** — individual error instance with span, correction, explanation, confidence

## Tech Stack

- **Web Frontend:** React 18, Vite, TypeScript, React Router, Recharts
- **Mobile Frontend:** React Native, Expo SDK 54, TypeScript, `react-native-svg`
- **Backend:** Python 3.11, FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **STT:** OpenAI Whisper API / faster-whisper / dummy
- **LLM:** Ollama / OpenAI / Anthropic
