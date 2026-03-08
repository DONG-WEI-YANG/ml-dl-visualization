# ML/DL Visualization Backend

FastAPI backend for the ML/DL interactive teaching platform.

## Requirements

- Python 3.11+
- pip

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start the server
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | (required for Claude) |
| `OPENAI_API_KEY` | OpenAI API key | (required for GPT) |
| `LLM_PROVIDER` | LLM provider: `anthropic`, `openai`, `ollama`, `local` | `anthropic` |
| `MODEL_NAME` | Model name | `claude-sonnet-4-20250514` |
| `JWT_SECRET` | JWT signing secret (change in production!) | default dev value |
| `DEFAULT_ADMIN_PASSWORD` | Initial admin password | `admin123` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:5173` |

## API Endpoints

### Auth (`/api/auth`)
- `POST /login` - Login, returns JWT token
- `GET /me` - Get current user info
- `POST /register` - Create new user (admin only)

### Admin (`/api/admin`)
- `GET /users` - List all users
- `GET/PUT /users/{id}` - Get/update user
- `DELETE /users/{id}` - Deactivate user
- `POST/DELETE /teachers/{tid}/students/{sid}` - Assign/remove student
- `GET/PUT /settings` - System settings
- `POST /train-nlp` - Train NLP models

### LLM (`/api/llm`)
- `POST /chat` - Chat with AI tutor
- `GET /model-info` - Current LLM config
- `WebSocket /ws/chat` - Streaming chat

### ML Models (`/api/models`)
- `POST /gradient-descent` - Run gradient descent
- `POST /loss-landscape` - Compute loss landscape
- `POST /decision-boundary` - Train and get decision boundary
- `POST /roc-pr` - ROC/PR curves
- `POST /tree` - Decision tree / random forest
- `GET /activations` - Activation function curves

### Analytics (`/api/analytics`)
- `POST /events` - Record learning event
- `GET /students/{id}` - Student analytics
- `GET /summary` - Class summary

### Quiz (`/api/quiz`)
- `GET /week/{week}` - Get quiz questions
- `POST /submit` - Submit and grade quiz

### RAG (`/api/rag`)
- `POST /search` - Search curriculum
- `POST /ingest` - Ingest curriculum content

### Health
- `GET /health` - Health check with DB status

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Project Structure

```
backend/
  app/
    main.py          # FastAPI app, middleware, startup
    config.py        # Environment settings
    db.py            # SQLite database
    api/             # Route handlers
    auth/            # JWT authentication
    llm/             # Multi-model LLM abstraction
    nlp/             # 7-layer NLP pipeline
    rag/             # Retrieval-Augmented Generation
    models/          # ML model computation
    analytics/       # Learning analytics
    quiz/            # Quiz question bank
  tests/             # pytest tests
  data/              # SQLite DB + trained models
```

## Docker

```bash
docker build -t ml-dl-backend .
docker run -p 8000:8000 --env-file .env ml-dl-backend
```
