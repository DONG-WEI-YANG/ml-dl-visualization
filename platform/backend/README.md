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
| `APP_ENV` | `production`, `development`, or `test`; `.env.example` explicitly selects local development | `production` |
| `DATABASE_PATH` | Shared SQLite path for users, events, quizzes and RAG | `data/app.db` beside backend code |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | (required for Claude) |
| `OPENAI_API_KEY` | OpenAI API key | (required for GPT) |
| `LLM_PROVIDER` | LLM provider: `anthropic`, `openai`, `ollama`, `local` | `anthropic` |
| `MODEL_NAME` | Model name | `claude-sonnet-4-20250514` |
| `JWT_SECRET` | Production requires a unique secret of at least 32 characters; default key refuses startup | development-only default |
| `DEFAULT_ADMIN_PASSWORD` | Production initial admin requires at least 12 characters; never resets an existing admin | `admin123` in development |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:5173` |

## API Endpoints

### Auth (`/api/auth`)
- `POST /login` - Login, returns JWT token
- `GET /me` - Get current user info
- `POST /register` - Create new user (admin only)
- `POST /change-password` - Returns a replacement `{access_token, token_type, user}` and invalidates all previous tokens for this account
- `POST /logout` - Revokes the current token; other sessions remain valid

Accounts requiring an initial password change can only use `/me`, `/change-password`, and `/logout`. All other authenticated routes and WebSockets enforce the restriction. HTTP and WebSocket requests check session revocation. Offline browser sign-out clears local state but cannot confirm server-side revocation until the logout request succeeds.

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
- `GET /activation-functions` - Activation function curves

Model jobs run outside the event loop, with two concurrent jobs per process. Overload returns 503 with Retry-After; invalid shapes, excessive limits and divergence return 422. Limits include 500 rows, 20 features, 1,000 epochs, landscape resolution 100, and 200 trees / depth 15.

### Analytics (`/api/analytics`)
- `POST /events` - Record an authenticated user's `viz_interaction`; score and client timestamp are forbidden
- `POST /assignments/grade` - Teacher/admin grading: `{student_id: string, week: 1..18, score: 0..100}`. Teachers must be assigned to the target student; the server records grader and audit information
- `GET /students/{id}` - Student analytics
- `GET /summary` - Class summary

### Quiz (`/api/quiz`)
- `GET /week/{week}` - Get quiz questions
- `POST /submit` - Submit and grade quiz

Initialization inserts a validated, versioned baseline of 54 questions (3 per week) without replacing existing question IDs or teacher edits. See [quiz seed notes](app/quiz/README.md). No student accounts are created by seeding.

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
docker run -p 8000:8000 --env-file .env -e APP_ENV=production -v backend-data:/data ml-dl-backend
```

Set production secrets first. Use the repository's Docker Compose setup to mount curriculum as well. Before updating an existing HF Space, follow [storage operations](../../docs/storage-operations.md); deployment is gated until verified backup and persistent storage migration. This release invalidates legacy JWTs, so existing sessions must log in again.
