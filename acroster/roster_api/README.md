# Officer Roster Optimization API

Production-grade FastAPI backend for optimizing officer allocation on counters.

## Architecture

```
app/
├── api/
│   ├── deps.py              # Dependency injection
│   └── v1/
│       ├── router.py        # API router aggregator
│       └── endpoints/       # Route handlers
│           ├── roster.py    # Roster generation
│           ├── history.py   # History management
│           └── edits.py     # Edit operations
├── core/
│   ├── config.py           # Pydantic settings
│   └── logging_config.py   # Structured logging
├── schemas/
│   └── roster.py           # Pydantic models
└── services/
    ├── database.py         # Database session management
    ├── roster.py           # Roster generation logic
    └── db_operations.py    # Database operations
```

## Setup

1. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Run application**

```bash
uvicorn acroster.roster_api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Roster Generation

- `POST /api/v1/roster/generate` - Generate optimized roster

### History Management

- `GET /api/v1/history/last-inputs` - Get last used inputs
- `POST /api/v1/history/last-inputs` - Save inputs
- `GET /api/v1/history/history` - Get roster history

### Edit Operations

- `POST /api/v1/edits/` - Create roster edit
- `GET /api/v1/edits/` - List roster edits
- `DELETE /api/v1/edits/{edit_id}` - Delete specific edit
- `DELETE /api/v1/edits/` - Clear all edits

### Health Check

- `GET /health` - Service health status

## API Documentation

Interactive API documentation available at:

- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

## Request/Response Format

All responses follow this structure:

**Success Response:**

```json
{
  "success": true,
  "data": {
    ...
  },
  ...
}
```

**Error Response:**

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": []
  }
}
```

## Environment Variables

See `.env.example` for all available configuration options.

Key variables:

- `CORS_ORIGINS` - Allowed frontend origins
- `DATABASE_PATH` - SQLite database path
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT` - Log format (json or text)

## Development

**Run with auto-reload:**

```bash
uvicorn acroster.roster_api.main:app --reload
```

**Run tests:**

```bash
pytest
```

## Production Deployment

**Using Gunicorn with Uvicorn workers:**

```bash
gunicorn acroster.roster_api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Features

- ✅ Async/await throughout
- ✅ Dependency injection
- ✅ Pydantic v2 validation
- ✅ Structured JSON logging
- ✅ Global exception handling
- ✅ CORS configured for Next.js
- ✅ OpenAPI/Swagger documentation
- ✅ Environment-based configuration
- ✅ Clean architecture with service layer