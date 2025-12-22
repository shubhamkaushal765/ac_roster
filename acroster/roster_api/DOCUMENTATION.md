# Officer Roster Optimization API - Complete Documentation

## Project Overview

Production-grade FastAPI backend for optimizing officer allocation on counters.
Built with clean architecture, async operations, and strict type safety.

## Technology Stack

- **Framework**: FastAPI 0.109.0
- **Python**: 3.11+
- **Validation**: Pydantic v2
- **Database**: SQLite via SQLAlchemy 2.0
- **Server**: Uvicorn with async/await
- **Logging**: Structured JSON logging

## Architecture Principles

1. **Clean Architecture**: Clear separation between API, business logic, and
   data layers
2. **Dependency Injection**: FastAPI Depends() for loose coupling
3. **Async Everywhere**: All I/O operations use async/await
4. **Type Safety**: Strict Pydantic models for validation
5. **Error Handling**: Global exception handlers with consistent responses
6. **CORS Ready**: Configured for Next.js frontend integration

## Project Structure

```
roster_api/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── __init__.py
│   │
│   ├── core/                      # Core configuration
│   │   ├── config.py             # Pydantic Settings
│   │   ├── logging_config.py     # Structured logging setup
│   │   └── __init__.py
│   │
│   ├── api/                       # API layer
│   │   ├── deps.py               # Dependency injection
│   │   ├── __init__.py
│   │   └── v1/                   # API version 1
│   │       ├── router.py         # Route aggregator
│   │       ├── __init__.py
│   │       └── endpoints/        # Route handlers
│   │           ├── roster.py     # Roster generation
│   │           ├── history.py    # History operations
│   │           ├── edits.py      # Edit operations
│   │           └── __init__.py
│   │
│   ├── schemas/                   # Pydantic models
│   │   ├── roster.py             # Request/Response schemas
│   │   └── __init__.py
│   │
│   └── services/                  # Business logic layer
│       ├── database.py           # Database session management
│       ├── roster.py             # Roster generation service
│       ├── db_operations.py      # Database CRUD operations
│       └── __init__.py
│
├── acroster/                      # Domain logic (existing module)
│   ├── __init__.py
│   ├── config.py
│   ├── counter.py
│   ├── database.py
│   ├── officer.py
│   ├── roster_builder.py
│   ├── sos_scheduler.py
│   ├── optimization.py
│   ├── assignment_engine.py
│   ├── statistics.py
│   └── ...
│
├── .env.example                   # Environment template
├── .env                          # Environment variables (gitignored)
├── .gitignore
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Docker Compose setup
├── run.sh                        # Startup script
└── test_api.py                   # API test suite
```

## API Endpoints Reference

### Health Check

```
GET /health
Response: { "status": "healthy", "version": "1.0.0" }
```

### Roster Generation

```
POST /api/v1/roster/generate
Content-Type: application/json

Request Body:
{
  "mode": "arrival" | "departure",
  "main_officers_reported": "1-18",
  "report_gl_counters": "4AC1, 8AC11, 12AC21, 16AC31",
  "handwritten_counters": "3AC12,5AC13",
  "ot_counters": "2,20,40",
  "ro_ra_officers": "3RO2100, 11RO1700",
  "sos_timings": "(AC22)1000-1300, 2000-2200",
  "beam_width": 20,
  "save_to_history": true
}

Response:
{
  "success": true,
  "data": {
    "officer_schedules": { ... },
    "counter_matrix": [ ... ],
    "mode": "arrival"
  },
  "officer_counts": {
    "main": 18,
    "sos": 5,
    "ot": 3,
    "total": 26
  },
  "optimization_penalty": 12.5,
  "statistics": {
    "stats1": "...",
    "stats2": "..."
  }
}
```

### History Management

```
GET /api/v1/history/last-inputs
Response: { "success": true, "data": { ... } }

POST /api/v1/history/last-inputs
Body: { "main_officers": "...", "beam_width": 20, ... }
Response: { "success": true, "message": "..." }

GET /api/v1/history/history?limit=10
Response: { "success": true, "data": [...], "count": 10 }
```

### Edit Operations

```
POST /api/v1/edits/
Body: {
  "edit_type": "delete" | "swap" | "add",
  "officer_id": "M1",
  "slot_start": 0,
  "slot_end": 5,
  "time_start": "1000",
  "time_end": "1130",
  ...
}
Response: { "success": true, "edit_id": 123, "message": "..." }

GET /api/v1/edits/?limit=20
Response: { "success": true, "data": [...], "count": 20 }

DELETE /api/v1/edits/{edit_id}
Response: { "success": true, "message": "..." }

DELETE /api/v1/edits/
Response: { "success": true, "message": "All edits cleared" }
```

## Error Response Format

All errors follow this structure:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": []
  }
}
```

**Error Codes:**

- `VALIDATION_ERROR` (422): Request validation failed
- `INVALID_INPUT` (400): Business logic validation failed
- `NOT_FOUND` (404): Resource not found
- `INTERNAL_ERROR` (500): Server error

## Request Flow

1. **Client Request** → FastAPI endpoint
2. **Validation** → Pydantic schema validation
3. **Dependency Injection** → Services injected via Depends()
4. **Business Logic** → Service layer processes request
5. **Database Operations** → SQLAlchemy ORM queries
6. **Response** → Pydantic model serialization
7. **Error Handling** → Global exception handlers

## Key Design Decisions

### 1. Dependency Injection Pattern

```python
async def generate_roster(
        request: RosterGenerationRequest,
        roster_service: RosterService,  # Injected
        db_session: DBSession,  # Injected
        db_ops: DBOperationsService  # Injected
):
    ...
```

**Why**: Loose coupling, easy testing, clear dependencies

### 2. Service Layer Separation

```
endpoints/ (HTTP) → services/ (Business Logic) → acroster/ (Domain)
```

**Why**: Single responsibility, testable business logic, domain isolation

### 3. Async Throughout

```python
async def generate_roster(...) -> Tuple[...]:
    roster_data, counts, penalty, stats = await roster_service.generate_roster(
        ...
        )
    await db_ops.save_last_inputs(...)
    await db_ops.save_roster_history(...)
```

**Why**: Non-blocking I/O, better performance under load

### 4. Pydantic v2 Models

```python
class RosterGenerationRequest(BaseModel):
    mode: OperationMode = Field(default=OperationMode.ARRIVAL)
    main_officers_reported: str = Field(..., min_length=1)
    beam_width: int = Field(default=20, ge=1, le=100)
```

**Why**: Runtime validation, automatic OpenAPI docs, type safety

### 5. Structured JSON Logging

```python
{
    "timestamp": "2024-12-22T07:30:00.000Z",
    "level":     "INFO",
    "logger":    "app.services.roster",
    "message":   "Generated roster with 26 total officers"
}
```

**Why**: Structured logs for production monitoring, easy parsing

### 6. Global Exception Handling

```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={...})
```

**Why**: Consistent error responses, centralized error handling

## Configuration

Environment variables (`.env`):

```bash
# Application
PROJECT_NAME=Officer Roster Optimization API
VERSION=1.0.0
API_V1_STR=/api/v1

# CORS (for Next.js frontend)
CORS_ORIGINS=["http://localhost:3000"]

# Database
DATABASE_PATH=acroster.db

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Algorithm Parameters
DEFAULT_BEAM_WIDTH=20
MAX_BEAM_WIDTH=100
```

## Running the Application

### Development

```bash
# Using run script
./run.sh

# Or directly
uvicorn acroster.roster_api.main:app --reload --host 0.0.0.0 --port 8000

```

### Production

```bash
# Using Gunicorn with multiple workers
gunicorn acroster.roster_api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or using Dockerfile
docker build -t roster-api .
docker run -p 8000:8000 roster-api
```

## Testing

```bash
# Run test suite
python test_api.py

# Or use curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/roster/generate \
  -H "Content-Type: application/json" \
  -d '{"mode":"arrival","main_officers_reported":"1-18",...}'
```

## Frontend Integration

The API is designed for seamless Next.js integration:

1. **CORS**: Pre-configured for localhost:3000
2. **JSON Responses**: Optimized structure for frontend consumption
3. **Error Handling**: Consistent error format across all endpoints
4. **Type Safety**: OpenAPI schema available at `/api/v1/docs`

### Example Frontend Usage (TypeScript)

```typescript
// types.ts
interface RosterGenerationRequest {
    mode: 'arrival' | 'departure';
    main_officers_reported: string;
    beam_width?: number;
    // ...
}

interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
    };
}

// api.ts
async function generateRoster(
    request: RosterGenerationRequest
): Promise<ApiResponse<RosterData>> {
    const response = await fetch('http://localhost:8000/api/v1/roster/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(request),
    });
    return response.json();
}
```

## Performance Considerations

1. **Async Operations**: All I/O is non-blocking
2. **Connection Pooling**: SQLAlchemy manages database connections
3. **Request Validation**: Fast fail on invalid input
4. **Structured Logging**: Minimal performance impact
5. **Stateless Design**: Horizontal scaling ready

## Security Considerations

1. **Input Validation**: Pydantic validates all inputs
2. **SQL Injection**: SQLAlchemy ORM prevents SQL injection
3. **CORS**: Restricted to configured origins
4. **Error Messages**: No sensitive information in errors
5. **Environment Variables**: Secrets in .env (not committed)

## Monitoring and Observability

1. **Health Check**: `/health` endpoint for load balancers
2. **Structured Logs**: JSON format for log aggregation
3. **Request Logging**: Automatic via Uvicorn
4. **Error Tracking**: Global exception handlers log all errors

## API Documentation

Interactive documentation available at:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## Deployment Checklist

- [ ] Set production environment variables
- [ ] Change LOG_LEVEL to WARNING or ERROR
- [ ] Update CORS_ORIGINS for production domain
- [ ] Use production-grade WSGI server (Gunicorn)
- [ ] Set up database backups
- [ ] Configure reverse proxy (Nginx)
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation (ELK, Datadog)
- [ ] Enable HTTPS/TLS
- [ ] Set resource limits (memory, CPU)

## Maintenance

### Adding New Endpoints

1. Create endpoint in `app/api/v1/endpoints/`
2. Add route to `app/api/v1/router.py`
3. Create Pydantic schemas in `app/schemas/`
4. Add service logic in `app/services/`

### Updating Dependencies

```bash
pip install --upgrade fastapi uvicorn pydantic
pip freeze > requirements.txt
```

### Database Migrations

For schema changes, use Alembic:

```bash
pip install alembic
alembic init migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Support

For issues or questions:

1. Check API documentation at `/api/v1/docs`
2. Review logs in JSON format
3. Run test suite with `python test_api.py`
4. Check health endpoint `/health`