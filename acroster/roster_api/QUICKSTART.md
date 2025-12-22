# Quick Start Guide - Officer Roster Optimization API

## 🚀 Get Started in 5 Minutes

### Prerequisites

- Python 3.11+
- pip
- Git (optional)

### Setup Steps

#### 1. Navigate to Project

```bash
cd roster_api
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (default values work out of the box)
```

#### 5. Start Server

```bash
# Option A: Using the run script
./run.sh

# Option B: Direct command
uvicorn acroster.roster_api.main:app --reload --host 0.0.0.0 --port 8000
```

#### 6. Verify Installation

Open your browser:

- API Docs: http://localhost:8000/api/v1/docs
- Health Check: http://localhost:8000/health

### Test the API

#### Using Python

```bash
python test_api.py
```

#### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Generate roster
curl -X POST http://localhost:8000/api/v1/roster/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "arrival",
    "main_officers_reported": "1-18",
    "report_gl_counters": "4AC1, 8AC11",
    "beam_width": 20
  }'
```

## 📦 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or using Docker directly
docker build -t roster-api .
docker run -p 8000:8000 roster-api
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# API Configuration
PROJECT_NAME=Officer Roster Optimization API
API_V1_STR=/api/v1

# CORS (add your frontend URL)
CORS_ORIGINS=["http://localhost:3000"]

# Database
DATABASE_PATH=acroster.db

# Logging
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json       # json or text

# Algorithm
DEFAULT_BEAM_WIDTH=20
MAX_BEAM_WIDTH=100
```

## 📝 Common Operations

### Generate a Roster

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/roster/generate",
    json={
        "mode":                   "arrival",
        "main_officers_reported": "1-18",
        "report_gl_counters":     "4AC1, 8AC11, 12AC21",
        "handwritten_counters":   "3AC12,5AC13",
        "ot_counters":            "2,20,40",
        "ro_ra_officers":         "3RO2100, 11RO1700",
        "sos_timings":            "(AC22)1000-1300, 2000-2200",
        "beam_width":             20,
        "save_to_history":        True
    }
)

data = response.json()
print(f"Officers: {data['officer_counts']}")
print(f"Penalty: {data['optimization_penalty']}")
```

### Get Last Inputs

```python
response = requests.get("http://localhost:8000/api/v1/history/last-inputs")
last_inputs = response.json()['data']
```

### View History

```python
response = requests.get(
    "http://localhost:8000/api/v1/history/history?limit=10"
    )
history = response.json()['data']
```

### Save Edit

```python
response = requests.post(
    "http://localhost:8000/api/v1/edits/",
    json={
        "edit_type":  "delete",
        "officer_id": "M1",
        "slot_start": 0,
        "slot_end":   5,
        "time_start": "1000",
        "time_end":   "1130"
    }
)
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process or use different port
uvicorn acroster.roster_api.main:app --port 8001
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify acroster module is accessible
python -c "import acroster; print('OK')"
```

### Database Issues

```bash
# Database is created automatically
# To reset, simply delete acroster.db
rm acroster.db
# Restart the application
```

### CORS Errors

```bash
# Add your frontend URL to .env
CORS_ORIGINS=["http://localhost:3000","http://your-domain.com"]
```

## 📚 Next Steps

1. **Read Full Documentation**: See `DOCUMENTATION.md`
2. **Explore API**: Visit http://localhost:8000/api/v1/docs
3. **Customize**: Modify `.env` for your needs
4. **Integrate**: Connect your Next.js frontend
5. **Deploy**: Follow production deployment guide

## 🔗 Important URLs

- **API Documentation**: http://localhost:8000/api/v1/docs
- **Alternative Docs**: http://localhost:8000/api/v1/redoc
- **Health Check**: http://localhost:8000/health
- **OpenAPI Schema**: http://localhost:8000/api/v1/openapi.json

## 💡 Tips

1. Use `--reload` flag during development for auto-restart
2. Check logs in JSON format for debugging
3. Use Swagger UI for testing endpoints interactively
4. Keep `.env` out of version control (already in `.gitignore`)
5. Run `python test_api.py` to verify everything works

## 🆘 Getting Help

1. Check logs for error messages
2. Visit `/api/v1/docs` for endpoint documentation
3. Run test suite: `python test_api.py`
4. Review `DOCUMENTATION.md` for detailed information

---

**Ready to use!** The API is now running and accepting requests. 🎉