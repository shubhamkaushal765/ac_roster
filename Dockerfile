FROM gdssingapore/airbase:python-3.13
ENV PYTHONUNBUFFERED=TRUE
# Copy pyproject.toml first
COPY --chown=app:app pyproject.toml ./

# Copy the acroster package directory
COPY --chown=app:app acroster ./acroster

# Install dependencies
RUN pip install --no-cache-dir .
COPY --chown=app:app . ./
USER app
CMD ["python", "nice_app.py"]
