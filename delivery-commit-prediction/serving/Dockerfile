# Build from the use-case folder:
#   docker build -f serving/Dockerfile -t delivery-commit-scoring .
# Run with a trained model mounted (or bake one in by training during build):
#   docker run -p 8000:8000 -v $(pwd)/artifacts/models:/app/artifacts/models delivery-commit-scoring
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY serving ./serving

ENV DELIVERY_COMMIT_MODEL_DIR=/app/artifacts/models
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
