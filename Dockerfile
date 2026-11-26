# Reproducible offline CPU inference & evaluation image for MonsoonVerse R1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# Install PyTorch CPU wheel and runtime dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install --no-cache-dir \
      "numpy>=1.26" \
      "scipy>=1.11" \
      "pandas>=2.1" \
      "fastapi>=0.110" \
      "uvicorn>=0.27" \
      "pg8000>=1.31" \
      "python-dotenv>=1.0" \
      "requests>=2.31" \
      "pytest>=8.0" \
      "reportlab>=4.0"

# Copy application source and committed weights/artifacts cleanly (no .env, no .nc)
COPY pyproject.toml README.md brief.pdf serve.py requirements.txt ./
COPY artifacts ./artifacts
COPY backend ./backend
COPY ml ./ml
COPY data_pipeline ./data_pipeline
COPY evaluation ./evaluation
COPY impact ./impact
COPY db ./db
COPY tests ./tests
COPY experiments/run_experiments.py experiments/run_exp3.py experiments/run_exp4.py experiments/run_mission100_exps.py ./experiments/

# Non-root execution user for read-only container operation
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "serve.py", "--port", "8000"]
