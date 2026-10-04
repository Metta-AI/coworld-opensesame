FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY opensesame ./opensesame
RUN pip install --no-cache-dir uv==0.11.7 && uv sync --frozen --no-dev --no-editable
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "opensesame.server"]
