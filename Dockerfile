FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY opensesame ./opensesame
RUN pip install --no-cache-dir .

CMD ["python", "-m", "opensesame.server"]
