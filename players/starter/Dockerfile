FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPEN_SESAME_PROFILE=starter
WORKDIR /app

COPY pyproject.toml README.md ./
COPY opensesame ./opensesame
RUN pip install --no-cache-dir .

CMD ["python", "-m", "opensesame.player"]
