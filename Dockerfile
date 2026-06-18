FROM python:3.12-slim

RUN apt-get update && apt-get install -y git gcc cron poppler-utils tesseract-ocr tesseract-ocr-spa tesseract-ocr-cat antiword

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY . /app

RUN touch /var/log/cron.log

ENV PATH="/app/.venv/bin:$PATH"

CMD ["/bin/sh", "-c", "/etc/init.d/cron start && tail -f /var/log/cron.log"]
