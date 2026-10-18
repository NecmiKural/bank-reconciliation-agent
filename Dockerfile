# The `recon` command in a small image. Dependencies come from uv.lock, so the image runs
# exactly what CI tests. It holds no keys: pass them at run time with --env-file.
FROM python:3.13.16-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.13.16-slim-bookworm
RUN useradd --create-home --uid 10001 recon
COPY --from=build /app/.venv /app/.venv
WORKDIR /home/recon
COPY tests/cassettes ./tests/cassettes
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER recon
ENTRYPOINT ["recon"]
CMD ["--help"]
