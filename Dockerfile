FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ciw
COPY pyproject.toml README.md LICENSE LICENSE-POLICY.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir . \
    && groupadd --gid 10001 ciw \
    && useradd --uid 10001 --gid 10001 --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin ciw \
    && mkdir /data \
    && chown 10001:10001 /data

USER 10001:10001
WORKDIR /data
EXPOSE 8765
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=7s --start-period=10s --retries=3 \
    CMD ["ciw", "health", "--url", "ws://127.0.0.1:8765"]

CMD ["ciw", "serve", "--bind", "0.0.0.0", "--resume", "--output-dir", "/data"]
