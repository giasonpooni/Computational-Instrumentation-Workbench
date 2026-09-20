# Container backend

Run these commands from the repository root with Docker Engine and Docker Compose v2 available. On Docker Desktop, use Linux containers. The container runs the Python backend; the terminal and optional Godot viewport remain clients on the host.

```text
docker compose build
docker compose up -d --wait --wait-timeout 60
docker compose ps
docker compose exec -T backend ciw health --url ws://127.0.0.1:8765
```

The host endpoint is `ws://127.0.0.1:8765`. By default, a local backend using that port must be stopped before starting this container. Likewise, stop the container before starting a local backend on port 8765. To run both at once, set `CIW_PORT=8766` for Compose and connect container clients to `ws://127.0.0.1:8766`; the container continues to use port 8765 internally. The container deliberately binds `0.0.0.0` inside its network namespace; Compose publishes only the host's loopback address. Keep this mapping: the prototype has no remote authentication.

The Dockerfile installs the package normally, without an editable source mount, using `python:3.12.14-slim-bookworm`. Only package source and build metadata enter the build context. The process runs as UID/GID `10001:10001`, with a read-only root filesystem, writable `/tmp` tmpfs, all Linux capabilities dropped, and privilege escalation disabled. `/data` is created with that user's ownership so Docker can initialize a writable named volume.

To choose port 8766 in PowerShell, run `$env:CIW_PORT = "8766"` before `docker compose up -d --wait --wait-timeout 60`. On Linux/macOS, use `CIW_PORT=8766 docker compose up -d --wait --wait-timeout 60`. Keep the same environment setting for subsequent Compose commands. Host CLI commands then use `--url ws://127.0.0.1:8766`, for example `ciw health --url ws://127.0.0.1:8766` and `ciw send session.get --url ws://127.0.0.1:8766`. Configure the optional viewport to that same host endpoint; commands executed inside the container keep its internal port 8765.

## Use the session

With the host Python environment installed as described in the [quickstart](../docs/quickstart.md), use the same client commands as the local backend:

```text
ciw health --url ws://127.0.0.1:8765
ciw send session.get
ciw watch
```

Godot uses that same loopback endpoint. Closing a client or viewport leaves the backend running. Commands can also run inside the container without installing Python on the host:

```text
docker compose exec -T backend ciw send session.get
docker compose exec -T backend ciw send analysis.stats
docker compose exec -T backend ciw send result.list
docker compose exec -T backend ciw send workspace.save
```

The first run initializes the synthetic oscillator recording when no saved workspace exists. Explicit `analysis.*` requests perform calculations and create new execution and result identities. The health probe makes a bounded, read-only protocol request; it does not run scientific calculations. Its request budget is three seconds, with a bounded connection close, and Docker allows seven seconds for process startup and completion. Docker checks every 30 seconds, allows a ten-second startup period, and marks the container unhealthy after three failed checks. Inspect failures with `docker compose logs --tail 100 backend`. An unhealthy status alone does not trigger a Docker restart; the configured `on-failure:3` policy retries a process that exits with an error at most three times.

## Stop and resume

The `workspace` named volume retains `/data/workspace.json`, the recording, and saved results across container replacement. `--resume` restores the saved selection, including its revision and cursor, and existing result records without scientific recomputation. The runtime session identity can change when the process restarts; retained evidence, execution and result identities do not.

For a normal stop, Docker sends SIGTERM directly to the Python process. Compose allows up to 30 seconds for the service to finish shutdown and save its workspace before forcing termination. SIGINT also requests graceful shutdown. Use process control for shutdown; there is no shutdown request in the client protocol.

```text
docker compose exec -T backend ciw send session.get
docker compose exec -T backend ciw send result.list
docker compose stop backend
docker compose up -d --wait --wait-timeout 60
docker compose exec -T backend ciw send session.get
docker compose exec -T backend ciw send result.list
```

Compare the selection and result identities before and after restart. An immediate restart is also available with `docker compose restart backend`; then wait for `docker compose ps` to report healthy or run `ciw health` before reconnecting clients. A forced kill or power loss cannot guarantee preservation of changes since the last completed workspace save. Use `workspace.save` whenever an explicit checkpoint is needed.

To rebuild after updating the source while keeping the same workspace:

```text
docker compose stop backend
docker compose build
docker compose up -d --wait --wait-timeout 60
```

`docker compose down` removes the container and network while retaining the named volume. Do not add `--volumes` or prune this volume if its saved investigation is needed. Run subsequent commands from the same checkout, or use the same Compose project name, to reconnect to the same project-scoped volume.

## Validation status

Docker Compose configuration validation passed. Docker Desktop is installed in the implementation environment, but its runtime is unavailable while the required WSL2 setup awaits user approval. An image build and container persistence smoke test have therefore not been run locally. The commands above are the deployment checks to run on a Docker host; source-level Python checks do not replace them.

Docker references: [Compose services](https://docs.docker.com/reference/compose-file/services/), [waiting for service health](https://docs.docker.com/reference/cli/docker/compose/up/), and [named-volume lifecycle](https://docs.docker.com/engine/storage/volumes/).
