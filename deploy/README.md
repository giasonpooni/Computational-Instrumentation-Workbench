# Deploy the workbench

Two supported prototype deployment paths share the same Python service and protocol: a native Windows backend and a Linux container backend. Godot remains an optional native client. Both preserve recorded evidence and result identities when restoring a saved workspace.

## Native Windows

From the repository root in PowerShell, with Python 3.11+ installed:

```powershell
.\scripts\workbench.ps1 Setup
.\scripts\workbench.ps1 Start
.\scripts\workbench.ps1 Status
```

Pass `-PythonExecutable C:\Path\To\python.exe` to Setup if Python is not on PATH. Setup creates the repository's `.venv` and installs the package. Start runs the backend as a hidden process on `ws://127.0.0.1:8765`. No Windows service or login task is registered. Keep the repository and `.venv` in place while this deployment is running.

The default `.ciw/` directory contains source recordings, result files, `workspace.json`, stdout/stderr logs and `service.json` ownership metadata. It is ignored by Git. You can set `-DataDirectory` and `-Port`; repeat the same data-directory option for subsequent control commands.

```powershell
.venv\Scripts\ciw.exe health
.venv\Scripts\ciw.exe send analysis.spectrum
.venv\Scripts\ciw.exe send workspace.save
.\scripts\workbench.ps1 Stop
.\scripts\workbench.ps1 Start
```

Stop verifies the owned process and session, saves the workspace, verifies the save, and stops only that process. It refuses to kill a different or unresponsive process. Start restores an existing saved workspace; corrupt or incompatible saved data causes an error rather than silently starting a new investigation. Reopening does not rerun saved analyses.

Save explicitly before a power loss or forced termination. The controller's normal Stop path persists state, but the prototype does not yet offer transactional autosave after every request. Before upgrading installed code, stop the backend, back up `.ciw`, run Setup, then Start. Setup must not replace the runtime of a running owned process.

Import `godot/project.godot` using Godot 4.5.2 Standard and run the scene to attach its phase portrait and energy-surface viewport. Terminal commands remain usable when that window is closed.

## Container backend

Install Docker Engine with Compose on Linux, or Docker Desktop with a working Linux backend on Windows/macOS. On Windows, [Docker's installation guide](https://docs.docker.com/desktop/setup/install/windows-install/) describes the WSL 2 prerequisites. Enabling Windows features may require a user-controlled restart.

```text
docker compose up --build --detach --wait
docker compose ps
docker compose logs --tail 50 backend
docker compose exec backend ciw health
docker compose exec backend ciw send analysis.spectrum
docker compose stop
docker compose up --detach --wait
```

The default container endpoint also uses host port 8765. If the native backend is already running, keep it on 8765 and publish the container on 8766:

```powershell
$env:CIW_PORT = '8766'
docker compose up --build --detach --wait
.venv\Scripts\ciw.exe health --url ws://127.0.0.1:8766
```

The named `workspace` volume persists `/data`. Normal container shutdown drains requests and saves `workspace.json`; startup uses `--resume`. `docker compose down` preserves the volume; `down --volumes` deletes it and should not be used on retained investigations. The image runs without root, with a read-only root filesystem and only a loopback host port. See [CONTAINER.md](CONTAINER.md) for the detailed container configuration.

The native and container deployments hold independent sessions and data directories. Identical demo run IDs identify the same deterministic input, not a shared live session. Connect to the intended endpoint explicitly.

## Verification and build artifacts

The repository CI runs numerical/protocol tests on Windows and Linux, imports the Godot project and checks two live clients, builds an installable Python wheel, and builds/runs/restarts an isolated container deployment. The wheel is retained as the `ciw-python-wheel` workflow artifact. A wheel contains the Python backend; obtain the Godot project from the same repository revision.

```text
python -m pytest -q
python scripts/check_godot.py --godot godot
python scripts/check_container.py
```

The Godot check needs a free port 8765. The container check uses a fresh Compose project, temporary host port and dedicated temporary volume, and removes only those test resources afterward. It checks persisted selection/result identity after SIGTERM and restart, without sending an explicit workspace-save request first.

The current deployment is a local synthetic-instrument prototype. It has no remote authentication, multi-user isolation, external device access, guaranteed latency, high availability or certified measurement behavior. Existing instrument and PayloadOS adapters remain the next integration milestone.
